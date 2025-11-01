"""
퀀트 트레이딩 시스템용 ML 학습 데이터 생성기

이 모듈은 원시 금융 데이터로부터 머신러닝 학습 데이터셋을 생성합니다.
완전한 데이터 준비 파이프라인을 수행합니다:

1. VIEW 파일에서 금융 데이터 로드 (종목 리스트, 가격, 재무제표, 메트릭)
2. 지정된 간격으로 리밸런싱 날짜 생성 (예: 분기별)
3. 가격 변동 계산 (ML 모델의 타겟 변수)
4. 각 종목별로 12개월 룩백 윈도우 생성
5. tsfresh를 사용한 시계열 특성 추출 (자기상관, FFT, AR 계수 등)
6. 커스텀 재무 비율 계산 (OverMC_*, adaptiveMC_*)
7. RobustScaler를 사용한 특성 정규화 (아웃라이어 저항성)
8. 분기별 ML 데이터셋을 Parquet 파일로 저장

출력 파일:
    - rnorm_fs_{year}_{quarter}.parquet: 타겟 변수 없는 특성 (최신 예측용)
    - rnorm_ml_{year}_{quarter}.parquet: 타겟 변수 포함 특성 (학습/테스트용)

사용 예시:
    from config.context_loader import load_config, MainContext
    from training.make_mldata import AIDataMaker

    config = load_config('config/conf.yaml')
    ctx = MainContext(config)
    maker = AIDataMaker(ctx, config)
    # ML data files created in /data/ml_per_year/

작성자: Quant Trading Team
날짜: 2025-10-29
"""

import datetime
import logging
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters, MinimalFCParameters
from dateutil.relativedelta import relativedelta
from functools import reduce
from config.g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map, cal_timefeature_col_list
from config.logger import get_logger
from sklearn.preprocessing import StandardScaler, RobustScaler
from warnings import simplefilter
import warnings

# 디버깅을 위한 pandas 디스플레이 설정
pd.options.display.width = 30

# 성능 경고 억제
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# 특정 경고 억제 (데이터 검증은 코드에서 수행됨)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*grouping columns.*')


class AIDataMaker:
    """
    재무제표와 가격 데이터로부터 ML 학습 데이터를 생성합니다.

    이 클래스는 완전한 ML 데이터 준비 파이프라인을 조율합니다:
    - VIEW 파일에서 원시 금융 데이터 로드
    - 리밸런싱 날짜 생성
    - 가격 변동 계산 (타겟 변수)
    - tsfresh로 시계열 특성 추출
    - 커스텀 재무 비율 계산
    - RobustScaler로 특성 정규화
    - 분기별 ML 데이터셋을 Parquet 파일로 저장

    출력 데이터셋은 주식 선택을 위한 ML 모델 학습에 바로 사용할 수 있습니다.

    Attributes:
        main_ctx (MainContext): 설정과 경로를 포함한 시스템 컨텍스트
        conf (Dict[str, Any]): YAML에서 로드한 설정 딕셔너리
        logger (logging.Logger): 이 클래스의 로거 인스턴스
        rebalance_period (int): 리밸런싱 간격(개월) (기본값: 3)
        symbol_table (pd.DataFrame): 주식 메타데이터 (종목코드, 섹터, IPO 날짜 등)
        price_table (pd.DataFrame): 과거 가격 및 거래량 데이터
        fs_table (pd.DataFrame): 재무제표 (손익계산서, 대차대조표, 현금흐름표)
        metrics_table (pd.DataFrame): 재무 메트릭 (P/E, ROE, 부채비율 등)
        date_table_list (List): 날짜 테이블 리스트 (사용 안 함, 레거시)
        trade_date_list (List[pd.Timestamp]): 리밸런싱을 위한 실제 거래 날짜

    Class Attributes:
        suffixes_dict (Dict[str, List[str]]): tsfresh 특성 선택을 위한 접미사 필터.
            각 특성 유형에 대해 주요 접미사만 선택하여 특성 차원을 축소합니다
            (예: r=0.0, 0.25, 0.6, 0.9에서의 standard_deviation).

    사용 예시:
        config = load_config('config/conf.yaml')
        ctx = MainContext(config)
        # 2015-2023년도 ML 데이터 생성
        maker = AIDataMaker(ctx, config)
        # 출력: ml_per_year/rnorm_ml_2015_Q1.parquet, rnorm_ml_2015_Q2.parquet, ...

    See Also:
        - config.g_variables: 특성 리스트 및 섹터 매핑
        - training.regressor: 생성된 ML 데이터를 모델 학습에 사용
        - tsfresh 문서: https://tsfresh.readthedocs.io/
    """

    # 차원 축소를 위한 tsfresh 특성 접미사 필터
    # 정보를 유지하면서 특성 수를 줄이기 위해 특정 접미사만 유지
    suffixes_dict = {
        "standard_deviation": ["__r_0.0", "__r_0.25", "__r_0.6", "__r_0.9"],  # 서로 다른 롤링 윈도우
        "quantile": ["__q_0.2", "__q_0.8"],  # 20번째 및 80번째 백분위수
        "autocorrelation": ["__lag_0", "__lag_5", "__lag_9"],  # 서로 다른 지연 기간
        "fft_coefficient": ["__coeff_0", "__coeff_33", "__coeff_99"],  # 주요 푸리에 성분
        "cwt_coefficients": ["__coeff_0", "__coeff_6", "__coeff_12"],  # 웨이블릿 변환 계수
        "symmetry_looking": ["__r_0.0", "__r_0.25", "__r_0.65", "__r_0.9"],  # 서로 다른 스케일의 대칭성
        "ar_coefficient": ["__coeff_0", "__coeff_3", "__coeff_6", "__coeff_10"]  # 자기회귀 계수
    }

    def __init__(self, main_ctx: 'MainContext', conf: Dict[str, Any]) -> None:
        """
        AIDataMaker를 초기화하고 완전한 데이터 준비 파이프라인을 실행합니다.

        파이프라인 실행 순서:
        1. VIEW 파일에서 데이터 로드 (load_bt_table)
        2. 리밸런싱 날짜 생성 (set_date)
        3. 가격 변동 계산 (process_price_table_wdate)
        4. 시계열 특성을 포함한 ML 데이터셋 생성 (make_ml_data)

        Args:
            main_ctx: 설정과 경로를 포함한 시스템 컨텍스트
            conf: DATA, ML, BACKTEST 섹션을 포함한 설정 딕셔너리

        Raises:
            FileNotFoundError: VIEW 파일이 누락된 경우
            ValueError: 유효한 거래 날짜를 찾을 수 없는 경우
        """
        self.main_ctx = main_ctx
        self.conf = conf
        self.logger = get_logger('AIDataMaker')

        # 설정에서 리밸런싱 기간 가져오기 (기본값: 3개월 = 분기별)
        backtest_config = conf.get('BACKTEST', {})
        self.rebalance_period = backtest_config.get('REBALANCE_PERIOD', 3)

        # 데이터 컨테이너 초기화
        self.symbol_table = pd.DataFrame()
        self.price_table = pd.DataFrame()
        self.fs_table = pd.DataFrame()
        self.metrics_table = pd.DataFrame()
        self.date_table_list = []
        self.trade_date_list = []

        # 날짜 테이블을 위한 출력 디렉토리 생성
        self.main_ctx.create_dir(self.main_ctx.root_path + "/DATE_TABLE")

        # 파이프라인 실행
        self.load_bt_table(main_ctx.start_year)
        self.set_date()
        self.process_price_table_wdate()
        self.make_ml_data(main_ctx.start_year, main_ctx.end_year)

    def load_bt_table(self, year: int) -> None:
        """
        VIEW 파일에서 금융 데이터를 로드합니다.

        4가지 유형의 데이터를 로드합니다:
        1. symbol_table: 주식 메타데이터 (종목코드, 섹터, 업종, IPO/상장폐지 날짜)
        2. price_table: 일별 가격 및 거래량 데이터
        3. fs_table: 재무제표 (손익계산서, 대차대조표, 현금흐름표) - 3년 전부터 로드
        4. metrics_table: 재무 메트릭 (P/E, ROE, 부채비율) - 3년 전부터 로드

        3년 룩백은 시계열 특성 추출을 위한 충분한 과거 데이터를 보장합니다.

        Args:
            year: 데이터 로딩 시작 연도 (사용 안 함, start_year-3부터 end_year까지 로드)

        Raises:
            FileNotFoundError: VIEW 디렉토리 또는 필수 파일이 누락된 경우

        Note:
            - 심볼 테이블은 symbol로 중복 제거됨 (첫 번째 발생 유지)
            - 재무제표와 메트릭은 대용량 데이터셋 처리를 위해 연도별로 로드
            - 누락된 파일은 로그에 기록되고 건너뜀 (부분 데이터 로딩 허용)

        TODO:
            - CSV/Parquet 파일의 대안으로 데이터베이스 지원 추가
            - 대용량 데이터셋을 위한 증분 로딩 구현
        """
        # 주식 메타데이터 로드
        self.symbol_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/symbol_list.csv")
        self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
        self.symbol_table['ipoDate'] = pd.to_datetime(self.symbol_table['ipoDate'])
        self.symbol_table['delistedDate'] = pd.to_datetime(self.symbol_table['delistedDate'])

        # 가격 데이터 로드
        self.price_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/price.csv")
        self.price_table['date'] = pd.to_datetime(self.price_table['date'])

        # 재무제표 로드 (시계열 특성을 위해 3년 과거 데이터)
        self.fs_table = pd.DataFrame()
        for year in range(self.main_ctx.start_year-3, self.main_ctx.end_year+1):
            fs_file = self.main_ctx.root_path + "/VIEW/financial_statement_" + str(year) + ".csv"
            if not os.path.exists(fs_file):
                self.logger.warning(f"Financial statement file not found, skipping: {fs_file}")
                print(f"WARNING: Financial statement file not found for year {year}")
                continue
            tmp_fs = pd.read_csv(fs_file,
                                    parse_dates=['fillingDate_x', 'acceptedDate_x'],
                                    dtype={'reportedCurrency_x': str, 'period_x': str,
                                        'link_x': str, 'finalLink_x': str})
            self.fs_table = pd.concat([tmp_fs, self.fs_table])

        # 날짜 컬럼을 datetime으로 변환
        if not self.fs_table.empty:
            self.fs_table['date'] = pd.to_datetime(self.fs_table['date'])
            self.fs_table['fillingDate'] = pd.to_datetime(self.fs_table['fillingDate'])
            self.fs_table['acceptedDate'] = pd.to_datetime(self.fs_table['acceptedDate'])

        # 재무 메트릭 로드 (3년 과거)
        self.metrics_table = pd.DataFrame()
        for year in range(self.main_ctx.start_year-3, self.main_ctx.end_year+1):
            metrics_file = self.main_ctx.root_path + "/VIEW/metrics_" + str(year) + ".csv"
            if not os.path.exists(metrics_file):
                self.logger.warning(f"Metrics file not found, skipping: {metrics_file}")
                print(f"WARNING: Metrics file not found for year {year}")
                continue
            tmp_metrics = pd.read_csv(metrics_file,
                                        dtype={'period_x': str, 'period_y': str})
            self.metrics_table = pd.concat([tmp_metrics, self.metrics_table])

        # 날짜 컬럼을 datetime으로 변환
        if not self.metrics_table.empty:
            self.metrics_table['date'] = pd.to_datetime(self.metrics_table['date'])

    def get_trade_date(self, pdate: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        달력 날짜를 실제 거래 날짜로 변환합니다.

        주어진 날짜 이전 10일 이내의 가장 가까운 거래 날짜(가격 데이터가 있는 날짜)를
        찾습니다. 이는 주말, 공휴일, 시장 휴장일을 처리합니다.

        Args:
            pdate: 목표 달력 날짜

        Returns:
            거래 날짜를 찾으면 반환, 그렇지 않으면 None

        사용 예시:
            # pdate가 일요일이면, 이전 금요일의 거래 날짜를 반환
            trading_date = maker.get_trade_date(pd.Timestamp('2023-01-01'))
        """
        # 목표 날짜 이전 10일 이내의 거래 날짜 검색
        post_date = pdate - relativedelta(days=10)
        res = self.price_table.query("date >= @post_date and date <= @pdate")
        if res.empty:
            return None
        else:
            return res.iloc[0].date

    def generate_date_list(self) -> List[datetime.datetime]:
        """
        리밸런싱을 위한 달력 날짜 리스트를 생성합니다.

        start_year-3부터 end_year까지 고정 간격(rebalance_period)으로 날짜를 생성합니다.
        3년 룩백은 충분한 과거 데이터를 보장합니다.

        Returns:
            rebalance_period 간격의 달력 날짜 리스트

        사용 예시:
            # REBALANCE_PERIOD=3 (분기별)인 경우
            dates = maker.generate_date_list()
            # 반환: [2012-01-01, 2012-04-01, 2012-07-01, 2012-10-01, 2013-01-01, ...]
        """
        date_list = []

        # 설정에서 시작 월/일 가져오기 (기본값: 1월 1일)
        backtest_config = self.conf.get('BACKTEST', {})
        start_month = backtest_config.get('START_MONTH', 1)
        start_date = backtest_config.get('START_DATE', 1)

        # 과거 데이터를 위해 start_year보다 3년 전부터 시작
        date = datetime.datetime(int(self.main_ctx.start_year)-3, start_month, start_date)
        print(date)

        # 사용 가능한 가격 데이터를 초과하지 않도록 함
        recent_date = self.price_table["date"].max()
        end_date = datetime.datetime(self.main_ctx.end_year, 12, 31)
        if end_date > recent_date:
            end_date = recent_date

        # rebalance_period 간격으로 날짜 생성
        while date <= end_date:
            date_list.append(date)
            date += relativedelta(months=self.rebalance_period)

        return date_list

    def set_trade_date_list(self, date_list: List[datetime.datetime]) -> List[pd.Timestamp]:
        """
        달력 날짜를 실제 거래 날짜로 변환합니다.

        각 달력 날짜에 대해 가격 데이터가 있는 가장 가까운 거래 날짜를 찾습니다.
        가격 데이터가 없는 날짜는 건너뜁니다.

        Args:
            date_list: 달력 날짜 리스트

        Returns:
            실제 거래 날짜 리스트

        Raises:
            ValueError: 유효한 거래 날짜를 찾을 수 없는 경우

        Note:
            - 가격 데이터가 없는 날짜는 건너뜀 (오류 아님)
            - 건너뛴 날짜에 대한 경고를 로그에 기록
            - 최소 하나의 유효한 거래 날짜가 필요
        """
        trade_date_list = []
        price_min_date = self.price_table["date"].min()
        price_max_date = self.price_table["date"].max()

        for date in date_list:
            tdate = self.get_trade_date(date)
            if tdate is None:
                # 이 날짜에 대한 가격 데이터가 없음 - 건너뜀
                self.logger.warning(f"⚠️  Cannot find tradable date for {date.strftime('%Y-%m-%d')}")
                self.logger.warning(f"   Price data range: {price_min_date.strftime('%Y-%m-%d')} to {price_max_date.strftime('%Y-%m-%d')}")
                self.logger.warning(f"   Skipping dates before price data is available...")
                print(f"⚠️  WARNING: Skipping {date.strftime('%Y-%m-%d')} - no price data available")
                continue
            trade_date_list.append(tdate)

        if not trade_date_list:
            error_msg = f"❌ FATAL: No valid trading dates found! Price data range: {price_min_date} to {price_max_date}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"✅ Found {len(trade_date_list)} valid trading dates")
        return trade_date_list


    def set_date(self) -> None:
        """
        리밸런싱을 위한 거래 날짜 리스트를 초기화합니다.

        달력 날짜를 생성하고 실제 거래 날짜로 변환합니다.
        데이터 준비 전반에 사용되는 self.trade_date_list를 설정합니다.
        """
        date_list = self.generate_date_list()
        self.trade_date_list = self.set_trade_date_list(date_list)


    def process_price_table_wdate(self) -> None:
        """
        리밸런싱 날짜에 대한 가격 변동(타겟 변수)을 계산합니다.

        처리 단계:
        1. 가격 테이블을 리밸런싱 날짜만으로 필터링
        2. 종목코드와 날짜로 정렬
        3. price_diff 계산: 이전 리밸런싱부터의 절대 가격 변동
        4. volume_mul_price 계산: 유동성 지표 (가격 × 거래량)
        5. price_dev 계산: 가격 변동률 (수익률)

        타겟 변수:
            price_dev = (price_t - price_{t-1}) / price_{t-1}
            이는 이전 리밸런싱부터 현재 리밸런싱까지 주식을 보유하여 달성한 수익률입니다.

        저장:
            참조용으로 VIEW 디렉토리에 price_diff.csv

        Note:
            - 각 종목의 첫 번째 행은 price_diff와 price_dev가 NaN (이전 데이터 없음)
            - 이러한 행은 ML 데이터 생성 중에 필터링됨
        """
        # 리밸런싱 날짜만으로 필터링
        self.price_table = self.price_table[self.price_table['date'].isin(self.trade_date_list)]

        # 순차 계산을 위해 종목코드와 날짜로 정렬
        self.price_table = self.price_table.sort_values(by=['symbol', 'date'])

        # 이전 리밸런싱부터의 절대 가격 변동 계산
        self.price_table['price_diff'] = self.price_table.groupby('symbol')['close'].diff()

        # 유동성 지표 계산 (저유동성 주식 필터링에 사용)
        self.price_table['volume_mul_price'] = self.price_table['close'] * self.price_table['volume']

        # 백분율 수익률 계산 (ML 모델의 타겟 변수)
        # 분모로 이전 종가 사용 (현재 종가 아님)
        self.price_table['price_dev'] = self.price_table['price_diff'] / self.price_table.groupby('symbol')['close'].shift(1)

        # 명확성을 위한 컬럼명 변경
        self.price_table.rename(columns={'close': 'price'}, inplace=True)

        # 참조를 위해 저장
        self.price_table.to_csv(self.main_ctx.root_path + "/VIEW/price_diff.csv", index=False)


    def filter_columns_by_suffixes(self, df: pd.DataFrame) -> List[str]:
        """
        접미사로 tsfresh 컬럼을 필터링하여 차원을 축소합니다.

        tsfresh는 입력 컬럼당 수백 개의 특성을 생성합니다. 이 메서드는
        suffixes_dict에 정의된 특정 접미사를 가진 특성만 유지하도록 필터링하여
        정보를 유지하면서 특성 수를 극적으로 줄입니다.

        Args:
            df: tsfresh 특성이 포함된 DataFrame

        Returns:
            유지할 컬럼 이름 리스트

        사용 예시:
            # 이전: ['revenue_ts_standard_deviation__r_0.0', '__r_0.05', '__r_0.1', ...]
            # 이후: ['revenue_ts_standard_deviation__r_0.0', '__r_0.25', '__r_0.6', '__r_0.9']
            filtered_cols = maker.filter_columns_by_suffixes(features_df)
        """
        filtered_cols = []
        for col in df.columns:
            include_col = True

            # 컬럼이 suffixes_dict의 키워드와 일치하는지 확인
            for keyword, suffixes in self.suffixes_dict.items():
                if keyword in col:
                    # 키워드가 발견되면, 컬럼이 유효한 접미사를 가진 경우만 포함
                    if any(suffix in col for suffix in suffixes):
                        break  # 유효한 접미사 발견, 컬럼 포함
                    else:
                        include_col = False  # 키워드는 있지만 유효한 접미사 없음, 제외
                        break

            if include_col:
                filtered_cols.append(col)

        return filtered_cols

    def filter_dates(self, df: pd.DataFrame, target_col_name: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        날짜 범위로 DataFrame을 필터링합니다.

        end_year의 모든 Q4 데이터를 확보하기 위해 데이터를
        [start_year-01-01, end_year+1-03-01]로 필터링합니다.

        Args:
            df: 필터링할 DataFrame
            target_col_name: 필터링할 날짜 컬럼 이름
            start_year: 시작 연도 (포함)
            end_year: 종료 연도 (포함, 다음 해 3월까지 확장)

        Returns:
            필터링된 DataFrame
        """
        df[target_col_name] = pd.to_datetime(df[target_col_name])
        start_date = pd.Timestamp(year=start_year, month=1, day=1)
        end_date = pd.Timestamp(year=end_year+1, month=3, day=1)  # Q4 신고를 포함하기 위해 확장
        filtered_df = df[(df[target_col_name] >= start_date) & (df[target_col_name] <= end_date)]
        return filtered_df

    def reorder_columns(self, df: pd.DataFrame, keywords: List[str] = ['symbol', 'date']) -> pd.DataFrame:
        """
        주요 컬럼을 앞으로 이동시키기 위해 DataFrame 컬럼을 재정렬합니다.

        Args:
            df: 재정렬할 DataFrame
            keywords: 앞으로 이동시킬 컬럼 키워드

        Returns:
            재정렬된 컬럼을 가진 DataFrame
        """
        key_cols = [col for col in df.columns if any(key.lower() in col.lower() for key in keywords)]
        other_cols = [col for col in df.columns if col not in key_cols]
        new_order = key_cols + other_cols
        return df[new_order]

    def efficient_merge_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame 조인에서 발생한 _x 및 _y 접미사 컬럼을 병합합니다.

        DataFrame 병합 후 중복 컬럼은 _x 및 _y 접미사를 갖게 됩니다.
        이 메서드는 어느 컬럼에서든 null이 아닌 값을 선호하여 다시 병합합니다.

        Args:
            df: _x 및 _y 접미사 컬럼이 있는 DataFrame

        Returns:
            병합된 컬럼을 가진 DataFrame

        사용 예시:
            # 이전: revenue_x, revenue_y
            # 이후: revenue (revenue_x.combine_first(revenue_y)로 결합)
        """
        # 고유한 기본 컬럼 이름 추출
        base_cols = set(col.rstrip('_x').rstrip('_y') for col in df.columns)

        result_df = pd.DataFrame(index=df.index)
        for base in base_cols:
            # 이 기본 이름으로 시작하는 모든 컬럼 찾기
            cols = [col for col in df.columns if col.startswith(base)]
            # combine_first를 사용하여 병합 (첫 번째 null이 아닌 값 선호)
            merged_col = reduce(lambda x, y: df[y].combine_first(x), cols, pd.Series([np.nan]*len(df), index=df.index))
            result_df[base] = merged_col

        result_df = self.reorder_columns(result_df)
        return result_df


    def make_ml_data(self, start_year: int, end_year: int) -> None:
        """
        시계열 특성을 포함한 ML 학습 데이터셋을 생성합니다.

        지정된 연도 범위에 대해 분기별 ML 데이터셋을 생성하는 메인 파이프라인 함수입니다.

        각 연도별 처리 과정:
        1. 고유동성 주식으로 필터링 (거래량 기준 상위 50%)
        2. 종목, 가격, 재무제표, 메트릭 병합
        3. 각 분기(Q1, Q2, Q3, Q4)마다:
           a. 12개월 룩백 윈도우 생성
           b. tsfresh 시계열 특성 추출
           c. 커스텀 재무 비율 계산 (OverMC_*, adaptiveMC_*)
           d. RobustScaler로 정규화
           e. 타겟 변수 계산 (price_dev, price_dev_subavg)
           f. Parquet 파일로 저장

        Args:
            start_year: 처리할 첫 번째 연도
            end_year: 처리할 마지막 연도

        출력 파일:
            각 연도와 분기별로:
            - rnorm_fs_{year}_{quarter}.parquet: 특성만 (최신 예측용)
            - rnorm_ml_{year}_{quarter}.parquet: 특성 + 타겟 (학습용)

        사용 예시:
            maker.make_ml_data(2015, 2023)
            # 생성: rnorm_ml_2015_Q1.parquet, rnorm_ml_2015_Q2.parquet, ...

        Notes:
            - 12개 이상의 분기 데이터가 있는 주식만 처리
            - 저유동성 주식 필터링 (거래량 하위 50%)
            - 데이터가 불충분한 분기는 건너뜀 (경고 로그)
            - 아웃라이어 저항성 정규화를 위해 RobustScaler 사용
            - 섹터 조정 수익률 계산 (sec_price_dev_subavg)

        TODO:
            - 룩백 윈도우 커스터마이즈 옵션 추가 (현재 12로 고정)
            - 유동성 임계값 커스터마이즈 옵션 추가 (현재 50%)
            - 커스텀 특성 추출 파라미터 지원 추가
        """
        # 출력 디렉토리 생성
        ml_dir = os.path.join(self.main_ctx.root_path, "ml_per_year")
        self.main_ctx.create_dir(ml_dir)

        for cur_year in range(start_year, end_year+1):
            # 종목 테이블로 시작
            table_for_ai = self.symbol_table.copy()

            # 가격 데이터를 현재 연도 ± 4년으로 필터링 (시계열 특성용)
            cur_price_table = self.price_table.copy()
            cur_price_table = self.filter_dates(cur_price_table, 'date', cur_year-4, cur_year)

            # 고유동성 주식으로 필터링 (평균 거래 금액 기준 상위 50%)
            symbol_means = cur_price_table.groupby('symbol')['volume_mul_price'].mean().reset_index()
            top_symbols = symbol_means.nlargest(int(len(symbol_means) * 0.50), 'volume_mul_price')
            cur_price_table = cur_price_table[cur_price_table['symbol'].isin(top_symbols['symbol'])]

            # 종목 테이블과 병합
            table_for_ai = pd.merge(table_for_ai, cur_price_table, how='inner', on='symbol')
            table_for_ai.rename(columns={'date': 'rebalance_date'}, inplace=True)

            # 재무제표 준비
            fs = self.fs_table.copy()
            fs = fs[fs['symbol'].isin(top_symbols['symbol'])]
            fs = self.filter_dates(fs, 'fillingDate', cur_year-4, cur_year)
            fs = fs.drop_duplicates(['symbol', 'date'], keep='first')

            # 메트릭 준비
            metrics = self.metrics_table.copy()
            metrics = metrics[metrics['symbol'].isin(top_symbols['symbol'])]
            metrics = metrics.drop_duplicates(['symbol', 'date'], keep='first')

            # 재무제표와 메트릭 병합
            common_columns = fs.columns.intersection(metrics.columns)
            fs = fs.drop(columns=common_columns.difference(['symbol', 'date']))
            fs_metrics = pd.merge(fs, metrics, how='inner', on=['symbol', 'date'])
            fs_metrics = fs_metrics.drop_duplicates(['symbol', 'date'], keep='first')

            # 날짜 준비
            fs_metrics['date'] = pd.to_datetime(fs_metrics['date'])
            fs_metrics.rename(columns={'date': 'report_date'}, inplace=True)
            fs_metrics['fillingDate'] = pd.to_datetime(fs_metrics['fillingDate'])

            # 각 신고 날짜를 다음 리밸런싱 날짜에 매핑
            # 이는 각 리밸런싱에서 사용 가능한 정보만 사용하도록 보장
            date_index = np.sort(pd.DatetimeIndex(self.trade_date_list.copy()))
            indices = np.searchsorted(date_index, fs_metrics['fillingDate'], side='right')
            fs_metrics['rebalance_date'] = [date_index[i] if i < len(date_index) else pd.NaT for i in indices]

            # 시간 순서를 위한 year_period 생성
            # Q1=.2, Q2=.4, Q3=.6, Q4=.8로 소수점 정렬 가능
            fs_metrics = fs_metrics.dropna(subset=['calendarYear'])
            period_map = {'Q1': 0.2, 'Q2': 0.4, 'Q3': 0.6, 'Q4': 0.8}
            fs_metrics['year_period'] = fs_metrics['calendarYear'] + fs_metrics['period'].map(period_map)
            fs_metrics = fs_metrics.sort_values(by=['symbol', 'year_period'])

            # tsfresh를 위한 시간 인덱스 할당 (12분기 윈도우의 경우 0, 1, 2, ..., 11)
            def assign_time(group):
                group = group.sort_values(by='year_period').reset_index(drop=True)
                group['time_for_sort'] = range(len(group))
                return group

            fs_metrics = fs_metrics.groupby('symbol', group_keys=False).apply(assign_time).reset_index(drop=True)

            # 커스텀 비율 계산: OverMC_* (메트릭 / 시가총액)
            for col in meaning_col_list:
                if col not in fs_metrics.columns:
                    continue
                new_col_name = 'OverMC_' + col
                fs_metrics[new_col_name] = np.where(fs_metrics['marketCap'] > 0,
                                                    fs_metrics[col]/fs_metrics['marketCap'], np.nan)

            # 커스텀 비율 계산: adaptiveMC_* (EV / 메트릭)
            # EV (기업가치) = 시가총액 + 순부채
            fs_metrics["adaptiveMC_ev"] = fs_metrics['marketCap'] + fs_metrics["netDebt"]
            for col in cal_ev_col_list:
                new_col_name = 'adaptiveMC_' + col
                fs_metrics[new_col_name] = np.where(fs_metrics[col] > 0,
                                                    fs_metrics['adaptiveMC_ev']/fs_metrics[col], np.nan)

            # 디버깅을 위한 스냅샷 저장
            print("*** fs_metrics w/ rebalance_date")
            print(fs_metrics)
            fs_metrics.head(1000).to_csv(self.main_ctx.root_path + f"/fs_metric_wdate_{str(cur_year)}.csv", index=False)

            # 각 분기 처리
            for quarter_str, quarter in [('Q1', 0.2), ('Q2', 0.4), ('Q3', 0.6), ('Q4', 0.8)]:
                base_year_period = cur_year + quarter

                # 출력 파일 경로
                file_path = os.path.join(self.main_ctx.root_path, "ml_per_year", f"rnorm_fs_{str(cur_year)}_{quarter_str}.parquet")
                file2_path = os.path.join(self.main_ctx.root_path, "ml_per_year", f"rnorm_ml_{str(cur_year)}_{quarter_str}.parquet")

                # 이미 존재하면 건너뛰기
                if os.path.isfile(file2_path):
                    print(f"*** there is parquet file {str(cur_year)}_{quarter_str}")
                    continue

                print(base_year_period)

                # 현재 분기까지의 데이터 가져오기
                filtered_data = fs_metrics[fs_metrics['year_period'] <= float(base_year_period)]

                # 12분기 룩백 윈도우 생성
                def get_last_12_rows(group):
                    return group.tail(12)

                window_data = filtered_data.groupby('symbol', group_keys=False).apply(get_last_12_rows).reset_index(drop=True)
                print(window_data)

                # 데이터가 없으면 건너뛰기
                if window_data.empty:
                    self.logger.warning(f"No data available for {cur_year}_{quarter_str}. Skipping...")
                    print(f"⚠️  WARNING: No data for {cur_year}_{quarter_str} - skipping file generation")
                    continue

                # 12분기 미만의 데이터를 가진 종목 필터링
                symbol_counts = window_data['symbol'].value_counts()
                symbols_to_remove = symbol_counts[symbol_counts < 12].index
                window_data = window_data[~window_data['symbol'].isin(symbols_to_remove)]

                if window_data.empty:
                    self.logger.warning(f"No symbols with sufficient data (12+ rows) for {cur_year}_{quarter_str}. Skipping...")
                    print(f"⚠️  WARNING: No symbols with 12+ data points for {cur_year}_{quarter_str} - skipping")
                    continue

                # tsfresh를 사용하여 시계열 특성 추출
                df_for_extract_feature = pd.DataFrame()

                for target_col in cal_timefeature_col_list:
                    # 유효한 (NaN이 아닌, 유한한) 데이터가 있는 컬럼만 처리
                    if (target_col in window_data.columns and
                        not window_data[target_col].isna().any() and
                        np.isfinite(window_data[target_col]).all()):

                        # tsfresh 형식으로 데이터 준비
                        temp_df = pd.DataFrame({
                            'id': window_data['symbol'],
                            'kind': target_col,
                            'time': window_data['time_for_sort'],
                            'value': window_data[target_col].values,
                            'year_period': window_data['year_period']
                        })
                        df_for_extract_feature = pd.concat([df_for_extract_feature, temp_df])

                if not df_for_extract_feature.empty:
                    # tsfresh로 특성 추출
                    features = extract_features(df_for_extract_feature,
                                               column_id='id',
                                               column_kind='kind',
                                               column_sort='time',
                                               column_value='value',
                                               default_fc_parameters=EfficientFCParameters())

                    # '_ts_' 마커를 포함하도록 컬럼명 변경
                    features = features.rename(columns=lambda x: f"{x.partition('__')[0]}_ts_{x.partition('__')[2]}")

                    # 차원 축소를 위해 접미사로 특성 필터링
                    filtered_columns = self.filter_columns_by_suffixes(features)
                    df_w_time_feature = features[filtered_columns].copy()
                    df_w_time_feature['symbol'] = features.index

                    # 현재 분기만으로 필터링
                    window_data = window_data[window_data['year_period'] == float(base_year_period)]
                    df_w_time_feature = pd.merge(window_data, df_w_time_feature, how='inner', on='symbol')

                    # 절대값 컬럼 제거 (ML에 유용하지 않음, 비율만 중요)
                    abs_col_list = list(set(meaning_col_list) - set(ratio_col_list))
                    for col in abs_col_list:
                        df_w_time_feature = df_w_time_feature.drop([col], axis=1, errors='ignore')

                    # 정규화하지 않을 컬럼 분리
                    excluded_columns = ['symbol', 'rebalance_date', 'report_date', 'fillingDate_x', 'year_period']
                    excluded_df = df_w_time_feature[excluded_columns]

                    # 정규화할 컬럼 선택
                    filtered_columns = [
                        col for col in df_w_time_feature.columns
                        if ('_ts_' in col) or  # 시계열 특성
                        (col in ratio_col_list) or  # 재무 비율
                        col.startswith('OverMC_') or  # 커스텀 비율
                        col.startswith('adaptiveMC_')  # 커스텀 EV 비율
                    ]
                    filtered_df = df_w_time_feature[filtered_columns]

                    # 빈 데이터 체크
                    if filtered_df.empty or len(filtered_df) == 0:
                        self.logger.warning(f"⚠️ No data to scale for {base_year_period}")
                        self.logger.warning(f"   Available columns in df_w_time_feature: {len(df_w_time_feature.columns)}")
                        self.logger.warning(f"   Columns after filtering: {len(filtered_columns)}")
                        self.logger.warning(f"   Rows in filtered_df: {len(filtered_df)}")

                        # 디버깅: 어떤 타입의 컬럼들이 있는지 확인
                        ts_cols = [col for col in df_w_time_feature.columns if '_ts_' in col]
                        ratio_cols = [col for col in df_w_time_feature.columns if col in ratio_col_list]
                        overmc_cols = [col for col in df_w_time_feature.columns if col.startswith('OverMC_')]
                        adaptive_cols = [col for col in df_w_time_feature.columns if col.startswith('adaptiveMC_')]

                        self.logger.warning(f"   Time series features (_ts_): {len(ts_cols)}")
                        self.logger.warning(f"   Ratio columns: {len(ratio_cols)}")
                        self.logger.warning(f"   OverMC_ columns: {len(overmc_cols)}")
                        self.logger.warning(f"   adaptiveMC_ columns: {len(adaptive_cols)}")
                        self.logger.warning(f"   Skipping {base_year_period} due to insufficient data")
                        continue

                    # RobustScaler로 정규화 (아웃라이어에 강함)
                    scaler = RobustScaler()
                    scaled_data = scaler.fit_transform(filtered_df)
                    scaled_df = pd.DataFrame(scaled_data, columns=filtered_df.columns)

                    # 정규화된 컬럼과 제외된 컬럼 결합
                    scaled_df = pd.concat([excluded_df, scaled_df], axis=1)

                    # 타겟 변수 없이 특성 저장 (최신 예측용)
                    symbol_industry = table_for_ai[['symbol', 'industry', 'volume_mul_price']]
                    symbol_industry = symbol_industry.drop_duplicates('symbol', keep='first')
                    fs_df = pd.merge(symbol_industry, scaled_df, how='inner', on=['symbol'])
                    fs_df["sector"] = fs_df["industry"].map(sector_map)
                    fs_df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)

                    # 학습 데이터를 위한 타겟 변수 추가
                    cur_table_for_ai = pd.merge(table_for_ai, scaled_df, how='inner', on=['symbol','rebalance_date'])
                    cur_table_for_ai["sector"] = cur_table_for_ai["industry"].map(sector_map)

                    # 시장 조정 수익률 계산
                    cur_table_for_ai['price_dev_subavg'] = cur_table_for_ai['price_dev'] - cur_table_for_ai['price_dev'].mean()

                    # 섹터 조정 수익률 계산
                    sector_list = list(cur_table_for_ai['sector'].unique())
                    sector_list = [x for x in sector_list if str(x) != 'nan']
                    for sec in sector_list:
                        sec_mask = cur_table_for_ai['sector'] == sec
                        sec_mean = cur_table_for_ai.loc[sec_mask, 'price_dev'].mean()
                        cur_table_for_ai.loc[sec_mask, 'sec_price_dev_subavg'] = cur_table_for_ai.loc[sec_mask, 'price_dev'] - sec_mean

                    # 타겟 변수가 포함된 완전한 데이터셋 저장
                    cur_table_for_ai.to_parquet(file2_path, engine='pyarrow', compression='snappy', index=False)
                    self.logger.info(f"✅ Saved ML data: {os.path.basename(file2_path)}")
                else:
                    # df_for_extract_feature가 비어있는 경우
                    self.logger.warning(f"⚠️ No features to extract for {base_year_period}")
                    self.logger.warning(f"   All time series data was filtered out (NaN or non-finite values)")
                    self.logger.warning(f"   Skipping {base_year_period} due to no extractable features")
                    continue
