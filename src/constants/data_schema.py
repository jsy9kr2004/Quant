"""
데이터 스키마 및 열 정의

이 모듈은 시스템 전체에서 사용되는 모든 열 이름, 메타데이터 정의, 데이터 계약에 대한
단일 진실 공급원(SINGLE SOURCE OF TRUTH)을 제공합니다.

중요: regressor.py와 ml_backtest.py가 동일한 열 정의를 사용하도록 보장하여
열 이름 불일치로 인한 버그를 방지합니다.

작성자: Quant Trading Team
날짜: 2025-12-01
"""

from typing import List, Set


class DataSchema:
    """
    통합 데이터 스키마 정의입니다.

    이 클래스는 ML 파이프라인에서 사용되는 모든 열 이름과 메타데이터를 정의합니다.

    목적:
    --------
    - 열 정의에 대한 단일 진실 공급원
    - regressor.py와 ml_backtest.py 간 열 이름 불일치 방지
    - 모든 모듈에서 일관된 feature engineering 지원

    사용 예시:
    ------
    # feature에서 제외할 열 가져오기
    excluded = DataSchema.get_excluded_cols()

    # DataFrame에서 feature 열 가져오기
    features = DataSchema.get_feature_cols(df)

    # 회귀 타겟 열 가져오기
    target = DataSchema.REGRESSION_TARGET
    """

    # ========================================================================
    # 메타데이터 열 (종목 식별자 및 기업 정보)
    # ========================================================================
    METADATA_COLS: List[str] = [
        "symbol",              # 종목 티커 심볼 (예: AAPL, TSLA)
        "exchangeShortName",   # 거래소 이름 (예: NASDAQ, NYSE)
        "type",                # 증권 유형 (예: 'stock', 'ETF')
        "delistedDate",        # 상장폐지 날짜 (해당되는 경우)
        "industry",            # 산업 분류
        "ipoDate",             # IPO 날짜
        "sector",              # 섹터 분류 (예: Technology, Financial)
    ]

    # ========================================================================
    # 날짜 열 (시간 정보)
    # ========================================================================
    DATE_COLS: List[str] = [
        "rebalance_date",      # 포트폴리오 리밸런싱 날짜
        "report_date",         # 재무 보고 날짜
        "fillingDate",         # 공시 날짜 (SEC 제출)
        "fillingDate_x",       # 공시 날짜 변형 (데이터 병합에서 발생)
        "acceptedDate",        # 보고서 승인 날짜
        "start_date",          # 기간 시작 날짜
        "year_period",         # 연도 기간 식별자
        "date",                # 일반 날짜 열
    ]

    # ========================================================================
    # 가격/거래량 열 (시장 데이터 - feature로 사용하지 않음)
    # ========================================================================
    # 미래 정보를 포함하거나 타겟/수익률 계산에만 사용되므로 제외
    PRICE_VOLUME_COLS: List[str] = [
        "price",               # 주가 (미래 유출 위험)
        "adjClose",            # 분할/배당 조정 종가 (수익률 계산용)
        "volume",              # 거래량 (미래 유출 위험)
        "marketCap",           # 시가총액 (주가에서 파생)
        "price_diff",          # 가격 차이 (타겟 관련)
        "volume_mul_price",    # 거래량 * 주가 (필터링에만 사용)
    ]

    # ========================================================================
    # 타겟 변수 열 (ML 모델 레이블)
    # ========================================================================
    TARGET_COLS: List[str] = [
        "price_dev",              # 이진 타겟: 가격 편차 (상승/하락)
        "price_dev_subavg",       # 주요 회귀 타겟: 가격 편차 - 평균
        "sec_price_dev_subavg",   # 섹터 조정 가격 편차
        "price_dev_prediction",   # 모델 예측 출력
        "price_diff_prediction",  # 가격 차이 예측
        "price_diff_3month",      # 3개월 가격 차이
        "price_diff_6month",      # 6개월 가격 차이
    ]

    # ========================================================================
    # 타겟 변수 이름 (단일 진실 공급원)
    # ========================================================================
    REGRESSION_TARGET = "price_dev_subavg"        # 주요 회귀 타겟
    CLASSIFICATION_TARGET = "price_dev"           # 이진 분류 타겟
    SECTOR_REGRESSION_TARGET = "sec_price_dev_subavg"  # 섹터별 타겟

    # ========================================================================
    # 예측 열 (ML 모델 출력)
    # ========================================================================
    PRED_RETURN = 'pred_return'           # 예측 수익률 (regressor 출력)
    PRED_PROBA = 'pred_proba'             # 예측 확률 (classifier 출력, 0~1)
    ML_SCORE = 'ml_score'                 # 최종 점수 (pred_proba x pred_return)
    RANK = 'rank'                         # ml_score 기준 순위 (1, 2, 3, ...)
    SELECTED = 'selected'                 # Top-K 선택 여부 (True/False)
    PRED_LABEL = 'pred_label'             # 예측 레이블 (UP/DOWN)

    # ========================================================================
    # 백테스트 열 (실제 거래 결과)
    # ========================================================================
    ACTUAL_RETURN = 'actual_return'       # 실제 수익률 ((매도가-매수가)/매수가)
    BUY_PRICE = 'buy_price'               # 진입 시 매수가
    SELL_PRICE = 'sell_price'             # 청산 시 매도가
    BUY_DATE = 'buy_date'                 # 실제 매수일 (거래일 조정)
    SELL_DATE = 'sell_date'               # 실제 매도일 (거래일 조정)
    HOLDING_DAYS = 'holding_days'         # 보유 기간 (일)

    # ========================================================================
    # 평가 열 (예측 vs 실제)
    # ========================================================================
    PREDICTION_ERROR = 'prediction_error' # abs(pred_return - actual_return)
    DIRECTION_CORRECT = 'direction_correct'  # 방향 예측 정확 여부 (True/False)
    ACTUAL_LABEL = 'actual_label'         # 실제 레이블 (UP/DOWN)

    # ========================================================================
    # 추가 정보 열
    # ========================================================================
    COMPANY_NAME = 'company_name'         # 기업명 (사용자 친화적 표시)

    # ========================================================================
    # 클래스 메서드
    # ========================================================================

    @classmethod
    def get_excluded_cols(cls) -> List[str]:
        """
        모델 feature에서 제외해야 하는 모든 열을 가져옵니다.

        Returns:
            List[str]: 모든 메타데이터, 날짜, 가격/거래량, 타겟 열의 통합 리스트

        사용 예시:
            excluded = DataSchema.get_excluded_cols()
            features = [col for col in df.columns if col not in excluded]
        """
        return (
            cls.METADATA_COLS +
            cls.DATE_COLS +
            cls.PRICE_VOLUME_COLS +
            cls.TARGET_COLS
        )

    @classmethod
    def get_excluded_cols_set(cls) -> Set[str]:
        """
        빠른 조회를 위해 제외 열을 set으로 가져옵니다.

        Returns:
            Set[str]: 모든 제외 열 이름의 set
        """
        return set(cls.get_excluded_cols())

    @classmethod
    def get_feature_cols(cls, df) -> List[str]:
        """
        DataFrame에서 feature 열 이름을 추출합니다.

        이 메서드는 모든 메타데이터, 날짜, 가격/거래량, 타겟 열을 자동으로 제외하고
        ML 모델의 feature로 사용해야 하는 열만 반환합니다.

        Args:
            df (pd.DataFrame): 입력 DataFrame

        Returns:
            List[str]: feature 열 이름 리스트

        사용 예시:
            feature_cols = DataSchema.get_feature_cols(train_df)
            X = train_df[feature_cols]
        """
        excluded = cls.get_excluded_cols_set()
        return [col for col in df.columns if col not in excluded]

    @classmethod
    def validate_dataframe(cls, df, require_target: bool = True) -> dict:
        """
        DataFrame이 필수 열을 포함하고 있는지 검증합니다.

        Args:
            df (pd.DataFrame): 검증할 DataFrame
            require_target (bool): True인 경우 회귀 타겟 존재 여부 확인

        Returns:
            dict: {
                'valid': bool,
                'missing_cols': List[str],
                'warnings': List[str]
            }
        """
        result = {
            'valid': True,
            'missing_cols': [],
            'warnings': []
        }

        # 필수 타겟 확인
        if require_target and cls.REGRESSION_TARGET not in df.columns:
            result['valid'] = False
            result['missing_cols'].append(cls.REGRESSION_TARGET)

        # 메타데이터 열 확인 (선택 사항이지만 권장)
        for col in ['symbol', 'sector']:
            if col not in df.columns:
                result['warnings'].append(f"Recommended column '{col}' not found")

        return result

    @classmethod
    def get_target_column(cls, target_type: str = 'regression') -> str:
        """
        작업 유형에 따른 타겟 열 이름을 가져옵니다.

        Args:
            target_type (str): 'regression', 'classification', 또는 'sector'

        Returns:
            str: 타겟 열 이름

        Raises:
            ValueError: target_type이 유효하지 않은 경우
        """
        if target_type == 'regression':
            return cls.REGRESSION_TARGET
        elif target_type == 'classification':
            return cls.CLASSIFICATION_TARGET
        elif target_type == 'sector':
            return cls.SECTOR_REGRESSION_TARGET
        else:
            raise ValueError(
                f"Invalid target_type: {target_type}. "
                f"Must be 'regression', 'classification', or 'sector'"
            )

    @classmethod
    def get_detailed_trades_columns(cls) -> List[str]:
        """
        Detailed Trades 레포트 시트의 열 순서를 가져옵니다.

        통합 레포트의 "Detailed Trades" 시트에 대한 표준 열 순서를 정의하며,
        regressor 예측과 백테스트 실제 결과를 결합합니다.

        Returns:
            List[str]: Detailed Trades 시트의 정렬된 열 이름 리스트

        사용 예시:
            detailed_df = detailed_df[DataSchema.get_detailed_trades_columns()]
        """
        return [
            # 기본 정보
            'rebalance_date',
            'symbol',
            'company_name',
            'sector',

            # 선택 정보
            'rank',
            'selected',

            # 예측 (regressor에서)
            'pred_return',
            'pred_proba',
            'ml_score',

            # 실제 결과 (백테스트에서)
            'actual_return',
            'buy_price',
            'sell_price',
            'buy_date',
            'sell_date',
            'holding_days',

            # 평가
            'prediction_error',
            'direction_correct'
        ]

    @classmethod
    def summary(cls) -> str:
        """
        데이터 스키마의 요약을 가져옵니다.

        Returns:
            str: 서식이 적용된 요약 문자열
        """
        return f"""
Data Schema Summary
===================
Metadata columns: {len(cls.METADATA_COLS)}
Date columns: {len(cls.DATE_COLS)}
Price/Volume columns: {len(cls.PRICE_VOLUME_COLS)}
Target columns: {len(cls.TARGET_COLS)}
Total excluded columns: {len(cls.get_excluded_cols())}

Primary Targets:
- Regression: {cls.REGRESSION_TARGET}
- Classification: {cls.CLASSIFICATION_TARGET}
- Sector Regression: {cls.SECTOR_REGRESSION_TARGET}

Detailed Trades Columns: {len(cls.get_detailed_trades_columns())}
"""


# ============================================================================
# 하위 호환성 별칭
# ============================================================================
# 기존 코드에서 점진적 마이그레이션을 허용합니다

# regressor.py의 기존 이름: y_col_list
# 새 통합 이름: DataSchema.get_excluded_cols()
y_col_list = DataSchema.get_excluded_cols()

# ml_backtest.py의 기존 이름: exclude_cols
# 새 통합 이름: DataSchema.get_excluded_cols()
exclude_cols = DataSchema.get_excluded_cols()


if __name__ == "__main__":
    # 자체 테스트 및 문서 출력
    print(DataSchema.summary())
    print("\nExcluded columns:")
    for i, col in enumerate(DataSchema.get_excluded_cols(), 1):
        print(f"  {i:2d}. {col}")

    print(f"\nTotal: {len(DataSchema.get_excluded_cols())} columns excluded from features")
