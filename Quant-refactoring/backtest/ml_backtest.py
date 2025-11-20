"""
ML 예측 기반 Walk-Forward 백테스트 시스템

이 모듈은 머신러닝 모델을 사용한 실제 백테스트를 수행합니다.
미래 유출(Future Leakage)을 방지하기 위해 Walk-Forward Analysis를 구현합니다.

핵심 원칙:
1. 각 리밸런싱 시점마다 그 시점까지의 데이터만으로 모델을 학습
2. Filing Date(공시일)를 엄격히 준수
3. 미래 데이터는 절대 사용하지 않음

작성자: Quant Trading Team
날짜: 2025-11-20
"""

import logging
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb


class MLBacktest:
    """
    머신러닝 모델을 사용한 Walk-Forward 백테스트

    백테스트 플로우:
    ---------------
    1. 리밸런싱 날짜 목록 생성 (예: 2023-03-13, 2023-06-13, ...)
    2. 각 리밸런싱 날짜마다:
       a. 그 날짜까지 사용 가능한 데이터 로드 (Filing Date 고려)
       b. 모델 학습 (또는 기존 모델 로드)
       c. 예측 수행
       d. 상위 K개 종목 선택
       e. 실제 수익률 계산
    3. 전체 성과 리포트 생성

    미래 유출 방지:
    -------------
    - Filing Date: 재무제표 공시 전에는 해당 데이터 사용 불가
    - Expanding Window: 시작~현재까지 데이터로 학습 (점점 증가)
    - 또는 Rolling Window: 최근 N년 데이터만 사용 (고정 크기)

    Parameters:
    ----------
    config : Dict[str, Any]
        설정 딕셔너리 (conf.yaml에서 로드)
    main_ctx : MainContext
        메인 컨텍스트 (root_path 등 포함)
    rebalance_period : int
        리밸런싱 주기 (개월), 기본값 3
    top_k : int
        선택할 종목 수, 기본값 20
    retrain_frequency : str
        모델 재학습 주기:
        - 'every': 매 리밸런싱마다 재학습 (가장 정확, 느림)
        - 'quarterly': 분기마다 재학습
        - 'yearly': 연도마다 재학습
        - 'once': 한 번만 학습 (현재 방식, 비추천)
    window_type : str
        학습 윈도우 타입:
        - 'expanding': 시작~현재 (점점 증가)
        - 'rolling': 최근 N년만 (고정)
    window_size : Optional[int]
        Rolling window 사용 시 윈도우 크기 (년), 기본값 3
    """

    def __init__(
        self,
        config: Dict[str, Any],
        main_ctx: Any,
        rebalance_period: int = 3,
        top_k: int = 20,
        retrain_frequency: str = 'quarterly',  # 'every', 'quarterly', 'yearly', 'once'
        window_type: str = 'expanding',  # 'expanding' or 'rolling'
        window_size: Optional[int] = 3  # rolling window size in years
    ):
        self.config = config
        self.main_ctx = main_ctx
        self.rebalance_period = rebalance_period
        self.top_k = top_k
        self.retrain_frequency = retrain_frequency
        self.window_type = window_type
        self.window_size = window_size

        # 로깅 설정
        self.logger = logging.getLogger('MLBacktest')

        # 경로 설정
        ml_config = config.get('ML', {})
        self.data_path = Path(main_ctx.root_path) / 'ml_per_year'
        self.model_path = Path(main_ctx.root_path) / 'MODELS_WALKFORWARD'
        self.model_path.mkdir(exist_ok=True)

        # 결과 저장용
        self.backtest_results = []
        self.predictions_history = []

    def _get_available_data_until(self, cutoff_date: datetime) -> pd.DataFrame:
        """
        특정 날짜까지 사용 가능한 데이터 로드 (Filing Date 고려)

        중요: 미래 유출 방지의 핵심 함수!

        Parameters:
        ----------
        cutoff_date : datetime
            기준 날짜 (이 날짜까지만 사용 가능)

        Returns:
        -------
        pd.DataFrame
            사용 가능한 학습 데이터

        예시:
        ----
        cutoff_date = 2023-06-13이라면:
        - 2023 Q1 실적이 2023-05-15에 공시되었다면 → 사용 가능 ✓
        - 2023 Q2 실적이 2023-08-15에 공시되었다면 → 사용 불가 ✗
        """
        all_data = []

        # 분기별 파일을 순회하며 로드
        for year in range(self.main_ctx.start_year, cutoff_date.year + 1):
            for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                file_path = self.data_path / f'rnorm_ml_{year}_{quarter}.parquet'

                if not file_path.exists():
                    continue

                try:
                    df = pd.read_parquet(file_path)

                    # Filing Date 확인
                    # 주의: make_mldata.py에서 이미 filing date를 고려하여
                    # rebalance_date를 할당했지만, 추가 검증
                    if 'fillingDate' in df.columns:
                        df['fillingDate'] = pd.to_datetime(df['fillingDate'])
                        # Filing Date가 cutoff_date 이전인 것만 사용
                        df = df[df['fillingDate'] <= cutoff_date]

                    if not df.empty:
                        all_data.append(df)
                        self.logger.debug(f"Loaded {file_path.name}: {len(df)} rows")

                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
                    continue

        if not all_data:
            raise ValueError(f"No data available until {cutoff_date}")

        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"📊 Available data until {cutoff_date.date()}: {len(combined_data)} rows")

        return combined_data

    def _should_retrain(self, current_date: datetime, last_train_date: Optional[datetime]) -> bool:
        """
        모델을 재학습해야 하는지 판단

        Parameters:
        ----------
        current_date : datetime
            현재 리밸런싱 날짜
        last_train_date : Optional[datetime]
            마지막 학습 날짜

        Returns:
        -------
        bool
            재학습 필요 여부
        """
        if last_train_date is None:
            return True  # 첫 학습

        if self.retrain_frequency == 'every':
            return True  # 매번 재학습

        if self.retrain_frequency == 'quarterly':
            # 분기가 바뀌었는지 확인
            return (current_date.year, (current_date.month - 1) // 3) != \
                   (last_train_date.year, (last_train_date.month - 1) // 3)

        if self.retrain_frequency == 'yearly':
            return current_date.year != last_train_date.year

        if self.retrain_frequency == 'once':
            return False  # 한 번만 학습

        return True

    def _train_model(self, train_data: pd.DataFrame, cutoff_date: datetime) -> Dict[str, Any]:
        """
        모델 학습

        Parameters:
        ----------
        train_data : pd.DataFrame
            학습 데이터
        cutoff_date : datetime
            학습 기준 날짜

        Returns:
        -------
        Dict[str, Any]
            학습된 모델 딕셔너리
            {
                'classifiers': [모델1, 모델2, ...],
                'regressors': [모델1, 모델2, ...],
                'features': [특성명 리스트],
                'scaler': 스케일러
            }
        """
        self.logger.info(f"🔧 Training models with data until {cutoff_date.date()}")
        self.logger.info(f"   Training samples: {len(train_data)}")

        # 특성과 타겟 분리
        # 제외할 컬럼 (메타데이터, 타겟 변수 등)
        exclude_cols = [
            'symbol', 'sector', 'industry', 'exchangeShortName',
            'label', 'price_dev_prediction', 'start_date',
            'fillingDate', 'acceptedDate', 'year_period'
        ]

        feature_cols = [col for col in train_data.columns if col not in exclude_cols]

        X = train_data[feature_cols].copy()
        y = train_data['label'].copy()  # 수익률

        # NaN 처리
        X = X.fillna(0)
        y = y.fillna(0)

        # 스케일링
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # 이진 분류용 타겟 (상승/하락)
        y_binary = (y > 0).astype(int)

        # 모델 학습
        models = {}

        # 분류 모델 (상승/하락 예측)
        self.logger.info("   Training classifier...")
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            tree_method='gpu_hist' if self._is_gpu_available() else 'hist',
            random_state=42
        )
        clf.fit(X_scaled, y_binary)
        models['classifier'] = clf

        # 회귀 모델 (수익률 크기 예측)
        self.logger.info("   Training regressor...")
        reg = XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            tree_method='gpu_hist' if self._is_gpu_available() else 'hist',
            random_state=42
        )
        reg.fit(X_scaled, y)
        models['regressor'] = reg

        models['features'] = feature_cols
        models['scaler'] = scaler

        # 모델 저장
        model_file = self.model_path / f'model_{cutoff_date.strftime("%Y%m%d")}.pkl'
        joblib.dump(models, model_file)
        self.logger.info(f"   ✅ Model saved: {model_file}")

        return models

    def _is_gpu_available(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import cupy
            return True
        except:
            return False

    def _predict(self, models: Dict[str, Any], test_data: pd.DataFrame) -> pd.DataFrame:
        """
        예측 수행

        Parameters:
        ----------
        models : Dict[str, Any]
            학습된 모델
        test_data : pd.DataFrame
            예측할 데이터

        Returns:
        -------
        pd.DataFrame
            예측 결과 포함 데이터프레임
        """
        feature_cols = models['features']
        scaler = models['scaler']

        X = test_data[feature_cols].copy()
        X = X.fillna(0)
        X_scaled = scaler.transform(X)

        # 분류 예측 (상승 확률)
        y_pred_proba = models['classifier'].predict_proba(X_scaled)[:, 1]

        # 회귀 예측 (예상 수익률)
        y_pred_return = models['regressor'].predict(X_scaled)

        # 결과 추가
        result = test_data.copy()
        result['pred_up_proba'] = y_pred_proba
        result['pred_return'] = y_pred_return

        # 최종 점수: 확률 * 예상 수익률
        result['ml_score'] = y_pred_proba * y_pred_return

        return result

    def _select_top_k(self, predictions: pd.DataFrame) -> List[str]:
        """
        상위 K개 종목 선택

        Parameters:
        ----------
        predictions : pd.DataFrame
            예측 결과

        Returns:
        -------
        List[str]
            선택된 종목 코드 리스트
        """
        # ml_score 기준으로 정렬
        sorted_df = predictions.sort_values('ml_score', ascending=False)

        # 상위 K개 선택
        top_k_df = sorted_df.head(self.top_k)

        return top_k_df['symbol'].tolist()

    def _calculate_period_return(
        self,
        selected_symbols: List[str],
        buy_date: datetime,
        sell_date: datetime,
        price_table: pd.DataFrame
    ) -> float:
        """
        기간 수익률 계산

        Parameters:
        ----------
        selected_symbols : List[str]
            선택된 종목 리스트
        buy_date : datetime
            매수 날짜
        sell_date : datetime
            매도 날짜
        price_table : pd.DataFrame
            가격 데이터

        Returns:
        -------
        float
            평균 수익률
        """
        returns = []

        for symbol in selected_symbols:
            symbol_prices = price_table[price_table['symbol'] == symbol]

            # 매수 가격
            buy_price_rows = symbol_prices[symbol_prices['date'] >= buy_date]
            if buy_price_rows.empty:
                continue
            buy_price = buy_price_rows.iloc[0]['close']

            # 매도 가격
            sell_price_rows = symbol_prices[symbol_prices['date'] >= sell_date]
            if sell_price_rows.empty:
                continue
            sell_price = sell_price_rows.iloc[0]['close']

            # 수익률
            ret = (sell_price - buy_price) / buy_price
            returns.append(ret)

        if not returns:
            return 0.0

        return np.mean(returns)

    def run(self) -> pd.DataFrame:
        """
        Walk-Forward 백테스트 실행

        Returns:
        -------
        pd.DataFrame
            백테스트 결과
        """
        self.logger.info("="*80)
        self.logger.info("ML Walk-Forward Backtest Starting")
        self.logger.info("="*80)
        self.logger.info(f"Rebalance period: {self.rebalance_period} months")
        self.logger.info(f"Top K: {self.top_k}")
        self.logger.info(f"Retrain frequency: {self.retrain_frequency}")
        self.logger.info(f"Window type: {self.window_type}")

        # 가격 데이터 로드 (수익률 계산용)
        price_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/price.parquet")
        price_table['date'] = pd.to_datetime(price_table['date'])

        # 리밸런싱 날짜 생성
        start_date = datetime(
            self.config.get('BACKTEST', {}).get('START_YEAR', 2023),
            self.config.get('BACKTEST', {}).get('START_MONTH', 3),
            self.config.get('BACKTEST', {}).get('START_DATE', 13)
        )
        end_date = datetime(
            self.config.get('BACKTEST', {}).get('END_YEAR', 2024),
            12, 31
        )

        rebalance_dates = []
        current = start_date
        while current <= end_date:
            rebalance_dates.append(current)
            current += relativedelta(months=self.rebalance_period)

        self.logger.info(f"\n📅 Rebalance dates: {len(rebalance_dates)}")
        for date in rebalance_dates:
            self.logger.info(f"   {date.date()}")

        # Walk-Forward 백테스트
        last_train_date = None
        current_models = None

        for i, rebalance_date in enumerate(rebalance_dates):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Rebalance #{i+1}: {rebalance_date.date()}")
            self.logger.info(f"{'='*80}")

            # 1. 사용 가능한 데이터 로드 (미래 유출 방지!)
            try:
                available_data = self._get_available_data_until(rebalance_date)
            except ValueError as e:
                self.logger.warning(f"⚠️ {e}, skipping this period")
                continue

            # 2. 모델 재학습 필요 여부 판단
            should_retrain = self._should_retrain(rebalance_date, last_train_date)

            if should_retrain:
                # Window 타입에 따라 학습 데이터 필터링
                if self.window_type == 'rolling' and self.window_size:
                    # Rolling: 최근 N년만
                    cutoff = rebalance_date - relativedelta(years=self.window_size)
                    train_data = available_data[
                        pd.to_datetime(available_data.get('fillingDate', available_data.index)) >= cutoff
                    ]
                else:
                    # Expanding: 전체
                    train_data = available_data

                # 모델 학습
                current_models = self._train_model(train_data, rebalance_date)
                last_train_date = rebalance_date
            else:
                self.logger.info(f"📦 Reusing existing model from {last_train_date.date()}")

            # 3. 예측용 데이터 로드 (현재 시점의 최신 데이터)
            predict_file = self.data_path / f'rnorm_fs_{rebalance_date.year}_Q{(rebalance_date.month-1)//3 + 1}.parquet'
            if not predict_file.exists():
                self.logger.warning(f"⚠️ Prediction file not found: {predict_file}")
                continue

            predict_data = pd.read_parquet(predict_file)

            # 4. 예측 수행
            predictions = self._predict(current_models, predict_data)

            # 5. 상위 K개 선택
            selected_symbols = self._select_top_k(predictions)
            self.logger.info(f"📊 Selected {len(selected_symbols)} stocks")

            # 6. 수익률 계산 (다음 리밸런싱 날짜까지)
            if i < len(rebalance_dates) - 1:
                next_rebalance = rebalance_dates[i + 1]
                period_return = self._calculate_period_return(
                    selected_symbols,
                    rebalance_date,
                    next_rebalance,
                    price_table
                )

                self.logger.info(f"💰 Period return: {period_return*100:.2f}%")

                # 결과 저장
                self.backtest_results.append({
                    'date': rebalance_date,
                    'selected_symbols': selected_symbols,
                    'return': period_return,
                    'retrained': should_retrain
                })

        # 7. 최종 리포트
        results_df = pd.DataFrame(self.backtest_results)
        self._print_summary(results_df)

        # 결과 저장
        output_file = Path('./reports') / f'ml_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        output_file.parent.mkdir(exist_ok=True)
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"\n✅ Results saved: {output_file}")

        return results_df

    def _print_summary(self, results: pd.DataFrame):
        """백테스트 요약 출력"""
        self.logger.info("\n" + "="*80)
        self.logger.info("BACKTEST SUMMARY")
        self.logger.info("="*80)

        total_return = (1 + results['return']).prod() - 1
        avg_return = results['return'].mean()
        std_return = results['return'].std()
        sharpe = avg_return / std_return * np.sqrt(12/self.rebalance_period) if std_return > 0 else 0

        # MDD 계산
        cumulative = (1 + results['return']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        mdd = drawdown.min()

        win_rate = (results['return'] > 0).sum() / len(results)

        self.logger.info(f"Total Periods: {len(results)}")
        self.logger.info(f"Total Return: {total_return*100:.2f}%")
        self.logger.info(f"Average Return: {avg_return*100:.2f}%")
        self.logger.info(f"Std Dev: {std_return*100:.2f}%")
        self.logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        self.logger.info(f"Max Drawdown: {mdd*100:.2f}%")
        self.logger.info(f"Win Rate: {win_rate*100:.1f}%")
        self.logger.info(f"Models Retrained: {results['retrained'].sum()} times")
