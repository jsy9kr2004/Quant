"""
Robust Backtester

개선된 백테스팅 시스템
- Cross-validation
- Walk-forward validation
- Performance monitoring
- Feature selection
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from validation.time_series_cv import TimeSeriesCV
from validation.walk_forward import WalkForwardValidator
from monitoring.performance_monitor import PerformanceMonitor
from feature_engineering.feature_selector import FeatureSelector


class RobustBacktester:
    """
    개선된 백테스팅 시스템

    특징:
    1. Time Series Cross-Validation
    2. Walk-Forward Validation (모델 재학습)
    3. 성능 모니터링 및 성능 저하 감지
    4. 피처 선택 (과적합 방지)
    """

    def __init__(self,
                 train_months: int = 24,
                 test_months: int = 3,
                 retrain_frequency: int = 3,
                 cv_splits: int = 5,
                 top_k_features: int = 50,
                 feature_selection_method: str = 'mutual_info',
                 performance_window: int = 10,
                 alert_threshold: float = 0.1):
        """
        Args:
            train_months: 학습 데이터 기간 (개월)
            test_months: 테스트 데이터 기간 (개월)
            retrain_frequency: 재학습 빈도 (개월)
            cv_splits: Cross-validation fold 수
            top_k_features: 선택할 피처 개수
            feature_selection_method: 피처 선택 방법
            performance_window: 성능 모니터링 윈도우 크기
            alert_threshold: 성능 저하 알림 임계값
        """
        # 1. 교차 검증
        self.cv = TimeSeriesCV(n_splits=cv_splits)

        # 2. Walk-forward validation
        self.wfv = WalkForwardValidator(
            train_months=train_months,
            test_months=test_months,
            retrain_frequency=retrain_frequency,
            anchored=False  # Rolling window
        )

        # 3. 성능 모니터링
        self.monitor = PerformanceMonitor(
            window_size=performance_window,
            alert_threshold=alert_threshold
        )

        # 4. 피처 선택
        self.feature_selector = FeatureSelector(
            method=feature_selection_method,
            top_k=top_k_features
        )

        self.selected_features = None
        self.results = []

    def run_backtest(self,
                    model_class: Any,
                    df: pd.DataFrame,
                    date_col: str,
                    feature_cols: List[str],
                    target_col: str,
                    start_date: datetime,
                    end_date: datetime,
                    model_params: Optional[Dict] = None,
                    use_feature_selection: bool = True,
                    use_cv: bool = True,
                    use_monitoring: bool = True,
                    save_results: bool = True,
                    output_dir: str = './results') -> pd.DataFrame:
        """
        강건한 백테스팅 실행

        Args:
            model_class: 모델 클래스
            df: 전체 데이터프레임
            date_col: 날짜 컬럼명
            feature_cols: 피처 컬럼 리스트
            target_col: 타겟 컬럼명
            start_date: 시작 날짜
            end_date: 종료 날짜
            model_params: 모델 파라미터
            use_feature_selection: 피처 선택 사용 여부
            use_cv: 교차검증 사용 여부
            use_monitoring: 성능 모니터링 사용 여부
            save_results: 결과 저장 여부
            output_dir: 결과 저장 디렉토리

        Returns:
            백테스팅 결과 데이터프레임
        """
        if model_params is None:
            model_params = {}

        logging.info(f"\n{'='*60}")
        logging.info("🚀 Robust Backtesting 시작")
        logging.info(f"{'='*60}")
        logging.info(f"기간: {start_date} ~ {end_date}")
        logging.info(f"피처 수: {len(feature_cols)}")
        logging.info(f"피처 선택: {use_feature_selection}")
        logging.info(f"교차검증: {use_cv}")
        logging.info(f"성능 모니터링: {use_monitoring}")
        logging.info(f"{'='*60}\n")

        # 1단계: 피처 선택 (선택 사항)
        if use_feature_selection:
            logging.info(f"\n{'='*60}")
            logging.info("1️⃣ 피처 선택")
            logging.info(f"{'='*60}")

            # 전체 데이터에서 샘플링하여 피처 선택
            sample_mask = (df[date_col] >= start_date) & \
                         (df[date_col] < start_date + pd.DateOffset(months=12))
            sample_df = df[sample_mask]

            if len(sample_df) > 0:
                X_sample = sample_df[feature_cols]
                y_sample = sample_df[target_col]

                # NaN 제거
                valid_mask = ~(X_sample.isnull().any(axis=1) | y_sample.isnull())
                X_sample = X_sample[valid_mask]
                y_sample = y_sample[valid_mask]

                self.selected_features = self.feature_selector.select_features(
                    X_sample, y_sample
                )
            else:
                logging.warning("샘플 데이터가 없어 피처 선택을 건너뜁니다.")
                self.selected_features = feature_cols
        else:
            self.selected_features = feature_cols

        # 2단계: Walk-Forward Validation
        logging.info(f"\n{'='*60}")
        logging.info("2️⃣ Walk-Forward Validation")
        logging.info(f"{'='*60}")

        windows = self.wfv.generate_windows(start_date, end_date)
        results = []

        for i, window in enumerate(windows, 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Window {i}/{len(windows)}")
            logging.info(f"  Train: {window['train_start'].strftime('%Y-%m-%d')} ~ {window['train_end'].strftime('%Y-%m-%d')}")
            logging.info(f"  Test:  {window['test_start'].strftime('%Y-%m-%d')} ~ {window['test_end'].strftime('%Y-%m-%d')}")
            logging.info(f"{'='*60}")

            # 데이터 분할
            train_mask = (df[date_col] >= window['train_start']) & \
                        (df[date_col] < window['train_end'])
            test_mask = (df[date_col] >= window['test_start']) & \
                       (df[date_col] < window['test_end'])

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                logging.warning(f"  ⚠️ Empty data in window {i}, skipping...")
                continue

            try:
                # 피처 및 타겟 추출
                X_train = train_df[self.selected_features]
                y_train = train_df[target_col]
                X_test = test_df[self.selected_features]
                y_test = test_df[target_col]

                # NaN 제거
                train_valid_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
                test_valid_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())

                X_train = X_train[train_valid_mask]
                y_train = y_train[train_valid_mask]
                X_test = X_test[test_valid_mask]
                y_test = y_test[test_valid_mask]

                if len(X_train) == 0 or len(X_test) == 0:
                    logging.warning(f"  ⚠️ No valid data after NaN removal, skipping...")
                    continue

                logging.info(f"  Train samples: {len(X_train):,}")
                logging.info(f"  Test samples:  {len(X_test):,}")

                # 모델 재학습 ⭐
                model = model_class(**model_params)

                if use_cv:
                    # 교차 검증
                    logging.info(f"  🔄 교차 검증 중...")
                    cv_scores, _ = model.cross_validate(
                        X_train, y_train, cv_splits=3, verbose=False
                    )
                    logging.info(f"  📊 CV 결과: {cv_scores}")

                # 학습
                logging.info(f"  🔄 모델 학습 중...")
                model.fit(X_train, y_train, verbose=0)

                # 평가
                metrics = model.evaluate(X_test, y_test)

                result = {
                    'window': i,
                    'train_start': window['train_start'],
                    'train_end': window['train_end'],
                    'test_start': window['test_start'],
                    'test_end': window['test_end'],
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    **metrics
                }

                if use_cv:
                    for k, v in cv_scores.items():
                        result[f'cv_{k}'] = v

                results.append(result)

                logging.info(f"  📊 Test Performance:")
                for metric, value in metrics.items():
                    logging.info(f"     {metric}: {value:.4f}")

                # 3단계: 성능 모니터링 (선택 사항)
                if use_monitoring:
                    period_label = window['test_start'].strftime('%Y-%m')
                    self.monitor.update_performance(metrics, period_label)

                    # 피처 드리프트 체크
                    drift_features = self.monitor.check_feature_drift(
                        X_test.values,
                        feature_names=self.selected_features
                    )

                    if drift_features:
                        logging.warning(f"  ⚠️ {len(drift_features)}개 피처에서 드리프트 감지")

                    # 재학습 필요 여부
                    if self.monitor.should_retrain():
                        logging.warning(f"  🔄 성능 저하로 인한 재학습 권장!")

            except Exception as e:
                logging.error(f"  ❌ Error in window {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 결과 요약
        if not results:
            logging.error("No valid results from backtesting!")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        logging.info(f"\n{'='*60}")
        logging.info("3️⃣ 백테스팅 결과 요약")
        logging.info(f"{'='*60}")

        # 평균 성능
        metric_cols = [col for col in results_df.columns if col not in [
            'window', 'train_start', 'train_end', 'test_start', 'test_end',
            'train_samples', 'test_samples'
        ]]

        logging.info("\n📊 평균 성능:")
        for metric in metric_cols:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            logging.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

        # 성능 모니터링 요약
        if use_monitoring:
            logging.info(f"\n{'='*60}")
            logging.info("4️⃣ 성능 모니터링 요약")
            logging.info(f"{'='*60}")
            self.monitor.print_summary()

        # 결과 저장
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 백테스팅 결과 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = output_path / f'backtest_results_{timestamp}.csv'
            results_df.to_csv(results_file, index=False)
            logging.info(f"\n💾 결과 저장: {results_file}")

            # 선택된 피처 저장
            if use_feature_selection:
                features_file = output_path / f'selected_features_{timestamp}.csv'
                self.feature_selector.save_selected_features(str(features_file))

            # 성능 알림 저장
            if use_monitoring:
                alerts_df = self.monitor.get_alerts_dataframe()
                if not alerts_df.empty:
                    alerts_file = output_path / f'performance_alerts_{timestamp}.csv'
                    alerts_df.to_csv(alerts_file, index=False)
                    logging.info(f"💾 알림 저장: {alerts_file}")

        self.results = results_df
        return results_df

    def get_results(self) -> pd.DataFrame:
        """백테스팅 결과 반환"""
        return self.results

    def get_selected_features(self) -> List[str]:
        """선택된 피처 반환"""
        return self.selected_features

    def get_performance_summary(self) -> Dict:
        """성능 요약 반환"""
        return self.monitor.get_performance_summary()
