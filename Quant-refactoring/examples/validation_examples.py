"""
Validation Examples

각 검증 방법의 개별 사용 예제
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from validation.time_series_cv import TimeSeriesCV, ExpandingWindowCV, RollingWindowCV
from validation.walk_forward import WalkForwardValidator
from monitoring.performance_monitor import PerformanceMonitor
from feature_engineering.feature_selector import FeatureSelector
from models.xgboost_model import XGBoostModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def example_time_series_cv():
    """Time Series Cross-Validation 예제"""
    logging.info("\n" + "="*60)
    logging.info("예제 1: Time Series Cross-Validation")
    logging.info("="*60)

    # 샘플 데이터 생성
    n_samples = 1000
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
    })
    y = pd.Series(np.random.choice([0, 1], n_samples))

    # TimeSeriesCV 사용
    cv = TimeSeriesCV(n_splits=5)

    # 모델 초기화
    model = XGBoostModel(n_estimators=50, max_depth=3)
    model.build_model({})

    # 교차 검증 수행
    avg_scores, all_scores = cv.cross_validate_model(model, X, y)

    logging.info("\n평균 점수:")
    for metric, value in avg_scores.items():
        logging.info(f"  {metric}: {value:.4f}")


def example_expanding_window_cv():
    """Expanding Window CV 예제"""
    logging.info("\n" + "="*60)
    logging.info("예제 2: Expanding Window CV")
    logging.info("="*60)

    # 샘플 데이터
    n_samples = 1000
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
    })

    # Expanding Window CV
    cv = ExpandingWindowCV(
        initial_train_size=200,
        test_size=100,
        step_size=50
    )

    splits = cv.split(X)
    logging.info(f"\n생성된 윈도우 수: {len(splits)}")


def example_rolling_window_cv():
    """Rolling Window CV 예제"""
    logging.info("\n" + "="*60)
    logging.info("예제 3: Rolling Window CV")
    logging.info("="*60)

    # 샘플 데이터
    n_samples = 1000
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
    })

    # Rolling Window CV
    cv = RollingWindowCV(
        train_size=300,
        test_size=100,
        step_size=50
    )

    splits = cv.split(X)
    logging.info(f"\n생성된 윈도우 수: {len(splits)}")


def example_walk_forward():
    """Walk-Forward Validation 예제"""
    logging.info("\n" + "="*60)
    logging.info("예제 4: Walk-Forward Validation")
    logging.info("="*60)

    # 샘플 데이터 생성 (날짜 포함)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='MS')
    n_stocks = 50

    data = []
    for date in dates:
        for stock_id in range(n_stocks):
            data.append({
                'date': date,
                'symbol': f'STOCK_{stock_id}',
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn(),
                'feature_3': np.random.randn(),
                'target': np.random.choice([0, 1])
            })

    df = pd.DataFrame(data)

    # Walk-Forward Validator
    wfv = WalkForwardValidator(
        train_months=12,
        test_months=3,
        retrain_frequency=3,
        anchored=False
    )

    # 백테스팅 실행
    results = wfv.validate(
        model_class=XGBoostModel,
        df=df,
        date_col='date',
        feature_cols=['feature_1', 'feature_2', 'feature_3'],
        target_col='target',
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        model_params={'n_estimators': 50, 'max_depth': 3}
    )

    logging.info(f"\n결과:\n{results}")


def example_performance_monitor():
    """Performance Monitor 예제"""
    logging.info("\n" + "="*60)
    logging.info("예제 5: Performance Monitoring")
    logging.info("="*60)

    # Performance Monitor 초기화
    monitor = PerformanceMonitor(
        window_size=10,
        alert_threshold=0.10,
        drift_threshold=0.05
    )

    # 시뮬레이션: 시간에 따른 성능 변화
    baseline_acc = 0.85
    for period in range(1, 16):
        # 성능이 점진적으로 저하되는 시뮬레이션
        noise = np.random.randn() * 0.02
        degradation = max(0, (period - 5) * 0.01)  # 5번째 기간부터 저하
        current_acc = baseline_acc - degradation + noise

        metrics = {
            'accuracy': max(0.5, current_acc),
            'f1': max(0.5, current_acc - 0.02)
        }

        monitor.update_performance(metrics, period_label=f'Period_{period}')

    # 재학습 필요 여부 확인
    should_retrain = monitor.should_retrain()
    logging.info(f"\n재학습 필요: {should_retrain}")

    # 성능 요약
    monitor.print_summary()


def example_feature_selection():
    """Feature Selection 예제"""
    logging.info("\n" + "="*60)
    logging.info("예제 6: Feature Selection")
    logging.info("="*60)

    # 샘플 데이터 (피처 많음)
    n_samples = 500
    n_features = 100

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([0, 1], n_samples))

    # 다양한 방법으로 피처 선택
    methods = ['mutual_info', 'f_test', 'tree_importance']

    for method in methods:
        logging.info(f"\n방법: {method}")
        logging.info("-" * 40)

        selector = FeatureSelector(
            method=method,
            top_k=10,
            correlation_threshold=0.95
        )

        selected = selector.select_features(X, y)
        logging.info(f"선택된 피처: {selected}")


def main():
    """모든 예제 실행"""
    example_time_series_cv()
    example_expanding_window_cv()
    example_rolling_window_cv()
    example_walk_forward()
    example_performance_monitor()
    example_feature_selection()

    logging.info("\n" + "="*60)
    logging.info("모든 예제 완료")
    logging.info("="*60)


if __name__ == '__main__':
    main()
