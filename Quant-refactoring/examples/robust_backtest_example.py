"""
Robust Backtest Example

RobustBacktester 사용 예제
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

from robust_backtester import RobustBacktester
from models.xgboost_model import XGBoostModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def generate_sample_data(start_date, end_date, n_stocks=100):
    """
    샘플 데이터 생성 (실제로는 실제 데이터를 사용)

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        n_stocks: 종목 수

    Returns:
        데이터프레임
    """
    logging.info("샘플 데이터 생성 중...")

    dates = pd.date_range(start_date, end_date, freq='MS')  # 월별
    data = []

    for date in dates:
        for stock_id in range(n_stocks):
            # 피처 생성 (실제로는 재무제표, 가격 데이터 등)
            features = {
                'date': date,
                'symbol': f'STOCK_{stock_id:03d}',
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn(),
                'feature_3': np.random.randn(),
                'feature_4': np.random.randn(),
                'feature_5': np.random.randn(),
                'feature_6': np.random.randn(),
                'feature_7': np.random.randn(),
                'feature_8': np.random.randn(),
                'feature_9': np.random.randn(),
                'feature_10': np.random.randn(),
                # 타겟: 다음 달 수익률이 양수면 1, 음수면 0
                'target': np.random.choice([0, 1], p=[0.45, 0.55])
            }
            data.append(features)

    df = pd.DataFrame(data)
    logging.info(f"샘플 데이터 생성 완료: {len(df)} rows, {len(dates)} months, {n_stocks} stocks")

    return df


def main():
    """메인 함수"""

    logging.info("="*60)
    logging.info("Robust Backtest Example")
    logging.info("="*60)

    # 1. 데이터 준비
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)

    # 샘플 데이터 생성 (실제로는 실제 데이터 로드)
    df = generate_sample_data(start_date, end_date, n_stocks=100)

    # 2. RobustBacktester 초기화
    backtester = RobustBacktester(
        train_months=24,              # 24개월 데이터로 학습
        test_months=3,                # 3개월 테스트
        retrain_frequency=3,          # 3개월마다 재학습
        cv_splits=3,                  # 3-fold cross-validation
        top_k_features=7,             # 상위 7개 피처 선택
        feature_selection_method='mutual_info',
        performance_window=10,        # 최근 10개 기간 성능 추적
        alert_threshold=0.15          # 15% 성능 저하 시 알림
    )

    # 3. 모델 파라미터 설정
    model_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1
    }

    # 4. 백테스팅 실행
    feature_cols = [f'feature_{i}' for i in range(1, 11)]

    results = backtester.run_backtest(
        model_class=XGBoostModel,
        df=df,
        date_col='date',
        feature_cols=feature_cols,
        target_col='target',
        start_date=start_date,
        end_date=end_date,
        model_params=model_params,
        use_feature_selection=True,   # 피처 선택 사용
        use_cv=True,                  # 교차검증 사용
        use_monitoring=True,          # 성능 모니터링 사용
        save_results=True,
        output_dir='./results'
    )

    # 5. 결과 출력
    logging.info("\n" + "="*60)
    logging.info("최종 결과")
    logging.info("="*60)

    if not results.empty:
        logging.info(f"\n백테스팅 결과:\n{results.to_string()}\n")

        # 선택된 피처
        selected_features = backtester.get_selected_features()
        logging.info(f"선택된 피처 ({len(selected_features)}개):")
        for feat in selected_features:
            logging.info(f"  - {feat}")

        # 성능 요약
        summary = backtester.get_performance_summary()
        logging.info(f"\n성능 요약: {summary}")
    else:
        logging.error("백테스팅 결과가 없습니다.")

    logging.info("\n" + "="*60)
    logging.info("완료")
    logging.info("="*60)


if __name__ == '__main__':
    main()
