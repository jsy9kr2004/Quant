"""
Comprehensive Example

3가지 주요 기능 통합 예제:
1. 리밸런싱 기간 최적화
2. 모델 성능 비교
3. 섹터별 모델 + 통합 선택
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

from optimization.rebalance_optimizer import RebalancingOptimizer
from optimization.model_comparator import ModelComparator
from strategy.sector_ensemble import SectorEnsemble, create_default_sector_configs
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def generate_sample_data_with_sector(start_date, end_date, n_stocks_per_sector=20):
    """
    섹터 정보가 포함된 샘플 데이터 생성

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        n_stocks_per_sector: 섹터당 주식 수

    Returns:
        데이터프레임
    """
    logging.info("섹터별 샘플 데이터 생성 중...")

    sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Industrial']
    dates = pd.date_range(start_date, end_date, freq='MS')  # 월별
    data = []

    for date in dates:
        for sector in sectors:
            for stock_id in range(n_stocks_per_sector):
                symbol = f'{sector[:4].upper()}_{stock_id:03d}'

                # 섹터별로 다른 피처 분포
                if sector == 'Technology':
                    # Tech는 R&D 비중이 높음
                    rd_ratio = np.random.uniform(0.1, 0.3)
                    revenue_growth = np.random.uniform(0.1, 0.5)
                elif sector == 'Financial':
                    # Finance는 부채비율이 중요
                    debt_equity = np.random.uniform(0.5, 2.0)
                    roe = np.random.uniform(0.05, 0.20)
                else:
                    rd_ratio = np.random.uniform(0.01, 0.1)
                    revenue_growth = np.random.uniform(0.0, 0.3)
                    debt_equity = np.random.uniform(0.3, 1.5)
                    roe = np.random.uniform(0.03, 0.15)

                # 피처 생성
                features = {
                    'date': date,
                    'symbol': symbol,
                    'sector': sector,
                    # 공통 피처
                    'revenue': np.random.uniform(1e6, 1e9),
                    'netIncome': np.random.uniform(1e5, 1e8),
                    'operatingCashFlow': np.random.uniform(1e5, 1e8),
                    'freeCashFlow': np.random.uniform(1e5, 1e8),
                    'totalAssets': np.random.uniform(1e6, 1e10),
                    'totalEquity': np.random.uniform(1e6, 1e9),
                    # 비율 피처
                    'OverMC_revenue': np.random.uniform(0.5, 2.0),
                    'OverMC_netIncome': np.random.uniform(0.05, 0.3),
                    'priceToBookRatio': np.random.uniform(1.0, 5.0),
                    'priceToEarningsRatio': np.random.uniform(10, 30),
                    'returnOnEquity': roe if sector == 'Financial' else np.random.uniform(0.05, 0.20),
                    # 타겟: 다음 달 수익률이 양수면 1
                    'target': np.random.choice([0, 1], p=[0.4, 0.6])
                }

                # 섹터별 특수 피처
                if sector == 'Technology':
                    features['OverMC_researchAndDevelopmentExpenses'] = rd_ratio
                    features['researchAndDevelopmentExpenses'] = features['revenue'] * rd_ratio
                elif sector == 'Financial':
                    features['debtToEquity'] = debt_equity
                    features['totalLiabilities'] = features['totalEquity'] * debt_equity

                data.append(features)

    df = pd.DataFrame(data)
    logging.info(f"샘플 데이터 생성 완료: {len(df):,} rows, {len(dates)} months, {len(sectors)} sectors")

    return df


def example_1_rebalancing_optimization():
    """예제 1: 리밸런싱 기간 최적화"""
    logging.info("\n" + "="*60)
    logging.info("예제 1: 리밸런싱 기간 최적화")
    logging.info("="*60)

    # 샘플 데이터 생성
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    df = generate_sample_data_with_sector(start_date, end_date)

    # 간단한 백테스팅 함수 (실제로는 더 복잡함)
    def simple_backtest(data, date_col, start_date, end_date, rebalance_months, top_k):
        """간단한 백테스팅 함수"""
        dates = pd.date_range(start_date, end_date, freq=f'{rebalance_months}MS')
        results = []

        for i in range(len(dates) - 1):
            period_start = dates[i]
            period_end = dates[i + 1]

            # 해당 기간 데이터
            period_data = data[(data[date_col] >= period_start) & (data[date_col] < period_end)]

            if len(period_data) == 0:
                continue

            # 랜덤 수익률 (실제로는 모델 예측 기반)
            returns = np.random.uniform(-10, 20, len(dates)-1)  # -10% ~ +20%

            results.append({
                'period_start': period_start,
                'period_end': period_end,
                'return': returns[i]
            })

        return pd.DataFrame(results)

    # 리밸런싱 기간 최적화
    optimizer = RebalancingOptimizer(
        periods_to_test=[1, 2, 3, 6],  # 1, 2, 3, 6개월 테스트
        optimization_metric='total_return',
        min_trades=5
    )

    # 최적화 실행
    optimization_result = optimizer.optimize(
        backtest_func=simple_backtest,
        data=df,
        date_col='date',
        start_date=start_date,
        end_date=end_date,
        top_k=5
    )

    # 결과 저장
    optimizer.save_results('./results/rebalancing_optimization.csv')

    # 시각화
    # optimizer.plot_results('./results/rebalancing_optimization.png')

    logging.info(f"\n최적 리밸런싱 기간: {optimization_result['optimal_period']['period_months']}개월")


def example_2_model_comparison():
    """예제 2: 모델 성능 비교"""
    logging.info("\n" + "="*60)
    logging.info("예제 2: 모델 성능 비교")
    logging.info("="*60)

    # 샘플 데이터 생성
    n_samples = 1000
    X_train = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
        'feature_5': np.random.randn(n_samples),
    })
    y_train = pd.Series(np.random.choice([0, 1], n_samples))

    X_test = pd.DataFrame({
        'feature_1': np.random.randn(200),
        'feature_2': np.random.randn(200),
        'feature_3': np.random.randn(200),
        'feature_4': np.random.randn(200),
        'feature_5': np.random.randn(200),
    })
    y_test = pd.Series(np.random.choice([0, 1], 200))

    # ModelComparator 초기화
    comparator = ModelComparator(experiment_name="model_version_comparison")

    # 모델 1: XGBoost 기본 버전
    model_v1 = XGBoostModel(n_estimators=50, max_depth=3, learning_rate=0.1)
    model_v1.build_model({})
    comparator.add_model(
        model_name="XGBoost_v1",
        model_instance=model_v1,
        description="기본 파라미터",
        hyperparameters={'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
    )

    # 모델 2: XGBoost 개선 버전
    model_v2 = XGBoostModel(n_estimators=100, max_depth=5, learning_rate=0.05)
    model_v2.build_model({})
    comparator.add_model(
        model_name="XGBoost_v2",
        model_instance=model_v2,
        description="파라미터 튜닝 (더 많은 트리, 더 깊은 depth)",
        hyperparameters={'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05}
    )

    # 모델 3: LightGBM
    model_v3 = LightGBMModel(n_estimators=100, max_depth=5, learning_rate=0.05)
    model_v3.build_model({})
    comparator.add_model(
        model_name="LightGBM_v1",
        model_instance=model_v3,
        description="LightGBM 알고리즘",
        hyperparameters={'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05}
    )

    # 모델 비교
    comparison_df = comparator.compare_models(X_train, y_train, X_test, y_test, cv_splits=3)

    # 개선 여부 확인
    is_improved = comparator.is_improved(
        new_model_name="XGBoost_v2",
        baseline_model_name="XGBoost_v1",
        metric='accuracy',
        threshold=0.01  # 1% 이상 개선
    )

    # 결과 저장
    comparator.save_results('./results/model_comparison')

    # 시각화
    # comparator.plot_comparison('./results/model_comparison.png')


def example_3_sector_ensemble():
    """예제 3: 섹터별 모델 + 통합 선택"""
    logging.info("\n" + "="*60)
    logging.info("예제 3: 섹터별 모델 + 통합 선택")
    logging.info("="*60)

    # 샘플 데이터 생성
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    df = generate_sample_data_with_sector(start_date, end_date)

    # 학습 데이터와 예측 데이터 분리
    train_end = datetime(2023, 6, 30)
    train_df = df[df['date'] <= train_end]
    predict_df = df[df['date'] > train_end]

    logging.info(f"\n학습 데이터: {len(train_df):,} rows")
    logging.info(f"예측 데이터: {len(predict_df):,} rows")

    # SectorEnsemble 초기화
    ensemble = SectorEnsemble(sector_col='sector')

    # 기본 섹터 설정 사용
    default_configs = create_default_sector_configs(XGBoostModel)

    # 각 섹터 설정
    for sector_name, config in default_configs.items():
        ensemble.configure_sector(
            sector_name=sector_name,
            **config
        )

    # 학습 → 예측 → 선택
    top_stocks = ensemble.fit_predict_select(
        train_df=train_df,
        predict_df=predict_df,
        target_col='target',
        top_k=10,
        use_feature_selection=True,
        symbol_col='symbol'
    )

    logging.info(f"\n선택된 주식 상세:")
    logging.info(top_stocks[['symbol', 'sector', 'predicted_score', 'predicted_class']])

    # 섹터 요약
    sector_summary = ensemble.get_sector_summary()
    logging.info(f"\n섹터별 모델 정보:")
    logging.info(sector_summary)

    # 설정 저장
    ensemble.save_config('./results/sector_ensemble_config.json')


def main():
    """메인 함수 - 모든 예제 실행"""
    logging.info("="*60)
    logging.info("Comprehensive Example - 3가지 핵심 기능")
    logging.info("="*60)

    # 예제 1: 리밸런싱 기간 최적화
    example_1_rebalancing_optimization()

    # 예제 2: 모델 성능 비교
    example_2_model_comparison()

    # 예제 3: 섹터별 모델 + 통합 선택
    example_3_sector_ensemble()

    logging.info("\n" + "="*60)
    logging.info("모든 예제 완료!")
    logging.info("="*60)
    logging.info("\n결과 확인:")
    logging.info("  - ./results/rebalancing_optimization.csv")
    logging.info("  - ./results/model_comparison/")
    logging.info("  - ./results/sector_ensemble_config.json")


if __name__ == '__main__':
    main()
