# 📈 고급 기능 가이드

실전 트레이딩 시스템을 위한 3가지 핵심 기능

---

## 🎯 1. 리밸런싱 기간 최적화

### 문제점
현재 3개월로 고정된 리밸런싱 기간을 사용하고 있으나, 최적의 기간을 알 수 없음.

### 해결 방안
과거 데이터를 기반으로 여러 리밸런싱 기간(1개월, 2개월, 3개월, 6개월 등)을 테스트하여 수익률이 가장 좋은 기간을 선택.

### 사용 방법

```python
from optimization.rebalance_optimizer import RebalancingOptimizer
from datetime import datetime

# 1. Optimizer 초기화
optimizer = RebalancingOptimizer(
    periods_to_test=[1, 2, 3, 4, 6, 12],  # 테스트할 기간들
    optimization_metric='total_return',    # 최적화 기준
    min_trades=10                          # 최소 거래 횟수
)

# 2. 백테스팅 함수 정의 (your_backtest_func)
def your_backtest_func(data, date_col, start_date, end_date, rebalance_months, top_k):
    # 백테스팅 로직
    # rebalance_months 파라미터를 사용하여 리밸런싱 기간 조정
    ...
    return results_df  # 'return' 컬럼 필요

# 3. 최적화 실행
result = optimizer.optimize(
    backtest_func=your_backtest_func,
    data=your_data,
    date_col='date',
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    top_k=5
)

# 4. 최적 기간 확인
print(f"최적 리밸런싱 기간: {result['optimal_period']['period_months']}개월")
print(f"예상 수익률: {result['optimal_period']['total_return']:.2f}%")

# 5. 결과 저장 및 시각화
optimizer.save_results('./results/rebalancing_optimization.csv')
optimizer.plot_results('./results/rebalancing_optimization.png')
```

### 최적화 기준

- `total_return`: 총 수익률 (기본값)
- `annualized_return`: 연평균 수익률
- `sharpe_ratio`: 샤프 비율 (위험 조정 수익률)
- `win_rate`: 승률

### 출력 메트릭

각 리밸런싱 기간에 대해:
- 총 수익률
- 연평균 수익률
- 샤프 비율
- 승률
- 거래 횟수
- 최대 낙폭(MDD)

---

## 🔬 2. 모델 성능 비교

### 문제점
모델이나 파라미터를 변경할 때 이전 모델보다 나아졌는지 알 수 없음.

### 해결 방안
동일한 테스트 셋에서 여러 모델 버전을 비교하고, 통계적 유의성 검정을 수행.

### 사용 방법

```python
from optimization.model_comparator import ModelComparator
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel

# 1. ModelComparator 초기화
comparator = ModelComparator(experiment_name="model_version_comparison")

# 2. 비교할 모델들 추가

# 기존 모델 (v1)
model_v1 = XGBoostModel(n_estimators=50, max_depth=3)
model_v1.build_model({})
comparator.add_model(
    model_name="XGBoost_v1",
    model_instance=model_v1,
    description="기존 모델 - 기본 파라미터",
    hyperparameters={'n_estimators': 50, 'max_depth': 3}
)

# 새 모델 (v2) - 파라미터 개선
model_v2 = XGBoostModel(n_estimators=100, max_depth=5)
model_v2.build_model({})
comparator.add_model(
    model_name="XGBoost_v2",
    model_instance=model_v2,
    description="개선 모델 - 파라미터 튜닝",
    hyperparameters={'n_estimators': 100, 'max_depth': 5}
)

# 새 모델 (v3) - 알고리즘 변경
model_v3 = LightGBMModel(n_estimators=100, max_depth=5)
model_v3.build_model({})
comparator.add_model(
    model_name="LightGBM_v1",
    model_instance=model_v3,
    description="LightGBM으로 변경",
    hyperparameters={'n_estimators': 100, 'max_depth': 5}
)

# 3. 모델 비교 (동일한 데이터 사용)
comparison_df = comparator.compare_models(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,  # 반드시 동일한 테스트 셋 사용!
    y_test=y_test,
    cv_splits=5
)

# 4. 결과 확인
print(comparison_df)

# 5. 개선 여부 확인 (통계적 유의성 검정)
is_improved = comparator.is_improved(
    new_model_name="XGBoost_v2",
    baseline_model_name="XGBoost_v1",
    metric='accuracy',
    threshold=0.01  # 1% 이상 개선되어야 함
)

if is_improved:
    print("✅ 새 모델이 통계적으로 유의미하게 개선되었습니다!")
else:
    print("❌ 새 모델이 충분히 개선되지 않았습니다.")

# 6. 결과 저장 및 시각화
comparator.save_results('./results/model_comparison')
comparator.plot_comparison('./results/model_comparison.png')
```

### 통계적 유의성 검정

- **McNemar's test** 사용 (분류 문제)
- p-value < 0.05이면 유의미한 차이
- 모든 모델 쌍에 대해 자동으로 검정 수행

### 출력 결과

- CSV: 모든 모델의 성능 메트릭
- JSON: 모델 메타데이터 (하이퍼파라미터, 설명 등)
- TXT: 요약 리포트
- PNG: 성능 비교 차트

---

## 🏢 3. 섹터별 모델 + 통합 선택

### 문제점
- 섹터마다 중요한 지표가 다름 (Tech: R&D 비중, Finance: 자본비율)
- 섹터별로 고정 투자가 아니라 전체에서 좋은 주식만 선택하고 싶음

### 해결 방안
1. 각 섹터마다 다른 피처/모델 사용
2. 섹터별로 수익률 예측
3. 모든 섹터의 예측 점수를 합쳐서 전체 top N개 선택

### 개념 설명

```
[Technology 섹터]
  피처: R&D 비중, 매출 성장률 등
  모델: XGBoost with specific params
  → 예측 점수: TECH_001(0.85), TECH_002(0.78), ...

[Financial 섹터]
  피처: 자본비율, ROE 등
  모델: XGBoost with different params
  → 예측 점수: FIN_001(0.92), FIN_002(0.71), ...

[Healthcare 섹터]
  피처: R&D 비중, 영업이익률 등
  모델: XGBoost with specific params
  → 예측 점수: HLTH_001(0.88), HLTH_002(0.76), ...

↓ 통합 ↓

전체 주식 풀:
  FIN_001(0.92), HLTH_001(0.88), TECH_001(0.85), TECH_002(0.78), ...

→ Top 5 선택: FIN_001, HLTH_001, TECH_001, TECH_002, FIN_002
```

### 사용 방법

```python
from strategy.sector_ensemble import SectorEnsemble, create_default_sector_configs
from models.xgboost_model import XGBoostModel

# 1. SectorEnsemble 초기화
ensemble = SectorEnsemble(sector_col='sector')

# 2. 각 섹터별 설정

# Technology 섹터
ensemble.configure_sector(
    sector_name='Technology',
    model_class=XGBoostModel,
    feature_cols=[
        'revenue', 'netIncome', 'researchAndDevelopmentExpenses',
        'OverMC_researchAndDevelopmentExpenses',  # R&D 비중 중요!
        'operatingCashFlow', 'freeCashFlow',
        'priceToBookRatio', 'priceToSalesRatio'
    ],
    model_params={'n_estimators': 100, 'max_depth': 6},
    feature_selection_params={'method': 'tree_importance', 'top_k': 8}
)

# Financial 섹터
ensemble.configure_sector(
    sector_name='Financial',
    model_class=XGBoostModel,
    feature_cols=[
        'totalAssets', 'totalLiabilities', 'totalEquity',
        'debtToEquity',  # 부채비율 중요!
        'returnOnEquity', 'returnOnAssets',
        'priceToBookRatio', 'dividendYield'
    ],
    model_params={'n_estimators': 100, 'max_depth': 5},
    feature_selection_params={'method': 'mutual_info', 'top_k': 8}
)

# Healthcare 섹터
ensemble.configure_sector(
    sector_name='Healthcare',
    model_class=XGBoostModel,
    feature_cols=[
        'revenue', 'grossProfit', 'netIncome',
        'researchAndDevelopmentExpenses',
        'OverMC_researchAndDevelopmentExpenses',  # R&D 비중 중요!
        'grossProfitRatio', 'operatingIncomeRatio'
    ],
    model_params={'n_estimators': 100, 'max_depth': 6},
    feature_selection_params={'method': 'tree_importance', 'top_k': 8}
)

# ... 다른 섹터들도 동일하게 설정 ...

# 또는 기본 설정 사용
# default_configs = create_default_sector_configs(XGBoostModel)
# for sector_name, config in default_configs.items():
#     ensemble.configure_sector(sector_name=sector_name, **config)

# 3. 학습 → 예측 → 선택 (한번에)
top_stocks = ensemble.fit_predict_select(
    train_df=train_df,      # 학습 데이터
    predict_df=predict_df,  # 예측할 데이터 (리밸런싱 날 데이터)
    target_col='target',
    top_k=10,               # 전체에서 10개 선택
    use_feature_selection=True,
    symbol_col='symbol'
)

# 4. 결과 확인
print("선택된 주식:")
print(top_stocks[['symbol', 'sector', 'predicted_score']])

# 섹터별 분포 확인
print("\n섹터별 분포:")
print(top_stocks['sector'].value_counts())
```

### 또는 단계별로 실행

```python
# 1. 학습
ensemble.fit(
    df=train_df,
    target_col='target',
    use_feature_selection=True
)

# 2. 예측
predicted_df = ensemble.predict_by_sector(predict_df)

# 3. 선택
top_stocks = ensemble.select_top_stocks(
    df=predicted_df,
    top_k=10,
    symbol_col='symbol'
)
```

### 섹터별 중요 피처 예시

| 섹터 | 중요 피처 | 이유 |
|------|-----------|------|
| **Technology** | `OverMC_researchAndDevelopmentExpenses` | R&D 투자 비중이 혁신과 성장에 중요 |
| | `revenue`, `netIncome` | 성장성 지표 |
| **Financial** | `debtToEquity` | 금융 섹터는 레버리지 관리가 핵심 |
| | `returnOnEquity` | 자본 효율성 |
| **Healthcare** | `OverMC_researchAndDevelopmentExpenses` | 신약 개발을 위한 R&D 투자 |
| | `grossProfitRatio` | 제약사의 마진율 |
| **Consumer** | `inventoryTurnover` | 재고 관리 효율성 |
| | `returnOnEquity` | 수익성 |
| **Industrial** | `propertyPlantEquipmentNet` | 고정자산 규모 |
| | `assetTurnover` | 자산 효율성 |

---

## 🚀 실전 사용 예제

### 시나리오: 3개월마다 리밸런싱하는 실전 시스템

```python
from datetime import datetime
from dateutil.relativedelta import relativedelta
from strategy.sector_ensemble import SectorEnsemble, create_default_sector_configs
from models.xgboost_model import XGBoostModel

# 1단계: 최적 리밸런싱 기간 찾기 (최초 1회)
from optimization.rebalance_optimizer import RebalancingOptimizer

optimizer = RebalancingOptimizer(periods_to_test=[1, 2, 3, 4, 6])
result = optimizer.optimize(...)
optimal_months = result['optimal_period']['period_months']
print(f"최적 리밸런싱 기간: {optimal_months}개월")

# 2단계: 섹터별 모델 설정
ensemble = SectorEnsemble(sector_col='sector')
default_configs = create_default_sector_configs(XGBoostModel)
for sector_name, config in default_configs.items():
    ensemble.configure_sector(sector_name=sector_name, **config)

# 3단계: 리밸런싱 날 (예: 2024-01-01)
rebalance_date = datetime(2024, 1, 1)

# FMP API로 최신 데이터 가져오기
latest_data = fetch_data_from_fmp(date=rebalance_date)  # 여기에 FMP 호출 로직

# 학습 데이터: 과거 3년
train_start = rebalance_date - relativedelta(years=3)
train_end = rebalance_date - relativedelta(days=1)
train_df = fetch_historical_data(train_start, train_end)

# 예측 데이터: 리밸런싱 날 데이터
predict_df = latest_data

# 학습 → 예측 → 선택
top_stocks = ensemble.fit_predict_select(
    train_df=train_df,
    predict_df=predict_df,
    target_col='future_return',  # 미래 수익률
    top_k=10,
    use_feature_selection=True
)

# 4단계: 주식 매수
print("매수할 주식:")
for idx, row in top_stocks.iterrows():
    print(f"{row['symbol']}: {row['predicted_score']:.4f} (섹터: {row['sector']})")
```

---

## 📊 통합 사용 예제

```python
# examples/comprehensive_example.py 실행
python examples/comprehensive_example.py
```

이 예제는 3가지 기능을 모두 통합하여 실행합니다:
1. 리밸런싱 기간 최적화
2. 모델 버전 비교
3. 섹터별 모델 + 통합 선택

---

## 💡 팁과 주의사항

### 리밸런싱 기간 최적화
- **충분한 데이터**: 최소 3년 이상의 데이터 권장
- **과최적화 주의**: 너무 많은 기간을 테스트하면 과최적화 위험
- **거래 비용 고려**: 짧은 기간은 거래 비용이 높을 수 있음

### 모델 성능 비교
- **동일한 테스트 셋**: 반드시 같은 테스트 데이터 사용
- **통계적 유의성**: p-value < 0.05일 때만 유의미
- **실전 검증**: 백테스팅뿐 아니라 paper trading으로 검증

### 섹터별 모델
- **데이터 충분성**: 각 섹터당 최소 100개 이상의 샘플 권장
- **피처 선택**: 섹터별 도메인 지식 활용
- **밸런스**: 특정 섹터로 편중되지 않도록 주의

---

## 📁 프로젝트 구조

```
Quant-refactoring/
├── optimization/
│   ├── rebalance_optimizer.py    # 리밸런싱 기간 최적화
│   └── model_comparator.py       # 모델 성능 비교
├── strategy/
│   └── sector_ensemble.py        # 섹터별 앙상블
├── examples/
│   └── comprehensive_example.py  # 통합 예제
└── ADVANCED_FEATURES_GUIDE.md    # 이 문서
```

---

## 🎓 참고 자료

- [리밸런싱 최적화](https://www.investopedia.com/terms/r/rebalancing.asp)
- [McNemar's Test](https://en.wikipedia.org/wiki/McNemar%27s_test)
- [Sector Rotation Strategy](https://www.investopedia.com/articles/trading/05/020305.asp)

---

## 🤝 기여

버그 리포트 및 개선 제안은 이슈로 등록해주세요.
