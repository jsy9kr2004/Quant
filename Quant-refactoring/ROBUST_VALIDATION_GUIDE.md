# 📊 Robust Validation Guide

백테스팅 및 모델 검증 개선 가이드

## 🎯 주요 문제점과 해결 방안

### 1️⃣ Cross-validation 없음 (단순 train/test split)

#### 문제점
- 단순히 시간 순서대로 데이터를 나누어 백테스팅만 수행
- 특정 기간의 데이터에만 과적합될 위험
- 모델의 안정성과 일반화 성능을 평가할 수 없음

#### 해결 방안
**Time Series Cross-Validation 구현**

```python
from validation.time_series_cv import TimeSeriesCV
from models.xgboost_model import XGBoostModel

# 1. TimeSeriesCV 초기화
cv = TimeSeriesCV(n_splits=5)

# 2. 모델 생성
model = XGBoostModel(n_estimators=100, max_depth=5)
model.build_model({})

# 3. 교차 검증 수행
avg_scores, all_scores = cv.cross_validate_model(
    model=model,
    X=X_train,
    y=y_train,
    dates=dates,
    verbose=True
)

print(f"평균 Accuracy: {avg_scores['accuracy_mean']:.4f}")
print(f"표준편차: {avg_scores['accuracy_std']:.4f}")
```

**BaseModel에 통합된 메서드 사용**

```python
from models.xgboost_model import XGBoostModel

model = XGBoostModel(n_estimators=100)
model.build_model({})

# 교차검증 + 전체 데이터 학습
avg_scores, all_scores = model.fit_with_cv(
    X=X_train,
    y=y_train,
    dates=dates,
    cv_splits=5
)
```

---

### 2️⃣ Walk-forward validation 없음

#### 문제점
- 한번 학습한 모델을 계속 사용 (모델 재학습 없음)
- 시장 환경 변화에 적응하지 못함
- 실제 트레이딩 환경과 다름

#### 해결 방안
**Walk-Forward Validation 구현**

```python
from validation.walk_forward import WalkForwardValidator
from models.xgboost_model import XGBoostModel
from datetime import datetime

# 1. Walk-Forward Validator 초기화
wfv = WalkForwardValidator(
    train_months=24,        # 24개월 데이터로 학습
    test_months=3,          # 3개월 테스트
    retrain_frequency=3,    # 3개월마다 재학습
    anchored=False          # Rolling window (False) or Anchored window (True)
)

# 2. 백테스팅 실행
results = wfv.validate(
    model_class=XGBoostModel,
    df=data_df,
    date_col='date',
    feature_cols=['feature_1', 'feature_2', ...],
    target_col='target',
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    model_params={'n_estimators': 100, 'max_depth': 5}
)

print(results)
```

**차이점 설명**

```python
# Rolling Window (anchored=False)
# Period 1: Train(2020-01~2021-12) → Test(2022-01~2022-03)
# Period 2: Train(2020-04~2022-03) → Test(2022-04~2022-06)
# Period 3: Train(2020-07~2022-06) → Test(2022-07~2022-09)

# Anchored Window (anchored=True)
# Period 1: Train(2020-01~2021-12) → Test(2022-01~2022-03)
# Period 2: Train(2020-01~2022-03) → Test(2022-04~2022-06)  ← 학습 시작점 고정
# Period 3: Train(2020-01~2022-06) → Test(2022-07~2022-09)
```

---

### 3️⃣ 모델 성능 저하 모니터링 없음

#### 문제점
- 모델이 언제 성능이 나빠지는지 감지하는 로직이 없음
- 모델 drift 탐지 불가
- 재학습 시점을 결정할 수 없음

#### 해결 방안
**Performance Monitor 구현**

```python
from monitoring.performance_monitor import PerformanceMonitor

# 1. Performance Monitor 초기화
monitor = PerformanceMonitor(
    window_size=10,          # 최근 10개 기간 추적
    alert_threshold=0.10,    # 10% 성능 저하 시 알림
    drift_threshold=0.05     # p-value < 0.05이면 drift
)

# 2. 백테스팅 루프에서 성능 모니터링
for period in periods:
    # 모델 학습 및 평가
    metrics = model.evaluate(X_test, y_test)

    # 성능 업데이트
    monitor.update_performance(metrics, period_label=period)

    # 피처 드리프트 체크
    drift_features = monitor.check_feature_drift(
        X_new=X_test.values,
        feature_names=feature_cols
    )

    if drift_features:
        print(f"⚠️ {len(drift_features)}개 피처에서 드리프트 감지")

    # 재학습 필요 여부
    if monitor.should_retrain():
        print("🔄 성능 저하로 인한 재학습 필요!")
        # 모델 재학습 로직

# 3. 성능 요약
monitor.print_summary()
```

**성능 모니터링 기능**
- ✅ Rolling window 성능 추적
- ✅ Baseline 대비 성능 저하 자동 감지
- ✅ 데이터 드리프트 감지 (Kolmogorov-Smirnov test)
- ✅ 재학습 권장 알림

---

### 4️⃣ tsfresh로 수백 개 피처 생성 → 과적합 가능성

#### 문제점
- `EfficientFCParameters()`는 794개의 피처를 생성
- 샘플 수 < 피처 수인 경우 과적합 심각
- 불필요한 피처들이 노이즈로 작용

#### 해결 방안

**Option 1: Feature Selection 사용**

```python
from feature_engineering.feature_selector import FeatureSelector

# 1. Feature Selector 초기화
selector = FeatureSelector(
    method='mutual_info',           # 'mutual_info', 'f_test', 'rfe', 'tree_importance'
    top_k=50,                       # 상위 50개 피처 선택
    correlation_threshold=0.95,     # 상관관계 0.95 이상이면 제거
    task='classification'           # or 'regression'
)

# 2. 피처 선택
selected_features = selector.select_features(
    X=X_train,
    y=y_train,
    feature_names=feature_names
)

print(f"선택된 피처: {selected_features}")

# 3. 데이터 변환
X_train_selected = selector.transform(X_train, feature_names)
X_test_selected = selector.transform(X_test, feature_names)

# 4. 피처 점수 확인
feature_scores = selector.get_feature_scores()
print(feature_scores.head(10))
```

**Option 2: tsfresh MinimalFCParameters 사용**

```python
from tsfresh.feature_extraction import MinimalFCParameters

# 794개 → 약 20개로 축소
settings = MinimalFCParameters()
extracted_features = extract_features(
    long_format_df,
    default_fc_parameters=settings
)
```

**Option 3: 사용자 정의 피처**

```python
from tsfresh.feature_extraction import extract_features

# 필요한 피처만 선택
fc_parameters = {
    "mean": None,
    "median": None,
    "std": None,
    "min": None,
    "max": None,
    "quantile": [{"q": 0.25}, {"q": 0.75}],
    "linear_trend": [{"attr": "slope"}],
    "autocorrelation": [{"lag": 1}, {"lag": 2}]
}

extracted_features = extract_features(
    long_format_df,
    default_fc_parameters=fc_parameters
)
```

---

## 🚀 통합 사용: RobustBacktester

모든 기능을 통합한 `RobustBacktester` 사용 예제:

```python
from robust_backtester import RobustBacktester
from models.xgboost_model import XGBoostModel
from datetime import datetime

# 1. RobustBacktester 초기화
backtester = RobustBacktester(
    train_months=24,                    # 24개월 학습
    test_months=3,                      # 3개월 테스트
    retrain_frequency=3,                # 3개월마다 재학습
    cv_splits=5,                        # 5-fold CV
    top_k_features=50,                  # 상위 50개 피처
    feature_selection_method='mutual_info',
    performance_window=10,              # 최근 10개 기간 추적
    alert_threshold=0.10                # 10% 성능 저하 시 알림
)

# 2. 백테스팅 실행
results = backtester.run_backtest(
    model_class=XGBoostModel,
    df=data_df,
    date_col='date',
    feature_cols=feature_columns,
    target_col='target',
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    model_params={'n_estimators': 100, 'max_depth': 5},
    use_feature_selection=True,         # 피처 선택 사용
    use_cv=True,                        # 교차검증 사용
    use_monitoring=True,                # 성능 모니터링 사용
    save_results=True,
    output_dir='./results'
)

# 3. 결과 확인
print(results)
print(f"\n선택된 피처: {backtester.get_selected_features()}")
print(f"\n성능 요약: {backtester.get_performance_summary()}")
```

---

## 📁 프로젝트 구조

```
Quant-refactoring/
├── validation/
│   ├── __init__.py
│   ├── time_series_cv.py          # Time Series Cross-Validation
│   └── walk_forward.py            # Walk-Forward Validation
├── monitoring/
│   ├── __init__.py
│   └── performance_monitor.py     # 성능 모니터링
├── feature_engineering/
│   ├── __init__.py
│   └── feature_selector.py        # 피처 선택
├── models/
│   ├── base_model.py              # BaseModel (CV 메서드 추가)
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   └── catboost_model.py
├── robust_backtester.py           # 통합 백테스터
└── examples/
    ├── robust_backtest_example.py # 통합 예제
    └── validation_examples.py     # 개별 기능 예제
```

---

## 🔧 실행 예제

### 예제 1: 통합 백테스팅

```bash
cd Quant-refactoring
python examples/robust_backtest_example.py
```

### 예제 2: 개별 검증 방법

```bash
python examples/validation_examples.py
```

---

## 📊 출력 결과

백테스팅 실행 시 다음과 같은 결과 파일들이 생성됩니다:

```
results/
├── backtest_results_20231215_143022.csv      # 백테스팅 결과
├── selected_features_20231215_143022.csv     # 선택된 피처
└── performance_alerts_20231215_143022.csv    # 성능 알림
```

---

## ⚠️ 주의사항

1. **데이터 누출 방지**: 모든 검증은 시계열 순서를 유지합니다.
2. **충분한 데이터**: Walk-forward에는 최소 3년 이상의 데이터 권장
3. **계산 시간**: CV + Walk-forward는 시간이 오래 걸릴 수 있습니다.
4. **메모리**: 큰 데이터셋의 경우 메모리 사용량 주의

---

## 📚 참고 문헌

- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - Marcos López de Prado
- [Sklearn Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Walk-Forward Optimization](https://www.investopedia.com/terms/w/walkforward.asp)

---

## 🤝 기여

버그 리포트 및 개선 제안은 이슈로 등록해주세요.
