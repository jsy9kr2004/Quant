# 아키텍처 상세 분석

> **작성일**: 2025-12-17 (최종 업데이트: 2026-01-17)
> **이전 문서**: [00_overview.md](./00_overview.md)
> **다음 문서**: [02_data_pipeline.md](./02_data_pipeline.md)

---

## 목차

1. [전체 아키텍처 개요](#1-전체-아키텍처-개요)
2. [2-Stage ML 아키텍처](#2-2-stage-ml-아키텍처)
3. [섹터 카테고리화 전략](#3-섹터-카테고리화-전략)
4. [통합 아키텍처 (Unified Architecture)](#4-통합-아키텍처)
5. [설계 패턴 분석](#5-설계-패턴-분석)
6. [아키텍처 평가](#6-아키텍처-평가)
7. [개선 제안](#7-개선-제안)

---

## 1. 전체 아키텍처 개요

### 1.1 아키텍처 철학

이 시스템의 아키텍처는 **"예측이 아닌 선별"**이라는 철학을 반영합니다:

```
철학                        아키텍처 구현
──────────────────────────────────────────────────────────
"안정성 먼저"       →      Stage 1: Classifier (필터링)
"그 다음 수익성"     →      Stage 2: Regressor (순위 매기기)
"섹터 특성 고려"     →      Sector Categorization
"코드 신뢰성"        →      Unified Architecture (단일 전처리)
"일원화 강제"        →      Prediction Cache 필수화 (2026-01-17)
"미래 유출 방지"     →      Walk-Forward Backtest
"현실적 백테스트"    →      거래 비용 반영 (Commission+Slippage)
```

### 1.2 아키텍처 계층 (Layered Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Configuration (설정 계층)                           │
│   - conf.yaml: 전역 설정                                     │
│   - DataSchema: 컬럼 정의 (Single Source of Truth)          │
│   - G_Variables: 경로 관리                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Data Pipeline (데이터 파이프라인)                   │
│   - FMP API → Parquet Storage → VIEW Builder                │
│   - AIDataMaker: Feature Engineering (tsfresh)              │
│   - DataProcessor: 통합 전처리 파이프라인                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: ML Core (ML 코어)                                   │
│   - ModelFactory: 모델 생성 (팩토리 패턴)                    │
│   - Regressor: 학습 (2-Stage)                                │
│   - OptunaOptimizer: 하이퍼파라미터 최적화                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Backtesting (백테스팅)                              │
│   - MLBacktest: Walk-Forward Analysis                       │
│   - Benchmark Comparison                                    │
│   - Report Generation                                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Orchestration (조율)                                │
│   - main.py: 전체 파이프라인 실행                            │
│   - Scripts: 개별 작업 실행 (run_ml_backtest.py 등)          │
│   - Prediction Mode: 사전 학습된 모델로 추천 종목 생성       │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 모듈 의존성 그래프

```
main.py
  │
  ├─→ config/
  │     ├─→ context_loader.py (MainContext)
  │     ├─→ logger.py
  │     └─→ g_variables.py
  │
  ├─→ src/constants/
  │     └─→ data_schema.py (DataSchema) ★ Single Source of Truth
  │
  ├─→ src/data_collector/
  │     └─→ fmp.py (FMP API 수집)
  │
  ├─→ src/storage/
  │     ├─→ parquet_storage.py
  │     └─→ parquet_converter.py
  │
  ├─→ src/training/
  │     ├─→ make_mldata.py (AIDataMaker)
  │     │     └─→ DataProcessor ★
  │     │     └─→ DataSchema ★
  │     │
  │     ├─→ data_processor.py (DataProcessor) ★ Unified Preprocessing
  │     │
  │     ├─→ regressor.py (Regressor)
  │     │     ├─→ DataProcessor ★
  │     │     ├─→ DataSchema ★
  │     │     ├─→ ModelFactory ★
  │     │     └─→ optuna_utils.py
  │     │
  │     └─→ optimizer.py (OptunaOptimizer)
  │
  ├─→ src/models/
  │     └─→ model_factory.py (ModelFactory) ★ Factory Pattern
  │           ├─→ xgboost_model.py
  │           ├─→ lightgbm_model.py
  │           └─→ catboost_model.py
  │
  └─→ src/backtest/
        └─→ ml_backtest.py (MLBacktest)
              ├─→ DataProcessor ★
              ├─→ DataSchema ★
              └─→ ModelFactory ★

★ = 핵심 통합 모듈 (Unified Modules)
```

**핵심 통합 모듈 (★)**:
- **DataSchema**: 컬럼 정의 단일화
- **DataProcessor**: 전처리 파이프라인 통합
- **ModelFactory**: 모델 생성 팩토리

---

## 2. 2-Stage ML 아키텍처

### 2.1 철학적 배경

**문제**: 전통적인 회귀 모델은 **"모든 종목의 수익률을 예측"**하려고 합니다.
하지만 펀더멘털이 깨진 종목(부실 기업)과 건전한 종목을 동시에 학습하면 노이즈가 발생합니다.

**해결**: 2단계로 분리
1. **Stage 1 (Classifier)**: "펀더멘털이 깨지지 않았는가?" (안전성 필터링)
2. **Stage 2 (Regressor)**: "무효화된 가치의 회복 가능성" (수익성 순위 매기기)

### 2.2 Stage 1: Classifier (안정성 필터링)

#### 목적
"**재무적으로 부실한 기업 제거**"

#### 학습 프로세스

**Step 1: Binary Target 생성**
```python
# data_processor.py
def create_binary_target(y_continuous):
    """
    타겟: label_binary = (price_dev > 0)
    1 = 다음 분기 가격 상승
    0 = 다음 분기 가격 하락
    """
    return (y_continuous > 0).astype(int)
```

**Step 2: 분류기 학습 (모든 데이터)**
```python
# regressor.py
# 4개 분류기 앙상블
clsmodels = [
    XGBClassifier(max_depth=8, ...),  # Optuna 최적화
    XGBClassifier(max_depth=9, ...),
    XGBClassifier(max_depth=10, ...),
    LGBMClassifier(max_depth=8, ...)
]

for clf in clsmodels:
    clf.fit(X_train, y_binary_train)
```

**Step 3: 최적 Threshold 자동 탐색**
```python
def _find_optimal_threshold(y_true, y_probs, ...):
    """
    Percentile 85~98 구간 탐색
    각 percentile에서 precision, recall 계산

    전략:
    - "precision": Precision 최대값 선택
    - "balance": Min precision 만족하면서 최대 데이터 선택
    """
    best_percentile = None
    best_precision = 0

    for pct in range(85, 99):
        threshold = np.percentile(y_probs, pct)
        mask = y_probs > threshold

        precision = precision_score(y_true[mask], y_pred[mask])
        recall = recall_score(y_true[mask], y_pred[mask])

        if strategy == "balance":
            if precision >= min_precision:
                if mask.sum() > best_n_selected:
                    best_percentile = pct
                    best_n_selected = mask.sum()
        elif strategy == "precision":
            if precision > best_precision:
                best_percentile = pct
                best_precision = precision

    # threshold_config.pkl 저장
    save_threshold_config({
        'percentile': best_percentile,
        'threshold_value': np.percentile(y_probs, best_percentile),
        'precision': precision,
        'recall': recall,
        'n_selected': n_selected
    })

    return best_percentile
```

**예시 출력**:
```
Searching optimal threshold...
Percentile 85: threshold=0.612, selected=15000, precision=0.523
Percentile 90: threshold=0.701, selected=10000, precision=0.610
Percentile 93: threshold=0.789, selected=7000,  precision=0.729 ← 선택!
Percentile 95: threshold=0.854, selected=5000,  precision=0.780
Percentile 98: threshold=0.921, selected=2000,  precision=0.850

Best: Percentile 93 (precision=0.729, n_selected=7000)
Saved to threshold_config.pkl
```

**Step 4: 학습 데이터 필터링**
```python
# 최적 threshold로 필터링
threshold = threshold_config['threshold_value']  # 0.789
safe_mask = y_probs > threshold  # 상위 7%

X_train_filtered = X_train[safe_mask]  # 7000개
y_train_filtered = y_train[safe_mask]

print(f"Filtered: {len(X_train)} → {len(X_train_filtered)}")
# Filtered: 100000 → 7000 (93% 제거!)
```

#### 예측 프로세스

**regressor.py (모델 평가)**:
```python
# threshold_config.pkl 로드
threshold_config = joblib.load('threshold_config.pkl')
THRESHOLD_PERCENTILE = threshold_config['percentile']  # 93

# 상승 확률 예측
y_probs = classifier.predict_proba(X)[:, 1]

# 학습 시와 동일한 percentile threshold 적용
threshold = np.percentile(y_probs, THRESHOLD_PERCENTILE)
y_predict_binary = (y_probs > threshold).astype(int)

# 하위 93% 패널티 (평가용)
prediction_wbinary = np.where(y_predict_binary == 0, -1, y_predict_return)
```

**ml_backtest.py (백테스트)**:
```python
# 상승 확률 예측
y_pred_proba = classifier.predict_proba(X)[:, 1]

# 수익률 예측
y_pred_return = regressor.predict(X)

# 최종 스코어: 확률 × 수익률 (Soft Filtering)
ml_score = y_pred_proba * y_pred_return

# 예시:
# 상승 확률 높음 (0.9) × 예측 수익률 (+10%) = +9.0
# 상승 확률 낮음 (0.1) × 예측 수익률 (+10%) = +1.0
```

#### 수학적 동등성

**Hard Filtering** (설계 의도):
```python
# 1. 상위 8% 선택
safe_stocks = stocks[y_probs > percentile(92)]

# 2. Top-5 선택
top_5 = safe_stocks.sort_values('y_pred', ascending=False).head(5)
```

**Soft Filtering** (현재 구현):
```python
# 1. 모든 종목에 확률 × 수익률
ml_score = y_pred_proba * y_pred_return

# 2. Top-5 선택 (자동으로 상위 확률 종목 선택됨)
top_5 = all_stocks.sort_values('ml_score', ascending=False).head(5)
```

**결론**: N > K/0.08 조건에서 두 방식 완전 동일

### 2.3 Stage 2: Regressor (수익률 예측)

#### 목적
"**무효화된 가치의 회복 가능성**"을 통계적으로 점수화

#### 학습 프로세스

**Step 1: 필터링된 데이터로 학습**
```python
# Stage 1에서 필터링된 "안전한" 종목만 사용
X_train_safe, y_train_safe = stage1_filter(X_train, y_train)

# 2개 회귀기 앙상블
models = [
    XGBRegressor(max_depth=8, ...),
    XGBRegressor(max_depth=10, ...)
]

for reg in models:
    reg.fit(X_train_safe, y_train_safe)
```

**효과**:
- 회귀기가 상위 7% "안전한" 종목의 패턴만 학습
- 하위 93% 노이즈 제거
- 학습 데이터는 감소하지만 품질 향상

#### 예측 프로세스

**모든 종목 예측 → Stage 1 확률과 곱셈**:
```python
# 1. 모든 종목에 대해 수익률 예측
y_pred_return = regressor.predict(X_all)

# 2. Stage 1 확률과 곱셈
ml_score = y_pred_proba * y_pred_return

# 3. Top-K 선정
top_k = df.sort_values('ml_score', ascending=False).head(K)
```

### 2.4 앙상블 전략

#### Classifier Ensemble (4 variants)
```python
clsmodels = [
    XGBClassifier(max_depth=8, n_estimators=optuna_best),  # Optuna 최적화
    XGBClassifier(max_depth=9, n_estimators=200),
    XGBClassifier(max_depth=10, n_estimators=200),
    LGBMClassifier(max_depth=8, n_estimators=150)
]

# 평균 확률
y_probs_avg = np.mean([clf.predict_proba(X)[:, 1] for clf in clsmodels], axis=0)
```

**효과**:
- 다양한 depth로 과적합 방지
- XGBoost + LightGBM 혼합으로 알고리즘 다양성
- 평균으로 안정화

#### Regressor Ensemble (2 variants)
```python
models = [
    XGBRegressor(max_depth=8, ...),
    XGBRegressor(max_depth=10, ...)
]

# 평균 예측
y_pred_avg = np.mean([reg.predict(X) for reg in models], axis=0)
```

**효과**:
- 서로 다른 depth로 학습
- 평균으로 분산 감소

---

## 3. 섹터 카테고리화 전략

### 3.1 문제 정의

**문제**: 일부 섹터는 샘플 수가 너무 적어 과적합 위험
```
Conglomerates: 118개 (너무 적음!)
Utilities: 342개
Healthcare: 1234개 (충분)
```

**해결**: 경제적 특성에 따라 섹터 통합

### 3.2 카테고리 정의

```
원본 섹터 (11개)          →    카테고리 (5개)             경제적 특성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Financials                →    Financial                이자율/신용 민감
Real Estate               ↗

Information Technology    →    Technology               혁신 중심, 고성장
Communication Services    ↗

Consumer Staples          →    Defensive                경기 방어
Utilities                 ↗
Healthcare                ↗

Industrials               →    Cyclical                 경기 순환
Materials                 ↗
Energy                    ↗
Consumer Discretionary    ↗

Conglomerates             →    Others                   기타
```

### 3.3 구현

**설정 (conf.yaml)**:
```yaml
ML:
  USE_SECTOR_MODEL: Y

  SECTOR_CATEGORIZATION:
    ENABLED: Y  # 카테고리화 활성화

    CATEGORIES:
      Financial:
        sectors: [Financials, Real Estate]
        description: "이자율 및 신용 주기에 민감"
        model_config:
          model: xgboost
          n_estimators: 200
          max_depth: 7

      Technology:
        sectors: [Information Technology, Communication Services]
        model_config:
          n_estimators: 250
          max_depth: 8

      # ... (Defensive, Cyclical, Others)

    MIN_SAMPLES_PER_CATEGORY: 1000
    FALLBACK_STRATEGY: "unified"  # or "skip"
```

**매핑 함수 (data_processor.py)**:
```python
@staticmethod
def map_sectors_to_categories(df, config, sector_column='sector', logger=None):
    """
    원본 섹터 → 카테고리 매핑

    ENABLED=Y: Financials → Financial
    ENABLED=N: Financials → Financials (원본 유지)
    """
    if not config.get('ML', {}).get('SECTOR_CATEGORIZATION', {}).get('ENABLED'):
        return df  # 원본 섹터 사용

    # 카테고리 매핑 딕셔너리 생성
    sector_to_category = {}
    categories = config['ML']['SECTOR_CATEGORIZATION']['CATEGORIES']

    for cat_name, cat_info in categories.items():
        for sector in cat_info.get('sectors', []):
            sector_to_category[sector] = cat_name

    # Others 카테고리 자동 할당
    for sector in df[sector_column].unique():
        if sector not in sector_to_category:
            sector_to_category[sector] = 'Others'

    # 매핑 적용
    df['category'] = df[sector_column].map(sector_to_category)

    return df
```

### 3.4 Fallback 전략

**Unified Fallback** (권장):
```python
if len(X_category) < MIN_SAMPLES_PER_CATEGORY:
    logger.warning(f"Category {cat} has only {len(X_category)} samples")
    logger.info("Using unified model as fallback")

    # 전체 데이터로 학습한 unified model 사용
    predictions = unified_model.predict(X_category)
```

**Skip Fallback**:
```python
if len(X_category) < MIN_SAMPLES_PER_CATEGORY:
    logger.warning(f"Skipping category {cat} (too few samples)")
    # 해당 카테고리 종목은 예측에서 제외
    continue
```

### 3.5 모델 구성

**CATEGORIZATION.ENABLED=Y**:
```
5개 카테고리 × 6 모델 (4 classifiers + 2 regressors) = 30개
+ 통합 모델 6개 = 총 36개
```

**CATEGORIZATION.ENABLED=N**:
```
11개 섹터 × 6 모델 = 66개
+ 통합 모델 6개 = 총 72개
```

**효과**: 36개 vs 72개 → **50% 모델 수 감소**

---

## 4. 통합 아키텍처 (Unified Architecture)

### 4.1 문제: 코드 이중화

**리팩토링 이전 (Before)**:
```python
# regressor.py
def preprocess_regressor(X, y):
    # Infinite 제거
    X = X.replace([np.inf, -np.inf], np.nan)

    # NaN 처리
    X = X.fillna(0)

    # Sparse column 제거
    missing_ratio = X.isnull().sum() / len(X)
    X = X.loc[:, missing_ratio < 0.8]

    return X, y

# ml_backtest.py
def preprocess_backtest(X, y):
    # Infinite 제거 (약간 다른 방식!)
    X = X.replace(np.inf, np.nan)
    X = X.replace(-np.inf, np.nan)

    # NaN 처리 (다른 기본값!)
    X = X.fillna(-999)

    # Sparse column 제거 (다른 threshold!)
    missing_ratio = X.isnull().sum() / len(X)
    X = X.loc[:, missing_ratio < 0.85]

    return X, y
```

**문제**:
1. **로직 불일치**: regressor와 ml_backtest의 전처리가 다름
2. **유지보수 어려움**: 한쪽만 수정되는 버그 발생
3. **검증 무효화**: "예측도는 좋은데 수익률이 낮다"는 괴리

### 4.2 해결: 통합 모듈

#### DataSchema (Single Source of Truth)

**이전 (Before)**:
```python
# regressor.py
y_col_list = ['symbol', 'sector', 'rebalance_date', 'price', ...]

# ml_backtest.py
exclude_cols = ['symbol', 'sec', 'date', 'price_col', ...]  # 다름!
```

**이후 (After)**:
```python
# data_schema.py
class DataSchema:
    METADATA_COLS = ['symbol', 'sector', 'industry', ...]
    DATE_COLS = ['rebalance_date', 'report_date', 'filingDate']
    PRICE_VOLUME_COLS = ['price', 'volume', 'marketCap', ...]
    TARGET_COLS = ['price_dev', 'price_dev_subavg', ...]

    @classmethod
    def get_excluded_cols(cls):
        return cls.METADATA_COLS + cls.DATE_COLS + cls.PRICE_VOLUME_COLS + cls.TARGET_COLS

# regressor.py, ml_backtest.py 모두
excluded = DataSchema.get_excluded_cols()
```

**효과**: 컬럼 정의 단일화 → 일관성 보장

#### DataProcessor (Unified Preprocessing)

**이전 (Before)**:
```python
# regressor.py (코드 200줄)
# ml_backtest.py (코드 200줄)
# 총 400줄 중복!
```

**이후 (After)**:
```python
# data_processor.py (통합, 200줄)
class DataProcessor:
    @staticmethod
    def preprocess_training_data(X, y, y_cls, config, logger):
        """
        통합 전처리 파이프라인
        regressor.py와 ml_backtest.py 모두 사용
        """
        # 1. Infinite 제거
        X = DataProcessor.remove_infinite_values(X, logger)

        # 2. Log transformation (선택사항)
        if config.get('USE_LOG_TRANSFORM', False):
            X = DataProcessor.apply_log_transform(X, logger)

        # 3. Sparse column/row 제거
        X = DataProcessor.drop_sparse_cols(X, threshold=0.8, logger=logger)
        X, y, y_cls = DataProcessor.drop_sparse_rows(X, y, y_cls, threshold=0.6, logger=logger)

        # 4. Winsorization (선택사항)
        if config.get('FEATURES', {}).get('USE_WINSORIZATION') == 'Y':
            X = DataProcessor.winsorize_features(X, lower=0.01, upper=0.99, logger=logger)

        # 5. Feature selection (선택사항)
        if config.get('FEATURES', {}).get('USE_FEATURE_SELECTION') == 'Y':
            X, selected_features = DataProcessor.select_features(X, y, logger=logger)

        return X, y, y_cls, selected_features

# regressor.py
X, y, y_cls, features = DataProcessor.preprocess_training_data(X, y, y_cls, config, logger)

# ml_backtest.py
X, y, y_cls, features = DataProcessor.preprocess_training_data(X, y, y_cls, config, logger)
```

**효과**: 400줄 → 200줄 (50% 감소), 로직 일원화

#### ModelFactory (Factory Pattern)

**이전 (Before)**:
```python
# regressor.py (모델 생성 100줄)
clsmodels = [
    XGBClassifier(...),
    XGBClassifier(...),
    XGBClassifier(...),
    LGBMClassifier(...)
]

models = [
    XGBRegressor(...),
    XGBRegressor(...)
]

# ml_backtest.py (모델 생성 75줄)
clf = XGBClassifier(...)
reg = XGBRegressor(...)
```

**이후 (After)**:
```python
# model_factory.py (통합, 531줄)
class ModelFactory:
    def create_ensemble_models(self):
        """regressor.py용: 4 classifiers + 2 regressors"""
        return classifiers, regressors

    def create_single_models(self, use_gpu):
        """ml_backtest.py용: 1 classifier + 1 regressor"""
        return classifier, regressor

    def create_sector_models(self, sector_list):
        """섹터별: 각 섹터당 6 모델"""
        return sector_classifiers, sector_regressors

# regressor.py
factory = ModelFactory(config, optuna_params, use_ensemble=True)
clsmodels, models = factory.create_ensemble_models()

# ml_backtest.py
factory = ModelFactory(config, optuna_params=None, use_ensemble=False)
clf, reg = factory.create_single_models(use_gpu=False)
```

**효과**: 175줄 → 531줄 (통합 코드), 중복 제거

### 4.3 통합 효과 요약

| 항목 | 이전 (Before) | 이후 (After) | 개선 |
|------|--------------|-------------|------|
| **컬럼 정의** | regressor/ml_backtest 각각 | DataSchema 단일 | 일관성 ✅ |
| **전처리** | 400줄 중복 | 200줄 통합 | -50% |
| **모델 생성** | 175줄 중복 | 531줄 통합 | 중복 제거 ✅ |
| **Feature Alignment** | 100줄 중복 | 50줄 통합 | -50% |
| **총 중복 코드** | ~825줄 | 0줄 | -100% ✅ |

---

## 5. 설계 패턴 분석

### 5.1 Factory Pattern (ModelFactory)

**정의**: 객체 생성 로직을 캡슐화

**구현**:
```python
class ModelFactory:
    def __init__(self, config, optuna_params, use_ensemble, logger):
        self.config = config
        self.optuna_params = optuna_params
        self.use_ensemble = use_ensemble

    def create_ensemble_models(self):
        """4 classifiers + 2 regressors"""
        if self.config['ML']['USE_CLASSIFIER'] == 'Y':
            clsmodels = self._create_classifiers(n=4)
        else:
            clsmodels = []

        models = self._create_regressors(n=2)
        return clsmodels, models
```

**장점**:
- 모델 생성 로직 단일화
- Optuna 파라미터 자동 적용
- regressor ↔ ml_backtest 일원화

### 5.2 Strategy Pattern (Preprocessing)

**정의**: 알고리즘을 캡슐화하여 교체 가능하게 만듦

**구현**:
```python
# 전략 1: Winsorization
if config['FEATURES']['USE_WINSORIZATION'] == 'Y':
    X = DataProcessor.winsorize_features(X)

# 전략 2: Feature Selection
if config['FEATURES']['USE_FEATURE_SELECTION'] == 'Y':
    X, features = DataProcessor.select_features(X, y)

# 전략 3: Log Transform
if config['USE_LOG_TRANSFORM']:
    X = DataProcessor.apply_log_transform(X)
```

**장점**:
- 전처리 단계를 설정으로 on/off 가능
- Ablation Study 용이

### 5.3 Template Method Pattern (Backtest)

**정의**: 알고리즘의 골격을 정의하고 세부 단계를 하위 클래스에서 구현

**구현**:
```python
class MLBacktest:
    def run(self):
        """Template Method: 백테스트 골격"""
        # 1. 데이터 준비
        self._prepare_data()

        # 2. 리밸런싱 날짜 생성
        rebalance_dates = self._generate_rebalance_dates()

        # 3. 각 날짜마다 반복
        for date in rebalance_dates:
            # a. 미래 유출 방지 데이터 로드
            X, y = self._get_available_data_until(date)

            # b. 모델 재학습 (필요 시)
            if self._should_retrain(date):
                self._train_model(X, y)

            # c. 예측
            predictions = self._predict(X)

            # d. Top-K 선택
            top_k = self._select_top_k(predictions)

            # e. 수익률 계산
            returns = self._calculate_period_return(top_k, date)

            results.append(returns)

        # 4. 리포트 생성
        self._generate_report(results)
```

**장점**:
- 백테스트 프로세스 표준화
- 확장 용이 (새로운 전략 추가 시)

### 5.4 Singleton Pattern (DataSchema)

**정의**: 클래스의 인스턴스가 하나만 존재하도록 보장

**구현**:
```python
class DataSchema:
    """
    Static class (Singleton 대신)
    모든 메서드가 @classmethod or @staticmethod
    """
    METADATA_COLS = [...]
    DATE_COLS = [...]

    @classmethod
    def get_excluded_cols(cls):
        return cls.METADATA_COLS + cls.DATE_COLS + ...
```

**장점**:
- 컬럼 정의 단일화 (Single Source of Truth)
- 메모리 효율

---

## 6. 아키텍처 평가

### 6.1 강점

| 항목 | 평가 | 설명 |
|------|------|------|
| **철학 일관성** | A+ | "예측 아닌 선별"이 아키텍처 전반에 반영 |
| **2-Stage 설계** | A | 안정성 필터링 + 수익률 예측, 투자 프로세스와 일치 |
| **통합 아키텍처** | A | 코드 이중화 제거 (825줄 → 0줄) |
| **섹터 카테고리화** | A- | 샘플 부족 문제 해결, 경제적 합리성 |
| **Factory Pattern** | A | 모델 생성 통합, 유지보수 용이 |
| **확장성** | B+ | 새로운 모델/전략 추가 비교적 용이 |

### 6.2 약점

| 항목 | 평가 | 설명 |
|------|------|------|
| **복잡도** | C | 파이프라인 단계가 너무 많음 (20+ 단계) |
| **과적합 위험** | C+ | 모델 수 과다 (36~72개), Feature 수 과다 |
| **테스트 커버리지** | D | 단위 테스트 없음 |
| **프로덕션 준비도** | C+ | 모니터링, Alert, 리스크 관리 미흡 |
| **문서화** | B | CLAUDE.md는 우수하나, API 문서 부족 |

### 6.3 퀀트 투자 관점 평가

#### 학술적 관점 (Academic)
```
논문으로 발표 가능? YES
이유:
  - 명확한 철학 (예측 아닌 선별)
  - 2-Stage 아키텍처 novel
  - 엄격한 백테스트 (Walk-Forward)
  - 미래 유출 방지 (Filing Date)
```

#### 실전 관점 (Practical)
```
실제 돈 투자 가능? YES, but with caution
이유:
  - 백테스트는 엄격하나 Out-of-Sample 기간 짧음 (2년)
  - 과적합 위험 높음 (모델 수, Feature 수)
  - 프로덕션 모니터링 미흡

권장:
  - 소액 파일럿 운용 (총 자산 5%)
  - 3~6개월 실전 검증 후 확대
```

#### 산업 관점 (Industry)
```
헤지펀드/자산운용사 도입 가능? YES, with improvements
필요 개선사항:
  - 단위 테스트 추가
  - 프로덕션 모니터링 (Prometheus + Grafana)
  - 리스크 관리 강화 (Stop-Loss, Position Sizing)
  - CI/CD 파이프라인
  - 클라우드 배포 (AWS/GCP)
```

---

## 7. 개선 제안

### 7.1 단기 개선 (1~2주)

#### 1. 복잡도 감소
```
현재: 20+ 단계
목표: 10 단계

방법:
  - Ablation Study 실행
  - 각 단계의 성능 기여도 측정
  - 불필요한 단계 제거
```

**예시**:
```python
# Ablation Study
baseline = train_and_evaluate(stages=['basic'])  # RMSE: 0.15
+ winsorization = train_and_evaluate(stages=['basic', 'winsor'])  # RMSE: 0.14 (-0.01)
+ feature_selection = train_and_evaluate(stages=['basic', 'winsor', 'feat_sel'])  # RMSE: 0.14 (no change)
# → Feature selection 제거 가능!
```

#### 2. 과적합 방지
```
현재: 36~72개 모델
목표: 10~20개 모델

방법:
  - 섹터별 모델의 실제 성능 검증
  - 통합 모델 vs 섹터 모델 비교
  - 성능 차이 없으면 통합 모델만 사용
```

**예시**:
```python
# 성능 비교
unified_sharpe = backtest(model='unified')  # Sharpe: 1.2
sector_sharpe = backtest(model='sector')    # Sharpe: 1.3 (+0.1)

if sector_sharpe - unified_sharpe < 0.2:
    print("섹터별 모델 효과 미미 → 통합 모델 사용 권장")
```

#### 3. Feature 선택
```
현재: 수백~수천 개 Feature
목표: Top-50 Feature만 사용

방법:
  - Feature Importance 분석 (XGBoost/SHAP)
  - 상위 50개만 선택
  - 성능 변화 측정
```

**예시**:
```python
# Feature Importance
importances = model.feature_importances_
top_50 = np.argsort(importances)[-50:]

X_reduced = X[:, top_50]
performance_reduced = train_and_evaluate(X_reduced)

print(f"Original: {performance_original}")
print(f"Reduced: {performance_reduced}")
# Original: RMSE 0.15, Sharpe 1.2
# Reduced: RMSE 0.16, Sharpe 1.1 (거의 비슷!)
# → Top-50만 사용해도 충분
```

### 7.2 중기 개선 (1~2개월)

#### 1. 단위 테스트
```python
# tests/test_data_processor.py
def test_remove_infinite_values():
    X = pd.DataFrame({'a': [1, np.inf, 3], 'b': [4, 5, -np.inf]})
    X_clean = DataProcessor.remove_infinite_values(X)
    assert not np.isinf(X_clean).any().any()

def test_align_features_to_model():
    X = pd.DataFrame({'feat_a': [1, 2], 'feat_b': [3, 4]})
    model = MockModel(features=['feat_b', 'feat_c'])  # feat_c 없음
    X_aligned = DataProcessor.align_features_to_model(X, model)
    assert 'feat_c' in X_aligned.columns
    assert X_aligned['feat_c'].isnull().all()
```

#### 2. Out-of-Sample 검증
```yaml
# 여러 기간 백테스트
BACKTEST:
  PERIODS:
    # 2008 금융위기
    - START_YEAR: 2008
      END_YEAR: 2009

    # 2020 코로나 충격
    - START_YEAR: 2020
      END_YEAR: 2021

    # 2022 금리 급등기
    - START_YEAR: 2022
      END_YEAR: 2023
```

#### 3. 시장 레짐 분석
```python
# 시장 상황별 성능 분석
def analyze_by_market_regime(backtest_results):
    """
    Bull Market: SPY 연간 +10% 이상
    Bear Market: SPY 연간 -10% 이하
    Sideways: 그 외
    """
    regime_performance = {}

    for year, performance in backtest_results.items():
        spy_return = get_spy_return(year)

        if spy_return > 0.10:
            regime = 'Bull'
        elif spy_return < -0.10:
            regime = 'Bear'
        else:
            regime = 'Sideways'

        regime_performance.setdefault(regime, []).append(performance)

    for regime, perfs in regime_performance.items():
        print(f"{regime}: Sharpe {np.mean(perfs):.2f}")

    # 예시 출력:
    # Bull: Sharpe 1.5 (좋음)
    # Bear: Sharpe 0.8 (보통)
    # Sideways: Sharpe 0.5 (나쁨) ← 문제 발견!
```

### 7.3 장기 개선 (3~6개월)

#### 1. Microservices 아키텍처
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Service│    │ ML Service  │    │ Backtest    │
│             │→→→→│             │→→→→│ Service     │
│ FMP API     │    │ Training    │    │ Walk-Fwd    │
└─────────────┘    └─────────────┘    └─────────────┘
      ↓                   ↓                   ↓
┌─────────────────────────────────────────────────────┐
│             Message Queue (RabbitMQ/Kafka)          │
└─────────────────────────────────────────────────────┘
      ↓                   ↓                   ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Monitoring  │    │ Alert       │    │ Dashboard   │
│ (Prometheus)│    │ (Slack/Email│    │ (Grafana)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

#### 2. Event-Driven 아키텍처
```python
# 이벤트 기반
class EventBus:
    def publish(self, event: Event):
        for handler in self.handlers[event.type]:
            handler.handle(event)

# 이벤트 정의
@dataclass
class ModelTrainedEvent:
    model_id: str
    performance: dict
    timestamp: datetime

# 핸들러
class AlertHandler:
    def handle(self, event: ModelTrainedEvent):
        if event.performance['sharpe'] < 0.5:
            send_alert("Low Sharpe Ratio detected!")
```

#### 3. 고급 ML 모델
```python
# Transformer 기반 시계열
from transformers import TimeSeriesTransformer

class TransformerRegressor:
    def __init__(self, seq_len=20):
        self.model = TimeSeriesTransformer(
            input_dim=100,
            output_dim=1,
            n_heads=8,
            n_layers=6
        )

    def fit(self, X_seq, y):
        # X_seq: (n_samples, seq_len, n_features)
        self.model.train(X_seq, y)

    def predict(self, X_seq):
        return self.model.predict(X_seq)
```

---

## 결론

### 아키텍처 평가 요약

**전체 등급**: **A- (우수)**

**핵심 강점**:
1. 명확한 철학 반영 (예측 아닌 선별)
2. 2-Stage 설계 (안정성 + 수익성)
3. 통합 아키텍처 (코드 이중화 제거)
4. 섹터 카테고리화 (샘플 부족 해결)

**주요 약점**:
1. 복잡도 높음 (20+ 단계)
2. 과적합 위험 (모델 수, Feature 수)
3. 테스트 부족 (단위 테스트 없음)
4. 프로덕션 준비도 미흡

**개선 우선순위**:
1. **복잡도 감소** (Ablation Study)
2. **단위 테스트** 추가
3. **Out-of-Sample 검증** 확대
4. **프로덕션 모니터링** 구축

---

**다음 문서**: [02_data_pipeline.md](./02_data_pipeline.md) - 데이터 파이프라인 상세 분석
