# ML 전략 상세 분석

> **작성일**: 2025-12-17 (최종 업데이트: 2026-01-17)
> **이전 문서**: [02_data_pipeline.md](./02_data_pipeline.md)
> **다음 문서**: [04_backtesting.md](./04_backtesting.md)

---

## 핵심 요약

### ML 전략 평가: B+ (정교하나 과적합 위험)

**강점**:
- 2-Stage 아키텍처 (Classifier → Regressor)
- 자동 Threshold 최적화
- Optuna 하이퍼파라미터 튜닝
- 섹터 카테고리화

**약점**:
- 모델 수 과다 (36~72개)
- Feature 수 과다 (500~1000개)
- 복잡도 높음

---

## 1. 2-Stage 전략 심층 분석

### 철학적 타당성

**전통적 회귀 (X)**:
```
문제: 모든 종목 동시 학습
→ 부실 기업(하위 50%) + 건전 기업(상위 50%)
→ 노이즈 많음, 패턴 불명확
```

**2-Stage (O)**:
```
Stage 1: 부실 기업 제거 (상위 7% 선택)
Stage 2: 건전 기업만으로 수익률 예측
→ Clean 데이터, 패턴 명확
```

### 수학적 분석

**Hard Filtering**:
```python
safe_stocks = stocks[y_proba > percentile(93)]  # 상위 7%
predictions = regressor.predict(safe_stocks)
top_k = safe_stocks.sort_values('prediction').head(K)
```

**Soft Filtering (현재 구현)**:
```python
ml_score = y_proba * y_pred_return
top_k = all_stocks.sort_values('ml_score').head(K)
```

**동등성 증명**:
- K < 7%: 두 방식 동일
- K > 7%: Soft가 약간 더 유연 (하위 종목도 포함 가능)

---

## 2. CLASSIFIER_MODE: 분류기 작동 모드 (2026-01 추가)

### 개요

CLASSIFIER_MODE는 분류기의 타겟 정의와 필터링 방향을 결정합니다.

```yaml
ML:
  CLASSIFIER_MODE: "negative_screen"  # 또는 "positive_screen"
  CLASSIFIER_REMOVE_PCT_MIN: 2        # 최소 제거 비율 (%)
  CLASSIFIER_REMOVE_PCT_MAX: 15       # 최대 제거 비율 (%)
  NEGATIVE_SCREEN:
    LOSS_THRESHOLD: -0.3              # -30% 이하 손실 = BAD
```

### Mode 1: negative_screen (권장)

**철학**: "나쁜 것을 피하는 것이 좋은 것 찾기보다 쉽다"

**타겟 정의**:
```python
# 극단적 손실 = BAD 레이블
label_binary = (price_dev < -0.3).astype(int)
# Class 0 = OK (안전), Class 1 = BAD (위험)
```

**필터링 로직**:
```python
y_probs = classifier.predict_proba(X)[:, 1]  # BAD 확률

# 상위 N% (BAD 확률 높음) 제거
threshold = np.percentile(y_probs, 100 - remove_pct)  # 예: 92 percentile
safe_mask = y_probs < threshold  # 상위 8% 제거 → 92% 유지
```

**장점**:
- ✅ 대부분의 데이터 유지 (85~98%)
- ✅ 파산/폭락 패턴은 명확하여 예측 용이
- ✅ Conservative 전략으로 리스크 최소화

### Mode 2: positive_screen

**철학**: "상승 가능성 높은 종목을 적극 선택"

**타겟 정의**:
```python
# 상승 = GOOD 레이블
label_binary = (price_dev > 0.0).astype(int)
# Class 0 = LOSS (하락), Class 1 = GOOD (상승)
```

**필터링 로직**:
```python
y_probs = classifier.predict_proba(X)[:, 1]  # GOOD 확률

# 하위 N% (GOOD 확률 낮음) 제거
threshold = np.percentile(y_probs, remove_pct)  # 예: 8 percentile
safe_mask = y_probs > threshold  # 하위 8% 제거 → 92% 유지
```

**단점**:
- ⚠️ "좋은 것 찾기"는 노이즈 많음
- ⚠️ 과적합 위험 (특정 시기 패턴에만 맞을 수 있음)

### 실제 구현 위치

| 파일 | 함수 | 역할 |
|------|------|------|
| `data_processor.py` | `create_binary_target()` | 모드에 따른 타겟 생성 |
| `regressor.py` | `_find_optimal_threshold()` | threshold 자동 탐색 |
| `ml_backtest.py` | `_calculate_threshold_config()` | 섹터별 threshold 계산 |
| `ml_backtest.py` | `_filter_by_classifier()` | hard filtering 적용 |

### threshold_config.pkl 구조

학습 시 저장되는 설정:
```python
threshold_config = {
    'mode': 'negative_screen',        # 또는 'positive_screen'
    'percentile': 92,                 # 자동 탐색된 최적 percentile
    'threshold_value': 0.15,          # 해당 percentile의 확률값
    'remove_pct': 8,                  # 실제 제거 비율 (100 - percentile)
    'precision': 0.78,                # 해당 threshold에서의 precision
    'recall': 0.65,                   # 해당 threshold에서의 recall
    'n_selected': 8500,               # 선택된 샘플 수
    'n_total': 9200                   # 전체 샘플 수
}
```

### 권장 설정

| 시장 환경 | 권장 모드 | 이유 |
|-----------|-----------|------|
| 불확실/하락장 | `negative_screen` | 리스크 회피 우선 |
| 상승장 | `positive_screen` | 공격적 수익 추구 |
| 기본값 | `negative_screen` | 안정성 우선 |

---

## 3. Threshold 최적화 전략

### 자동 탐색 알고리즘

```python
def _find_optimal_threshold(y_true, y_probs, strategy='balance'):
    best_pct = None
    
    for pct in range(85, 99):  # 상위 15% ~ 2%
        threshold = np.percentile(y_probs, pct)
        mask = y_probs > threshold
        
        precision = precision_score(y_true[mask], y_pred[mask])
        n_selected = mask.sum()
        
        if strategy == 'balance':
            # 최소 precision 만족하면서 최대 데이터
            if precision >= 0.65 and n_selected > best_n:
                best_pct = pct
                best_n = n_selected
        
        elif strategy == 'precision':
            # 최대 precision
            if precision > best_precision:
                best_pct = pct
                best_precision = precision
    
    return best_pct
```

**전략 비교**:
| 전략 | 장점 | 단점 |
|------|------|------|
| precision | 정확도 최대화 | 데이터 손실 큼 |
| balance | 정확도 + 데이터 균형 | Precision 약간 희생 |

**권장**: balance (기본값)

---

## 4. Optuna 하이퍼파라미터 최적화

### 탐색 공간

```yaml
OPTUNA_SEARCH_SPACE:
  n_estimators: [100, 500]      # 메모리 안전
  learning_rate: [0.01, 0.3]    # 학습 속도
  max_depth: [3, 8]             # 과적합 방지
  subsample: [0.5, 1.0]         # 행 샘플링
  colsample_bytree: [0.5, 1.0]  # 열 샘플링
  gamma: [0, 10]                # 분할 최소 손실
```

### 최적화 프로세스

```python
def objective(trial):
    # 1. 하이퍼파라미터 샘플링
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        ...
    }
    
    # 2. Cross-Validation
    scores = []
    for train_idx, val_idx in cv_splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)
    
    # 3. 평균 성능 반환
    return np.mean(scores)

# 4. 최적화 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=3600)

best_params = study.best_params
```

**효과**:
- 수동 튜닝 대비 5~10% 성능 향상
- 시간: 50 trials × 5 CV folds = ~40시간 (섹터별)

**개선안**:
- Pruning 적용 (중간 성능 낮으면 조기 중단)
- Multi-objective (Accuracy + MDD)

---

## 5. 섹터별 모델 전략

### 카테고리화 효과

**BEFORE (11개 섹터)**:
```
Conglomerates: 118개 (너무 적음!)
→ 과적합 위험 높음
→ 11 × 6 = 66개 모델
```

**AFTER (5개 카테고리)**:
```
Financial: 1500개 (Financials 1200 + Real Estate 300)
Technology: 2000개 (IT 1500 + Communication 500)
Defensive: 1800개 (Staples 600 + Utilities 400 + Healthcare 800)
Cyclical: 2200개 (Industrials 800 + Materials 500 + Energy 400 + Discretionary 500)
Others: 300개 (Conglomerates 등)
→ 5 × 6 = 30개 모델
```

**효과**:
- 샘플 수 부족 해결
- 모델 수 50% 감소 (66 → 30)

### Fallback 전략

```python
if len(X_category) < MIN_SAMPLES (1000):
    if FALLBACK == 'unified':
        # 전체 데이터 모델 사용
        predictions = unified_model.predict(X_category)
    elif FALLBACK == 'skip':
        # 해당 카테고리 스킵
        continue
```

**권장**: unified (모든 종목 예측 가능)

---

## 6. 앙상블 전략

### Classifier Ensemble (4 variants)

```python
clsmodels = [
    XGBClassifier(max_depth=8, ...),  # Optuna 최적화
    XGBClassifier(max_depth=9, ...),  # 깊은 트리
    XGBClassifier(max_depth=10, ...), # 더 깊은 트리
    LGBMClassifier(max_depth=8, ...)  # 알고리즘 다양성
]

# 평균 확률
y_probs = np.mean([clf.predict_proba(X)[:, 1] for clf in clsmodels], axis=0)
```

**효과**:
- Depth 다양성 → 과적합 방지
- 알고리즘 다양성 (XGB + LGBM) → robust
- 분산 감소

### Regressor Ensemble (2 variants)

```python
models = [
    XGBRegressor(max_depth=8, ...),
    XGBRegressor(max_depth=10, ...)
]

# 평균 예측
y_pred = np.mean([reg.predict(X) for reg in models], axis=0)
```

**효과**:
- Depth 다양성
- 분산 감소

---

## 7. 과적합 방지 메커니즘

### 1. Early Stopping

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,  # 10 round 동안 개선 없으면 중단
    verbose=False
)
```

### 2. L1/L2 Regularization

```python
XGBRegressor(
    reg_alpha=0.1,   # L1 regularization
    reg_lambda=1.0   # L2 regularization
)
```

### 3. Max Depth 제한

```yaml
OPTUNA_SEARCH_SPACE:
  max_depth: [3, 8]  # 8로 제한 (메모리 + 과적합 방지)
```

### 4. Subsample

```yaml
subsample: [0.5, 1.0]         # 행 샘플링
colsample_bytree: [0.5, 1.0]  # 열 샘플링
```

---

## 8. 문제점 및 개선안

### 문제 1: 모델 수 과다

**현재**:
- USE_SECTOR_MODEL=Y, CATEGORIZATION.ENABLED=Y: 36개
- USE_SECTOR_MODEL=Y, CATEGORIZATION.ENABLED=N: 72개

**우려**:
- 학습 시간 증가 (40시간+)
- 메모리 사용량 증가 (10GB+)
- 유지보수 어려움

**개선안**:
```python
# 성능 비교
unified_perf = backtest(model='unified')
sector_perf = backtest(model='sector')

if sector_perf - unified_perf < 0.2:  # Sharpe Ratio 차이 0.2 미만
    print("섹터별 모델 효과 미미 → 통합 모델 권장")
```

### 문제 2: Feature 수 과다

**현재**: 500~1000개

**우려**:
- Spurious correlation
- 과적합
- 학습 시간 증가

**개선안**:
```python
# Feature Importance 기반 선택
importances = model.feature_importances_
top_50 = np.argsort(importances)[-50:]

X_reduced = X[:, top_50]
performance = train_and_evaluate(X_reduced)
# 성능 차이 미미하면 Top-50만 사용
```

### 문제 3: Threshold 전략 검증

**현재**: balance (precision >= 0.65, 최대 데이터)

**우려**:
- 0.65는 임의 설정
- 최적값 검증 필요

**개선안**:
```python
# Grid Search
for min_precision in [0.60, 0.65, 0.70, 0.75]:
    threshold = find_optimal_threshold(min_precision=min_precision)
    backtest_perf = run_backtest(threshold)
    
    print(f"Min Precision {min_precision}: Sharpe {backtest_perf}")

# 최적 min_precision 선택
```

---

## 결론

### ML 전략 평가: A- (2026-01 업데이트)

**강점**:
- 철학적 타당성 (2-Stage)
- 자동화 (Threshold, Optuna)
- 섹터 특성 고려
- ✅ **CLASSIFIER_MODE 유연성** (negative_screen/positive_screen) - 시장 환경별 전략 전환 가능
- ✅ **Hard Filtering 구현 완료** - 확률 가중치 대신 명확한 cutoff 적용

**약점**:
- 복잡도 높음
- 과적합 위험
- 검증 부족

### 최근 개선 (2026-01)
1. ✅ **CLASSIFIER_MODE** 구현 완료 (`negative_screen` / `positive_screen`)
2. ✅ **threshold_config.pkl** 구조 개선 (mode, remove_pct 필드 추가)
3. ✅ **Hard Filtering** 적용 (`ml_score = np.where(pass_mask, return, -np.inf)`)
4. ✅ **4-classifier 앙상블 평균**으로 섹터 threshold 계산

### 개선 우선순위
1. Feature 선택 (Top-50)
2. 섹터 모델 효과 검증
3. Threshold 전략 Grid Search
4. ⏳ positive_screen 모드 실전 검증

---

**다음 문서**: [04_backtesting.md](./04_backtesting.md)
