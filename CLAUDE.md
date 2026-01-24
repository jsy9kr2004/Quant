# Claude AI 작업 가이드

## 🌟 시스템 철학 (System Philosophy)

### 목적: "예측이 아닌 선별 (Selection over Prediction)"

이 시스템은 **"미래 가격 맞추기"가 아니라 "상대적 저평가(Mispricing) 종목의 정렬(Ranking)"**을 목표로 합니다.

#### 핵심 개념

1. **펀더멘털 중심 (Fundamental-Anchored)**
   - 적정가치 계산은 불가능함을 인정
   - 재무제표 기반의 **안정성 필터링**을 1차 목표로 함
   - 단기 모멘텀/흥행주는 배제하고, **내재가치 회귀** 가능성이 있는 종목에 집중

2. **선별 전략 (Stock Selection)**
   - ML 모델의 출력값은 절대적인 가격이 아닌, **상대적인 순위(Score)**로 활용
   - 비대칭적 기대수익을 가진 후보를 골라내는 것이 목표
   - Top-K 선정: 상위 종목만 선택하여 포트폴리오 구성

3. **수익률의 다원성 인정**
   - 수익률은 단일 요인이 아님
   - 매크로/섹터 변수를 고려하여 **상대적 수익률**을 예측
   - `price_dev_subavg`: 전체 평균 대비 상대 수익률
   - `sec_price_dev_subavg`: 섹터 평균 대비 상대 수익률

---

## ⚠️ 데이터 처리 핵심 원칙 (Critical Data Handling Principles)

### 원칙 1: NaN ≠ 0 (Missing vs Zero)

**절대 동일시하지 말 것!**

#### NaN (Not a Number) - Missing Data
**의미**: 재무제표 항목 자체가 존재하지 않음

**발생 원인**:
- 산업별 특성: 은행의 대출이자율 (제조업에는 없음)
- 기업 성장 단계: 스타트업의 영업이익 (아직 발생하지 않음)
- 회계 시점: 상장 전 분기 데이터
- 상장폐지/파산: 미래 가격 데이터 없음

**올바른 처리**:
```python
# ✅ CORRECT
# 1. 타겟 변수의 NaN: 완전히 제거
if y.isna().any():
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

# 2. Feature의 NaN: 컨텍스트 고려하여 imputation
# - 업종별/규모별 중앙값
# - 또는 "결측" 자체를 정보로 활용 (별도 indicator)
```

**잘못된 처리**:
```python
# ❌ WRONG
df.fillna(0)  # "데이터 없음" = "실제로 0" (완전히 다른 의미!)
→ 모델이 "영업이익 항목 없는 회사 = 영업이익 0인 회사"로 잘못 학습
```

#### 0 (Zero) - Actual Zero Value
**의미**: 재무제표 항목이 존재하지만 실제 값이 0

**발생 원인**:
- 손익분기점: 영업이익 0원
- 경영 전략: 무차입 경영 (부채 0원), 무배당 정책 (배당 0원)
- 사업 특성: 계절성 매출 (비수기 매출 0원)

**올바른 처리**:
```python
# ✅ CORRECT
# 0은 유효한 데이터로 학습에 사용
model.fit(X, y)  # y에 0이 포함되어 있어도 OK
```

#### 실전 예시

**시나리오**: 두 기업의 영업이익

| 기업 | 영업이익 | 의미 |
|------|----------|------|
| A사 | 0 | 손익분기점 (사업은 하지만 이익 없음) |
| B사 | NaN | 스타트업 (아직 본격 영업 전) |

**잘못된 처리**:
```python
df['operating_income'].fillna(0)
→ A사와 B사를 동일하게 취급 (완전히 다른 상황인데!)
```

**올바른 처리**:
```python
# B사(NaN) 제거 또는 별도 그룹으로 분리
# A사(0)는 유효한 데이터로 학습
```

---

### 원칙 2: inf ≠ too large (Infinity vs Overflow)

**절대 동일시하지 말 것!**

#### inf (Infinity) - Mathematical Error
**의미**: 수학적 무한대 (계산 오류)

**발생 원인**:
- Division by zero: `PER = 주가 / EPS`, EPS = 0 → inf
- Logarithm of zero: `log(0)` → -inf
- 계산 오류: `ROE = 순이익 / 자본`, 자본 = 0 → inf

**올바른 처리**:
```python
# ✅ CORRECT - inf는 계산 오류이므로 제거
df = df[~np.isinf(df).any(axis=1)]

# 또는 계산 방식 수정
# PER = 주가 / max(EPS, 0.01)  # 0 방지
```

#### too large (Overflow) - Valid but Out of Range
**의미**: 유효한 값이지만 데이터 타입의 표현 한계 초과

**발생 원인**:
- **현대 기업의 거대화**:
  - 2000년대: 시가총액 억 단위
  - 2020년대: 조 단위 (Apple 3조 달러)
  - 1000배 증가!
- 중간 계산값 overflow:
  - `market_cap × revenue × assets`
  - 각각 10^12 수준이면 곱하면 10^36
  - float64 최대값(~10^308)은 괜찮지만, 계산 중 overflow 가능

**올바른 처리**:
```python
# ✅ CORRECT - 큰 값을 적절히 처리

# 방법 1: Log scaling (곱셈 → 덧셈)
df['log_market_cap'] = np.log1p(df['market_cap'])

# 방법 2: 업종별 상대값 정규화
df['market_cap_rank_in_sector'] = df.groupby('sector')['market_cap'].rank(pct=True)

# 방법 3: Robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['market_cap_scaled'] = scaler.fit_transform(df[['market_cap']])

# 방법 4: 데이터 타입 업그레이드 (필요시)
import decimal
df['market_cap'] = df['market_cap'].astype('float128')  # 또는 Decimal
```

**잘못된 처리**:
```python
# ❌ WRONG - 큰 값을 무시
df[df > 1e15] = np.nan  # 대형 기업 데이터를 NaN으로!
df = df.clip(upper=1e12)  # 대형 기업을 중형으로 축소!

→ 모델이 "대형 기업 = 없음" 또는 "대형 기업 = 중형 기업"으로 잘못 학습
→ Apple, Microsoft 같은 초대형주를 제대로 평가 못 함
```

#### 실전 예시

**시나리오**: PER 계산

```python
# Case 1: EPS = 0 (손실) → PER = inf (수학적 오류)
# Case 2: EPS = 0.000001 (미미한 이익) → PER = 10,000,000 (유효하지만 매우 큼)
```

**잘못된 처리**:
```python
# ❌ WRONG
df['PER'] = df['price'] / df['EPS']
df[df['PER'] > 1000] = np.nan  # Case 2도 제거됨!
```

**올바른 처리**:
```python
# ✅ CORRECT
# inf 제거 (수학적 오류)
df = df[~np.isinf(df['PER'])]

# 큰 값은 유지하되 log scaling
df['log_PER'] = np.log1p(df['PER'].clip(lower=0))
```

---

### 체크리스트

**모든 데이터 전처리 시 확인**:

- [ ] `fillna(0)` 사용? → ❌ NaN과 0의 의미 확인 필수!
- [ ] `df[df == np.inf] = np.nan` 사용? → ❌ inf(오류)와 large(유효) 구분!
- [ ] `clip(upper=threshold)` 사용? → ⚠️ 큰 값이 유효한 데이터인지 확인!
- [ ] 타겟 변수의 NaN? → ✅ 반드시 제거 (학습 불가)
- [ ] Feature의 NaN? → ⚠️ 컨텍스트 고려하여 처리 (제거 또는 imputation)

**원칙 요약**:
1. **NaN = Missing**: 제거 또는 의미 있는 imputation
2. **0 = Zero**: 유효한 데이터로 학습
3. **inf = Error**: 제거 또는 계산 수정
4. **too large = Valid**: Log scaling, normalization, 상대값 변환

---

## 🏗️ System Architecture: 2-Stage ML Structure

안정성과 수익성을 동시에 잡기 위해 모델을 **두 단계로 분리**하여 운용합니다.

### Stage 1: Risk Screening (리스크 스크리닝)

**설계 철학**: Classifier는 **"동전 던지기"가 아닌 "위험 탐지기"**로 설계됨

#### 두 가지 스크리닝 모드

##### Mode 1: Negative Screening (권장) - "위험 제거"

**목적**: 재무적으로 명백히 위험한 종목 제거

**설계 의도**:
- "이 재무제표는 무조건 떨어진다" 종목 걸러내기
- Conservative 전략: 나쁜 것을 피하는 것이 좋은 것 찾기보다 쉬움
- 대량 파산/대폭락(-30% 이상) 종목을 학습하여 패턴 인식

**Binary Target**:
```python
# 극단적 손실을 "BAD" 레이블로 정의
label_binary = (price_dev < -0.3).astype(int)  # -30% 이하 = 1 (BAD), 나머지 = 0 (OK)
```

**Classifier 학습**:
- Class 0 = OK (안전한 종목, -30% 이상 수익률)
- Class 1 = BAD (위험한 종목, -30% 이하 손실)
- 목표: "이 재무제표는 파산/폭락 위험이 있다" 식별

**Threshold 적용** (핵심!):
```python
# 예측 확률
y_probs = classifier.predict_proba(X)[:, 1]  # BAD일 확률

# 상위 2~15% (BAD 확률이 가장 높은 종목) 제거
remove_pct = 10  # 예: 상위 10% 제거
threshold = np.percentile(y_probs, 100 - remove_pct)  # 90 percentile
safe_mask = y_probs < threshold  # BAD 확률 하위 90% = 안전

# Regressor에게 안전한 종목만 전달
x_train_filtered = x_train[safe_mask]
```

**Config 설정**:
```yaml
CLASSIFIER_MODE: "negative_screen"
NEGATIVE_SCREEN:
  LOSS_THRESHOLD: -0.3  # -30% 이하 손실 = BAD
CLASSIFIER_REMOVE_PCT_MIN: 2   # 최소 상위 2% 제거 (98% 유지)
CLASSIFIER_REMOVE_PCT_MAX: 15  # 최대 상위 15% 제거 (85% 유지)
```

**장점**:
- ✅ 많은 데이터 활용 (85~98% 유지)
- ✅ Regressor가 충분한 샘플로 학습
- ✅ "나쁜 것 피하기"는 예측하기 쉬움 (파산 패턴은 명확)

---

##### Mode 2: Positive Screening - "좋은 종목 선택"

**목적**: 상승 가능성이 높은 종목만 선택

**설계 의도**:
- "이 재무제표는 무조건 오른다" 종목 찾기
- Aggressive 전략: 좋은 것을 적극적으로 선택
- 고성장 종목 패턴 학습

**Binary Target**:
```python
# 상승을 "GOOD" 레이블로 정의
label_binary = (price_dev > 0.0).astype(int)  # 0% 이상 = 1 (GOOD), 나머지 = 0 (LOSS)
```

**Classifier 학습**:
- Class 0 = LOSS (하락 종목)
- Class 1 = GOOD (상승 종목)
- 목표: "이 재무제표는 상승 가능성이 높다" 식별

**Threshold 적용**:
```python
# 예측 확률
y_probs = classifier.predict_proba(X)[:, 1]  # GOOD일 확률

# 하위 2~15% (GOOD 확률이 가장 낮은 종목) 제거
remove_pct = 10  # 예: 하위 10% 제거
threshold = np.percentile(y_probs, remove_pct)  # 10 percentile
safe_mask = y_probs > threshold  # GOOD 확률 상위 90% = 유망

# Regressor에게 유망한 종목만 전달
x_train_filtered = x_train[safe_mask]
```

**Config 설정**:
```yaml
CLASSIFIER_MODE: "positive_screen"
CLASSIFIER_REMOVE_PCT_MIN: 2   # 하위 2% 제거
CLASSIFIER_REMOVE_PCT_MAX: 15  # 하위 15% 제거
```

**단점**:
- ⚠️ "좋은 것 찾기"는 어려움 (노이즈 많음)
- ⚠️ 과적합 위험 (특정 시기 패턴에만 맞을 수 있음)

---

#### 학습 프로세스 (Mode 공통)

**1. Binary Target 생성**:
```python
# DataProcessor가 모드에 따라 자동 생성
y_train_binary = DataProcessor.create_binary_target(
    y_train,
    config=conf,  # CLASSIFIER_MODE 읽음
    logger=logger
)
```

**2. Classifier 학습**:
```python
# 4개 앙상블 모델 학습
for model in [XGBClassifier, XGBClassifier, XGBClassifier, LGBMClassifier]:
    model.fit(x_train, y_train_binary)
```

**3. 최적 Threshold 자동 탐색**:
```python
# remove_pct 2~15% 범위에서 탐색
for remove_pct in range(2, 16):
    if mode == "negative_screen":
        threshold = np.percentile(y_probs, 100 - remove_pct)  # 상위 제거
        mask = y_probs < threshold
    else:  # positive_screen
        threshold = np.percentile(y_probs, remove_pct)  # 하위 제거
        mask = y_probs > threshold

    # Precision, Recall 계산
    precision = calculate_precision(mask)

# 최적 remove_pct 선택: Min precision 조건 만족하면서 최대 데이터
optimal_pct = best_remove_pct
```

**4. 학습 데이터 필터링**:
```python
# 최적 threshold로 필터링
x_train_filtered = x_train[safe_mask]
y_train_filtered = y_train[safe_mask]

# Regressor 학습
regressor.fit(x_train_filtered, y_train_filtered)
```

---

#### 예측 시 (Evaluation/Backtest)

**프로세스**:
1. 저장된 `threshold_config.pkl` 로드
2. 분류기로 예측 확률 계산
3. 학습 시와 동일한 로직으로 필터링
4. ml_backtest.py: 필터링된 종목 중 수익률 상위 Top-K 선택

---

#### 모델 구성

```python
# Global Classifiers (USE_CLASSIFIER=Y인 경우)
clsmodels[0]: XGBClassifier (Optuna 최적화)
clsmodels[1]: XGBClassifier (max_depth=9)
clsmodels[2]: XGBClassifier (max_depth=10)
clsmodels[3]: LGBMClassifier (max_depth=8)

# Sector Classifiers (USE_SECTOR_MODEL=Y인 경우)
각 섹터당 4개 분류기 (같은 구조)
```

---

#### ⚠️ 중요: Classifier의 역할

**Classifier는 절대 "동전 던지기"가 아닙니다**:

| 지표 | 의미 | 평가 |
|------|------|------|
| Accuracy 50% | 동전 던지기 | ❌ 쓸모없음 |
| Accuracy 60% | 약간 나음 | ⚠️ 불충분 |
| Accuracy 65%+ | 유의미한 신호 | ✅ 사용 가능 |
| Precision 70%+ | 선택한 종목의 70%가 실제로 안전/유망 | ✅ 좋음 |

**검증 방법**:
- Regressor 학습 후 로그에서 Classifier Accuracy, Precision 확인
- 65% 미만이면 모델 재학습 또는 Feature 개선 필요
- Threshold 자동 탐색이 최소 precision 조건 만족하는지 확인

### Stage 2: Return Forecast (수익률 예측)

**목적**: 단기적으로 무효화된 가치의 회복 가능성을 통계적으로 점수화

#### 학습 시 (Training)

**타겟**: `price_dev_subavg` - 다음 분기 상대 수익률

**프로세스**:
1. Stage 1에서 필터링된 "안전한" 종목 데이터만 사용
2. **회귀 모델 (Regression)** 학습:
   - 입력: 필터링된 종목들의 재무제표 features
   - 출력: 다음 분기 상대 수익률 예측
3. 노이즈(하위 종목)가 제거되어 더 clean한 학습

**장점**:
- 회귀기가 "안전한" 종목들의 패턴만 학습
- 하위 종목 노이즈가 회귀기 학습에 방해하지 않음
- 학습 데이터는 감소하지만 품질 향상

#### 예측 시 (Evaluation/Backtest)

**프로세스**:
1. 모든 종목에 대해 회귀기로 수익률 예측
2. Stage 1 필터링 (확률 weighting 또는 threshold cutoff)
3. 최종 스코어: `ml_score = y_pred_proba × y_pred_return`
4. Top-K 선정하여 포트폴리오 구성

**모델 구성**:
```python
# Global Regressors
models[0]: XGBRegressor (max_depth=8)
models[1]: XGBRegressor (max_depth=10)

# Sector Regressors (USE_SECTOR_MODEL=Y인 경우)
각 섹터당 2개 회귀기
```

### Model Ensemble Strategy

**Classifier Ensemble** (4 variants):
- 다양한 depth로 학습하여 과적합 방지
- XGBoost + LightGBM 혼합으로 알고리즘 다양성 확보
- 평균 또는 투표 방식으로 최종 필터링 결정

**Regressor Ensemble** (2 variants):
- 서로 다른 depth로 학습
- 평균값을 최종 수익률 예측치로 사용

### Architecture Configurations

설정 파일(`config/conf.yaml`)에서 제어:

| 설정 | 설명 | 모델 구성 |
|------|------|-----------|
| `USE_CLASSIFIER: Y`<br>`USE_SECTOR_MODEL: N` | 전역 2-Stage | 4 classifiers + 2 regressors |
| `USE_CLASSIFIER: Y`<br>`USE_SECTOR_MODEL: Y` | 섹터별 2-Stage | 각 섹터: 4 classifiers + 2 regressors |
| `USE_CLASSIFIER: N`<br>`USE_SECTOR_MODEL: N` | 전역 Regression만 | 2 regressors |
| `USE_CLASSIFIER: N`<br>`USE_SECTOR_MODEL: Y` | 섹터별 Regression만 | 각 섹터: 2 regressors |

### 📊 Sector Categorization (섹터 카테고리화)

**목적**: 작은 섹터들을 경제적 특성에 따라 통합하여 샘플 수 부족 문제 해결

#### 문제 상황
- 일부 섹터는 샘플 수가 너무 적어 과적합 위험 (예: Conglomerates 118개)
- 섹터별 모델 학습 시 데이터 부족으로 일반화 성능 저하

#### 해결 방법: 계층적 카테고리 통합
```
원본 섹터 (11개)          →    카테고리 (5개)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Financials                →    Financial
Real Estate               ↗

Information Technology    →    Technology
Communication Services    ↗

Consumer Staples          →    Defensive
Utilities                 ↗
Healthcare                ↗

Industrials               →    Cyclical
Materials                 ↗
Energy                    ↗
Consumer Discretionary    ↗

Conglomerates             →    Others
```

#### 설정 구조
```yaml
ML:
  USE_SECTOR_MODEL: Y  # 섹터별 모델 사용

  # 원본 섹터별 설정 (CATEGORIZATION.ENABLED=N일 때)
  SECTOR_CONFIG:
    Financials:  # 원본 섹터 이름
      model: xgboost
      n_estimators: 200
      max_depth: 7

  # 카테고리 통합 설정 (CATEGORIZATION.ENABLED=Y일 때)
  SECTOR_CATEGORIZATION:
    ENABLED: N  # Y = 카테고리 사용, N = 원본 섹터 사용

    CATEGORIES:
      Financial:  # 카테고리 이름
        sectors: [Financials, Real Estate]
        description: "이자율 및 신용 주기에 민감"
        model_config:  # 카테고리별 모델 설정
          model: xgboost
          n_estimators: 200
          max_depth: 7
          learning_rate: 0.05
```

#### 동작 방식

**ENABLED=N (기본값)**:
1. 원본 섹터 이름 사용 (Financials, Healthcare, ...)
2. `SECTOR_CONFIG`에서 모델 설정 로드
3. 섹터별로 독립적인 모델 학습

**ENABLED=Y (카테고리화 활성화)**:
1. 원본 섹터를 카테고리로 매핑 (Financials → Financial)
2. `SECTOR_CATEGORIZATION.CATEGORIES[].model_config`에서 설정 로드
3. 카테고리별로 통합 모델 학습
4. **레포트에는 원본 섹터 이름 표시** (사용자 친화성)

#### 구현 위치
- **설정**: `config/conf.yaml.template` → `ML.SECTOR_CATEGORIZATION`
- **매핑 로직**: `src/training/data_processor.py` → `map_sectors_to_categories()`
- **모델 설정**: `src/models/model_factory.py` → `_extract_category_configs()`
- **학습/예측**: `src/backtest/ml_backtest.py` → `_train_model_sector()`, `_predict_sector()`

#### 주의사항
- **설정 일관성**: CATEGORIZATION.ENABLED 변경 시 모델 재학습 필요
- **샘플 수 검증**: 카테고리별 최소 샘플 수 확인 (MIN_SAMPLES_PER_CATEGORY)
- **Fallback 전략**: 샘플 부족 시 unified model 사용 또는 skip

### 🔬 분류기 작동 방식 상세 (Classifier Implementation Details)

#### 학습 단계 (Training)

**Step 1: 분류기 학습 (모든 데이터)**

타겟 생성 (`DataProcessor.create_binary_target`):
```python
# 실제 다음 분기 가격 변동 기준
label_binary = (price_dev > 0).astype(int)
# 1 = 가격 상승, 0 = 가격 하락
```

의미:
- 분류기는 "재무제표 → 다음 분기 상승/하락" 패턴을 학습
- 재무적으로 건전한 기업 ≈ 상승 확률 높음 (간접적 proxy)
- 4개 분류기 앙상블로 과적합 방지

**Step 2: 최적 Threshold 자동 탐색** (`_find_optimal_threshold()`)

```python
# 학습된 분류기로 학습 데이터 예측
y_probs = classifier.predict_proba(x_train)[:, 1]

# 여러 percentile 시도 (85~98)
for pct in range(85, 99):
    threshold = np.percentile(y_probs, pct)
    mask = y_probs > threshold

    # 선택된 종목의 precision 계산
    precision = precision_score(y_true[mask], y_pred[mask])
    recall = recall_score(y_true[mask], y_pred[mask])

# 최적 percentile 선택 (balance 모드)
optimal_pct = max(pct where precision >= 0.65, key=n_selected)
```

결과 예시:
```
Percentile 85: threshold=0.612, selected=15000, precision=0.523
Percentile 90: threshold=0.701, selected=10000, precision=0.610
Percentile 93: threshold=0.789, selected=7000,  precision=0.729  ← 선택!
Percentile 95: threshold=0.854, selected=5000,  precision=0.780
```

저장:
- `threshold_config.pkl`: {'percentile': 93, 'threshold_value': 0.789, 'precision': 0.729, ...}

**Step 3: 학습 데이터 필터링**

```python
# 최적 threshold로 필터링
threshold = threshold_config['threshold_value']
safe_mask = y_probs > threshold

x_train_filtered = x_train[safe_mask]  # 상위 7% (93 percentile)
y_train_filtered = y_train[safe_mask]

# 필터링된 데이터로 회귀기 학습
regressor.fit(x_train_filtered, y_train_filtered)
```

효과:
- 회귀기가 상위 7% "안전한" 종목의 패턴만 학습
- 하위 93% 노이즈 제거
- 학습 데이터는 감소하지만 품질 향상

#### 예측 및 필터링 단계 (Prediction & Filtering)

**1. regressor.py (모델 평가)**:
```python
# 저장된 threshold config 로드
threshold_config = joblib.load('threshold_config.pkl')
THRESHOLD_PERCENTILE = threshold_config['percentile']  # 93

# 상승 확률 예측
y_probs = classifier.predict_proba(X)[:, 1]

# 학습 시와 동일한 percentile threshold 적용
threshold = np.percentile(y_probs, THRESHOLD_PERCENTILE)
y_predict_binary = (y_probs > threshold).astype(int)

# 하위 93% 패널티 적용 (평가용)
prediction_wbinary = np.where(y_predict_binary == 0, -1, y_predict_return)
```

**2. ml_backtest.py (백테스트)**:
```python
# 상승 확률 예측
y_pred_proba = classifier.predict_proba(X)[:, 1]

# 수익률 예측
y_pred_return = regressor.predict(X)

# 최종 스코어: 확률 × 수익률 (down-weighting)
ml_score = y_pred_proba * y_pred_return
```

**효과**:
- 상승 확률 높음 (0.9) × 예측 수익률 (+10%) = +9.0
- 상승 확률 낮음 (0.1) × 예측 수익률 (+10%) = +1.0
- Down-weighting으로 하위 종목 점수 자동 하락

#### 수학적 동등성 증명

설정:
- 전체 종목 수: N
- THRESHOLD = 92 (상위 8% 선택)
- Top-K = 5 (포트폴리오 5개 종목)

**Hard Filtering (설계 의도)**:
```
1. 상위 8% 선택: 0.08N 종목
2. Top-5 선택: min(5, 0.08N) 종목
3. N > 62이면 항상 5개 선택
```

**Soft Filtering (현재 구현)**:
```
1. 하위 92% 점수: -1 (또는 매우 낮음)
2. 상위 8% 점수: 원래 예측값 (양수)
3. Top-5 정렬 → 자동으로 상위 8%에서만 선택
```

**결론**: N > K/0.08 조건에서 두 방식 완전 동일

#### 왜 이렇게 구현했는가?

**장점**:
1. **유연성**: Hard cutoff 없이 연속적인 점수 부여
2. **앙상블**: 여러 분류기 확률을 평균/투표로 결합 가능
3. **해석 가능성**: 확률값 자체가 신뢰도 지표
4. **구현 단순성**: 행 제거 없이 벡터 연산만 사용

**단점**:
1. **개념적 혼란**: "필터링"이지만 실제로는 "가중치 조정"
2. **메모리**: 모든 종목 예측 후 정렬 (vs 사전 필터링)

#### 실전 적용 시 주의사항

- **Top-K > 8%인 경우**: 하위 종목도 포함될 수 있음
  - 해결: THRESHOLD 조정 또는 hard filtering 추가
- **분류기 정확도**: 상승/하락 예측이 부정확하면 필터링 효과 저하
  - 해결: 분류기 성능 모니터링 (Accuracy, Precision, Recall)
- **시장 환경 변화**: 과거 패턴이 깨지면 분류기 무용
  - 해결: 주기적 재학습 및 walk-forward validation

---

## 🎯 핵심 원칙: regressor.py ↔ ml_backtest.py 일원화

### 왜 중요한가?

"**모델의 예측도**"(regressor 평가)와 "**수익률**"(ml_backtest 평가)은 별개의 지표입니다.
- 예측도가 좋다고 수익률이 좋은 것은 아닙니다
- 두 평가를 **함께** 봐야 모델의 실제 가치를 판단할 수 있습니다
- **따라서 두 시스템은 반드시 동일한 로직으로 동작해야 합니다**

### 이중화의 위험

코드 이중화가 발생하면:

1. **유지보수 문제**: 한쪽만 수정되는 버그 발생
2. **검증 무효화**: 실수로 한쪽만 달라지면 종합 평가가 무의미해짐
3. **신뢰성 하락**: 백테스트 결과를 믿을 수 없게 됨

### 해결 방법: 아키텍처 기반 일원화 ✅

**핵심 아이디어**: 코드 중복을 줄이는 것을 넘어, **아키텍처 자체가 일원화를 강제**하도록 설계

#### 1단계: Prediction Cache 공유 (2025-12-21)

regressor.py가 생성한 예측 결과를 ml_backtest.py가 **재사용**하는 구조:

```
regressor.py (학습 & 예측)
    ↓
    예측 결과 저장: MODELS/regressor_predictions.pkl
    ↓
ml_backtest.py (수익률 계산만)
    ↑
    캐시 로드 & 재사용 (모델 학습/예측 스킵)
```

**효과**:
- 두 시스템이 **물리적으로 동일한 예측값** 사용
- 코드가 달라도 결과는 동일 (캐시 공유)

#### 2단계: Fallback 제거로 강제 (2025-01-17)

**문제**: 캐시가 없을 때 ml_backtest.py가 자체 학습하면 일원화 깨짐

**기존 코드 (위험)**:
```python
if cache_path.exists():
    self.predictions_cache = joblib.load(cache_path)
else:
    # ❌ Silent fallback - 일원화 위반 가능!
    self.logger.warning("Falling back to normal training mode")
    self.use_cached_predictions = False
```

**수정된 코드 (안전)**:
```python
if cache_path.exists():
    self.predictions_cache = joblib.load(cache_path)
else:
    # ✅ 에러 발생 - 일원화 강제
    raise FileNotFoundError(
        "Predictions cache not found!\n"
        "Run regressor.py first, or set USE_CACHED_PREDICTIONS=N"
    )
```

**효과**:
- 캐시 없이 백테스트 실행 자체가 **불가능**
- 실수로 다른 예측값 사용하는 것 **원천 차단**
- 유닛테스트로 일원화 검증할 필요 없음 (아키텍처가 보장)

#### 결과: 일원화 보장 수준

| 접근 방식 | 일원화 보장 | 단점 |
|-----------|------------|------|
| 코드 리뷰 | ⚠️ 사람 의존 | 실수 가능 |
| 유닛테스트 | ⚠️ 테스트 커버리지 의존 | 누락 가능 |
| **아키텍처 강제** | ✅ 100% 보장 | 없음 |

### 작업 시 필수 체크리스트

**모든 수정 작업 시 다음을 확인:**

- [ ] 이 변경이 regressor.py에만 적용되는가?
- [ ] ml_backtest.py에도 동일하게 적용되어야 하는가?
- [ ] 코드가 두 곳에 중복되고 있는가?
- [ ] 공통 함수로 통합할 수 있는가?
- [ ] DataProcessor나 별도 유틸리티로 빼야 하는가?

### 거래일 조정 (Trading Day Adjustment)

**문제**: 리밸런싱 날짜가 휴장일(주말, 공휴일)일 경우 거래 불가
- 예: 2025-01-01 (New Year's Day) → 거래 불가 → 0% 수익률
- regressor.py는 `searchsorted`로 가장 가까운 거래일 사용
- ml_backtest.py도 동일한 방식으로 조정 필요

**해결책**: `_get_trade_date()` 함수 사용하여 실제 거래일로 조정

**구현** (`src/backtest/ml_backtest.py`):
```python
# 리밸런싱 날짜 생성 후
for target_date in rebalance_dates:
    # target_date 이전 10일 내 가장 최근 거래일 반환
    actual_trade_date = self._get_trade_date(target_date, price_table)

    if actual_trade_date is None:
        # 10일 내 거래일 없으면 스킵
        continue

    adjusted_dates.append(actual_trade_date)

rebalance_dates = adjusted_dates
```

**효과**:
- 2025-01-01 → 2024-12-31 (이전 거래일)
- 실제 거래 가능한 날짜만 사용
- regressor.py와 ml_backtest.py 일관성 확보
- 0% 수익률 문제 해결

**주의사항**:
- `_get_trade_date()`는 **이전** 거래일을 찾음 (이후 X)
- 10일 내 거래일 없으면 해당 리밸런싱 스킵
- 조정 내역은 로그에 기록됨

## 🛡️ Data Leakage Prevention (미래 정보 유출 방지)

### 핵심 원칙: filingDate 기준 Cutoff

**문제**: 재무제표 데이터는 **분기 종료일**과 **공시일**이 다릅니다.
- 예: 2024 Q1 (종료일: 2024-03-31) → 공시일: 2024-05-15
- 종료일 기준으로 사용하면 미래 정보 유출 발생!

**해결책**: `filingDate` (공시일) 기준으로 엄격하게 cutoff

### 구현 방법 (make_mldata.py)

```python
# ❌ 나쁜 예: 분기 종료일 기준
fs_metrics['rebalance_date'] = fs_metrics['report_date']  # Future leakage!

# ✅ 좋은 예: 공시일 기준
indices = np.searchsorted(date_index, fs_metrics['filingDate'], side='right')
fs_metrics['rebalance_date'] = [date_index[i] if i < len(date_index) else pd.NaT
                                  for i in indices]
```

**작동 원리**:
1. `filingDate` (공시일)를 기준으로 리밸런싱 날짜 인덱스 검색
2. `side='right'`: 공시일 **이후**의 첫 번째 리밸런싱 날짜 선택
3. 해당 리밸런싱 시점에만 해당 재무 데이터 사용 가능

### Validation (검증)

```python
# 공시 지연 검증: filingDate와 분기 종료일 간격 분석
current_quarter_data['filling_delay_days'] = (
    pd.to_datetime(current_quarter_data['filingDate']) -
    pd.to_datetime(current_quarter_data['report_date'])
).dt.days
```

- 일반적으로 공시 지연: 30~90일
- 이상치 확인: 너무 짧거나 긴 지연은 데이터 오류 가능성

### 체크리스트

데이터 로딩/전처리 시 항상 확인:

- [ ] `filingDate` 컬럼이 존재하는가?
- [ ] 리밸런싱 날짜 매핑이 `filingDate` 기준인가?
- [ ] 테스트 데이터가 학습 데이터의 미래 정보를 포함하지 않는가?
- [ ] Walk-forward 백테스트에서 각 구간이 독립적인가?

---

## 📂 코드 구조

### 통합되어야 하는 로직

다음 로직들은 **반드시 단일 함수**로 관리:

#### 1. 데이터 전처리
- **위치**: `src/training/data_processor.py`
- **함수**:
  - `preprocess_training_data()` - 학습 데이터 전처리
  - `prepare_sector_data()` - 섹터 데이터 준비
  - `normalize_feature_names()` - Feature 이름 정규화
  - `winsorize_features()` - Winsorization
  - `align_features_to_model()` - Feature alignment ✅ 완료

#### 2. Feature Engineering
- **위치**: `src/training/make_mldata.py`
- tsfresh 파라미터
- Feature 선택 기준
- 정규화 로직

#### 3. 모델 예측
- regressor.py와 ml_backtest.py가 **정확히 동일한 순서**로:
  1. Feature alignment
  2. Preprocessing
  3. Model prediction
  4. Post-processing

### 현재 상태

✅ **통합 완료**:
- `DataProcessor.preprocess_training_data()` - 공통 전처리
- `DataProcessor.prepare_sector_data()` - 섹터 데이터 준비
- `DataProcessor.normalize_feature_names()` - Feature 정규화

⚠️ **통합 필요**:
- Feature alignment 로직 (현재 regressor.py, ml_backtest.py에 중복)

---

## 📊 핵심 분석 레포트 (Codebase Report)

> **위치**: `docs/codebase-report/`

프로젝트의 전반적인 코드 분석 레포트입니다. 자주 참조하고 업데이트합니다.

### 레포트 목록

| 파일 | 주제 | 용도 |
|------|------|------|
| `00_overview.md` | 전체 개요 | 프로젝트 철학, 종합 평가, 핵심 요약 |
| `01_architecture.md` | 아키텍처 | 시스템 구조, 모듈 관계, 데이터 흐름 |
| `02_data_pipeline.md` | 데이터 파이프라인 | FMP API, 전처리, 저장 구조 |
| `03_ml_strategy.md` | ML 전략 | 2-Stage 모델, Classifier/Regressor |
| `04_backtesting.md` | 백테스팅 | Walk-Forward, 거래 비용, 벤치마크 |
| `05_code_quality.md` | 코드 품질 | 가독성, 유지보수성, 테스트 |
| `06_quant_perspective.md` | 퀀트 관점 | 시장 효율성, 리스크, 알파 |
| `07_recommendations.md` | 개선 권고 | 우선순위별 TODO 목록 |
| `08_recent_changes.md` | 최신 변경사항 | 최근 업데이트 내역 |

### 업데이트 규칙

1. **기능 추가/변경 시**: 관련 레포트 업데이트
2. **버그 수정 시**: `08_recent_changes.md`에 기록
3. **TODO 완료 시**: `07_recommendations.md`에서 체크
4. **날짜 갱신**: 각 문서 헤더의 "최종 업데이트" 수정

### 활용 방법

- **프로젝트 이해**: 00 → 01 → 02 순서로 읽기
- **작업 계획**: 07_recommendations.md 참조
- **변경 추적**: 08_recent_changes.md 확인

---

## 🗂️ 프로젝트 관리 (Project Management)

### Configuration 파일 관리 (민감 정보 분리)

**2-파일 구조로 민감 정보 분리**

실수로 인한 민감 정보 유출을 방지하기 위해 설정 파일을 두 개로 분리합니다:

| 파일 | 용도 | Git 포함 | 공유 가능 |
|------|------|----------|-----------|
| `conf.yaml` | 일반 설정 (파라미터, 기간 등) | ❌ | ⚠️ 가능하지만 권장 안함 |
| `secrets.yaml` | 민감 정보 (API 키, 인증) | ❌ | ❌ 절대 금지 |
| `conf.yaml.template` | conf.yaml 템플릿 | ✅ | ✅ |
| `secrets.yaml.template` | secrets.yaml 템플릿 | ✅ | ✅ |

**secrets.yaml에서 관리하는 민감 정보**:
- `FMP_API_KEY`: FMP API 키
- `GOOGLE_SERVICE_ACCOUNT_KEY_PATH`: Google 서비스 계정 키 파일 경로
- `GOOGLE_SHEETS_ID`: Google Sheets ID
- `GITHUB_PAT`: GitHub Personal Access Token (Gist용)
- `MLFLOW_TRACKING_URI`: MLflow 서버 URL (내부 인프라)

**초기 설정 방법**:
```bash
# 1. 템플릿 복사
cp config/conf.yaml.template config/conf.yaml
cp config/secrets.yaml.template config/secrets.yaml

# 2. secrets.yaml에 실제 API 키 입력
# 3. conf.yaml은 일반 설정만 수정
```

**장점**:
1. **유출 방지**: conf.yaml을 실수로 커밋해도 API 키 안전
2. **팀 협업**: conf.yaml 공유 가능 (secrets.yaml만 개인 관리)
3. **명확한 분리**: 민감 정보 관리 단순화

**동작 원리** (`context_loader.py`):
```python
# 1. conf.yaml 로드
config = load_config('config/conf.yaml')

# 2. secrets.yaml 자동 merge (있으면)
# FMP_API_KEY -> config['DATA']['API_KEY']
# GOOGLE_SHEETS_ID -> config['EXPERIMENT_TRACKING']['GOOGLE_SHEETS']['SHEET_ID']
```

**하위 호환성**:
- secrets.yaml이 없으면 기존 방식(conf.yaml만 사용)으로 동작
- 경고 메시지 출력 후 정상 진행

**작업 원칙**:
1. **새 민감 정보 추가**: secrets.yaml.template + SECRETS_KEY_MAPPING 업데이트
2. **새 일반 설정 추가**: conf.yaml.template에 추가 (주석과 예시 포함)
3. **보안 정보**: 절대 template 파일에 실제 값 입력 금지
4. **기본값**: 합리적인 기본값 제공 (사용자가 바로 테스트 가능하도록)

### Dependencies 관리 (requirements.txt)

**작업 원칙**:
1. **새 패키지 설치**: requirements.txt에 반드시 추가
   ```bash
   pip install new-package
   pip freeze | grep new-package >> requirements.txt
   ```

2. **패키지 제거**: requirements.txt에서도 삭제
   ```bash
   pip uninstall old-package
   # requirements.txt에서 해당 줄 삭제
   ```

3. **버전 명시**: 주요 패키지는 버전 고정
   ```
   pandas==1.5.3  # 고정 버전 (재현성)
   numpy>=1.24.0  # 최소 버전 (호환성)
   ```

4. **주석 추가**: 용도가 명확하지 않은 패키지는 주석 작성
   ```
   optuna==3.1.0  # Hyperparameter optimization
   plotly==5.14.0  # Optuna visualization charts
   ```

**체크리스트**:
- [ ] 새 import 문 추가 시 requirements.txt 확인
- [ ] 에러 발생 시 버전 충돌 가능성 확인
- [ ] 주기적으로 `pip list --outdated` 실행하여 업데이트 검토

### 루트 디렉토리 관리

**원칙**: 루트 디렉토리는 깨끗하게 유지합니다.

**루트에 허용되는 파일**:
- ✅ `README.md` - 프로젝트 메인 문서
- ✅ `CLAUDE.md` - AI 작업 가이드 (이 문서)
- ✅ `.gitignore`, `.gitattributes` - Git 설정
- ✅ `requirements.txt` - Python 의존성
- ✅ `main.py` - 메인 실행 파일
- ✅ 기타 설정 파일 (`.env`, `pyproject.toml` 등)

**루트에서 제외할 파일**:
- ❌ 문서 파일 (`.md`) - `docs/` 디렉토리로 이동
  - 예: `TROUBLESHOOTING.md` → `docs/TROUBLESHOOTING.md`
  - 예: `API_GUIDE.md` → `docs/API_REFERENCE.md`
- ❌ 임시 파일, 테스트 파일
- ❌ 개인 메모, 실험 스크립트

**문서 배치 규칙**:
```
프로젝트 루트/
├── README.md              ✅ 프로젝트 소개
├── CLAUDE.md              ✅ AI 작업 가이드
├── requirements.txt       ✅ 의존성
├── main.py                ✅ 실행 파일
├── docs/                  📁 모든 문서 파일
│   ├── codebase-report/   🔍 핵심 분석 레포트 (AI 작성)
│   │   ├── 00_overview.md
│   │   ├── 01_architecture.md
│   │   ├── ...
│   │   └── 08_recent_changes.md
│   ├── analysis/          📊 작업용 분석 문서
│   ├── QUICK_START.md
│   └── ...
├── src/                   📁 소스 코드
└── config/                📁 설정 파일
```

**작업 시 체크리스트**:
- [ ] 새 문서 작성 시 `docs/` 디렉토리에 생성
- [ ] 루트에 파일 생성 시 반드시 필요한지 확인
- [ ] 커밋 전 `ls -la` 또는 `git status`로 루트 확인
- [ ] 불필요한 파일 발견 시 즉시 제거 또는 이동

**잘못된 예**:
```bash
# ❌ 나쁜 예: 루트에 문서 생성
touch TROUBLESHOOTING.md
touch OPTIMIZATION_GUIDE.md
```

**올바른 예**:
```bash
# ✅ 좋은 예: docs/ 하위에 생성
touch docs/TROUBLESHOOTING.md
touch docs/OPTIMIZATION_GUIDE.md
```

### 문서 관리 (README.md)

**업데이트 필요 시점**:
1. **구조 변경**: 폴더/파일 구조가 크게 바뀐 경우
   - 새 디렉토리 추가
   - 주요 파일 이동/이름 변경
   - 모듈 재구성

2. **주요 기능 추가**: 사용자가 알아야 할 새 기능
   - 새로운 ML 모델 추가
   - 백테스트 방식 변경
   - 설정 옵션 추가

3. **설치 방법 변경**: 의존성이나 설치 절차 변경
   - 새 필수 패키지
   - 설정 파일 형식 변경
   - 환경 요구사항 변경

4. **사용법 변경**: 실행 방법이나 워크플로우 변경
   - CLI 인터페이스 변경
   - 입력 데이터 형식 변경
   - 출력 파일 위치 변경

**README.md 필수 섹션**:
- **Installation**: requirements.txt 설치 방법
- **Configuration**: conf.yaml.template 사용법
- **Project Structure**: 주요 디렉토리 및 파일 설명
- **Usage**: 실행 예시 및 워크플로우
- **Development**: 개발자를 위한 가이드

**예시**:
```markdown
## Installation

1. Clone repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy config template:
   ```bash
   cp config/conf.yaml.template config/conf.yaml
   ```
4. Edit `config/conf.yaml` and add your API key
```

### 작업 시 체크리스트

**코드 수정 완료 후 반드시 확인**:

- [ ] 새 설정 추가 → `conf.yaml.template` 업데이트
- [ ] 새 패키지 사용 → `requirements.txt` 추가
- [ ] 구조 변경 → `README.md` 업데이트
- [ ] API 키 노출 → `.gitignore` 확인
- [ ] 커밋 전 → `git status`로 conf.yaml 포함 여부 확인

**자주 하는 실수**:
- ❌ conf.yaml을 실수로 커밋 (보안 위험!)
- ❌ requirements.txt 업데이트 없이 새 패키지 사용 (타인이 실행 불가)
- ❌ README.md 업데이트 없이 구조 변경 (혼란 야기)

## 🔧 작업 중 발견 시 즉시 조치

### 이중화 발견 시

```python
# ❌ 나쁜 예 - 코드 이중화
# regressor.py
for col in missing_features:
    X[col] = np.nan
X = X[model_features]

# ml_backtest.py
for col in missing_features:
    X[col] = np.nan
X = X[feature_cols]

# ✅ 좋은 예 - 단일 함수
# data_processor.py
@staticmethod
def align_features_to_model(X, model, logger=None):
    """공통 feature alignment"""
    # ... 로직 한 곳에만 ...

# regressor.py
X = DataProcessor.align_features_to_model(X, model)

# ml_backtest.py
X = DataProcessor.align_features_to_model(X, model)
```

### 에러 수정 시

1. 문제가 regressor.py에서 발생했다면:
   - ml_backtest.py에서도 같은 문제 발생 가능성 확인
   - 공통 원인이면 DataProcessor로 통합

2. 한쪽에만 수정하는 경우:
   - 반드시 주석으로 이유 명시
   - 다른 쪽에 영향 없는지 확인

## 📊 평가 체계

### regressor.py (모델 평가)
- **목적**: 모델의 예측 정확도 측정
- **지표**: RMSE, MAE, R², Accuracy, Precision, Recall
- **의미**: "모델이 잘 예측하는가?"

### ml_backtest.py (수익률 평가)
- **목적**: 실제 트레이딩 수익성 측정
- **지표**: 수익률, MDD, Sharpe Ratio, Win Rate
- **의미**: "실제로 돈을 버는가?"

### 종합 평가
- **필수**: 두 평가를 함께 봐야 의미 있음
- **전제**: 두 시스템이 **동일한 로직**으로 동작해야 함
- **검증**: 백테스트 결과가 실제 예측과 일치하는지 확인

## 🚀 작업 프로세스

### 새 기능 추가 시

1. **설계**: 어디에 구현할지 결정
   - 공통 로직 → DataProcessor
   - regressor 전용 → regressor.py
   - backtest 전용 → ml_backtest.py

2. **구현**: 코드 작성

3. **통합 체크**:
   - 다른 쪽에도 필요한가?
   - 중복 코드가 생기는가?
   - 통합 가능한가?

4. **테스트**: 양쪽에서 동일하게 동작하는지 확인

### 버그 수정 시

1. **원인 파악**: 어디서 발생했는가?

2. **범위 확인**:
   - 한쪽만의 문제인가?
   - 공통 로직의 문제인가?

3. **수정**:
   - 공통 문제 → DataProcessor 수정
   - 개별 문제 → 해당 파일만 수정 (이유 명시)

4. **검증**: 양쪽 모두 정상 동작 확인

## 📝 커밋 메시지 가이드

명확한 커밋 메시지로 의도 전달:

```
✅ 좋은 예:
"Refactor: Unify feature alignment in DataProcessor
- Move duplicated alignment logic from regressor/ml_backtest
- Both now use DataProcessor.align_features_to_model()
- Ensures consistency between training and backtesting"

❌ 나쁜 예:
"Fix feature alignment"
```

## 🔍 코드 리뷰 포인트

Pull Request 시 확인:

- [ ] 이중화된 코드가 없는가?
- [ ] 공통 로직은 DataProcessor에 있는가?
- [ ] regressor와 ml_backtest가 같은 함수를 호출하는가?
- [ ] 변경이 양쪽에 일관되게 적용되었는가?
- [ ] 테스트 케이스가 양쪽을 커버하는가?

## 📌 현재 진행 중인 통합 작업

### 완료된 항목
- ✅ Feature name normalization (DataProcessor.normalize_feature_names)
  - tsfresh 특수문자 제거 → XGBoost/LightGBM/CatBoost 호환
- ✅ Preprocessing pipeline (DataProcessor.preprocess_training_data)
  - Infinite 제거, NaN 처리, Outlier clipping 통합
- ✅ Sector data preparation (DataProcessor.prepare_sector_data)
  - 섹터별 모델 전처리 단일화
- ✅ Feature alignment (DataProcessor.align_features_to_model)
  - Missing features NaN fill, 순서 정렬 통합
  - regressor.py와 ml_backtest.py 중복 제거 완료

### 진행 중인 항목
- 없음

### 계획된 항목
- ⏳ GPU prediction wrapper 통합
  - 현재 regressor.py에만 `predict_with_gpu_support()` 존재
  - ml_backtest.py도 동일 함수 사용하도록 통합 필요
- ⏳ Evaluation metrics 통합
  - 평가 지표 계산 로직 중복 제거
- ⏳ Top-K selection 통합
  - 상위 종목 선정 로직 표준화

---

## 🔄 Development Workflow (개발 워크플로우)

### 🎯 NEW: Walk-Forward Evaluation & Prediction Cache (2025-12-21)

**문제점**: 기존에는 regressor.py와 ml_backtest.py가 독립적으로 모델을 학습하여 시간 낭비 발생
- regressor.py: 단일 train/test split으로 평가
- ml_backtest.py: Walk-forward로 재학습하여 백테스트
- **결과**: 동일한 모델을 두 번 학습 (시간 2배 소모)

**해결책**: regressor.py도 walk-forward 방식 사용 + 예측 결과 캐시 공유

#### 새로운 워크플로우

**Mode 1: 캐시 생성 모드 (regressor.py 먼저 실행)**
```yaml
# config/conf.yaml
EVALUATION:
  USE_WALK_FORWARD: Y          # Walk-forward 활성화
  USE_CACHED_PREDICTIONS: N    # 캐시 생성 모드
  TRAIN_START_YEAR: 1996
  PERIODS:
    - START_YEAR: 2020
      END_YEAR: 2021
    - START_YEAR: 2022
      END_YEAR: 2023
  REBALANCE_PERIOD: 3  # 분기별
  TOP_K_NUM: 10
```

실행 흐름:
```
1. regressor.py 실행
   ↓ Walk-forward 학습: 각 cutoff_date마다
   ↓   - Train: 1996 ~ cutoff_date (expanding window)
   ↓   - Predict: cutoff_date 시점 종목들
   ↓   - Top-K selection: 10개 선정
   ↓ 예측 결과 저장 → MODELS/regressor_predictions.pkl
   ↓
   ↓ Cache 구조:
   ↓ {
   ↓   '2020-01-01': {
   ↓     'predictions_df': DataFrame(symbol, sector, pred_return, pred_proba, ml_score, rank, selected),
   ↓     'top_k_selected': ['AAPL', 'MSFT', ...],
   ↓     'top_k_details': DataFrame (top 10 stocks only),
   ↓     'models_used': {...},
   ↓     'train_samples': 15000,
   ↓     'predict_samples': 3000
   ↓   },
   ↓   '2020-04-01': {...},
   ↓   ...
   ↓ }
   ↓
   ↓ Evaluation metrics 계산 (RMSE, MAE, R², Accuracy)
   ↓ outputs/reports/integrated_report_TIMESTAMP.xlsx 생성
```

**Mode 2: 캐시 재사용 모드 (ml_backtest.py 실행)**
```yaml
# config/conf.yaml
EVALUATION:
  USE_WALK_FORWARD: Y
  USE_CACHED_PREDICTIONS: Y    # ✅ 캐시 재사용
  PREDICTIONS_CACHE_FILE: "regressor_predictions.pkl"

BACKTEST:
  PERIODS:
    - START_YEAR: 2020
      END_YEAR: 2021
    - START_YEAR: 2022
      END_YEAR: 2023
  REBALANCE_PERIOD: 3
```

실행 흐름:
```
2. ml_backtest.py 실행
   ↓ Cache 로드: MODELS/regressor_predictions.pkl
   ↓
   ↓ For each rebalance_date:
   ↓   - Cache hit? → ✅ 예측 재사용 (모델 학습/예측 스킵)
   ↓   - Cache miss? → ⚠️ 일반 학습/예측 모드로 fallback
   ↓
   ↓ 실제 거래 시뮬레이션:
   ↓   - 가격 데이터 로드
   ↓   - Buy/Sell 시뮬레이션
   ↓   - 수익률 계산
   ↓
   ↓ 백테스트 결과 → outputs/reports/integrated_report_TIMESTAMP.xlsx
   ↓ (regressor.py 결과와 동일한 파일에 추가됨)
```

#### 통합 레포트 구조

**outputs/reports/integrated_report_TIMESTAMP.xlsx**

Sheet 1: Summary
```
Period    | Total Return | Avg Return | Sharpe | MDD    | Win Rate
----------|--------------|------------|--------|--------|----------
2020-2021 | 15.3%        | 3.8%       | 1.2    | -8.5%  | 62.5%
2022-2023 | -2.1%        | -0.5%      | -0.3   | -12.3% | 45.0%
```

Sheet 2: Regressor Metrics (from regressor.py)
```
Period     | RMSE  | MAE   | R²    | Accuracy | Precision | Recall
-----------|-------|-------|-------|----------|-----------|--------
2020-01-01 | 0.045 | 0.032 | 0.312 | 65.2%    | 68.1%     | 62.3%
2020-04-01 | 0.038 | 0.029 | 0.348 | 67.8%    | 71.2%     | 64.5%
```

Sheet 3: Backtest Performance (from ml_backtest.py)
```
Rebalance  | Buy Date   | Sell Date  | Stocks | Period Return | Cumulative
-----------|------------|------------|--------|---------------|------------
2020-01-01 | 2020-01-02 | 2020-04-01 | 10     | 8.2%          | 8.2%
2020-04-01 | 2020-04-02 | 2020-07-01 | 10     | 3.1%          | 11.5%
```

Sheet 4: Detailed Trades (17 columns)
```
Rebalance | Symbol | Company     | Sector | Rank | Selected | Pred Return | Pred Proba | ML Score | Actual Return | Buy Price | Sell Price | ...
----------|--------|-------------|--------|------|----------|-------------|------------|----------|---------------|-----------|------------|----
2020-01-01| AAPL   | Apple Inc.  | Tech   | 1    | True     | 0.082       | 0.75       | 0.0615   | 0.095         | 150.00    | 164.25     | ...
2020-01-01| MSFT   | Microsoft   | Tech   | 2    | True     | 0.075       | 0.78       | 0.0585   | 0.088         | 180.00    | 195.84     | ...
```

Sheet 5: Benchmark Comparison
```
Strategy   | Total Return | Sharpe | MDD    | Win Rate
-----------|--------------|--------|--------|----------
ML Model   | 15.3%        | 1.2    | -8.5%  | 62.5%
SPY (S&P)  | 12.1%        | 0.9    | -11.2% | 58.3%
QQQ (Nasdaq| 18.5%        | 1.1    | -15.3% | 60.0%
```

#### 장점 및 효과

**시간 절약 (50%)**:
- 기존: regressor 30분 + ml_backtest 30분 = 60분
- 신규: regressor 30분 (캐시 생성) + ml_backtest 5분 (캐시 재사용) = 35분

**완벽한 일관성**:
- regressor.py 평가 결과와 ml_backtest.py 백테스트가 동일한 예측 사용
- "예측도는 좋은데 수익률이 나쁜" 문제 원인 분석 가능

**정확한 평가**:
- 기존: regressor는 단일 train/test split (비현실적)
- 신규: regressor도 walk-forward (현실적)

**통합 레포팅**:
- 예측 정확도 + 실제 수익률을 한 눈에 비교
- 각 리밸런싱 시점의 상세 내역 추적 가능

---

### 전체 파이프라인

```
1. Data Collection (data_collector/)
   ↓ FMP API → Parquet files

2. Feature Engineering (make_mldata.py)
   ↓ tsfresh → ML-ready dataset
   ↓ filingDate cutoff (leakage prevention)

3. Walk-Forward Training (regressor.py) ✨ NEW
   ↓ For each cutoff_date in EVALUATION.PERIODS:
   ↓   - Load data: TRAIN_START_YEAR ~ cutoff_date
   ↓   - Train models (expanding window)
   ↓   - Predict at cutoff_date
   ↓   - Top-K selection
   ↓ Save predictions → MODELS/regressor_predictions.pkl
   ↓ Optuna → Hyperparameter tuning (optional)
   ↓ Stage 1: Classifiers (4 models)
   ↓ Stage 2: Regressors (2 models)

4. Evaluation Metrics (regressor.py)
   ↓ Prediction accuracy metrics (per period)
   ↓ RMSE, MAE, R², Accuracy, Precision, Recall
   ↓ Sheet 2 of integrated report

5. Backtesting (ml_backtest.py) ✨ UPDATED
   ↓ Load cache: MODELS/regressor_predictions.pkl
   ↓ For each rebalance_date:
   ↓   - Use cached predictions (skip training/prediction) ✅
   ↓   - OR fallback to training if cache miss ⚠️
   ↓   - Calculate actual returns
   ↓ Sheet 3, 4, 5 of integrated report
   ↓ Performance metrics: Return, MDD, Sharpe

6. Integrated Report (IntegratedReportWriter) ✨ NEW
   ↓ Combine regressor + backtest results
   ↓ 5 sheets: Summary, Regressor Metrics, Backtest Performance,
   ↓            Detailed Trades, Benchmark Comparison
   ↓ outputs/reports/integrated_report_TIMESTAMP.xlsx

7. Live Prediction (regressor.py)
   ↓ Load models → Latest data
   ↓ DataProcessor → Same preprocessing
   ↓ Generate rankings
```

### 일반적인 작업 시나리오

**시나리오 1: 새로운 Feature 추가**
1. `make_mldata.py`에서 feature 생성 로직 추가
2. `DataProcessor`에서 전처리 필요 시 추가
3. regressor.py 학습 실행
4. ml_backtest.py로 백테스트 검증
5. 두 결과 함께 평가

**시나리오 2: 모델 파라미터 튜닝**
1. `config/conf.yaml`에서 `OPTUNA_*` 설정 조정
2. `USE_OPTUNA: Y`로 설정
3. regressor.py 학습 실행 (자동 튜닝)
4. ml_backtest.py로 실전 수익률 검증
5. 예측도 ↔ 수익률 트레이드오프 분석

**시나리오 3: 버그 수정**
1. 어디서 발생했는지 파악 (regressor? ml_backtest? 공통?)
2. 공통 원인 → DataProcessor 수정
3. 개별 원인 → 해당 파일만 수정 (주석 명시)
4. 양쪽 모두 테스트 실행
5. 일관성 확인

**시나리오 4: 성능 개선**
1. Profiling으로 병목 지점 파악
2. 공통 로직 → DataProcessor에서 최적화
3. 개별 로직 → 해당 파일에서 최적화
4. 벤치마크: 학습 시간, 예측 속도 측정
5. 정확도 저하 없는지 확인

### 중요한 파일들

| 파일 | 역할 | 수정 빈도 | 주의사항 |
|------|------|-----------|----------|
| `CLAUDE.md` | AI 작업 가이드 | 낮음 | 프로젝트 철학 문서화 |
| `config/conf.yaml` | 전역 설정 | 높음 | 실험마다 변경 |
| `src/training/data_processor.py` | **통합 전처리** | 중간 | 변경 시 양쪽 영향 |
| `src/training/make_mldata.py` | Feature 생성 | 중간 | Leakage 주의 |
| `src/training/regressor.py` | 학습 & 평가 | 높음 | ml_backtest 일관성 |
| `src/backtest/ml_backtest.py` | 백테스트 | 높음 | regressor 일관성 |
| `src/models/config.py` | 모델 설정 | 낮음 | Optuna와 연계 |

### 테스트 전략

**Quick Test** (빠른 동작 확인):
```yaml
# config/conf.yaml
TRAIN_START_YEAR: 2020
TRAIN_END_YEAR: 2021
OPTUNA_TRIALS: 3
OPTUNA_CV_FOLDS: 2
```

**Production Test** (실전 투자):
```yaml
# config/conf.yaml
TRAIN_START_YEAR: 1996
TRAIN_END_YEAR: 2022
OPTUNA_TRIALS: 50
OPTUNA_CV_FOLDS: 5
```

---

**마지막 업데이트**: 2025-12-21
**작성자**: Development Team

**최근 변경사항 (2025-12-21)**:
- ✨ Walk-Forward Evaluation: regressor.py도 walk-forward 방식으로 평가
- 🔄 Prediction Cache: regressor.py와 ml_backtest.py 간 예측 결과 공유
- 📊 Integrated Report: 5개 시트로 구성된 통합 Excel 레포트
- 🚀 Performance: 50% 시간 절약 (중복 학습 제거)
- 🎯 Consistency: 평가와 백테스트의 완벽한 일관성 보장
