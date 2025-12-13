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

## 🏗️ System Architecture: 2-Stage ML Structure

안정성과 수익성을 동시에 잡기 위해 모델을 **두 단계로 분리**하여 운용합니다.

### Stage 1: Stability Filtering (안정성 필터링)

**설계 의도**: "이 회사는 펀더멘털이 깨져 있지 않다"는 신호만 남기는 **Negative Screening**

#### 학습 시 (Training)

**타겟**: `label_binary = (price_dev > 0)` - 다음 분기 가격 상승/하락

**프로세스**:
1. 모든 학습 데이터로 분류기 학습
2. **자동 Threshold 탐색** (CLASSIFIER_THRESHOLD_AUTO_SEARCH=Y):
   - 학습된 분류기로 학습 데이터 예측 확률 계산
   - Percentile 85~98 구간 탐색
   - 각 percentile에서 precision, recall 계산
   - 최적 threshold 선정:
     - "precision" 모드: Precision 최대값
     - "balance" 모드: Min precision 조건 만족하면서 최대 데이터
   - 결과를 `threshold_config.pkl`로 저장
3. 또는 **고정 Threshold** 사용 (AUTO_SEARCH=N):
   - Config에서 지정한 percentile 사용 (기본값: 92)

**결과**: 최적 threshold로 학습 데이터 필터링 → Stage 2로 전달

#### 예측 시 (Evaluation/Backtest)

**프로세스**:
1. 저장된 `threshold_config.pkl` 로드
2. 분류기로 예측 확률 계산
3. 학습 시와 동일한 percentile threshold 적용
4. ml_backtest.py: 확률 × 수익률 (down-weighting)

**수학적 동등성**:
- "재무적으로 부실한 기업 제거" (설계 의도)
- ≈ "상승 확률 낮은 종목 제거" (현재 구현)
- → Top-K < threshold%이면 두 방식 모두 동일한 결과

**모델 구성**:
```python
# Global Classifiers (USE_CLASSIFIER=Y인 경우)
clsmodels[0]: XGBClassifier (Optuna 최적화)
clsmodels[1]: XGBClassifier (max_depth=9)
clsmodels[2]: XGBClassifier (max_depth=10)
clsmodels[3]: LGBMClassifier (max_depth=8)

# Sector Classifiers (USE_SECTOR_MODEL=Y인 경우)
각 섹터당 4개 분류기 (같은 구조)
```

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

### 작업 시 필수 체크리스트

**모든 수정 작업 시 다음을 확인:**

- [ ] 이 변경이 regressor.py에만 적용되는가?
- [ ] ml_backtest.py에도 동일하게 적용되어야 하는가?
- [ ] 코드가 두 곳에 중복되고 있는가?
- [ ] 공통 함수로 통합할 수 있는가?
- [ ] DataProcessor나 별도 유틸리티로 빼야 하는가?

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

## 🗂️ 프로젝트 관리 (Project Management)

### Configuration 파일 관리

**conf.yaml vs conf.yaml.template**

- **conf.yaml**: 실제 사용하는 설정 파일 (Git에 포함 안 됨)
  - API_KEY 등 보안 정보 포함
  - `.gitignore`에 등록되어 Git 추적 제외
  - 사용자가 직접 생성하고 관리

- **conf.yaml.template**: 설정 파일 템플릿 (Git에 포함)
  - API_KEY 자리는 placeholder로 표시
  - 모든 설정 항목의 기본값 제공
  - 코드 수정 시 이 파일을 업데이트

**작업 원칙**:
1. **코드 작성 시**: 사용자가 conf.yaml.template을 복사하여 conf.yaml을 만들었다고 가정
2. **새 설정 추가**: conf.yaml.template에 추가 (주석과 예시 포함)
3. **보안 정보**: conf.yaml.template에는 절대 실제 키 입력 금지
4. **기본값**: 합리적인 기본값 제공 (사용자가 바로 테스트 가능하도록)

**예시**:
```yaml
# conf.yaml.template
DATA:
  API_KEY: "your_fmp_api_key_here"  # ← placeholder

# 사용자가 만드는 conf.yaml
DATA:
  API_KEY: "abc123xyz789real"  # ← 실제 키
```

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

### 전체 파이프라인

```
1. Data Collection (data_collector/)
   ↓ FMP API → Parquet files

2. Feature Engineering (make_mldata.py)
   ↓ tsfresh → ML-ready dataset
   ↓ filingDate cutoff (leakage prevention)

3. Training (regressor.py)
   ↓ DataProcessor → Preprocessing
   ↓ Optuna → Hyperparameter tuning
   ↓ Stage 1: Classifiers (4 models)
   ↓ Stage 2: Regressors (2 models)
   ↓ Save models → {ROOT_PATH}/models/

4. Evaluation (regressor.py)
   ↓ Prediction accuracy metrics
   ↓ RMSE, MAE, R², Accuracy, Precision, Recall

5. Backtesting (ml_backtest.py)
   ↓ Walk-forward validation
   ↓ DataProcessor → Same preprocessing
   ↓ Load models → Predict
   ↓ Top-K selection → Portfolio
   ↓ Performance metrics: Return, MDD, Sharpe

6. Live Prediction (regressor.py)
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

**마지막 업데이트**: 2025-12-13
**작성자**: Development Team
