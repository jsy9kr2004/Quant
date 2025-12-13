# Claude AI 작업 가이드

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
  - `align_features_to_model()` - Feature alignment (추가 필요)

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
- ✅ Preprocessing pipeline (DataProcessor.preprocess_training_data)
- ✅ Sector data preparation (DataProcessor.prepare_sector_data)

### 진행 중인 항목
- 🔄 Feature alignment 통합 (현재 중복 코드 상태)

### 계획된 항목
- ⏳ Prediction pipeline 통합
- ⏳ Evaluation metrics 통합

---

**마지막 업데이트**: 2025-12-13
**작성자**: Development Team
