# 최신 변경사항 분석 (2026-01)

> **작성일**: 2026-01-10
> **이전 문서**: [07_recommendations.md](./07_recommendations.md)
> **관련 커밋**: `3e73dc1`, `ba3b312`, `551fb8a`

---

## 목차

1. [변경사항 요약](#1-변경사항-요약)
2. [예측 전용 모드 (Prediction-Only Mode)](#2-예측-전용-모드)
3. [CLASSIFIER_MODE 개선](#3-classifier_mode-개선)
4. [거래일 조정 수정](#4-거래일-조정-수정)
5. [향후 과제](#5-향후-과제)

---

## 1. 변경사항 요약

### 최근 커밋 히스토리

| 커밋 | 설명 | 영향 범위 |
|------|------|----------|
| `3e73dc1` | feat: Add prediction-only mode for quick stock recommendations | regressor.py, main.py, conf.yaml.template |
| `ba3b312` | refactor: Improve negative_screen implementation in ml_backtest.py | ml_backtest.py, regressor.py |
| `551fb8a` | Fix get_trade_date() returning wrong trading date | ml_backtest.py |

### 주요 변경점

1. **예측 전용 모드**: 학습 없이 기존 모델로 빠른 추천 생성
2. **CLASSIFIER_MODE**: negative_screen / positive_screen 모드 선택 가능
3. **Hard Filtering**: 확률 가중치 → 명확한 cutoff 방식으로 변경
4. **거래일 조정**: 휴장일 처리 버그 수정

---

## 2. 예측 전용 모드

### 2.1 구현 배경

**문제점**:
- 기존에는 추천을 받으려면 전체 학습 파이프라인 실행 필요
- 모델이 이미 학습되어 있어도 재학습 필요
- 특정 과거 날짜 기준 시뮬레이션 불가능

**해결책**:
- `PREDICTION.ENABLED: Y` 설정으로 예측 전용 모드 활성화
- 학습된 모델(`MODELS/*.sav`)을 로드하여 즉시 예측
- 특정 날짜 또는 최신 데이터 기준으로 추천 생성

### 2.2 설정 방법

```yaml
# config/conf.yaml
PREDICTION:
  ENABLED: Y                    # 예측 전용 모드 활성화
  TARGET_DATE: "2025-01-11"     # 특정 날짜 또는 "latest"
  TOP_K: 10                     # 추천 종목 수
```

### 2.3 구현 상세

**파일 위치**: `src/training/regressor.py` (4700번째 줄 부근)

```python
def predict_for_date(self, target_date: str = "latest", top_k: int = 10) -> pd.DataFrame:
    """
    특정 날짜 기준으로 주식 추천 생성

    Args:
        target_date: 예측 기준 날짜 ("latest" 또는 "YYYY-MM-DD")
        top_k: 반환할 상위 종목 수

    Returns:
        DataFrame: 상위 K개 추천 종목 (symbol, company, ml_score, pred_return, ...)
    """
    # 1. 모델 로드
    clsmodels, models = self._load_models()
    threshold_config = joblib.load('MODELS/threshold_config.pkl')
    feature_columns = joblib.load('MODELS/feature_columns.pkl')

    # 2. 데이터 로드 및 필터링
    df = self._load_all_parquet_data()
    df = self._filter_by_filing_date(df, target_date)
    df = self._deduplicate_by_symbol(df)

    # 3. 예측 수행
    y_pred_proba = self._predict_classifier_ensemble(X, clsmodels)
    y_pred_return = self._predict_regressor_ensemble(X, models)

    # 4. Hard Filtering (classifier_mode 기반)
    pass_mask = self._apply_classifier_filter(y_pred_proba, threshold_config)
    ml_score = np.where(pass_mask, y_pred_return, -np.inf)

    # 5. Top-K 선정 및 저장
    top_k_df = df.nlargest(top_k, 'ml_score')
    top_k_df.to_csv(f'MODELS/prediction_{date_str}_top{top_k}.csv')

    return top_k_df
```

### 2.4 출력 파일

| 파일 | 설명 |
|------|------|
| `MODELS/prediction_{날짜}.csv` | 전체 예측 결과 (모든 종목) |
| `MODELS/prediction_{날짜}_top{K}.csv` | 상위 K개 추천 종목 |

### 2.5 사용 시나리오

**시나리오 1: 오늘 기준 추천**
```yaml
PREDICTION:
  ENABLED: Y
  TARGET_DATE: "latest"
  TOP_K: 10
```

**시나리오 2: 과거 시점 시뮬레이션**
```yaml
PREDICTION:
  ENABLED: Y
  TARGET_DATE: "2024-06-30"
  TOP_K: 10
```

---

## 3. CLASSIFIER_MODE 개선

### 3.1 구현 배경

**문제점**:
- 기존에는 분류기가 단순 상승/하락 예측
- Soft filtering (확률 × 수익률)은 경계가 불명확
- 시장 환경에 따른 전략 조절 불가능

**해결책**:
- `negative_screen`: 극단적 손실 종목 제거 (보수적)
- `positive_screen`: 상승 확률 낮은 종목 제거 (공격적)
- Hard filtering으로 명확한 cutoff 적용

### 3.2 모드 비교

| 항목 | negative_screen | positive_screen |
|------|-----------------|-----------------|
| 타겟 정의 | 손실 < -30% = BAD | 수익 > 0% = GOOD |
| 제거 대상 | BAD 확률 상위 N% | GOOD 확률 하위 N% |
| 전략 성격 | 보수적 (위험 회피) | 공격적 (수익 추구) |
| 권장 환경 | 불확실/하락장 | 상승장 |

### 3.3 구현 상세

**타겟 생성** (`data_processor.py`):
```python
def create_binary_target(y_train, config, logger=None):
    classifier_mode = config.get('ML', {}).get('CLASSIFIER_MODE', 'negative_screen')

    if classifier_mode == 'negative_screen':
        loss_threshold = config['ML']['NEGATIVE_SCREEN']['LOSS_THRESHOLD']
        label_binary = (y_train < loss_threshold).astype(int)
        # Class 1 = BAD (극단적 손실)
    else:  # positive_screen
        label_binary = (y_train > 0).astype(int)
        # Class 1 = GOOD (상승)

    return label_binary
```

**Threshold 계산** (`ml_backtest.py`):
```python
def _calculate_threshold_config(self, y_probs, classifier_mode, remove_pct):
    if classifier_mode == 'negative_screen':
        # 상위 N% (BAD 확률 높음) 제거
        threshold = np.percentile(y_probs, 100 - remove_pct)
        pass_mask = y_probs < threshold
    else:  # positive_screen
        # 하위 N% (GOOD 확률 낮음) 제거
        threshold = np.percentile(y_probs, remove_pct)
        pass_mask = y_probs > threshold

    return threshold, pass_mask
```

**Hard Filtering 적용**:
```python
# 기존 (Soft Filtering)
ml_score = y_pred_proba * y_pred_return  # 확률 가중치

# 현재 (Hard Filtering)
ml_score = np.where(pass_mask, y_pred_return, -np.inf)  # 명확한 cutoff
```

### 3.4 threshold_config.pkl 구조

```python
{
    'mode': 'negative_screen',        # 분류기 모드
    'percentile': 92,                 # 자동 탐색된 최적 percentile
    'threshold_value': 0.15,          # 해당 percentile의 확률값
    'remove_pct': 8,                  # 실제 제거 비율
    'precision': 0.78,                # 해당 threshold에서의 precision
    'recall': 0.65,                   # 해당 threshold에서의 recall
    'n_selected': 8500,               # 선택된 샘플 수
    'n_total': 9200                   # 전체 샘플 수
}
```

### 3.5 섹터별 Threshold 개선

**기존 문제**:
- 단일 분류기로 섹터 threshold 계산
- 분류기 간 차이가 큼

**개선 사항**:
- 4-classifier 앙상블 평균으로 threshold 계산
- 더 안정적인 필터링

```python
def _calculate_sector_threshold(self, X, clsmodels, classifier_mode, remove_pct):
    # 4개 분류기의 예측 확률 평균
    y_probs_list = [clf.predict_proba(X)[:, 1] for clf in clsmodels]
    y_probs_avg = np.mean(y_probs_list, axis=0)

    # 앙상블 평균으로 threshold 계산
    threshold, pass_mask = self._calculate_threshold_config(
        y_probs_avg, classifier_mode, remove_pct
    )

    return threshold, pass_mask
```

---

## 4. 거래일 조정 수정

### 4.1 문제점

**이슈**: 리밸런싱 날짜가 휴장일(주말, 공휴일)인 경우
- 가격 데이터 없음 → 0% 수익률
- regressor.py와 ml_backtest.py 간 불일치

**예시**:
- 2025-01-01 (New Year's Day) → 휴장일
- 리밸런싱 시도 → 거래 불가 → 잘못된 결과

### 4.2 수정 내용

**함수**: `_get_trade_date()` (ml_backtest.py)

```python
def _get_trade_date(self, target_date, price_table, lookback_days=10):
    """
    target_date 이전 가장 최근 거래일 반환

    Args:
        target_date: 목표 날짜
        price_table: 가격 테이블 (인덱스 = 거래일)
        lookback_days: 최대 탐색 일수

    Returns:
        datetime: 실제 거래일 (없으면 None)
    """
    available_dates = price_table.index

    for i in range(lookback_days):
        check_date = target_date - timedelta(days=i)
        if check_date in available_dates:
            if i > 0:
                self.logger.info(f"Adjusted {target_date} → {check_date}")
            return check_date

    self.logger.warning(f"No trading day found near {target_date}")
    return None
```

### 4.3 적용 위치

```python
# 리밸런싱 날짜 생성 후
adjusted_dates = []
for target_date in rebalance_dates:
    actual_trade_date = self._get_trade_date(target_date, price_table)
    if actual_trade_date is not None:
        adjusted_dates.append(actual_trade_date)
    else:
        self.logger.warning(f"Skipping {target_date}: no trading day")

rebalance_dates = adjusted_dates
```

### 4.4 효과

- 휴장일 0% 수익률 문제 해결
- regressor.py와 ml_backtest.py 간 일관성 확보
- 백테스트 정확도 향상

---

## 5. 향후 과제

### 5.1 예측 전용 모드 관련

| 과제 | 우선순위 | 설명 |
|------|----------|------|
| 모델 버전 관리 | 높음 | 여러 모델 버전 지원 (날짜별, 설정별) |
| 증분 예측 | 중간 | 새 데이터만 예측하여 기존 결과에 추가 |
| API 서버화 | 낮음 | REST API로 예측 서비스 제공 |

### 5.2 CLASSIFIER_MODE 관련

| 과제 | 우선순위 | 설명 |
|------|----------|------|
| positive_screen 검증 | 높음 | 실전 백테스트로 성능 비교 |
| 동적 모드 전환 | 중간 | 시장 환경에 따라 자동 전환 |
| 혼합 모드 | 낮음 | 두 모드 결합하여 더 정교한 필터링 |

### 5.3 일반적인 개선 사항

| 과제 | 우선순위 | 설명 |
|------|----------|------|
| 단위 테스트 | 높음 | 새 기능에 대한 테스트 코드 작성 |
| 문서 자동화 | 중간 | 코드 변경 시 문서 자동 업데이트 |
| 성능 모니터링 | 중간 | 예측 정확도 실시간 추적 |

---

## 결론

2026년 1월의 주요 변경사항은 **실전 투자 편의성**에 초점을 맞추었습니다:

1. **예측 전용 모드**: 학습 없이 빠른 추천 → 실전 활용성 극대화
2. **CLASSIFIER_MODE**: 시장 환경별 전략 선택 → 유연성 향상
3. **Hard Filtering**: 명확한 cutoff → 해석 용이성 향상
4. **거래일 조정**: 휴장일 처리 → 백테스트 정확도 향상

### 평가

| 항목 | 이전 | 현재 | 변화 |
|------|------|------|------|
| 실전 준비도 | B | A- | +1 등급 |
| 유연성 | B | A | +1 등급 |
| 코드 품질 | B | B+ | +0.5 등급 |

**종합**: 이번 업데이트로 시스템의 **실전 투자 준비도**가 크게 향상되었습니다.

---

**END OF DOCUMENT**

전체 분석 문서:
- [00_overview.md](./00_overview.md)
- [01_architecture.md](./01_architecture.md)
- [02_data_pipeline.md](./02_data_pipeline.md)
- [03_ml_strategy.md](./03_ml_strategy.md)
- [04_backtesting.md](./04_backtesting.md)
- [05_code_quality.md](./05_code_quality.md)
- [06_quant_perspective.md](./06_quant_perspective.md)
- [07_recommendations.md](./07_recommendations.md)
- [08_recent_changes.md](./08_recent_changes.md) (현재 문서)
