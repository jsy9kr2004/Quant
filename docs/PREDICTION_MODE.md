# 예측 전용 모드 (Prediction-Only Mode)

이미 학습된 모델을 사용하여 특정 날짜 기준으로 주식 추천만 수행합니다.

---

## 🎯 개요

**목적**: 학습(train), 평가(evaluation), 백테스트(backtest)를 건너뛰고 빠르게 추천을 받기

**사용 시나리오**:
1. 모델 학습 완료 후, 매일/매주 최신 데이터로 추천만 받고 싶을 때
2. 특정 과거 날짜 기준으로 "그때 뭘 샀어야 했나?" 확인하고 싶을 때
3. FMP 데이터 업데이트 후 빠르게 추천 확인하고 싶을 때

**소요 시간**: 수 초 (학습 없이 예측만 수행)

---

## 📋 전제 조건

예측 전용 모드를 사용하려면 **먼저 모델 학습이 완료**되어 있어야 합니다.

### 필수 파일 (MODELS/ 디렉토리)
```
MODELS/
├── clsmodel_0.sav          # 분류기 모델 (4개)
├── clsmodel_1.sav
├── clsmodel_2.sav
├── clsmodel_3.sav
├── model_0.sav             # 회귀기 모델 (2개)
├── model_1.sav
├── feature_columns.pkl     # 학습 시 사용한 피처 목록
└── threshold_config.pkl    # 분류기 임계값 설정
```

### 필수 데이터 (processed/ml_data/per_year/)
```
processed/ml_data/per_year/
├── rnorm_fs_2024.parquet   # 피처 데이터 (연도별)
├── rnorm_fs_2025.parquet
└── ...
```

> **Tip**: 먼저 `ML.RUN_REGRESSION=Y`로 학습을 완료한 후 예측 전용 모드를 사용하세요.

---

## ⚙️ 설정 방법

### config/conf.yaml

```yaml
PREDICTION:
  # 예측 전용 모드 활성화
  ENABLED: Y                    # Y = 예측만, N = 기존 파이프라인

  # 예측 기준 날짜
  TARGET_DATE: "2025-01-11"     # 특정 날짜 또는 "latest"

  # 추천 종목 수
  TOP_K: 10
```

### TARGET_DATE 옵션

| 값 | 설명 | 사용 예 |
|----|------|--------|
| `"latest"` | 가장 최근 데이터 사용 | 오늘 기준 추천 |
| `"2025-01-11"` | 해당 날짜 기준 | 특정 시점 시뮬레이션 |
| `"2024-06-30"` | 과거 날짜 | 백테스트 검증 |

> **Note**: `filingDate <= TARGET_DATE`인 데이터 중 각 종목의 가장 최근 데이터를 사용합니다.

---

## 🚀 실행 방법

### 1. 기본 실행

```bash
# config/conf.yaml에서 PREDICTION.ENABLED: Y 설정 후
python main.py
```

### 2. 출력 예시

```
================================================================================
🎯 PREDICTION-ONLY MODE
   Target Date: 2025-01-11
   Top-K: 10
================================================================================
✅ Models loaded successfully
✅ Threshold config loaded: Percentile 92, Mode: negative_screen
✅ Feature columns loaded: 847 features

📊 Loading data...
   Loaded 125000 rows from 30 files
   Filtering data for target_date: 2025-01-11
   After filingDate filter: 98000 rows (removed 27000)
   After deduplication: 4500 unique symbols
   After NaN filter: 4200 rows

🔮 Running predictions...
   Classifier filter (negative_screen): 3864/4200 passed

✅ Full predictions saved: MODELS/prediction_20250111.csv
✅ Top-10 recommendations saved: MODELS/prediction_20250111_top10.csv

================================================================================
🏆 TOP 10 RECOMMENDATIONS (as of 2025-01-11)
================================================================================
   AAPL   | Apple Inc.                     | Technology           | Score: +0.0823 | Return: +8.23%
   MSFT   | Microsoft Corporation          | Technology           | Score: +0.0756 | Return: +7.56%
   ...
================================================================================
```

---

## 📁 출력 파일

### 저장 위치: `MODELS/`

| 파일 | 설명 |
|------|------|
| `prediction_{날짜}.csv` | 전체 예측 결과 (모든 종목) |
| `prediction_{날짜}_top{K}.csv` | 상위 K개 추천 종목 |

### CSV 컬럼 설명

| 컬럼 | 설명 |
|------|------|
| `symbol` | 종목 심볼 |
| `company` | 회사명 |
| `sector` | 섹터 |
| `ml_score` | 최종 점수 (필터 통과 시 = pred_return, 미통과 시 = -inf) |
| `pred_return` | 예측 수익률 (회귀 모델 출력) |
| `pred_up_proba` | 상승 확률 (분류 모델 출력, 앙상블 평균) |
| `passed_filter` | 분류기 필터 통과 여부 |
| `filingDate` | 사용된 데이터의 공시일 |

---

## 🔄 워크플로우 예시

### 시나리오: 매주 월요일 추천 받기

```bash
# 1. FMP 데이터 업데이트 (필요 시)
# config: DATA.GET_FMP: Y, ML.RUN_REGRESSION: N, PREDICTION.ENABLED: N
python main.py

# 2. 예측 전용 모드로 추천 받기
# config: DATA.GET_FMP: N, PREDICTION.ENABLED: Y, TARGET_DATE: "latest"
python main.py

# 3. 결과 확인
cat MODELS/prediction_latest_top10.csv
```

### 시나리오: 과거 특정 시점 시뮬레이션

```yaml
# config/conf.yaml
PREDICTION:
  ENABLED: Y
  TARGET_DATE: "2024-06-30"  # 2024년 6월 30일 기준
  TOP_K: 10
```

```bash
python main.py
# → MODELS/prediction_20240630_top10.csv 생성
```

---

## ⚠️ 주의사항

### 1. 모델이 없으면 실패

```
❌ Model files not found: [Errno 2] No such file or directory: 'MODELS/clsmodel_0.sav'
   Please run training first (ML.RUN_REGRESSION=Y)
```

**해결**: 먼저 `ML.RUN_REGRESSION=Y`로 학습 실행

### 2. 데이터가 없으면 실패

```
❌ No data files found in processed/ml_data/per_year/
```

**해결**: `AIDataMaker` 실행 (main.py에서 자동 실행됨)

### 3. TARGET_DATE가 너무 오래되면 데이터 부족

```
❌ No data available for the specified date
```

**해결**: 더 최근 날짜 사용 또는 데이터 범위 확인

---

## 🔧 고급 설정

### Python에서 직접 호출

```python
from src.training.regressor import Regressor
import yaml

# Config 로드
with open('config/conf.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Regressor 인스턴스 생성
regressor = Regressor(config)

# 예측 수행
top_stocks = regressor.predict_for_date(
    target_date="2025-01-11",
    top_k=10
)

# 결과 확인
print(top_stocks[['symbol', 'company', 'ml_score', 'pred_return']])
```

### 여러 날짜에 대해 예측

```python
dates = ["2024-12-01", "2024-12-15", "2025-01-01", "2025-01-11"]

for date in dates:
    top_stocks = regressor.predict_for_date(target_date=date, top_k=10)
    print(f"\n=== {date} ===")
    print(top_stocks[['symbol', 'ml_score']].head())
```

---

## 📊 기존 방식과 비교

| 항목 | 기존 (latest_prediction) | 예측 전용 모드 |
|------|-------------------------|---------------|
| 날짜 지정 | ❌ 항상 최신 | ✅ 특정 날짜 가능 |
| 실행 시점 | train/eval 이후 자동 | 독립적으로 실행 가능 |
| 출력 파일 | `latest_prediction*.csv` | `prediction_{날짜}*.csv` |
| 설정 방법 | 없음 (자동) | `PREDICTION` 섹션 |

---

## 관련 문서

- **[QUICK_START.md](QUICK_START.md)** - 기본 실행 방법
- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - 전체 파이프라인 이해
- **[../CLAUDE.md](../CLAUDE.md)** - 시스템 철학 및 2-Stage ML 구조
