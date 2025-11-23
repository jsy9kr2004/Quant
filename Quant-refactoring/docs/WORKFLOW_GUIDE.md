# Quant Trading System - 개발자 워크플로우 가이드

> **목적**: 새로운 개발자 온보딩 및 시스템 아키텍처 이해
> **작성일**: 2025-11-17
> **대상**: 신규 개발자, 투자자를 위한 기술 리뷰

---

## 📖 이 문서는 누구를 위한 것인가?

- ✅ **개발자** - 코드를 수정하거나 확장하려는 분 (필독!)
- ✅ **시스템 아키텍처를 깊이 이해하고 싶은 분** - 전체 구조 파악
- ✅ **투자자** - 시스템이 어떻게 동작하는지 기술적으로 이해하려는 분

**이 문서를 읽기 전에:**
- [../README.md](../README.md)에서 프로젝트 구조 먼저 확인하세요

**이 문서를 읽은 후:**
- API 상세 정보 → [API_REFERENCE.md](API_REFERENCE.md)
- 개선 로드맵 → [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md)
- 로깅 시스템 → [LOGGING_GUIDE.md](LOGGING_GUIDE.md)
- 백테스팅 검증 → [ROBUST_VALIDATION_GUIDE.md](ROBUST_VALIDATION_GUIDE.md)

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [전체 아키텍처](#-전체-아키텍처)
3. [데이터 파이프라인](#-데이터-파이프라인)
4. [AI 모델 전략](#-ai-모델-전략)
5. [실행 플로우](#-실행-플로우)
6. [요약](#-요약)
7. [참고 자료](#-참고-자료)

---

## 🎯 프로젝트 개요

### 시스템 목적
주식 시장에서 **우수한 성과를 낼 것으로 예측되는 종목을 자동으로 선별**하는 Quantitative Trading 시스템입니다.

### 핵심 기능
- **데이터 수집**: Financial Modeling Prep (FMP) API를 통한 재무제표, 가격 데이터 수집
- **특성 엔지니어링**: 재무 비율 + 시계열 특성 추출 (tsfresh)
- **머신러닝 예측**: XGBoost + LightGBM + CatBoost 앙상블 모델
- **백테스팅**: 과거 데이터 기반 전략 성과 검증
- **자동화**: 데이터 수집 → 학습 → 예측 → 백테스팅 전 과정 자동화

### 기술 스택
```
언어: Python 3.8+
ML 프레임워크: XGBoost, LightGBM, CatBoost, scikit-learn
데이터 처리: Pandas, NumPy, PyArrow (Parquet)
특성 추출: tsfresh (시계열 특성)
실험 추적: MLflow
최적화: Optuna
데이터 소스: FMP API
```

---

## 🏗 전체 아키텍처

### 시스템 플로우 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. Configuration Loading                      │
│                    (config/conf.yaml)                            │
│         ├─ DATA: API Keys, Paths, Years                         │
│         ├─ ML: Model settings, MLflow config                    │
│         └─ BACKTEST: Strategy parameters                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              2. Data Collection (Optional, GET_FMP=Y)            │
│                                                                  │
│  [FMP API] → src/data_collector/fmp.py                           │
│     ├─ Stock List (NASDAQ, NYSE)                                │
│     ├─ Delisted Companies                                       │
│     ├─ Financial Statements (Income, Balance, CashFlow)         │
│     ├─ Key Metrics (P/E, ROE, Debt Ratios...)                   │
│     └─ Historical Price Data                                    │
│            ↓                                                     │
│     Parquet Files → ROOT_PATH/fmp_raw/{category}/{symbol}.parquet│
│            ↓                                                     │
│  src/storage/parquet_converter.py                               │
│     → VIEW 테이블 생성 (통합 뷰)                                   │
│            ↓                                                     │
│     ROOT_PATH/processed/views/ (통합 뷰)                          │
│       ├─ symbol_list.parquet                                    │
│       ├─ price.parquet                                          │
│       ├─ financial_statement_{year}.parquet                     │
│       └─ metrics_{year}.parquet                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│           3. ML Data Preparation (src/training/make_mldata.py)   │
│                                                                  │
│  VIEW 데이터 로드 → Merge (symbol, date 기준)                    │
│            ↓                                                     │
│  시계열 윈도우 생성 (12개월 lookback)                              │
│            ↓                                                     │
│  tsfresh 특성 추출 (EfficientFCParameters)                       │
│    ├─ standard_deviation, quantile                             │
│    ├─ autocorrelation, fft_coefficient                         │
│    └─ ar_coefficient (총 36개 시계열 특성)                       │
│            ↓                                                     │
│  재무 비율 계산 (139개 ratio features)                            │
│    ├─ ROE, ROIC, Profit Margins                                │
│    ├─ P/E, P/B, EV/EBITDA                                       │
│    └─ Debt Ratios, Coverage Ratios                             │
│            ↓                                                     │
│  RobustScaler 정규화 (Outlier-resistant)                         │
│            ↓                                                     │
│  Target 변수 생성: price_dev, price_dev_subavg                  │
│            ↓                                                     │
│  ROOT_PATH/processed/ml_data/per_year/rnorm_ml_{year}_{quarter}.parquet│
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              4. Model Training (src/training/regressor.py)       │
│                                                                  │
│  데이터 로드 (2015-2021 학습, 2022-2023 테스트)                   │
│            ↓                                                     │
│  데이터 클리닝                                                    │
│    ├─ 80% 이상 결측치 컬럼 제거                                   │
│    ├─ 95% 이상 동일 값 컬럼 제거                                  │
│    └─ 60% 이상 결측치 행 제거                                     │
│            ↓                                                     │
│  ┌─────────────────────────────────────────────────┐            │
│  │         Classification Models (Binary)          │            │
│  │  (상승/하락 예측 → 필터링 용도)                   │            │
│  ├─────────────────────────────────────────────────┤            │
│  │  clsmodel_0-3: XGBoost + LightGBM              │            │
│  │  → Ensemble Voting (상위 92% threshold)        │            │
│  │  → Binary Filter: 상승 예상 종목만 선택           │            │
│  └─────────────────────────────────────────────────┘            │
│            ↓                                                     │
│  ┌─────────────────────────────────────────────────┐            │
│  │         Regression Models (Continuous)          │            │
│  │  (상승폭 예측 → 랭킹 용도)                         │            │
│  ├─────────────────────────────────────────────────┤            │
│  │  model_0-1: XGBoost Regressor                  │            │
│  │  → Average Prediction                          │            │
│  │  → 상승폭 큰 순서로 종목 랭킹                      │            │
│  └─────────────────────────────────────────────────┘            │
│            ↓                                                     │
│  Final Strategy:                                                │
│    1. Classification 모델로 필터링 (상승 예상 종목)                │
│    2. Regression 모델로 상승폭 예측                               │
│    3. 예측값 높은 순으로 Top-K 종목 선정                          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│          5. Evaluation & Backtesting (Optional)                  │
│                                                                  │
│  Test Set 각 분기별 평가                                          │
│    ├─ Classification Accuracy                                  │
│    ├─ Regression RMSE                                          │
│    └─ Top-K 종목의 실제 수익률 계산                               │
│            ↓                                                     │
│  백테스팅 (backtest.py)                                           │
│    ├─ 리밸런싱 기간별 성과 측정                                     │
│    ├─ Sharpe Ratio, Max Drawdown                               │
│    └─ 리포트 생성                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙

1. **모듈화**: 각 컴포넌트가 독립적으로 실행 가능
2. **재현성**: 모든 설정이 YAML로 관리
3. **확장성**: 새로운 모델/데이터 소스 쉽게 추가
4. **성능**: Parquet, GPU, 멀티프로세싱 활용

---

## 📊 데이터 파이프라인

### 1. FMP API 데이터 수집
**파일**: `src/data_collector/fmp.py`

**수집 데이터**:
- `stock_list`: 상장 종목 리스트 (symbol, sector, industry, ipoDate)
- `delisted_companies`: 상장폐지 종목
- `income_statement`: 손익계산서
- `balance_sheet_statement`: 재무상태표
- `cash_flow_statement`: 현금흐름표
- `key_metrics`: 핵심 재무비율 (P/E, ROE, Debt/Equity...)
- `financial_growth`: 성장률 지표
- `historical_price_full`: 일별 가격 데이터

**특징**:
- **병렬 다운로드**: `src/data_collector/fmp_fetch_worker.py`에서 멀티프로세싱
- **NASDAQ, NYSE만**: 미국 주요 거래소 필터링
- **저장 위치**: `ROOT_PATH/fmp_raw/{category}/{symbol}.parquet`

### 2. Parquet 변환 및 VIEW 생성
**파일**: `src/storage/parquet_converter.py`

**변환 프로세스**:
```python
FMP 원본 데이터 (ROOT_PATH/fmp_raw/)
├─ 이미 Parquet 형식으로 저장됨
└─ VIEW 테이블 재구성 → ROOT_PATH/processed/views/
   ├─ symbol_list.parquet
   ├─ price.parquet
   ├─ financial_statement_{year}.parquet
   └─ metrics_{year}.parquet
```

**VIEW의 역할**: 여러 Parquet 파일을 통합하여 ML 학습에 바로 사용 가능한 형태로 재구성

### 3. ML 데이터 생성
**파일**: `src/training/make_mldata.py`

**프로세스**:
```python
For each year, quarter:
  1. 12개월 lookback window 생성
  2. tsfresh 특성 추출 (36개 시계열 특성)
  3. 재무 비율 계산 (139개)
  4. RobustScaler 정규화
  5. Target 변수 생성 (price_dev, price_dev_subavg)
  6. Parquet 저장 → ROOT_PATH/processed/ml_data/per_year/rnorm_ml_{year}_{quarter}.parquet
```

**생성되는 Feature**:
- **재무 비율 (139개)**: ROE, ROA, P/E, Debt/Equity, Current Ratio 등
- **시계열 특성 (36개)**: Standard Deviation, Autocorrelation, FFT Coefficient 등
- **Target 변수**:
  - `price_dev`: 가격 변화율 (수익률)
  - `price_dev_subavg`: 평균 대비 초과 수익률

---

## 🤖 AI 모델 전략

### 두 단계 예측 (Two-Stage Prediction)

#### Stage 1: Classification (필터링)
**목적**: "상승할 종목 vs 하락할 종목" 구분

**모델 구성**:
- XGBoost Classifier (depth=8, 9, 10) × 3개
- LightGBM Classifier (depth=8) × 1개

**앙상블**: Voting (4개 모델 중 과반수가 "상승" 예측)
**Threshold**: 상위 92% (aggressive filtering)

**Output**: 상승 예상 종목만 필터링

#### Stage 2: Regression (랭킹)
**목적**: "얼마나 상승할 것인가?" 예측

**모델 구성**:
- XGBoost Regressor (depth=8, 10) × 2개

**앙상블**: Simple Average
**Output**: 예측 수익률 → Top-K 종목 선정

### 모델 선택 이유

**XGBoost**:
- Gradient Boosting 기반 → 높은 정확도
- 결측치 자동 처리 (재무 데이터는 결측치 많음)
- Feature Importance 제공 → 해석 가능성
- GPU 지원 → 빠른 학습

**LightGBM**:
- XGBoost보다 2-3배 빠름
- Categorical Feature 직접 처리 (sector 등)
- 앙상블 다양성 확보

### 대체 모델 옵션

1. **CatBoost**: Categorical Feature 최적화, Overfitting 방지 강함
2. **Stacking Ensemble**: Meta-learner가 최적 가중치 학습
3. **Deep Learning (LSTM/Transformer)**: 시계열 패턴 학습 강력 (데이터 충분 시)

자세한 모델 API는 [API_REFERENCE.md](API_REFERENCE.md)를 참조하세요.

---

## 🚀 실행 플로우

### 전체 파이프라인 실행

#### 1. Configuration 설정
`config/conf.yaml` 편집:
```yaml
DATA:
  ROOT_PATH: /mnt/external_hdd/quant_data  # 외장하드 경로
  START_YEAR: 2015
  END_YEAR: 2023
  GET_FMP: Y  # 새 데이터 수집 여부
  API_KEY: "your_fmp_api_key"

ML:
  RUN_REGRESSION: Y  # ML 학습 실행 여부
  USE_NEW_MODELS: Y  # 새 모델 아키텍처 사용
  USE_MLFLOW: Y      # MLflow 추적 활성화
  TRAIN_START_YEAR: 2015
  TRAIN_END_YEAR: 2021
  TEST_START_YEAR: 2022
  TEST_END_YEAR: 2023

BACKTEST:
  RUN_BACKTEST: Y
  REBALANCE_PERIOD: 3  # 3개월 (분기별)
  TOP_K_NUM: 100
```

#### 2. Main Script 실행
```bash
cd /home/user/Quant/Quant-refactoring
python main.py
```

**실행 순서**:
```python
# main.py 내부 흐름

# 1. Configuration 로드
from config.context_loader import load_config, MainContext
config = load_config('config/conf.yaml')
ctx = MainContext(config)

# 2. 데이터 수집 (GET_FMP=Y일 때)
if config['DATA']['GET_FMP'] == 'Y':
    from src.data_collector.fmp import FMP
    from src.storage.parquet_storage import ParquetStorage
    from src.storage.parquet_converter import ParquetConverter

    fmp = FMP(ctx)
    fmp.collect()

    # VIEW 테이블 생성
    storage = ParquetStorage(ctx.root_path)
    converter = ParquetConverter(ctx, storage)
    converter.rebuild_table_view()

# 3. ML 데이터 준비 및 학습
if config['ML']['RUN_REGRESSION'] == 'Y':
    from src.training.make_mldata import AIDataMaker
    from src.training.regressor import Regressor

    aidata_maker = AIDataMaker(ctx, config)

    regressor = Regressor(config)
    regressor.dataload()
    regressor.train()
    regressor.evaluation()
    regressor.latest_prediction()

# 4. 백테스팅
if config['BACKTEST']['RUN_BACKTEST'] == 'Y':
    from src.backtest.ml_backtest import Backtest, PlanHandler

    plan_handler = PlanHandler(ctx, 'plan.csv')
    bt = Backtest(ctx, config, plan_handler)
    bt.run()
```

### 스크립트 기반 실행

더 자세한 스크립트 사용법은 [QUICK_START.md](QUICK_START.md)를 참조하세요.

#### 전체 파이프라인
```bash
python src/scripts/run_full_pipeline.py
```

#### 모델 비교
```bash
python src/scripts/run_model_comparison.py
```

#### 리밸런싱 최적화
```bash
python src/scripts/run_rebalance_optimization.py
```

#### 섹터별 트레이딩
```bash
python src/scripts/run_sector_trading.py
```

---

## 📝 요약

### 시스템 강점
1. **End-to-End 자동화**: 데이터 수집 → 학습 → 예측 → 백테스팅
2. **모듈화 설계**: 각 컴포넌트 독립적 실행 가능
3. **앙상블 전략**: 여러 모델 조합으로 안정성 확보
4. **확장 가능**: 새 모델, 데이터, 전략 쉽게 추가
5. **실험 추적**: MLflow로 모든 실험 기록
6. **성능 최적화**: Parquet, GPU, Multiprocessing

### 개선 가능 영역

현재 시스템의 한계점과 개선 방향은 [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md)를 참조하세요.

주요 개선 영역:
- Feature Selection 및 자동화
- Stacking Ensemble 도입
- 거래 비용 반영
- Risk Management 모듈
- 실시간 트레이딩 지원

### 다음 학습 단계

1. **API 상세 정보** → [API_REFERENCE.md](API_REFERENCE.md)
2. **개선 로드맵** → [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md)
3. **빠른 시작** → [QUICK_START.md](QUICK_START.md)
4. **고급 기능** → [ADVANCED_FEATURES_GUIDE.md](ADVANCED_FEATURES_GUIDE.md)

---

## 📚 참고 자료

### 주요 파일 위치
- **Configuration**: `config/conf.yaml`
- **Main Entry**: `main.py`
- **Data Collection**: `src/data_collector/fmp.py`
- **Feature Engineering**: `src/training/make_mldata.py`
- **Model Training**: `src/training/regressor.py`
- **Models**: `src/models/*.py`
- **Backtesting**: `src/backtest/ml_backtest.py`
- **Examples**: `src/examples/*.py`
- **Scripts**: `src/scripts/*.py`

### 로깅 확인
```bash
tail -f outputs/logs/main.log
```

### MLflow UI
```bash
mlflow ui --backend-store-uri {ROOT_PATH}/outputs/mlruns
# http://localhost:5000
```

### 데이터 위치
```
{ROOT_PATH}/  (외장하드)
├── fmp_raw/                        # FMP 원본 데이터
│   ├── income_statement/
│   ├── balance_sheet_statement/
│   └── ...
├── processed/
│   ├── views/                      # 통합 뷰
│   └── ml_data/per_year/           # ML 학습 데이터
└── models/                         # 학습된 모델
    ├── production/
    └── walkforward/
```

---

**문서 버전**: 2.0 (간소화됨)
**마지막 업데이트**: 2025-11-17
**작성자**: Claude AI + Development Team

궁금한 점이나 개선 제안이 있으면 GitHub Issues에 등록해주세요.
