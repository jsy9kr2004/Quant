# Quant Trading System - 개발자 워크플로우 가이드

> **목적**: 새로운 개발자 온보딩 및 코드 리뷰를 위한 종합 가이드
> **작성일**: 2025-10-27
> **대상**: 신규 개발자, 투자자를 위한 기술 리뷰

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [전체 아키텍처](#-전체-아키텍처)
3. [데이터 파이프라인](#-데이터-파이프라인)
4. [AI 모델 상세](#-ai-모델-상세)
5. [주요 컴포넌트](#-주요-컴포넌트)
6. [실행 플로우](#-실행-플로우)
7. [현재 한계점 및 개선 방향](#-현재-한계점-및-개선-방향)

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
│  [FMP API] → data_collector/fmp.py                              │
│     ├─ Stock List (NASDAQ, NYSE)                                │
│     ├─ Delisted Companies                                       │
│     ├─ Financial Statements (Income, Balance, CashFlow)         │
│     ├─ Key Metrics (P/E, ROE, Debt Ratios...)                   │
│     └─ Historical Price Data                                    │
│            ↓                                                     │
│     CSV Files → /data/{category}/{symbol}.csv                   │
│            ↓                                                     │
│  storage/parquet_converter.py                                   │
│     → Parquet Files (5-10x faster, 85-90% compressed)           │
│            ↓                                                     │
│     /data/VIEW/ (통합 뷰)                                         │
│       ├─ symbol_list.csv                                        │
│       ├─ price.csv                                              │
│       ├─ financial_statement_{year}.csv                         │
│       └─ metrics_{year}.csv                                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│           3. ML Data Preparation (training/make_mldata.py)       │
│                                                                  │
│  VIEW 데이터 로드 → Merge (symbol, date 기준)                    │
│            ↓                                                     │
│  시계열 윈도우 생성 (12개월 lookback)                              │
│            ↓                                                     │
│  tsfresh 특성 추출 (EfficientFCParameters)                       │
│    ├─ standard_deviation                                        │
│    ├─ quantile                                                  │
│    ├─ autocorrelation                                           │
│    ├─ fft_coefficient                                           │
│    └─ ar_coefficient (36개 시계열 특성)                          │
│            ↓                                                     │
│  재무 비율 계산 (139개 ratio features)                            │
│    ├─ ROE, ROIC, Profit Margins                                │
│    ├─ P/E, P/B, EV/EBITDA                                       │
│    ├─ Debt Ratios, Coverage Ratios                             │
│    └─ Customized: OverMC_*, adaptiveMC_*                       │
│            ↓                                                     │
│  RobustScaler 정규화 (Outlier-resistant)                         │
│            ↓                                                     │
│  Target 변수 생성: price_dev, price_dev_subavg                  │
│            ↓                                                     │
│  /data/ml_per_year/rnorm_ml_{year}_{quarter}.parquet           │
│    (예: rnorm_ml_2015_Q1.parquet, rnorm_ml_2015_Q2.parquet...) │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              4. Model Training (training/regressor.py)           │
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
│  │  clsmodel_0: XGBoost Classifier (depth=8)      │            │
│  │  clsmodel_1: XGBoost Classifier (depth=9)      │            │
│  │  clsmodel_2: XGBoost Classifier (depth=10)     │            │
│  │  clsmodel_3: LightGBM Classifier (depth=8)     │            │
│  │                                                 │            │
│  │  → Ensemble Voting (상위 92% threshold)        │            │
│  │  → Binary Filter: 상승 예상 종목만 선택           │            │
│  └─────────────────────────────────────────────────┘            │
│            ↓                                                     │
│  ┌─────────────────────────────────────────────────┐            │
│  │         Regression Models (Continuous)          │            │
│  │  (상승폭 예측 → 랭킹 용도)                         │            │
│  ├─────────────────────────────────────────────────┤            │
│  │  model_0: XGBoost Regressor (depth=8)          │            │
│  │  model_1: XGBoost Regressor (depth=10)         │            │
│  │                                                 │            │
│  │  → Average Prediction                          │            │
│  │  → 상승폭 큰 순서로 종목 랭킹                      │            │
│  └─────────────────────────────────────────────────┘            │
│            ↓                                                     │
│  Final Prediction Strategy:                                     │
│    1. Classification 모델로 필터링 (상승 예상 종목만)              │
│    2. Regression 모델로 상승폭 예측                               │
│    3. 예측값 높은 순으로 Top-K 종목 선정                          │
│            ↓                                                     │
│  Model 저장: /data/MODELS/                                       │
│    ├─ clsmodel_{0,1,2,3}.sav                                    │
│    └─ model_{0,1}.sav                                           │
│            ↓                                                     │
│  Optional: MLflow 실험 추적                                       │
│    ├─ 하이퍼파라미터 로깅                                          │
│    ├─ 메트릭 로깅 (Accuracy, RMSE)                               │
│    └─ 모델 아티팩트 저장                                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│          5. Evaluation (training/regressor.py:evaluation())      │
│                                                                  │
│  Test Set 각 분기별 평가                                          │
│    ├─ Classification Accuracy (threshold=92%)                  │
│    ├─ Regression RMSE                                          │
│    └─ Top-K 종목의 실제 수익률 계산                               │
│            ↓                                                     │
│  예측 결과 저장                                                   │
│    ├─ prediction_ai_{year}_{quarter}.csv (전체 예측)            │
│    ├─ prediction_ai_{year}_{quarter}_{model}_top0-3.csv        │
│    ├─ prediction_ai_{year}_{quarter}_{model}_top0-7.csv        │
│    └─ pred_df_topk.csv (요약 통계)                              │
│            ↓                                                     │
│  Latest Prediction (최신 데이터 예측)                             │
│    → latest_prediction.csv                                      │
│    → latest_prediction_{model}_top0-3.csv                       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│               6. Backtesting (backtest.py) - Optional            │
│                                                                  │
│  plan.csv 전략 로드                                               │
│    (key, key_dir, weight, diff, base, base_dir)                │
│            ↓                                                     │
│  For each rebalancing date:                                     │
│    1. Load 재무 데이터 snapshot                                  │
│    2. Calculate scores (plan 기준)                              │
│    3. Select Top-K stocks                                       │
│    4. Record entry prices                                       │
│    5. Wait until next rebalance                                 │
│    6. Record exit prices                                        │
│    7. Calculate returns                                         │
│            ↓                                                     │
│  Generate Reports:                                              │
│    ├─ EVAL: 각 리밸런싱 기간별 상세 메트릭                         │
│    ├─ RANK: 종목 랭킹                                            │
│    └─ AVG: 요약 통계 (Sharpe, Max DD, Win Rate...)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 데이터 파이프라인

### 1. FMP API 데이터 수집
**파일**: `data_collector/fmp.py`

```python
# 주요 흐름
FMP.__get_api_list()           # target_api_list.csv 읽기
  ↓
FMP.__fetch_ticker_list()      # stock_list, delisted_companies
  ↓
FMP.__set_symbol()              # NASDAQ, NYSE 필터링 → symbol_list
  ↓
FMP.__fetch_data()              # 나머지 데이터 (FS, metrics, price)
  ↓
CSV Files saved to /data/{category}/
```

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
- **Multiprocessing**: `fmp_fetch_worker.py`에서 병렬 다운로드
- **Filtering**: NASDAQ, NYSE만 수집 (미국 주요 거래소)
- **Update Check**: `config/update_date.txt`로 중복 수집 방지

### 2. Parquet 변환 및 VIEW 생성
**파일**: `storage/parquet_converter.py`

```python
# CSV → Parquet 변환
ParquetConverter.insert_csv()
  ↓
압축률: 85-90%
읽기 속도: CSV 대비 5-10배 빠름
  ↓
ParquetConverter.rebuild_table_view()
  → VIEW/symbol_list.parquet
  → VIEW/price.parquet
  → VIEW/financial_statement_{year}.parquet
  → VIEW/metrics_{year}.parquet
  (선택사항: SAVE_DEBUG_CSV=Y일 때 .csv 파일도 함께 생성)
```

**VIEW의 역할**: 여러 Parquet 파일을 통합하여 ML 학습에 바로 사용 가능한 형태로 재구성

### 3. ML 데이터 생성
**파일**: `training/make_mldata.py`

#### 3-1. 데이터 로드
```python
AIDataMaker.load_bt_table()
  - symbol_table: 종목 정보
  - price_table: 가격 데이터
  - fs_table: 재무제표 (3년 전~현재)
  - metrics_table: 재무비율
```

#### 3-2. 리밸런싱 날짜 설정
```python
AIDataMaker.set_date()
  - REBALANCE_PERIOD: 3개월 (분기별)
  - 시작: start_year-3
  - 종료: end_year
  → trade_date_list (실제 거래일로 보정)
```

#### 3-3. 가격 변화량 계산
```python
AIDataMaker.process_price_table_wdate()
  - price_diff: 다음 기간 가격 변화 (절대값)
  - price_dev: 가격 변화율 (수익률) → Target 변수
  - price_dev_subavg: 평균 대비 초과 수익률
```

#### 3-4. 시계열 특성 추출
```python
AIDataMaker.make_ml_data()
  For each year, quarter:
    1. 12개월 lookback window 생성
    2. tsfresh.extract_features()
       - EfficientFCParameters 사용
       - 36개 시계열 특성 추출
    3. 재무 비율 계산 (139개)
       - ratio_col_list: 기본 비율
       - OverMC_*: 시총 대비 비율
       - adaptiveMC_*: EV 대비 비율
    4. RobustScaler 정규화
       - Outlier에 강건한 정규화
       - Median, IQR 기반
    5. Sector 정보 추가
       - industry → sector mapping
       - sec_price_dev_subavg: 섹터 대비 초과 수익률
    6. Parquet 저장
       → rnorm_ml_{year}_{quarter}.parquet
```

**tsfresh 특성 상세**:
```python
suffixes_dict = {
    "standard_deviation": ["__r_0.0", "__r_0.25", "__r_0.6", "__r_0.9"],
    "quantile": ["__q_0.2", "__q_0.8"],
    "autocorrelation": ["__lag_0", "__lag_5", "__lag_9"],
    "fft_coefficient": ["__coeff_0", "__coeff_33", "__coeff_99"],
    "cwt_coefficients": ["__coeff_0", "__coeff_6", "__coeff_12"],
    "symmetry_looking": ["__r_0.0", "__r_0.25", "__r_0.65", "__r_0.9"],
    "ar_coefficient": ["__coeff_0", "__coeff_3", "__coeff_6", "__coeff_10"]
}
```
→ Feature 차원 축소: 모든 특성 대신 주요 Suffix만 선택

---

## 🤖 AI 모델 상세

### 모델 아키텍처 개요

#### 두 단계 예측 전략 (Two-Stage Prediction)

```
┌─────────────────────────────────────────────────────────────┐
│                 Stage 1: Classification                      │
│            "상승할 종목 vs 하락할 종목"                         │
├─────────────────────────────────────────────────────────────┤
│  Input: 350+ features (재무비율 + 시계열 특성)                 │
│  Target: price_dev > 0 → 1 (상승), else → 0 (하락)           │
│                                                             │
│  Models:                                                    │
│    1. XGBoost Classifier (depth=8)                         │
│    2. XGBoost Classifier (depth=9)                         │
│    3. XGBoost Classifier (depth=10)                        │
│    4. LightGBM Classifier (depth=8)                        │
│                                                             │
│  Ensemble: Voting (majority vote)                          │
│  Threshold: 상위 92% (aggressive filtering)                │
│                                                             │
│  Output: Binary mask (상승 예상 종목만 True)                  │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                 Stage 2: Regression                         │
│              "얼마나 상승할 것인가?"                            │
├─────────────────────────────────────────────────────────────┤
│  Input: Stage 1에서 필터링된 종목만                            │
│  Target: price_dev_subavg (평균 대비 초과 수익률)             │
│                                                             │
│  Models:                                                    │
│    1. XGBoost Regressor (depth=8)                          │
│    2. XGBoost Regressor (depth=10)                         │
│                                                             │
│  Ensemble: Average prediction                              │
│                                                             │
│  Output: 예측 수익률 (연속값)                                 │
│          → Top-K 종목 선정                                   │
└─────────────────────────────────────────────────────────────┘
```

### 모델별 세부 설정

#### 1. XGBoost Classifier
**파일**: `models/xgboost_model.py`

```python
# 공통 설정
tree_method='gpu_hist'          # GPU 가속
n_estimators=500                # 트리 개수
learning_rate=0.1               # 학습률
gamma=0                         # 최소 loss reduction
subsample=0.8                   # 샘플 샘플링 비율
colsample_bytree=0.8            # Feature 샘플링 비율
objective='binary:logistic'     # Binary classification
eval_metric='logloss'           # Log loss

# Depth 변형 (모델 다양성 확보)
clsmodel_0: max_depth=8         # Shallow (overfitting 방지)
clsmodel_1: max_depth=9         # Medium
clsmodel_2: max_depth=10        # Deep (복잡한 패턴 학습)
```

**왜 XGBoost인가?**
- **Gradient Boosting 기반**: 순차적으로 오차 보정 → 높은 정확도
- **Regularization**: L1/L2 정규화 내장 → Overfitting 방지
- **Missing Value 처리**: 결측치 자동 처리 (재무 데이터는 결측치 많음)
- **Feature Importance**: 해석 가능성 (어떤 재무비율이 중요한지 확인)
- **GPU 지원**: 대용량 데이터 빠른 학습

#### 2. LightGBM Classifier
```python
boosting_type='gbdt'            # Gradient Boosting Decision Tree
objective='binary'              # Binary classification
n_estimators=1000               # 트리 개수 (XGBoost보다 많음)
max_depth=8                     # 트리 깊이
learning_rate=0.1               # 학습률
device='gpu'                    # GPU 사용
boost_from_average=False        # Class imbalance 대응
```

**왜 LightGBM인가?**
- **속도**: XGBoost보다 2-3배 빠름 (Leaf-wise growth)
- **메모리 효율**: 대용량 데이터 처리 유리
- **Categorical Feature 지원**: Sector 등 범주형 변수 직접 처리
- **앙상블 다양성**: XGBoost와 다른 알고리즘 → 앙상블 효과 극대화

#### 3. XGBoost Regressor
```python
tree_method='gpu_hist'
n_estimators=1000               # Classification보다 많음
learning_rate=0.1
gamma=0
subsample=0.8
colsample_bytree=0.8
objective='reg:squarederror'    # Regression
eval_metric='rmse'              # Root Mean Squared Error

model_0: max_depth=8            # Conservative
model_1: max_depth=10           # Aggressive
```

**Regression 전략**:
- Classification 필터 통과 종목만 대상
- 예측값 높은 순으로 랭킹 → Top 3, Top 8, Top 16 선정

### 앙상블 전략

#### Classification Ensemble
```python
# Voting: 4개 모델 중 과반수가 "상승" 예측한 종목만 선택
y_probs = [model.predict_proba(X)[:, 1] for model in clsmodels]
threshold = np.percentile(y_probs, 92)  # 상위 8%만 필터링
ensemble_pred = (sum(y_probs > threshold) >= 2)  # 2개 이상 동의
```

**Threshold 조정**:
- 기본: 50% (balanced)
- 현재: 92% (aggressive) → False Positive 최소화

#### Regression Ensemble
```python
# Simple Average (weighted average도 가능)
pred_0 = model_0.predict(X)
pred_1 = model_1.predict(X)
final_pred = (pred_0 + pred_1) / 2

# Classification 필터 적용
final_pred = np.where(cls_ensemble_pred == 0, -1, final_pred)
```

### 대체 모델 옵션

#### 1. CatBoost
**장점**:
- Categorical Feature 최적화 (Sector, Industry)
- Overfitting 방지 강함
- Ordered Boosting (Target Leakage 방지)

**적용 방법**:
```python
from models.catboost_model import CatBoostModel

model = CatBoostModel(task='classification')
model.build_model()
model.fit(X_train, y_train, cat_features=['sector', 'industry'])
```

#### 2. Neural Network (LSTM/Transformer)
**장점**:
- 시계열 패턴 학습 강력
- 복잡한 비선형 관계 포착

**단점**:
- 해석 불가능 (Black Box)
- 학습 데이터 대량 필요 (현재 데이터 부족 가능성)
- Overfitting 위험

**적용 가능성**: 데이터 축적 후 (5년 이상 권장)

#### 3. Stacking Ensemble
**파일**: `models/ensemble.py`

```python
from models.ensemble import StackingEnsemble

# Base models
base_models = [
    ('xgb_8', xgb_model_8),
    ('xgb_9', xgb_model_9),
    ('xgb_10', xgb_model_10),
    ('lgb', lgb_model)
]

# Meta-learner로 Ridge Regression 사용
ensemble = StackingEnsemble(
    base_models=base_models,
    task='classification',
    meta_learner='ridge',
    cv=5
)
ensemble.build_ensemble().fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

**장점**: Simple Voting보다 성능 향상 (Meta-learner가 최적 가중치 학습)

#### 4. AutoML (H2O, Auto-sklearn)
**장점**: 자동 모델 선택, 하이퍼파라미터 최적화
**단점**: 해석성 낮음, 계산 비용 높음

---

## 🔧 주요 컴포넌트

### 1. Configuration (`config/`)

#### `context_loader.py`
```python
class MainContext:
    def __init__(self, config_dict):
        # DATA 설정
        self.root_path = config['DATA']['ROOT_PATH']
        self.start_year = config['DATA']['START_YEAR']
        self.end_year = config['DATA']['END_YEAR']
        self.fmp_api_key = config['DATA']['API_KEY']

        # ML 설정
        self.train_start_year = config['ML']['TRAIN_START_YEAR']
        self.train_end_year = config['ML']['TRAIN_END_YEAR']

        # BACKTEST 설정
        self.rebalance_period = config['BACKTEST']['REBALANCE_PERIOD']
        self.top_k_num = config['BACKTEST']['TOP_K_NUM']

        # Logger 초기화
        self.setup_logging()
```

**사용 예**:
```python
from config.context_loader import load_config, MainContext

config = load_config('config/conf.yaml')
ctx = MainContext(config)
```

#### `g_variables.py`
전역 변수 및 Feature 리스트 정의

```python
# 139개 재무 비율
ratio_col_list = [
    'roic', 'roe', 'roa',                  # 수익성
    'priceToBookRatio', 'priceEarningsRatio',  # 밸류에이션
    'debtToEquity', 'currentRatio',        # 재무건전성
    ...
]

# 158개 절대값 지표
meaning_col_list = [
    'revenue', 'netIncome', 'eps',
    'totalAssets', 'totalDebt',
    ...
]

# 36개 시계열 특성 대상
cal_timefeature_col_list = [
    'roic', 'roe', 'netProfitMargin',
    ...
]

# Sector Mapping
sector_map = {
    'Software': 'Technology',
    'Semiconductors': 'Technology',
    'Pharmaceuticals': 'Healthcare',
    ...
}
```

#### `logger.py`
멀티프로세싱 안전 로깅 시스템

```python
from config.logger import setup_logging, get_logger

setup_logging(log_level=20)  # INFO
logger = get_logger('my_module')

logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

**특징**:
- **QueueHandler**: 멀티프로세싱 환경에서 안전
- **Color Output**: 콘솔 로그 가독성
- **Rotation**: 로그 파일 10MB마다 자동 분할

### 2. Models (`models/`)

#### Base Model 구조
```python
from models.base_model import BaseModel

class CustomModel(BaseModel):
    def build_model(self, params=None):
        # 모델 생성
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # 학습
        pass

    def predict(self, X):
        # 예측
        pass

    def evaluate(self, X_test, y_test):
        # 평가
        pass
```

**사용 예**:
```python
from models.xgboost_model import XGBoostModel

model = XGBoostModel(task='classification', config_name='depth_9')
model.build_model()
model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

### 3. Training Pipeline (`training/`)

#### Hyperparameter Optimization
**파일**: `training/optimizer.py`

```python
from training.optimizer import OptunaOptimizer
from models.xgboost_model import XGBoostModel

# Search space 정의
search_space = {
    'max_depth': (5, 12),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'gamma': (0, 5)
}

optimizer = OptunaOptimizer(
    model_class=XGBoostModel,
    search_space=search_space,
    n_trials=100,
    cv_folds=5,
    scoring='accuracy'
)

best_params = optimizer.optimize(X_train, y_train)
```

**Optuna 장점**:
- **TPE Sampler**: Bayesian Optimization (Grid Search보다 효율적)
- **Pruning**: 성능 낮은 trial 조기 종료
- **Parallel**: 여러 trial 병렬 실행

#### MLflow Tracking
**파일**: `training/mlflow_tracker.py`

```python
from training.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(experiment_name='quant_trading')

with tracker.start_run(run_name='xgb_depth9'):
    tracker.log_params({
        'max_depth': 9,
        'learning_rate': 0.1,
        'n_estimators': 500
    })

    # 학습
    model.fit(X_train, y_train)

    # 평가
    metrics = model.evaluate(X_test, y_test)
    tracker.log_metrics(metrics)

    # 모델 저장
    tracker.log_model(model, 'xgboost_model')
```

**MLflow UI**:
```bash
mlflow ui --backend-store-uri /path/to/mlruns
# http://localhost:5000
```

### 4. Backtesting (`backtest.py`)

```python
from backtest import Backtest, PlanHandler, DateHandler

# Plan 로드 (사용자 정의 전략)
plan_handler = PlanHandler('plan.csv')

# Backtest 실행
bt = Backtest(ctx, config, plan_handler)
bt.run()

# 리포트 생성
bt.generate_reports(['EVAL', 'RANK', 'AVG'])
```

**Plan 예시** (`plan.csv`):
```
key,key_dir,weight,diff,base,base_dir
roe,descending,1.0,False,,
priceEarningsRatio,ascending,0.8,False,,
debtToEquity,ascending,0.5,False,,
price_dev,descending,2.0,True,,
```

**리포트 종류**:
- **EVAL**: 각 리밸런싱 기간별 수익률, Sharpe Ratio, Max Drawdown
- **RANK**: 선정 종목 리스트 및 스코어
- **AVG**: 전체 기간 평균 통계

---

## 🚀 실행 플로우

### 전체 파이프라인 실행

#### 1. Configuration 설정
`config/conf.yaml` 편집:
```yaml
DATA:
  ROOT_PATH: /home/user/Quant/data
  START_YEAR: 2015
  END_YEAR: 2023
  GET_FMP: Y  # 새 데이터 수집할지 여부
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
  TOP_K_NUM: 100       # 상위 100개 종목
  REPORT_LIST: [EVAL, RANK, AVG]
```

#### 2. Main Script 실행
```bash
cd Quant-refactoring
python main.py
```

**실행 순서**:
```python
# main.py 내부 흐름

# 1. Configuration 로드
config = load_config('config/conf.yaml')
ctx = MainContext(config)

# 2. 데이터 수집 (GET_FMP=Y일 때)
if config['DATA']['GET_FMP'] == 'Y':
    fmp = FMP(ctx)
    fmp.collect()

    # Parquet 변환
    storage = ParquetStorage(ctx.root_path)
    converter = ParquetConverter(ctx, storage)
    converter.insert_csv()
    converter.rebuild_table_view()

# 3. ML 데이터 준비
if config['ML']['RUN_REGRESSION'] == 'Y':
    aidata_maker = AIDataMaker(ctx, config)

    # 4. 모델 학습
    regressor = Regressor(config)
    regressor.dataload()
    regressor.train()
    regressor.evaluation()
    regressor.latest_prediction()

# 5. 백테스팅
if config['BACKTEST']['RUN_BACKTEST'] == 'Y':
    plan_handler = PlanHandler(ctx, 'plan.csv')
    bt = Backtest(ctx, config, plan_handler)
    bt.run()
```

### 개별 컴포넌트 실행

#### 데이터 수집만
```bash
python -c "
from config.context_loader import load_config, MainContext
from data_collector.fmp import FMP

config = load_config('config/conf.yaml')
ctx = MainContext(config)

fmp = FMP(ctx)
fmp.collect()
"
```

#### ML 학습만
```bash
python -c "
from config.context_loader import load_config, MainContext
from training.regressor import Regressor

config = load_config('config/conf.yaml')
regressor = Regressor(config)

regressor.dataload()
regressor.train()
regressor.evaluation()
"
```

#### 최신 예측만
```bash
python -c "
from config.context_loader import load_config
from training.regressor import Regressor

config = load_config('config/conf.yaml')
regressor = Regressor(config)

regressor.dataload()
regressor.latest_prediction()
# → /data/MODELS/latest_prediction.csv
"
```

### 스크립트 기반 실행

#### 전체 파이프라인
```bash
python scripts/run_full_pipeline.py
```

#### 모델 비교
```bash
python scripts/run_model_comparison.py
# 여러 모델 설정 성능 비교
```

#### 리밸런싱 최적화
```bash
python scripts/run_rebalance_optimization.py
# 1개월, 3개월, 6개월, 12개월 리밸런싱 성과 비교
```

#### 섹터별 트레이딩
```bash
python scripts/run_sector_trading.py
# 섹터별 모델 학습 및 예측
```

---

## ⚠ 현재 한계점 및 개선 방향

### 한계점

#### 1. 데이터 품질
**문제**:
- FMP API 데이터 결측치 많음 (특히 소형주)
- 일부 재무지표 오류 (API 제공사 문제)
- 상장폐지 종목 데이터 불완전

**개선 방향**:
- 다중 데이터 소스 통합 (Yahoo Finance, Alpha Vantage)
- 데이터 검증 로직 강화 (`validation/` 모듈 활용)
- 이상치 탐지 및 자동 보정

#### 2. Feature Engineering
**문제**:
- 고정된 Feature 리스트 (139개 ratio + 36개 시계열)
- Feature Selection 미흡 (중요도 낮은 feature 다수)
- Sector/Industry 정보 활용 부족

**개선 방향**:
- **Feature Selection**: `feature_engineering/feature_selector.py` 활용
  ```python
  from feature_engineering.feature_selector import FeatureSelector

  selector = FeatureSelector(method='importance', threshold=0.01)
  selected_features = selector.fit_transform(X_train, y_train)
  ```
- **Automated Feature Engineering**: Featuretools 도입
- **Sector-specific Features**: 섹터별 중요 지표 다르게 적용

#### 3. 모델 성능
**문제**:
- Classification Accuracy: 55-60% (높지 않음)
- Regression RMSE: 개선 여지 있음
- 시장 변동성 높은 시기 성능 저하

**개선 방향**:
- **앙상블 고도화**: Stacking Ensemble 적용 (현재 Simple Voting)
  ```python
  from models.ensemble import StackingEnsemble

  ensemble = StackingEnsemble(
      base_models=[...],
      meta_learner='ridge',
      cv=5
  )
  ```
- **하이퍼파라미터 최적화**: Optuna로 전역 최적화
  ```bash
  python scripts/optimize_hyperparameters.py --n_trials 200
  ```
- **Deep Learning 실험**: LSTM, Transformer (데이터 충분 시)

#### 4. 백테스팅 한계
**문제**:
- 거래 비용 미반영 (수수료, 슬리피지)
- 유동성 제약 미고려 (대량 매수/매도 시 가격 영향)
- Market Regime 변화 미반영 (불장/약장)

**개선 방향**:
- **거래 비용 모델링**:
  ```python
  # backtest.py 수정
  entry_cost = entry_price * 0.001  # 0.1% 수수료
  exit_cost = exit_price * 0.001
  net_return = gross_return - entry_cost - exit_cost
  ```
- **Slippage 모델**:
  ```python
  slippage = entry_price * 0.002  # 0.2% 슬리피지
  actual_entry = entry_price * (1 + slippage)
  ```
- **Volume Filter**: 일평균 거래대금 하위 50% 제외 (이미 적용 중)

#### 5. 실시간 트레이딩 지원 부족
**문제**:
- 일별 가격만 지원 (분/초 단위 미지원)
- 실시간 데이터 수집 없음
- 자동 주문 기능 없음

**개선 방향**:
- **실시간 데이터**: WebSocket API 연동
- **자동 매매**: Interactive Brokers API, Alpaca API 연동
- **모니터링**: `monitoring/performance_monitor.py` 활용
  ```python
  from monitoring.performance_monitor import PerformanceMonitor

  monitor = PerformanceMonitor()
  monitor.track_prediction_accuracy(y_true, y_pred)
  monitor.track_portfolio_value(current_value)
  monitor.send_alert_if_drawdown_exceeds(threshold=0.1)
  ```

### 추가 필요 기능

#### 1. Risk Management
**현재**: 단순 Top-K 선정 (동일 가중)

**개선**:
```python
# portfolio_optimization/ 추가
from portfolio_optimization.risk_manager import RiskManager

risk_mgr = RiskManager(
    max_position_size=0.05,    # 종목당 5% 제한
    max_sector_exposure=0.30,  # 섹터당 30% 제한
    stop_loss=0.10,            # 10% 손절
    take_profit=0.20           # 20% 익절
)

portfolio = risk_mgr.allocate_weights(predictions, prices)
```

#### 2. 포트폴리오 최적화
**추가 알고리즘**:
- **Mean-Variance Optimization** (Markowitz)
- **Black-Litterman Model** (예측 결합)
- **Risk Parity** (위험 균등 배분)

```python
from portfolio_optimization.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(method='mean_variance')
weights = optimizer.optimize(
    expected_returns=predictions,
    covariance_matrix=cov_matrix,
    risk_aversion=2.5
)
```

#### 3. Time Series CV 강화
**현재**: Train/Test 단순 분할

**개선**:
```python
from validation.time_series_cv import TimeSeriesCV

cv = TimeSeriesCV(n_splits=5, gap=3)  # 3개월 gap
for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # 각 fold별 학습 및 평가
```

**Walk-Forward Analysis**:
```python
from validation.walk_forward import WalkForwardAnalysis

wfa = WalkForwardAnalysis(
    train_period=12,  # 12개월 학습
    test_period=3,    # 3개월 테스트
    step=3            # 3개월씩 이동
)
results = wfa.run(X, y, model)
```

#### 4. Explainability (설명 가능성)
**SHAP 도입**:
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 종목별 예측 이유 시각화
shap.waterfall_plot(shap_values[0])

# Feature Importance
shap.summary_plot(shap_values, X_test)
```

**사용 사례**:
- "왜 이 종목을 추천했나?" → SHAP 값으로 설명
- 투자자 신뢰도 향상

#### 5. 대체 데이터 통합
**추가 데이터 소스**:
- **뉴스 감성 분석**: FinBERT, GPT API
- **소셜 미디어**: Twitter/Reddit 감성 (WallStreetBets)
- **거래량 이상 탐지**: Unusual Volume Analysis
- **옵션 데이터**: Put/Call Ratio

```python
# alternative_data/ 추가
from alternative_data.news_sentiment import NewsSentiment

news = NewsSentiment(api_key='news_api_key')
sentiment_scores = news.get_sentiment(symbols, start_date, end_date)

# Feature로 추가
X['news_sentiment'] = sentiment_scores
```

#### 6. Regime Detection (시장 국면 감지)
**목적**: 불장/약장에 따라 전략 변경

```python
from market_analysis.regime_detector import RegimeDetector

detector = RegimeDetector(method='hmm')  # Hidden Markov Model
regimes = detector.fit_predict(price_history)

# Regime별 모델 학습
bull_model = train_model(X[regimes == 'bull'], y[regimes == 'bull'])
bear_model = train_model(X[regimes == 'bear'], y[regimes == 'bear'])
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

### 개선 우선순위
1. **Feature Selection** (즉시 적용 가능, 큰 효과)
2. **Stacking Ensemble** (성능 향상 즉시)
3. **거래 비용 반영** (백테스트 현실화)
4. **Risk Management** (안정성 확보)
5. **SHAP Explainability** (신뢰도 향상)

### 다음 단계 제안
1. **단기** (1-2주):
   - Feature Selection 적용 → 학습 속도 향상
   - Stacking Ensemble 실험 → 성능 개선
   - 거래 비용 모델링 → 백테스트 정확도 향상

2. **중기** (1-2개월):
   - Optuna 하이퍼파라미터 최적화 → 모든 모델 최적화
   - Time Series CV 강화 → Overfitting 방지
   - Risk Management 모듈 추가 → 안정성 확보

3. **장기** (3-6개월):
   - 대체 데이터 통합 (뉴스, 소셜미디어)
   - Deep Learning 실험 (LSTM, Transformer)
   - 실시간 트레이딩 시스템 구축

---

## 📚 참고 자료

### 주요 파일 위치
- **Configuration**: `config/conf.yaml`
- **Main Entry**: `main.py`
- **Data Collection**: `data_collector/fmp.py`
- **Feature Engineering**: `training/make_mldata.py`
- **Model Training**: `training/regressor.py`
- **Models**: `models/*.py`
- **Backtesting**: `backtest.py`
- **Examples**: `examples/*.py`
- **Scripts**: `scripts/*.py`

### 로깅 확인
```bash
tail -f logs/quant_trading.log
```

### MLflow UI
```bash
mlflow ui --backend-store-uri /home/user/Quant/data/mlruns
# http://localhost:5000
```

### 데이터 위치
```
/home/user/Quant/data/
├── stock_list/
├── financial_statements/
├── key_metrics/
├── historical_price/
├── VIEW/                   # 통합 뷰
├── ml_per_year/            # ML 학습 데이터
└── MODELS/                 # 학습된 모델
```

---

**문서 버전**: 1.0
**마지막 업데이트**: 2025-10-27
**작성자**: Claude AI + Development Team

궁금한 점이나 개선 제안이 있으면 GitHub Issues에 등록해주세요.
