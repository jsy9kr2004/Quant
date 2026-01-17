# Quant Trading System - Refactored

개선된 퀀트 트레이딩 시스템 (2025)

---

## 📚 문서 가이드

이 프로젝트의 문서는 **사용자 유형에 따라** 읽는 순서가 다릅니다. 자신에게 맞는 경로를 선택하세요!

### 🎯 처음 사용하시는 분
**추천 읽기 순서:**
1. **[docs/QUICK_START.md](docs/QUICK_START.md)** ⭐ 5-10분 안에 실행 시작
2. **[README.md](#빠른-시작)** (현재 문서) - 프로젝트 전체 개요 이해
3. **[docs/FIX_ERRORS.md](docs/FIX_ERRORS.md)** - 문제 발생 시 참고

### 👨‍💻 개발자 / 시스템 깊이 이해가 필요한 분
**추천 읽기 순서:**
1. **[README.md](#프로젝트-구조)** (현재 문서) - 프로젝트 구조 먼저 확인
2. **[docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md)** ⭐ 전체 아키텍처 상세 설명 (필독)
3. **[docs/LOGGING_GUIDE.md](docs/LOGGING_GUIDE.md)** - 로깅 시스템 이해
4. **[docs/ROBUST_VALIDATION_GUIDE.md](docs/ROBUST_VALIDATION_GUIDE.md)** - 백테스팅 및 검증 시스템

### 📊 고급 기능을 활용하고 싶은 분
**추천 읽기 순서:**
1. **[docs/QUICK_START.md](docs/QUICK_START.md)** - 기본 실행 확인
2. **[docs/ADVANCED_FEATURES_GUIDE.md](docs/ADVANCED_FEATURES_GUIDE.md)** ⭐ 리밸런싱 최적화, 모델 비교, 섹터별 모델
3. **[docs/ROBUST_VALIDATION_GUIDE.md](docs/ROBUST_VALIDATION_GUIDE.md)** - 성능 검증 방법

### 🔄 리팩토링 관련 문서 (2025-12-01 추가)
**코드 이중화 제거 및 통합 아키텍처:**
1. **[docs/REFACTORING_GUIDE.md](docs/REFACTORING_GUIDE.md)** ⭐ 리팩토링 전체 가이드
2. **[docs/DATA_SCHEMA_REFERENCE.md](docs/DATA_SCHEMA_REFERENCE.md)** - DataSchema API 레퍼런스

### 🎯 실제 투자에 활용하고 싶은 분
**추천 읽기 순서:**
1. **[docs/PREDICTION_MODE.md](docs/PREDICTION_MODE.md)** ⭐ 예측 전용 모드 (오늘 뭘 사야 하나?)
2. **[docs/QUICK_START.md](docs/QUICK_START.md)** - 기본 실행 확인

### 🔍 시스템 전체 분석 레포트
**코드베이스 심층 분석 (AI 작성):**

> 📂 **[docs/codebase-report/](docs/codebase-report/)** - 프로젝트의 핵심 분석 레포트

| 문서 | 내용 |
|------|------|
| [00_overview.md](docs/codebase-report/00_overview.md) | 프로젝트 개요 및 종합 평가 |
| [01_architecture.md](docs/codebase-report/01_architecture.md) | 시스템 아키텍처 분석 |
| [02_data_pipeline.md](docs/codebase-report/02_data_pipeline.md) | 데이터 파이프라인 분석 |
| [03_ml_strategy.md](docs/codebase-report/03_ml_strategy.md) | ML 전략 분석 |
| [04_backtesting.md](docs/codebase-report/04_backtesting.md) | 백테스팅 시스템 분석 |
| [05_code_quality.md](docs/codebase-report/05_code_quality.md) | 코드 품질 분석 |
| [06_quant_perspective.md](docs/codebase-report/06_quant_perspective.md) | 퀀트 관점 분석 |
| [07_recommendations.md](docs/codebase-report/07_recommendations.md) | 개선 권고사항 |
| [08_recent_changes.md](docs/codebase-report/08_recent_changes.md) | 최신 변경사항 |

### 📁 기타 문서
- **[docs/archive/](docs/archive/)** - 과거 분석 리포트 및 참고 자료

---

## 주요 개선사항

### 1. Parquet 저장소 (검증 기능 포함)
- ✅ 자동 데이터 검증
- ✅ 샘플 CSV 자동 생성 (빠른 확인용)
- ✅ 70-90% 압축률 (CSV 대비)
- ✅ 컬럼별 선택적 읽기 (빠른 성능)

### 2. CatBoost 모델 추가
- ✅ 오버피팅에 강함
- ✅ Ordered boosting
- ✅ GPU 가속

### 3. Optuna 자동 하이퍼파라미터 튜닝
- ✅ Bayesian optimization
- ✅ 자동 Pruning
- ✅ Cross-validation 지원

### 4. Stacking 앙상블
- ✅ 단순 평균 대신 학습된 가중치
- ✅ Cross-validation 기반 메타 학습
- ✅ 더 나은 예측 성능

### 5. MLflow 실험 추적
- ✅ 모든 실험 자동 기록
- ✅ 파라미터/메트릭 비교
- ✅ 모델 버전 관리

### 6. Ray 기반 병렬 데이터 수집 (NEW)
- ✅ 멀티프로세싱보다 효율적인 분산 처리
- ✅ 동적 작업 스케줄링
- ✅ API rate limit 최적화 (8 workers)

### 7. 깔끔한 프로젝트 구조 (NEW)
- ✅ 모듈별 명확한 분리
- ✅ 루트 디렉토리 정리 (7개 → 2개 파일)
- ✅ 일관된 패키지 구조

### 8. 코드 이중화 제거 리팩토링 (2025-12-01) ✨ NEW
- ✅ **DataSchema**: 컬럼 정의 단일화 (regressor ↔ ml_backtest 동기화)
- ✅ **DataProcessor**: 전처리 파이프라인 통일
- ✅ **ModelFactory**: 모델 생성 일관성 보장
- ✅ **825줄의 중복 코드 제거** (-100% 중복)
- ✅ **예측도 ↔ 백테스트 수익률 동일선상 비교 가능**
- ✅ **아키텍처 기반 일원화 강제** (2025-01-17)
  - Prediction Cache 공유: regressor.py 예측 → ml_backtest.py 재사용
  - Fallback 제거: 캐시 없으면 에러 발생 (silent fallback 금지)
  - 유닛테스트 없이도 일원화 100% 보장

> 📖 **상세 가이드:** [docs/REFACTORING_GUIDE.md](docs/REFACTORING_GUIDE.md)

### 9. 예측 전용 모드 (2025-01) ✨ NEW
- ✅ **학습 없이 예측만**: 이미 학습된 모델로 빠르게 추천
- ✅ **특정 날짜 지정**: `TARGET_DATE: "2025-01-11"` 형식
- ✅ **실시간 활용**: FMP 데이터 업데이트 후 즉시 추천 확인
- ✅ **과거 시점 시뮬레이션**: "그때 뭘 샀어야 했나?" 분석

> 📖 **상세 가이드:** [docs/PREDICTION_MODE.md](docs/PREDICTION_MODE.md)

## 프로젝트 구조

### 📂 로컬 프로젝트 (Quant-refactoring/)

```
Quant-refactoring/
│
├── 📜 main.py                          # 🎯 메인 실행 진입점
├── 📄 README.md                        # 프로젝트 문서
├── 📄 requirements.txt                 # Python 패키지 의존성
│
├── 📁 src/                             # 모든 소스코드
│   ├── 📁 data_collector/              # 📡 데이터 수집 (Ray)
│   │   ├── fmp.py                     # FMP 메인
│   │   ├── fmp_api.py                 # API 관리
│   │   ├── fmp_fetch_worker.py        # 병렬 처리
│   │   └── target_api_list.csv        # API 목록
│   │
│   ├── 📁 storage/                     # 💾 데이터 저장소
│   │   ├── parquet_storage.py         # Parquet 저장
│   │   ├── parquet_converter.py       # 변환기
│   │   └── data_validator.py          # 검증
│   │
│   ├── 📁 constants/                   # ✨ NEW: 통합 상수 정의
│   │   ├── __init__.py
│   │   └── data_schema.py             # 컬럼 정의 단일화
│   │
│   ├── 📁 training/                    # 🎓 ML 학습
│   │   ├── data_processor.py          # ✨ NEW: 통합 전처리
│   │   ├── make_mldata.py             # 데이터 전처리
│   │   ├── regressor.py               # 학습 모델 (DataSchema 사용)
│   │   ├── optimizer.py               # Optuna 튜닝
│   │   └── mlflow_tracker.py          # MLflow 추적
│   │
│   ├── 📁 models/                      # 🤖 ML 모델
│   │   ├── base_model.py              # 기본 클래스
│   │   ├── xgboost_model.py           # XGBoost
│   │   ├── lightgbm_model.py          # LightGBM
│   │   ├── catboost_model.py          # CatBoost
│   │   └── ensemble.py                # 앙상블
│   │
│   ├── 📁 backtest/                    # 📊 백테스트
│   │   └── ml_backtest.py             # Walk-Forward
│   │
│   ├── 📁 validation/                  # ✅ 검증
│   │   ├── walk_forward.py
│   │   └── time_series_cv.py
│   │
│   ├── 📁 optimization/                # 🎯 최적화
│   │   ├── rebalance_optimizer.py
│   │   └── model_comparator.py
│   │
│   ├── 📁 scripts/                     # 🚀 실행 스크립트
│   │   ├── run_ml_backtest.py
│   │   ├── run_model_comparison.py
│   │   └── debug/
│   │
│   ├── 📁 examples/                    # 📚 예제
│   │   └── example_complete_pipeline.py
│   │
│   └── 📁 tools/                       # 🔧 도구
│       ├── parquet_viewer.py
│       └── rank_processing.py
│
├── 📁 config/                          # ⚙️ 설정 (코드 + 파일)
│   ├── context_loader.py              # 설정 로더
│   ├── logger.py                      # 로깅 시스템
│   ├── g_variables.py                 # 전역 변수
│   ├── file_utils.py                  # 파일 유틸리티
│   ├── conf.yaml.template             # 설정 템플릿
│   └── config_quick_test.json         # 테스트 설정
│
├── 📁 docs/                            # 📖 문서
│   ├── QUICK_START.md                 # 빠른 시작
│   ├── WORKFLOW_GUIDE.md              # 워크플로우
│   ├── MIGRATION_GUIDE.md             # 마이그레이션
│   ├── codebase-report/               # 🔍 핵심 분석 레포트 (AI 작성)
│   ├── analysis/                      # 작업용 분석 문서
│   └── archive/                       # 레거시 문서
│
└── 📁 outputs/                         # 실행 결과물 (gitignore)
    ├── logs/                          # 로그 파일
    │   ├── main.log
    │   └── archived/
    ├── reports/                       # 백테스트 결과
    ├── debug/                         # 디버그 파일
    ├── mlruns/                        # MLflow 추적
    └── temp/                          # 임시 파일
```

### 📂 외장하드 (ROOT_PATH)

**이동성을 위한 분리된 데이터 저장소** (외장하드에 저장하여 다양한 컴퓨터에서 사용)

```
{ROOT_PATH}/  (예: /mnt/external_hdd/quant_data/)
│
├── 📁 fmp_raw/                         # FMP 원본 데이터
│   ├── income_statement/              # API 카테고리별
│   ├── balance_sheet_statement/
│   ├── cash_flow_statement/
│   ├── key_metrics/
│   ├── stock_list/
│   └── ... (심볼별 .parquet 파일)
│
├── 📁 processed/                       # 가공된 데이터
│   ├── views/                         # VIEW 테이블
│   │   ├── price.parquet
│   │   ├── symbol_list.parquet
│   │   └── financial_statement_*.parquet
│   │
│   ├── ml_data/                       # ML 학습 데이터
│   │   └── per_year/
│   │       ├── rnorm_ml_2020_Q1.parquet
│   │       └── rnorm_fs_2020_Q1.parquet
│   │
│   └── intermediate/                  # 중간 데이터
│       ├── DATE_TABLE/
│       └── parquet/
│
├── 📁 models/                          # 학습된 모델
│   ├── production/                    # 프로덕션
│   │   ├── best_model.pkl
│   │   └── ensemble_final.pkl
│   │
│   └── walkforward/                   # Walk-Forward
│       ├── model_20230313.pkl
│       └── model_sector_20230613.pkl
│
├── 📁 analysis/                        # 분석 결과
│   ├── nan_analysis/                  # NaN 분석
│   └── nan_removal/                   # NaN 제거 상세
│
└── 📁 debug/                           # 디버그 파일
    └── fs_metric_wdate_*.parquet
```

### 📋 주요 파일 설명

#### 로컬 프로젝트 (소스 코드)
| 파일/디렉토리 | 설명 | 필수 여부 |
|--------------|------|----------|
| `main.py` | 메인 실행 진입점 | ✅ 필수 |
| `src/` | 모든 소스코드 | ✅ 필수 |
| `config/conf.yaml` | 설정 파일 (생성 필요) | ✅ 필수 |
| `docs/` | 문서 | ⚪ 권장 |
| `outputs/` | 실행 결과 (자동 생성, gitignore) | ⚪ 자동 |

#### 외장하드 (데이터)
| 파일/디렉토리 | 설명 | 크기 |
|--------------|------|------|
| `fmp_raw/` | FMP 원본 데이터 | 대용량 |
| `processed/views/` | 가공 데이터 | 중간 |
| `processed/ml_data/` | ML 학습 데이터 | 중간 |
| `models/` | 학습된 모델 | 소규모 |

### 🔑 핵심 개념

**데이터 흐름:**
```
FMP API → ROOT_PATH/fmp_raw/ (원본)
       → ROOT_PATH/processed/views/ (가공)
       → ROOT_PATH/processed/ml_data/ (ML 데이터)
       → 학습 → ROOT_PATH/models/ (모델 저장)
       → 백테스트 → outputs/reports/ (결과)
```

**코드와 데이터 분리:**
- **로컬**: 코드만 (Git 관리)
- **외장하드**: 데이터만 (이동 가능)
- **이동성**: ROOT_PATH만 변경하면 어디서든 실행

**설정 우선순위:**
1. `config/conf.yaml` (사용자 설정)
2. `config/conf.yaml.template` (기본값)
3. 코드 내 하드코딩 (fallback)

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 설정 파일 생성

```bash
cd /home/user/Quant/Quant-refactoring

# 템플릿 복사
cp config/conf.yaml.template config/conf.yaml

# 설정 편집 (YOUR_FMP_API_KEY_HERE를 실제 API 키로 변경)
nano config/conf.yaml  # 또는 vim, vi
```

**최소 필수 설정**:
```yaml
DATA:
  ROOT_PATH: /home/user/Quant/data  # 데이터 저장 경로
  API_KEY: YOUR_ACTUAL_API_KEY      # FMP API 키
  GET_FMP: N                         # 처음엔 N (기존 데이터 사용)

ML:
  RUN_REGRESSION: Y                  # 모델 학습
  EXIT_AFTER_ML: Y                   # ML만 실행

LOG_LVL: 20                          # 로그 레벨 (INFO)
```

### 3. 실행

```bash
# Quant-refactoring 디렉토리에서
python main.py
```

---

## 📋 실행 시나리오

### 시나리오 1: 기존 데이터로 ML 학습 (가장 빠름) ⭐

```yaml
DATA:
  GET_FMP: N  # 기존 데이터 사용

ML:
  RUN_REGRESSION: Y  # 학습 실행
  USE_NEW_MODELS: N  # 기존 regressor.py 사용
  EXIT_AFTER_ML: Y   # ML만 하고 종료
```

```bash
python main.py
```

**예상 시간**: 데이터 크기에 따라 10분 ~ 1시간

---

### 시나리오 2: 새 데이터 수집 + ML 학습

```yaml
DATA:
  GET_FMP: Y  # FMP에서 데이터 수집
  API_KEY: your_actual_api_key  # ⚠️ 필수!

ML:
  RUN_REGRESSION: Y
  EXIT_AFTER_ML: Y
```

```bash
python main.py
```

**예상 시간**: 3-4시간 (데이터 수집이 오래 걸림)

---

### 시나리오 3: 백테스팅만 실행

```yaml
DATA:
  GET_FMP: N

ML:
  RUN_REGRESSION: N  # ML 스킵
  EXIT_AFTER_ML: N

BACKTEST:
  RUN_BACKTEST: Y  # 백테스트 실행
```

```bash
python main.py
```

**주의**: plan.csv 파일이 필요합니다.

---

## 🔍 실행 확인 및 문제 해결

### 성공적인 실행

```
================================================================================
Quant Trading System - Refactored Version
================================================================================
[2025-10-24][INFO] ✅ Configuration loaded from: config/conf.yaml
[2025-10-24][INFO] Quant Trading System - Refactored
[2025-10-24][INFO] Data period: 2015 - 2023
[2025-10-24][INFO] Root path: /home/user/Quant/data
[2025-10-24][INFO] ✅ Configuration validated
...
```

### 흔한 에러와 해결

#### 1. Config file not found

```
❌ Config file not found. Please create config/conf.yaml
```

**해결**:
```bash
cp config/conf.yaml.template config/conf.yaml
```

#### 2. ROOT_PATH not set

```
❌ ROOT_PATH not set in config
```

**해결**: conf.yaml에서 `DATA.ROOT_PATH` 설정

#### 3. Module not found

```
ModuleNotFoundError: No module named 'xxx'
```

**해결**:
```bash
pip install -r requirements.txt
```

#### 4. Legacy regressor not found

```
⚠️ Legacy regressor not found, using new models only
```

**문제 아님**: regressor.py가 없어도 동작합니다. 새 모델 사용.

---

## 🐛 디버깅

### 로그 확인

```bash
# 실시간 로그
tail -f log.txt

# 에러만 보기
grep ERROR log.txt

# 마지막 100줄
tail -100 log.txt
```

### Verbose 로그

conf.yaml:
```yaml
LOG_LVL: 10  # DEBUG 레벨
```

---

## 💡 다음 단계

1. **기본 실행 성공** → ML 모델 학습 확인
2. **새 모델 시도** → `USE_NEW_MODELS: Y` + `USE_MLFLOW: Y`
3. **하이퍼파라미터 튜닝** → `USE_OPTUNA: Y`
4. **백테스팅** → `RUN_BACKTEST: Y`

## requirements.txt

```
# Core
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0

# Data Processing
pyarrow>=12.0.0
tqdm>=4.65.0

# ML Models
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
scikit-learn>=1.3.0

# Hyperparameter Tuning
optuna>=3.0.0

# Experiment Tracking
mlflow>=2.8.0

# Distributed Processing
ray>=2.0.0

# Time Series
tsfresh>=0.20.0
pmdarima>=2.0.0
statsmodels>=0.14.0

# Utilities
joblib>=1.3.0
requests>=2.31.0
```

## 사용법

### 1. 데이터 수집 (Ray 기반 병렬 처리)

```python
from data_collector.fmp import FMP

# FMP 데이터 수집 (자동으로 Ray workers 생성)
fmp = FMP(config, main_ctx)
fmp.collect()  # Ray로 병렬 데이터 수집

# 최대 8개 workers로 API rate limit 방지
```

### 2. Parquet 저장소 사용

```python
from storage import ParquetStorage

# 초기화 (자동 검증 활성화)
storage = ParquetStorage(
    root_path='/home/user/Quant/data',
    auto_validate=True
)

# 데이터 저장 (자동으로 검증 + 샘플 CSV 생성)
storage.save_parquet(df_price, 'price')
# ✅ Saved: price.parquet (1,234,567 rows, 45.2 MB)
# 📄 Sample saved: price_sample.csv
# ✅ Validation passed

# 데이터 로드 (특정 컬럼만)
df = storage.load_parquet('price', columns=['symbol', 'date', 'close'])

# 전체 검증
results = storage.validate_all_tables()
```

### 3. Parquet 뷰어 CLI

```bash
# 기본 사용 (처음 10개 행)
python tools/parquet_viewer.py data/parquet/price.parquet

# 자세한 정보
python tools/parquet_viewer.py data/parquet/price.parquet -a

# 특정 컬럼만 보기
python tools/parquet_viewer.py data/parquet/price.parquet -c "symbol,date,close"

# 쿼리 필터링
python tools/parquet_viewer.py data/parquet/price.parquet -q "close > 100"

# 랜덤 샘플
python tools/parquet_viewer.py data/parquet/price.parquet -s 50
```

### 4. 모델 학습

```python
from models import XGBoostModel, LightGBMModel, CatBoostModel

# XGBoost
xgb = XGBoostModel(task='classification', config_name='default')
xgb.build_model()
xgb.fit(X_train, y_train, X_val, y_val)
metrics = xgb.evaluate(X_test, y_test)

# CatBoost (신규)
cat = CatBoostModel(task='classification', config_name='default')
cat.build_model()
cat.fit(X_train, y_train, X_val, y_val)
metrics = cat.evaluate(X_test, y_test)

# 특징 중요도
importance = cat.get_feature_importance(top_n=20)
print(importance)
```

### 5. Optuna 하이퍼파라미터 튜닝

```python
from training import OptunaOptimizer
from models import CatBoostModel
from models.config import OPTUNA_SEARCH_SPACE

# Optimizer 초기화
optimizer = OptunaOptimizer(
    model_class=CatBoostModel,
    search_space=OPTUNA_SEARCH_SPACE['catboost'],
    n_trials=100,
    cv_folds=5
)

# 최적화 실행
best_params = optimizer.optimize(X_train, y_train, task='classification')

# 최적 모델 생성
best_model = optimizer.get_best_model(task='classification')
best_model.fit(X_train, y_train)

# 최적화 히스토리 플롯
optimizer.plot_optimization_history('optimization_history.png')
```

### 6. Stacking 앙상블

```python
from models import StackingEnsemble
from models import XGBoostModel, LightGBMModel, CatBoostModel

# Base models 생성
xgb1 = XGBoostModel(task='classification', config_name='default')
xgb1.build_model().fit(X_train, y_train)

lgb1 = LightGBMModel(task='classification')
lgb1.build_model().fit(X_train, y_train)

cat1 = CatBoostModel(task='classification')
cat1.build_model().fit(X_train, y_train)

# Stacking 앙상블
base_models = [
    ('xgb', xgb1.model),
    ('lgb', lgb1.model),
    ('cat', cat1.model)
]

ensemble = StackingEnsemble(
    base_models=base_models,
    task='classification',
    meta_learner='ridge',
    cv=5
)

ensemble.build_ensemble()
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

### 7. MLflow 실험 추적

```python
from training import MLflowTracker

# Tracker 초기화
tracker = MLflowTracker(experiment_name='quant_trading_v2')

# 모델 학습 및 자동 추적
tracker.log_training_run(
    model_name='catboost_default',
    model=model.model,
    model_type='catboost',
    params=model.get_params(),
    train_metrics={'accuracy': 0.85, 'f1': 0.82},
    test_metrics={'accuracy': 0.83, 'f1': 0.80},
    feature_importance=importance_df,
    tags={'version': 'v2.0', 'dataset': '2015-2023'}
)

# 최고 성능 모델 로드
best_model = tracker.load_best_model(metric='test_accuracy', model_type='catboost')

# Run 비교
comparison = tracker.compare_runs(metric='test_accuracy', top_n=10)
print(comparison)
```

## VSCode에서 Parquet 파일 보기

1. VSCode Extension 설치: **Parquet Viewer**
2. `.parquet` 파일 클릭 → 자동으로 테이블 뷰

## 성능 개선

### 저장소 성능
| 항목 | 기존 (CSV) | 개선 (Parquet) | 비율 |
|------|-----------|--------------|------|
| 파일 크기 | 500 MB | 50 MB | 10x |
| 읽기 속도 | 10초 | 1초 | 10x |
| 메모리 | 2 GB | 500 MB | 4x |

### ML 파이프라인 성능
| 항목 | 기존 | 개선 | 비율 |
|------|------|------|------|
| 모델 종류 | 2 (XGBoost, LightGBM) | 3 (+CatBoost) | +50% |
| 하이퍼파라미터 튜닝 | 수동 GridSearch | 자동 Optuna | 10x 빠름 |
| 앙상블 | 단순 평균 | Stacking | +3-5% 성능 |
| 실험 관리 | 수동 | MLflow 자동 | ∞ |

### 데이터 수집 성능
| 항목 | 기존 (multiprocessing) | 개선 (Ray) | 개선사항 |
|------|----------------------|-----------|---------|
| 병렬 처리 | Pool (비효율적 IPC) | Ray (효율적 분산) | 메모리 공유 최적화 |
| API rate limit | cpu_count() workers | 8 workers 제한 | Rate limit 방지 |
| 에러 처리 | 기본 | 향상된 재시도 로직 | 안정성 증가 |

## 최근 업데이트 (2025)

### v2.1 - 프로젝트 구조 개선
- ✅ 루트 디렉토리 정리: 7개 → 2개 파일
- ✅ 모듈별 명확한 분리 (config, storage, models, training, tools)
- ✅ 일관된 패키지 구조 (모든 폴더에 `__init__.py`)
- ✅ Import 경로 최적화

### v2.0 - 멀티프로세싱 최적화
- ✅ parquet.py: 비효율적인 파일 기반 IPC 제거 (30-50% 속도 향상)
- ✅ fmp.py: API rate limit 방지 (worker 수 제한)
- ✅ Ray 기반 데이터 수집 (효율적 분산 처리)

## 마이그레이션 가이드

기존 코드에서 리팩토링 버전으로 마이그레이션:

### 파일 위치 변경
| 기존 | 신규 |
|------|------|
| `g_variables.py` | `config/g_variables.py` |
| `make_mldata.py` | `training/make_mldata.py` |
| `regressor.py` | `training/regressor.py` |
| `parquet.py` | `storage/parquet_converter.py` |
| `rank_processing.py` | `tools/rank_processing.py` |

### Import 변경
```python
# 기존
from g_variables import ratio_col_list
from make_mldata import AIDataMaker
from regressor import Regressor
from parquet import Parquet

# 신규
from config.g_variables import ratio_col_list
from training.make_mldata import AIDataMaker
from training.regressor import Regressor
from storage.parquet_converter import Parquet
```

### 설정 파일
```yaml
# config/conf.yaml
DATA:
  TARGET_API_LIST: src/data_collector/target_api_list.csv  # 경로 변경
  STORAGE_TYPE: PARQUET

ML:
  USE_NEW_MODELS: Y  # 새 모델 사용
  USE_MLFLOW: Y      # MLflow 추적
```

## 라이선스

MIT

## 기여

이슈 및 PR 환영합니다.

## 문의

버그 리포트 및 기능 제안은 GitHub Issues를 이용해 주세요.
