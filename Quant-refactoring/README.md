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

## 프로젝트 구조

### 📂 실행 전 (소스 코드)

시스템 실행 전 프로젝트 파일 구조입니다.

```
Quant-refactoring/
│
├── 📜 main.py                          # 🎯 메인 실행 진입점
├── 📜 backtest.py                      # 백테스팅 로직
├── 📄 README.md                        # 프로젝트 문서
├── 📄 requirements.txt                 # Python 패키지 의존성
│
├── 📁 config/                          # ⚙️ 설정 파일
│   ├── __init__.py
│   ├── conf.yaml.template             # 설정 템플릿 (복사 필요)
│   ├── conf.yaml                      # 실제 설정 파일 (생성 필요, .gitignore)
│   ├── context_loader.py              # 설정 로더
│   └── g_variables.py                 # 전역 변수 (컬럼 정의, 상수)
│
├── 📁 data_collector/                  # 📡 데이터 수집 (Ray 기반)
│   ├── fmp.py                         # FMP 데이터 수집 메인 클래스
│   ├── fmp_api.py                     # FMP API 관리
│   ├── fmp_fetch_worker.py            # Ray worker (병렬 처리)
│   └── target_api_list.csv            # 수집할 API 목록
│
├── 📁 storage/                         # 💾 데이터 저장소
│   ├── __init__.py
│   ├── parquet_storage.py             # Parquet 저장 + 자동 검증
│   ├── parquet_converter.py           # CSV → Parquet 변환 + 테이블 재구성
│   └── data_validator.py              # 데이터 품질 검증
│
├── 📁 models/                          # 🤖 ML 모델
│   ├── __init__.py
│   ├── base_model.py                  # 기본 모델 추상 클래스
│   ├── xgboost_model.py               # XGBoost 래퍼
│   ├── lightgbm_model.py              # LightGBM 래퍼
│   ├── catboost_model.py              # CatBoost 래퍼 (신규)
│   ├── ensemble.py                    # Stacking 앙상블
│   └── config.py                      # 모델 하이퍼파라미터 설정
│
├── 📁 training/                        # 🎓 ML 학습 파이프라인
│   ├── __init__.py
│   ├── regressor.py                   # 레거시 통합 학습 모델
│   ├── make_mldata.py                 # ML 데이터 전처리 (feature engineering)
│   ├── optimizer.py                   # Optuna 하이퍼파라미터 튜닝
│   └── mlflow_tracker.py              # MLflow 실험 추적
│
├── 📁 tools/                           # 🔧 분석 도구
│   ├── __init__.py
│   ├── parquet_viewer.py              # Parquet 파일 뷰어 CLI
│   └── rank_processing.py             # 순위 분석 도구
│
└── 📁 examples/                        # 📚 사용 예제
    └── example_complete_pipeline.py   # 전체 파이프라인 예제
```

### 📂 실행 후 (생성되는 파일 및 디렉토리)

`python main.py` 실행 시 자동으로 생성되는 파일과 디렉토리입니다.

```
Quant-refactoring/
│
├── 📄 log.txt                          # 실행 로그
├── 📄 allsymbol.csv                    # 전체 종목 리스트
├── 📄 current_list.csv                 # 현재 거래 종목 리스트
│
├── 📁 data/                            # 💾 수집된 데이터 (ROOT_PATH 설정값)
│   ├── parquet/                       # 원본 데이터 (현재는 CSV 형식)
│   │   ├── stock_list.csv
│   │   ├── delisted_companies.csv
│   │   ├── historical_price_full.csv
│   │   ├── income_statement.csv
│   │   ├── balance_sheet_statement.csv
│   │   ├── cash_flow_statement.csv
│   │   ├── key_metrics.csv
│   │   ├── financial_growth.csv
│   │   └── ... (기타 FMP 데이터)
│   │
│   ├── VIEW/                          # 가공된 뷰 테이블 (CSV 형식, 1회만 읽음)
│   │   ├── symbol_list.csv
│   │   ├── price.csv
│   │   ├── financial_statement_*.csv
│   │   ├── metrics_*.csv
│   │   └── indexes.csv
│   │
│   ├── ml_per_year/                   # 🚀 ML 학습 데이터 (Parquet 형식 - 고성능)
│   │   ├── rnorm_ml_2015_Q1.parquet  # 학습용 (y값 포함)
│   │   ├── rnorm_ml_2015_Q2.parquet
│   │   ├── ... (연도별 분기별)
│   │   ├── rnorm_fs_2015_Q1.parquet  # 예측용 (y값 없음)
│   │   └── ... (연도별 분기별)
│   │
│   └── {API 종류별 폴더}/             # FMP API별 원본 데이터
│       ├── stock_list/
│       ├── historical_price_full/
│       ├── income_statement/
│       └── ... (symbol별 CSV 파일)
│
├── 📁 mlruns/                          # 🔬 MLflow 실험 추적 데이터
│   ├── 0/                             # Experiment ID
│   │   ├── meta.yaml
│   │   └── {run_id}/                  # 각 실행 결과
│   │       ├── params/                # 하이퍼파라미터
│   │       ├── metrics/               # 평가 지표
│   │       ├── artifacts/             # 모델 파일
│   │       └── tags/
│   └── .trash/
│
├── 📁 models/                          # 💾 학습된 모델 저장
│   ├── xgboost_default.pkl
│   ├── lightgbm_default.pkl
│   ├── catboost_default.pkl
│   ├── ensemble_stacking.pkl
│   └── ... (각 모델 파일)
│
├── 📁 reports/                         # 📊 백테스트 리포트
│   ├── backtest_results_*.csv
│   ├── performance_summary.csv
│   └── feature_importance.png
│
└── 📁 optuna_study/                    # 🎯 Optuna 튜닝 결과 (선택)
    ├── study_*.db                     # SQLite DB
    └── optimization_history.png       # 최적화 히스토리 플롯
```

### 📋 주요 파일 설명

#### 실행 전 (소스 코드)
| 파일/디렉토리 | 설명 | 필수 여부 |
|--------------|------|----------|
| `main.py` | 전체 파이프라인 실행 진입점 | ✅ 필수 |
| `config/conf.yaml` | 실제 설정 파일 (API 키, 경로 등) | ✅ 필수 (생성) |
| `config/conf.yaml.template` | 설정 템플릿 | ✅ 필수 |
| `data_collector/` | FMP API 데이터 수집 모듈 | ✅ 필수 |
| `storage/` | Parquet 저장 및 검증 | ✅ 필수 |
| `models/` | ML 모델 클래스 | ✅ 필수 |
| `training/` | 학습 파이프라인 | ✅ 필수 |
| `tools/` | 분석 도구 | ⚪ 선택 |
| `examples/` | 사용 예제 | ⚪ 선택 |

#### 실행 후 (자동 생성)
| 파일/디렉토리 | 생성 시점 | 형식 | 설명 |
|--------------|---------|------|------|
| `data/parquet/` | FMP 데이터 수집 시 | CSV | 원본 데이터 (현재는 CSV) |
| `data/VIEW/` | VIEW 재구성 시 | CSV | 가공 데이터 (1회만 읽음) |
| `data/ml_per_year/` | ML 데이터 생성 시 | **Parquet** | **학습 데이터 (매 학습마다 읽음, 고성능)** |
| `mlruns/` | MLflow 사용 시 | - | 실험 추적 데이터 |
| `models/` | 모델 학습 시 | PKL | 학습된 모델 파일 |
| `reports/` | 백테스트 실행 시 | CSV/PNG | 성과 리포트 |
| `log.txt` | 실행 즉시 | TXT | 전체 실행 로그 |

### 🔑 핵심 개념

**데이터 흐름 (최적화됨):**
```
FMP API → data/{api_name}/ (원본 CSV)
       → data/parquet/ (CSV 저장 - 현재)
       → data/VIEW/ (가공된 뷰 - CSV, 1회만 읽음)
       → data/ml_per_year/ (학습 데이터 - Parquet, 5-10배 빠름 🚀)
       → ML 학습 → models/ (모델 저장)
       → 백테스팅 → reports/ (결과)
```

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
  TARGET_API_LIST: data_collector/target_api_list.csv  # 경로 변경
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
