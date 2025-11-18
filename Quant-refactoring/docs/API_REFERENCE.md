# API Reference

> **목적**: 각 컴포넌트의 API 및 사용법 상세 가이드
> **작성일**: 2025-11-17
> **대상**: 개발자

---

## 📖 이 문서는 누구를 위한 것인가?

- ✅ **개발자** - 코드를 수정하거나 확장하려는 분
- ✅ **새로운 모듈을 추가하려는 분** - 기존 API 구조 이해
- ✅ **디버깅이 필요한 분** - 각 컴포넌트의 정확한 사용법 확인

**이 문서를 읽기 전에:**
- [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)에서 전체 아키텍처 먼저 확인하세요

---

## 📋 목차

1. [Configuration (`config/`)](#1-configuration-config)
2. [Models (`models/`)](#2-models-models)
3. [Training Pipeline (`training/`)](#3-training-pipeline-training)
4. [Backtesting (`backtest.py`)](#4-backtesting-backtestpy)
5. [Storage (`storage/`)](#5-storage-storage)
6. [Data Collector (`data_collector/`)](#6-data-collector-data_collector)

---

## 1. Configuration (`config/`)

### 1.1 `context_loader.py`

#### MainContext 클래스

```python
from config.context_loader import load_config, MainContext

# 설정 파일 로드
config = load_config('config/conf.yaml')

# MainContext 생성
ctx = MainContext(config)

# 주요 속성
ctx.root_path              # 데이터 루트 경로
ctx.start_year             # 데이터 시작 연도
ctx.end_year               # 데이터 종료 연도
ctx.fmp_api_key           # FMP API 키
ctx.train_start_year      # 학습 시작 연도
ctx.train_end_year        # 학습 종료 연도
ctx.rebalance_period      # 리밸런싱 기간 (월)
ctx.top_k_num             # Top K 종목 개수
```

**주요 메서드**:
```python
# 로거 설정
ctx.setup_logging(
    log_level=20,          # INFO (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)
    log_file='quant.log',
    max_bytes=10*1024*1024,
    backup_count=5
)

# 경로 생성
ctx.get_data_path('VIEW')                    # /data/VIEW/
ctx.get_model_path()                         # /data/MODELS/
ctx.get_ml_data_path('2023_Q1')             # /data/ml_per_year/rnorm_ml_2023_Q1.parquet
```

### 1.2 `g_variables.py`

전역 변수 및 Feature 리스트 정의

```python
from config.g_variables import (
    ratio_col_list,              # 139개 재무 비율
    meaning_col_list,            # 158개 절대값 지표
    cal_timefeature_col_list,    # 36개 시계열 특성 대상
    sector_map                   # Industry → Sector 매핑
)

# 재무 비율 예시
print(ratio_col_list[:5])
# ['roic', 'roe', 'roa', 'priceToBookRatio', 'priceEarningsRatio']

# Sector 매핑 예시
sector = sector_map.get('Software', 'Technology')
```

### 1.3 `logger.py`

멀티프로세싱 안전 로깅 시스템

```python
from config.logger import setup_logging, get_logger

# 전역 로깅 설정
setup_logging(log_level=20)  # INFO

# 모듈별 로거 생성
logger = get_logger(__name__)

# 로그 작성
logger.info("Processing started")
logger.warning("Low memory")
logger.error("Failed to connect", extra={'host': 'localhost', 'port': 5432})
```

**특징**:
- **QueueHandler**: 멀티프로세싱 환경에서 안전
- **Color Output**: 콘솔 로그 가독성 (DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red)
- **Rotation**: 로그 파일 10MB마다 자동 분할

자세한 사용법은 [LOGGING_GUIDE.md](LOGGING_GUIDE.md)를 참조하세요.

---

## 2. Models (`models/`)

### 2.1 Base Model 구조

모든 모델은 `BaseModel`을 상속받습니다:

```python
from models.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, task='classification', config_name='default'):
        super().__init__(task, config_name)
        self.model = None

    def build_model(self, params=None):
        """모델 생성"""
        # 모델 초기화 로직
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """모델 학습"""
        # 학습 로직
        return self

    def predict(self, X):
        """예측"""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """평가"""
        from sklearn.metrics import accuracy_score, mean_squared_error

        y_pred = self.predict(X_test)
        if self.task == 'classification':
            return {'accuracy': accuracy_score(y_test, y_pred)}
        else:
            return {'rmse': mean_squared_error(y_test, y_pred, squared=False)}
```

### 2.2 XGBoost Model

```python
from models.xgboost_model import XGBoostModel

# Classification
xgb_cls = XGBoostModel(task='classification', config_name='depth_9')
xgb_cls.build_model()
xgb_cls.fit(
    X_train, y_train,
    X_val, y_val,
    early_stopping_rounds=50,
    verbose=100
)
predictions = xgb_cls.predict(X_test)
metrics = xgb_cls.evaluate(X_test, y_test)

# Regression
xgb_reg = XGBoostModel(task='regression', config_name='depth_10')
xgb_reg.build_model()
xgb_reg.fit(X_train, y_train, X_val, y_val)
predictions = xgb_reg.predict(X_test)
```

**주요 파라미터**:
```python
{
    'tree_method': 'gpu_hist',      # GPU 가속
    'n_estimators': 500,            # 트리 개수
    'max_depth': 9,                 # 트리 깊이
    'learning_rate': 0.1,           # 학습률
    'gamma': 0,                     # 최소 loss reduction
    'subsample': 0.8,               # 샘플 샘플링 비율
    'colsample_bytree': 0.8         # Feature 샘플링 비율
}
```

### 2.3 LightGBM Model

```python
from models.lightgbm_model import LightGBMModel

lgb = LightGBMModel(task='classification')
lgb.build_model()
lgb.fit(X_train, y_train, X_val, y_val)
predictions = lgb.predict(X_test)
```

**주요 파라미터**:
```python
{
    'boosting_type': 'gbdt',
    'objective': 'binary',          # classification
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.1,
    'device': 'gpu'
}
```

### 2.4 CatBoost Model

```python
from models.catboost_model import CatBoostModel

cat = CatBoostModel(task='classification')
cat.build_model()
cat.fit(
    X_train, y_train,
    cat_features=['sector', 'industry'],  # Categorical features
    verbose=100
)
predictions = cat.predict(X_test)

# Feature Importance
importance = cat.get_feature_importance(top_n=20)
```

### 2.5 Ensemble

```python
from models.ensemble import StackingEnsemble

# Base models
base_models = [
    ('xgb_8', xgb_model_8.model),
    ('xgb_9', xgb_model_9.model),
    ('xgb_10', xgb_model_10.model),
    ('lgb', lgb_model.model)
]

# Stacking Ensemble
ensemble = StackingEnsemble(
    base_models=base_models,
    task='classification',
    meta_learner='ridge',  # or 'logistic', 'xgboost'
    cv=5
)
ensemble.build_ensemble()
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
probas = ensemble.predict_proba(X_test)
```

---

## 3. Training Pipeline (`training/`)

### 3.1 Hyperparameter Optimization

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

# Optimizer 초기화
optimizer = OptunaOptimizer(
    model_class=XGBoostModel,
    search_space=search_space,
    n_trials=100,
    cv_folds=5,
    scoring='accuracy',      # or 'roc_auc', 'f1', 'rmse'
    n_jobs=-1                # 병렬 실행
)

# 최적화 실행
best_params = optimizer.optimize(X_train, y_train, task='classification')
print(f"Best params: {best_params}")

# 최적 모델 생성
best_model = optimizer.get_best_model(task='classification')
best_model.fit(X_train, y_train)

# 최적화 히스토리 플롯
optimizer.plot_optimization_history('optimization_history.png')
optimizer.plot_param_importances('param_importances.png')
```

### 3.2 MLflow Tracking

```python
from training.mlflow_tracker import MLflowTracker

# Tracker 초기화
tracker = MLflowTracker(
    experiment_name='quant_trading',
    tracking_uri='/home/user/Quant/data/mlruns'
)

# Run 시작
with tracker.start_run(run_name='xgb_depth9_v2'):
    # 파라미터 로깅
    tracker.log_params({
        'max_depth': 9,
        'learning_rate': 0.1,
        'n_estimators': 500
    })

    # 모델 학습
    model.fit(X_train, y_train)

    # 메트릭 로깅
    metrics = model.evaluate(X_test, y_test)
    tracker.log_metrics(metrics)

    # 모델 저장
    tracker.log_model(model, 'xgboost_model')

    # Feature Importance 저장 (optional)
    importance_df = model.get_feature_importance()
    tracker.log_artifact(importance_df, 'feature_importance.csv')

# 최고 성능 모델 로드
best_model = tracker.load_best_model(
    metric='accuracy',
    model_type='xgboost'
)

# Run 비교
comparison = tracker.compare_runs(
    metric='accuracy',
    top_n=10,
    order='desc'
)
print(comparison)
```

**MLflow UI 실행**:
```bash
mlflow ui --backend-store-uri /home/user/Quant/data/mlruns
# http://localhost:5000
```

### 3.3 ML Data Maker

```python
from training.make_mldata import AIDataMaker

# 초기화
aidata_maker = AIDataMaker(ctx, config)

# 데이터 로드
aidata_maker.load_bt_table()

# 리밸런싱 날짜 설정
aidata_maker.set_date()

# ML 데이터 생성
aidata_maker.make_ml_data()

# 생성된 파일: /data/ml_per_year/rnorm_ml_{year}_{quarter}.parquet
```

### 3.4 Regressor (Legacy)

```python
from training.regressor import Regressor

# 초기화
regressor = Regressor(config)

# 데이터 로드
regressor.dataload()

# 학습
regressor.train()

# 평가
regressor.evaluation()

# 최신 예측
regressor.latest_prediction()
# → /data/MODELS/latest_prediction.csv
```

---

## 4. Backtesting (`backtest.py`)

### 4.1 Plan Handler

```python
from backtest import PlanHandler, Backtest

# Plan 로드
plan_handler = PlanHandler(ctx, 'plan.csv')

# Plan 예시 (plan.csv)
"""
key,key_dir,weight,diff,base,base_dir
roe,descending,1.0,False,,
priceEarningsRatio,ascending,0.8,False,,
debtToEquity,ascending,0.5,False,,
price_dev,descending,2.0,True,,
"""
```

### 4.2 Backtest 실행

```python
# Backtest 초기화
bt = Backtest(ctx, config, plan_handler)

# 실행
bt.run()

# 리포트 생성
bt.generate_reports(['EVAL', 'RANK', 'AVG'])

# 리포트 저장 위치
# - EVAL: 각 리밸런싱 기간별 수익률, Sharpe Ratio, Max Drawdown
# - RANK: 선정 종목 리스트 및 스코어
# - AVG: 전체 기간 평균 통계
```

---

## 5. Storage (`storage/`)

### 5.1 Parquet Storage

```python
from storage.parquet_storage import ParquetStorage

# 초기화
storage = ParquetStorage(
    root_path='/home/user/Quant/data',
    auto_validate=True
)

# 데이터 저장
storage.save_parquet(df_price, 'price')
# ✅ Saved: price.parquet (1,234,567 rows, 45.2 MB)
# 📄 Sample saved: price_sample.csv
# ✅ Validation passed

# 데이터 로드
df = storage.load_parquet('price', columns=['symbol', 'date', 'close'])

# 전체 검증
results = storage.validate_all_tables()
```

### 5.2 Parquet Converter

```python
from storage.parquet_converter import ParquetConverter

converter = ParquetConverter(ctx, storage)

# CSV → Parquet 변환
converter.insert_csv()

# VIEW 테이블 재구성
converter.rebuild_table_view()
```

---

## 6. Data Collector (`data_collector/`)

### 6.1 FMP API

```python
from data_collector.fmp import FMP

# 초기화
fmp = FMP(ctx)

# 데이터 수집
fmp.collect()

# 수집되는 데이터
# - stock_list
# - delisted_companies
# - income_statement
# - balance_sheet_statement
# - cash_flow_statement
# - key_metrics
# - financial_growth
# - historical_price_full
```

---

## 📝 사용 예제

### Example 1: 새 모델 추가

```python
from models.base_model import BaseModel
import some_ml_library

class MyCustomModel(BaseModel):
    def __init__(self, task='classification'):
        super().__init__(task)
        self.model = None

    def build_model(self, params=None):
        self.model = some_ml_library.Model(**params)
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X_test)
        return {'accuracy': accuracy_score(y_test, y_pred)}

# 사용
model = MyCustomModel()
model.build_model({'param1': 10})
model.fit(X_train, y_train)
```

### Example 2: 전체 파이프라인 커스터마이징

```python
from config.context_loader import load_config, MainContext
from data_collector.fmp import FMP
from training.make_mldata import AIDataMaker
from training.regressor import Regressor
from backtest import Backtest, PlanHandler

# 1. Config
config = load_config('config/conf.yaml')
ctx = MainContext(config)

# 2. 데이터 수집 (필요 시)
if config['DATA']['GET_FMP'] == 'Y':
    fmp = FMP(ctx)
    fmp.collect()

# 3. ML 데이터 준비
aidata = AIDataMaker(ctx, config)
aidata.make_ml_data()

# 4. 모델 학습
regressor = Regressor(config)
regressor.dataload()
regressor.train()
regressor.evaluation()

# 5. 백테스팅
plan = PlanHandler(ctx, 'plan.csv')
bt = Backtest(ctx, config, plan)
bt.run()
```

---

## 🔗 관련 문서

- **전체 아키텍처** → [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)
- **빠른 시작** → [QUICK_START.md](QUICK_START.md)
- **로깅 가이드** → [LOGGING_GUIDE.md](LOGGING_GUIDE.md)
- **개선 로드맵** → [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md)

---

**문서 버전**: 1.0
**마지막 업데이트**: 2025-11-17
**작성자**: Claude AI + Development Team
