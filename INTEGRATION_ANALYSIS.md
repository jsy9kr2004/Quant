# Quant System Integration Analysis

## 📊 기존 시스템 분석 (Legacy)

### 코드 규모
- **총 라인 수**: 3,841 lines
- **주요 모듈**: 8개

### 파일별 라인 수
| 파일 | 라인 수 | 역할 |
|------|---------|------|
| backtest.py | 1,024 | 백테스팅 엔진 |
| regressor.py | 769 | ML 모델 학습 |
| fmp.py | 526 | FMP 데이터 수집 |
| g_variables.py | 449 | 전역 변수 |
| make_mldata.py | 411 | ML 데이터 생성 |
| parquet.py | 237 | Parquet 저장 (CSV 실제 사용) |
| database.py | 175 | MariaDB 연동 |
| main.py | 149 | 메인 파이프라인 |
| rank_processing.py | 101 | 순위 처리 |

---

## 🔄 기존 시스템 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN PIPELINE (main.py)                  │
└─────────────────────────────────────────────────────────────┘

1. Configuration Loading
   ├─ get_config() → conf.yaml 로드
   └─ MainCtx(conf) → 컨텍스트 초기화

2. Data Collection (선택적, GET_FMP=Y)
   ├─ FMP(conf, main_ctx)
   ├─ fmp.get_new() → FMP API에서 데이터 수집
   └─ 저장:
       ├─ STORAGE_TYPE=DB → Database.insert_csv() + rebuild_table_view()
       └─ STORAGE_TYPE=PARQUET → Parquet.insert_csv() + rebuild_table_view()

3. ML Data Preparation (RUN_REGRESSION=Y)
   ├─ AIDataMaker(main_ctx, conf)
   │   ├─ load_bt_table() → VIEW에서 데이터 로드
   │   ├─ set_date() → 리밸런싱 날짜 생성
   │   ├─ process_price_table_wdate() → 가격 변동 계산
   │   └─ make_ml_data() → tsfresh 시계열 피처 추출
   │       └─ 저장: ROOT_PATH/ml_per_year/rnorm_ml_{year}_{Q}.parquet (Parquet 형식, 5-10배 빠름 🚀)

4. Model Training
   ├─ Regressor(conf)
   ├─ regressor.dataload()
   │   ├─ train_files 로드 (2015-2021)
   │   ├─ test_files 로드 (2022-2023)
   │   ├─ 의미없는 컬럼 제거 (missing > 80%, same_value > 95%)
   │   ├─ 행 필터링 (60% 이상 NaN인 행 제거)
   │   └─ sector 매핑
   ├─ regressor.train()
   │   ├─ XGBoost Classifier × 3 (depth 8, 9, 10)
   │   ├─ LightGBM Classifier × 1
   │   ├─ XGBoost Regressor × 2 (depth 8, 10)
   │   └─ 저장: ROOT_PATH/MODELS/*.sav
   ├─ regressor.evaluation()
   │   ├─ 각 test 파일별 예측
   │   ├─ 분류기 앙상블 (3개 모델 조합)
   │   ├─ 회귀 예측 (binary 필터링 적용)
   │   └─ 저장: MODELS/prediction_ai_{date}.csv
   └─ regressor.latest_prediction()
       ├─ 2024 최신 데이터로 예측
       └─ 저장: MODELS/latest_prediction.csv

5. Backtesting (RUN_REGRESSION=N)
   ├─ PlanHandler → 종목 점수화 전략
   ├─ Backtest(main_ctx, conf, plan_handler)
   │   ├─ load_bt_table() → 데이터 로드
   │   ├─ run() → 백테스팅 실행
   │   │   ├─ DateHandler → 날짜별 데이터 스냅샷
   │   │   ├─ PlanHandler → 점수 계산 및 상위 K개 선정
   │   │   └─ EvaluationHandler → 수익률 계산
   │   └─ 리포트 생성:
   │       ├─ EVAL_REPORT_*.csv
   │       ├─ RANK_REPORT_*.csv
   │       └─ AVG_REPORT_*.csv
   └─ exit()
```

---

## 🆚 기존 vs 리팩토링 비교

### 1. Storage Layer

| 항목 | 기존 (Legacy) | 리팩토링 |
|------|---------------|----------|
| **구현** | `parquet.py` (237 lines) | `storage/parquet_storage.py` (200 lines) |
| **실제 포맷** | CSV (Parquet 주석처리) | 진짜 Parquet |
| **검증** | 없음 | ✅ 자동 검증 (DataValidator) |
| **샘플 생성** | 없음 | ✅ 자동 샘플 CSV |
| **압축** | 없음 | ✅ snappy/gzip/zstd |
| **VIEW 생성** | rebuild_table_view() | 동일 패턴 사용 가능 |

**데이터 플로우:**
```
기존: FMP → CSV → CSV 읽기 → Pandas → VIEW CSV 생성
리팩: FMP → CSV → Parquet 변환 → Parquet 읽기 → VIEW Parquet 생성
```

### 2. ML Models Layer

| 항목 | 기존 (Legacy) | 리팩토링 |
|------|---------------|----------|
| **구현** | `regressor.py` (769 lines) | `models/*.py` (500 lines) |
| **모델** | XGBoost×3, LightGBM×1 | +CatBoost×2 ✅ |
| **구조** | 단일 클래스 (Regressor) | 모듈화 (BaseModel 상속) |
| **하이퍼파라미터** | 수동 GridSearch (주석) | ✅ Optuna 자동 |
| **앙상블** | 단순 평균 (np.average) | ✅ Stacking |
| **저장** | joblib (.sav) | joblib + MLflow |
| **추적** | 수동 로깅 | ✅ MLflow 자동 |

**모델 학습 플로우:**
```
기존:
  dataload() → clean → sector mapping → fit() → evaluate() → save .sav

리팩:
  ParquetStorage.load() → Model.build() → Model.fit() →
  OptunaOptimizer (선택) → StackingEnsemble (선택) →
  MLflowTracker.log() → Model.save()
```

### 3. Feature Engineering Layer

| 항목 | 기존 (Legacy) | 리팩토링 |
|------|---------------|----------|
| **구현** | `make_mldata.py` (411 lines) | 아직 없음 ❌ |
| **tsfresh** | ✅ 구현됨 | 그대로 사용 |
| **정규화** | RobustScaler | 설정 가능 |
| **섹터 매핑** | ✅ g_variables.py | 그대로 사용 |

**통합 필요**: 기존 make_mldata.py를 그대로 사용하되, Parquet 읽기로 변경

### 4. Backtesting Layer

| 항목 | 기존 (Legacy) | 리팩토링 |
|------|---------------|----------|
| **구현** | `backtest.py` (1,024 lines) | 아직 없음 ❌ |
| **PlanHandler** | ✅ 구현됨 | 그대로 사용 |
| **DateHandler** | ✅ 구현됨 | 그대로 사용 |
| **EvaluationHandler** | ✅ 구현됨 | 그대로 사용 |

**통합 필요**: 기존 backtest.py를 그대로 사용하되, Parquet 읽기로 변경

---

## 🔗 통합 포인트 (Integration Points)

### ✅ 즉시 통합 가능 (최소 변경)

**1. Parquet Storage 교체**
```python
# 기존 parquet.py (L61, L82, L114, L143)
# all_symbol.to_csv(self.view_path + "symbol_list.csv", index=False)

# 리팩토링
from storage import ParquetStorage
storage = ParquetStorage(root_path)
storage.save_parquet(all_symbol, 'symbol_list')
```

**2. CatBoost 모델 추가**
```python
# 기존 regressor.py의 def_model()에 추가
from models import CatBoostModel

cat = CatBoostModel(task='classification')
cat.build_model()
self.clsmodels[4] = cat.model  # 기존 구조에 추가
```

### ⚠️ 부분 수정 필요

**3. 데이터 로딩 변경**
```python
# 기존 make_mldata.py (L66-92)
self.symbol_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/symbol_list.csv")

# 리팩토링
storage = ParquetStorage(root_path)
self.symbol_table = storage.load_parquet('symbol_list')
```

**4. 모델 학습 래핑**
```python
# 기존 regressor.py의 train() 유지하되, MLflow 추가
from training import MLflowTracker

tracker = MLflowTracker('quant_trading')
# 학습 코드...
tracker.log_training_run(model_name='xgb_0', model=model, ...)
```

### 🔨 대규모 리팩토링 필요

**5. Regressor 클래스 재구성**
- 현재: 단일 769줄 클래스
- 목표: 모듈화된 구조
- 작업량: 3-4시간

**6. Optuna 통합**
- 현재: GridSearch 주석처리 (L254-299)
- 목표: Optuna로 자동화
- 작업량: 1시간

**7. Stacking Ensemble**
- 현재: 단순 평균 (L475)
- 목표: Stacking
- 작업량: 30분

---

## 📋 통합 전략 (3가지 옵션)

### 옵션 A: 점진적 통합 (추천) ⭐

**장점**: 안전, 빠름, 기존 코드 유지
**단점**: 코드 중복 존재
**시간**: 1-2시간

**순서**:
1. ✅ Parquet 진짜 사용 (30분)
   - `parquet.py`에서 주석 해제
   - `.to_parquet()` 사용
   - `.read_parquet()` 사용

2. ✅ CatBoost 추가 (30분)
   - `regressor.py`에 CatBoost 모델 추가
   - `def_model()`에 1개 추가

3. ✅ 검증 기능 추가 (30분)
   - 리팩토링의 DataValidator 복사
   - Parquet 저장 후 자동 검증

**변경 파일**: 2개 (`parquet.py`, `regressor.py`)

### 옵션 B: 하이브리드 (균형) ⚖️

**장점**: 새 기능 대부분 활용
**단점**: 테스트 필요
**시간**: 3-4시간

**순서**:
1. 옵션 A 완료
2. MLflow 추적 추가 (1시간)
3. Optuna 튜닝 추가 (1시간)
4. 기존 코드에서 새 모듈 import

**변경 파일**: 4개 (+ `main.py`, `regressor.py` 수정)

### 옵션 C: 완전 마이그레이션 🚀

**장점**: 깨끗한 코드, 모든 신기능
**단점**: 리스크, 시간 소요
**시간**: 6-8시간

**순서**:
1. 새 `main.py` 작성
2. `regressor.py` → `models/*.py` 완전 전환
3. `make_mldata.py` Parquet 버전
4. `backtest.py` Parquet 버전
5. 전체 통합 테스트

**변경 파일**: 전체

---

## 💡 제 추천: 옵션 A (점진적 통합)

### 단계별 구체적 계획

**Phase 1: Parquet 진짜 사용 (30분)**

`parquet.py` 수정:
```python
# Line 61: CSV → Parquet
# all_symbol.to_csv(self.view_path + "symbol_list.csv", index=False)
all_symbol.to_parquet(
    self.view_path + "symbol_list.parquet",
    engine='pyarrow',
    compression='snappy',
    index=False
)

# Line 66-92: 다른 테이블도 동일하게
# 총 4개 위치 변경 (symbol_list, price, financial_statement, metrics)
```

**Phase 2: CatBoost 추가 (30분)**

`regressor.py` 수정:
```python
# Line 231 이후 추가
from catboost import CatBoostClassifier, CatBoostRegressor

def def_model(self):
    # 기존 코드...

    # CatBoost 추가
    self.clsmodels[4] = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=8,
        task_type='GPU',
        loss_function='Logloss',
        eval_metric='AUC'
    )
```

**Phase 3: 검증 추가 (30분)**

`parquet.py`에 검증 추가:
```python
from storage.data_validator import DataValidator

def rebuild_table_view(self):
    # 기존 저장 코드...

    # 검증 추가
    validator = DataValidator()
    result = validator.validate_file(
        str(self.view_path + "symbol_list.parquet"),
        'symbol_list'
    )
    if not result['passed']:
        logging.error("Validation failed!")
```

---

## 🎯 다음 단계

어떤 옵션으로 진행하시겠습니까?

1. **옵션 A**: 지금 바로 30분만에 통합 (Parquet + CatBoost)
2. **옵션 B**: 3-4시간 투자해서 대부분 기능 활용
3. **옵션 C**: 완전 새로 시작 (6-8시간)

또는:
4. **더 분석**: 특정 부분 더 자세히 보기

선택해주시면 바로 시작하겠습니다!
