# 리팩토링 분석 보고서

> **작성일**: 2026-02-07
> **이전 문서**: [08_recent_changes.md](./08_recent_changes.md)
> **관련 문서**: [05_code_quality.md](./05_code_quality.md), [07_recommendations.md](./07_recommendations.md)

---

## 핵심 요약

### 전체 코드베이스 평가: B (Good, 구조적 리팩토링 필요)

| 영역 | 등급 | 상태 |
|------|------|------|
| 아키텍처 | B+ | 관심사 분리 우수, 일부 God Class 존재 |
| PEP8 준수 | B- | 라인 길이 초과, bare except, import 정리 필요 |
| Docstring | B+ | 대부분 양호, backtest.py 등 일부 미흡 |
| 가독성 | C+ | 초장 메서드 다수, 매직 넘버, 중복 코드 |
| 테스트 | B- | 8개 테스트 파일 존재, 커버리지 불명확 |
| 에러 처리 | B | 대체로 양호, bare except 수정 필요 |

**총 파일 수**: 64개 (활성 소스), **총 코드량**: ~44,500줄

---

## 파일별 상세 분석

### 등급 기준

| 등급 | 의미 | 기준 |
|------|------|------|
| A | 우수 | PEP8 준수, docstring 완비, 가독성 우수, 리팩토링 불필요 |
| B | 양호 | 경미한 이슈, 선택적 리팩토링 |
| C | 보통 | 다수 이슈, 리팩토링 권장 |
| D | 미흡 | 심각한 이슈, 리팩토링 필수 |
| F | 불량 | 전면 재작성 필요 |

---

### 1. `src/training/` (핵심 학습 모듈)

#### 1.1 `regressor.py` — 4,962줄 | 등급: C

**가장 큰 파일이자 가장 시급한 리팩토링 대상**

| 항목 | 평가 | 상세 |
|------|------|------|
| PEP8 | C | 라인 길이 초과 다수, 주석 처리된 코드 블록 잔존 |
| Docstring | B+ | 모듈/클래스/메서드 docstring 충실 (한국어) |
| 가독성 | C- | 5,000줄 단일 파일, God Class, 초장 메서드 |
| 에러 처리 | B | try-except 패턴 적절, GPU fallback 처리 양호 |

**리팩토링 필요 여부**: ✅ **필수 (P0)**

**핵심 이슈**:

1. **God Class — `Regressor` 클래스 (~4,300줄)**
   - 데이터 로딩, 전처리, 모델 학습, 평가, 예측, Walk-Forward, Ray 병렬 처리까지 모든 기능 포함
   - 권장: 최소 3개 클래스로 분리
     ```
     Regressor (현재)
       ├── DataLoader         — 데이터 로딩/전처리 담당
       ├── ModelTrainer       — 모델 학습/저장 담당
       ├── ModelEvaluator     — 평가/예측 담당
       └── WalkForwardRunner  — Walk-Forward 실행 담당
     ```

2. **초장 메서드**
   - `train()`: ~800줄 — Optuna, 모델 학습, 섹터별 학습 혼재
   - `dataload()`: ~400줄 — 5단계 처리를 단일 메서드에서 수행
   - `evaluation()`: ~500줄+ (offset 3400 이후)
   - `_train_walk_forward_sequential()`: ~200줄

3. **주석 처리된 코드 (Dead Code)**
   - 줄 2148-2182: GridSearchCV, RandomizedSearchCV 주석 코드 (~35줄)
   - 줄 2767-2773: 특성 중요도 분석 주석 코드
   - 권장: 완전 삭제 (Git 히스토리에서 복구 가능)

4. **`print()` 사용**
   - 줄 709, 748, 750, 773, 774: `print()` 직접 사용
   - 권장: `logging.info()` 또는 `logging.debug()`로 통일

5. **전역 변수**
   - 줄 107-111: `MODEL_SAVE_PATH`, `THRESHOLD`, `TARGET_FEATURES` 모듈 레벨 전역 변수
   - 권장: 인스턴스 변수 또는 config로 이동

6. **미사용 import 의심**
   - `torch.nn`, `torch.nn.functional`, `torch.optim`: PyTorch NN 관련 — regressor에서 사용 안 함
   - `RandomForestRegressor`, `LinearRegression`, `MLPRegressor`: sklearn 모델 — 현재 사용 안 함
   - `seaborn`: 시각화 — 현재 코드에서 호출 불확실

7. **코드 내 TODO**
   ```
   줄 55-58: 섹터 매핑 제거, GridSearchCV 코드 마이그레이션
   줄 269: y_col_list → DataSchema 마이그레이션
   ```

---

#### 1.2 `data_processor.py` — 2,823줄 | 등급: B+

| 항목 | 평가 | 상세 |
|------|------|------|
| PEP8 | B | 일부 라인 100자 초과 |
| Docstring | A- | 95% 커버리지, Parameters/Returns/Examples 형식 우수 |
| 가독성 | B- | 장 메서드 존재하나 섹션 구분자 활용 양호 |
| 에러 처리 | A | try-except 블록 우수, 상세 로깅 |

**리팩토링 필요 여부**: ⚠️ 권장 (P1)

**핵심 이슈**:

1. **초장 메서드**
   - `preprocess_training_data()`: 366줄 — 8단계 전처리를 단일 메서드에서 수행
     ```python
     # 권장: 단계별 분리
     preprocess_training_data()
       ├── _validate_input_data()
       ├── _remove_infinite_values()
       ├── _apply_log_transform()
       ├── _remove_sparse_columns()
       ├── _remove_sparse_rows()
       ├── _apply_winsorization()
       ├── _apply_feature_selection()
       └── _finalize_output()
     ```
   - `full_pipeline()`: 224줄
   - `log_transform_features()`: 123줄

2. **매직 넘버**
   - `0.5`: NaN 임계값 (줄 1241)
   - `1000`: 목표 feature 수 (줄 1317) — 상수로 정의 필요
   - `1e10`: 클리핑 값 (줄 897) — 문서화됨, 양호

---

#### 1.3 `make_mldata.py` — 1,983줄 | 등급: C+

| 항목 | 평가 | 상세 |
|------|------|------|
| PEP8 | B- | 긴 import 문, 일부 라인 120자+ |
| Docstring | B+ | 98% 커버리지, 2개 메서드 누락 |
| 가독성 | C | 807줄 메서드 존재 (**가장 긴 단일 메서드**) |
| 에러 처리 | B- | 일부 섹션 try-except 부재 |

**리팩토링 필요 여부**: ✅ **필수 (P0)**

**핵심 이슈**:

1. **`make_ml_data()` 메서드 — 807줄** ⚠️ **코드베이스 최대 단일 메서드**
   ```python
   # 현재 (문제)
   def make_ml_data(self):
       # 807줄의 혼재된 관심사

   # 권장 (분리)
   def make_ml_data(self):
       self._load_and_validate_data()        # Phase 1: 데이터 로드
       self._extract_time_series_features()   # Phase 2: tsfresh 특성 추출
       self._calculate_financial_ratios()     # Phase 3: 재무 비율 계산
       self._apply_winsorization()            # Phase 4: 극단값 처리
       self._export_results()                 # Phase 5: 결과 저장
   ```

2. **누락된 Docstring**
   - `assign_time()`: docstring 없음
   - `get_last_12_rows()`: docstring 없음

3. **매직 넘버**
   - `12`: lookback 기간 (상수 `LOOKBACK_MONTHS`로 정의 필요)
   - `0.25, 0.6, 0.9`: 롤백 윈도우 파라미터
   - `100`: 카운트 기준 (문맥 부재)

4. **코드 내 TODO**
   ```
   줄 211: TODO 마커 존재
   줄 567: TODO 마커 존재
   ```

---

#### 1.4 기타 training 모듈

| 파일 | 줄수 | 등급 | 리팩토링 필요 | 핵심 이슈 |
|------|------|------|-------------|-----------|
| `data_quality.py` | 679 | A- | ❌ 불필요 | 매직 넘버 경미 |
| `feature_selector.py` | 368 | A | ❌ 불필요 | 우수한 구조 |
| `optuna_utils.py` | 379 | B | ⚠️ 경미 | **bare `except:` (줄 50)** |
| `walk_forward.py` | 304 | A | ❌ 불필요 | 우수한 구조 |
| `mlflow_tracker.py` | 254 | B | ⚠️ 경미 | **bare `except:` (줄 50)** |
| `time_series_cv.py` | 241 | A- | ❌ 불필요 | 라인 길이 경미 |
| `optimizer.py` | 214 | A | ❌ 불필요 | 우수한 코드 |

---

### 2. `src/backtest/` (백테스트 모듈)

#### 2.1 `ml_backtest.py` — 2,003줄 | 등급: C+

| 항목 | 평가 | 상세 |
|------|------|------|
| PEP8 | C | 라인 길이 초과 59건, 중복 import |
| Docstring | C+ | ~65% 커버리지, 주요 메서드 불완전 |
| 가독성 | C | 600줄+ 메서드, 깊은 중첩 (5-7레벨) |
| 에러 처리 | C+ | **bare `except:` (줄 875, 1908)**, 너무 넓은 Exception 처리 |

**리팩토링 필요 여부**: ✅ **필수 (P0)**

**핵심 이슈**:

1. **God Class — `MLBacktest`**
   - 데이터 로딩, 모델 관리, 백테스트 실행, 리포팅, 상장폐지 검증까지 담당
   - 권장: 책임 분리
     ```
     MLBacktest (현재)
       ├── BacktestDataLoader  — 데이터/캐시 로딩
       ├── BacktestEngine      — 백테스트 실행 로직
       ├── PortfolioManager    — 포트폴리오/매매 관리
       └── BacktestReporter    — 결과 리포팅
     ```

2. **중복 import**
   - `import joblib` — 줄 18과 줄 138에서 이중 import

3. **매직 넘버**
   - `0.001`: 수수료/슬리피지 (줄 163-164) — config에서 읽어야 함
   - `10`: 거래일 탐색 범위 (줄 248) — 상수 정의 필요
   - `2 *`: 곱셈 팩터 의미 불명 (줄 167)

4. **bare `except:` 구문**
   - 줄 875, 1908: `except:` → `except Exception as e:` 변경 필요

5. **한국어/영어 혼용 주석**
   - 일관성 부재, 영어로 통일 권장

---

#### 2.2 `backtest.py` — 1,014줄 | 등급: C

| 항목 | 평가 | 상세 |
|------|------|------|
| PEP8 | C- | 라인 길이 초과 79건, 중복 import |
| Docstring | D | ~30% 커버리지, 대부분 메서드 미작성 |
| 가독성 | C- | 매직 넘버, 약어 변수명, 레거시 코드 |
| 에러 처리 | D | **bare `except:` (줄 560)**, 주석 처리된 핸들러 |

**리팩토링 필요 여부**: ✅ **필수 (P1)**

**핵심 이슈**:

1. **중복 import**
   - `from collections import defaultdict` — 줄 15와 22에서 이중 import

2. **Docstring 부재**
   - `get_trade_date()` (줄 177): docstring 없음
   - 대부분의 메서드에 docstring 미작성 (30% 커버리지)

3. **코드 내 TODO — 5개 발견**
   ```
   줄 212: "TODO: plan 안쓸 때 loop/함수 분리"
   줄 286: "TODO multiprocessing 처리"
   줄 379: "TODO: pd assign 시 경고 수정"
   줄 394: "TODO: get_trade_date() 함수는 어느 class가"
   줄 427: "TODO threshold"
   ```

4. **매직 넘버**
   - `[3, 6, 9, 12, 15, 18, 21, 24]`: 시간 기간 (줄 373) — 설명 없음
   - `CHUNK_SIZE = 20480`: 임의 값 (줄 29) — 근거 불명

5. **pandas 안티패턴**
   - `pd.set_option('mode.chained_assignment', None)` (줄 359) — 경고 숨김

6. **레거시 코드**
   - 주석 처리된 multiprocessing 코드 (줄 281-285)
   - 주석 처리된 exception handler (줄 638)

---

#### 2.3 기타 backtest 모듈

| 파일 | 줄수 | 등급 | 리팩토링 필요 | 핵심 이슈 |
|------|------|------|-------------|-----------|
| `model_comparator.py` | 427 | B | ❌ | 매직 넘버 (p < 0.05) |
| `sector_ensemble.py` | 428 | C+ | ⚠️ 경미 | 과도한 로깅, 넓은 Exception |
| `etf_data_loader.py` | 395 | B- | ❌ | 캐시 만료 로직 부재 |
| `rebalance_optimizer.py` | 314 | B- | ❌ | 매직 넘버 경미 |

---

### 3. `src/models/` (모델 모듈)

| 파일 | 줄수 | 등급 | 리팩토링 필요 | 핵심 이슈 |
|------|------|------|-------------|-----------|
| `config.py` | 273 | **A** | ❌ 불필요 | 우수한 설정 관리, 타입 힌트 완비 |
| `base_model.py` | 560 | B | ❌ | 추상 메서드 docstring 일부 미완 |
| `ensemble.py` | 683 | B- | ⚠️ 경미 | Stacking/Voting 공유 로직 중복 |
| `model_factory.py` | 624 | B- | ⚠️ 경미 | 하드코딩된 하이퍼파라미터 (줄 198-206) |
| `xgboost_model.py` | 275 | **B+** | ❌ 불필요 | 깔끔한 구조 |
| `lightgbm_model.py` | 342 | **B+** | ❌ 불필요 | 잘 문서화됨 |
| `catboost_model.py` | 368 | B | ❌ | 양호 |

**model_factory.py 매직 넘버 상세**:
```python
# 줄 198-206: 하드코딩된 값
'verbose': False        # → config로 이동 필요
'l2_leaf_reg': 5.0      # → config로 이동 필요
'rsm': 0.8              # → config로 이동 필요
'iterations': 800       # → config로 이동 필요
'learning_rate': 0.05   # → config로 이동 필요
'depth': 7              # → config로 이동 필요
```

---

### 4. `src/infra/` (인프라 모듈)

| 파일 | 줄수 | 등급 | 리팩토링 필요 | 핵심 이슈 |
|------|------|------|-------------|-----------|
| `g_variables.py` | 820 | **A** | ❌ | TODO 3개 (YAML config 마이그레이션), 양호 |
| `context_loader.py` | 684 | B+ | ⚠️ 경미 | **bare `except:` (줄 403)**, `create_dir()` 중복 |
| `logger.py` | 682 | A- | ❌ | 우수한 멀티프로세싱 지원 |
| `file_utils.py` | 110 | **A** | ❌ | 완벽한 docstring |

**context_loader.py `create_dir()` 중복**:
- `ContextLoader.create_dir()` (줄 457-486)
- `MainContext.create_dir()` (줄 594-626)
- 동일 로직 → 공유 유틸리티로 추출 필요

---

### 5. `src/data/` (데이터 모듈)

| 파일 | 줄수 | 등급 | 리팩토링 필요 | 핵심 이슈 |
|------|------|------|-------------|-----------|
| `fmp_api.py` | 432 | B | ⚠️ 경미 | 매직 넘버 `min(8, os.cpu_count())`, URL 처리 취약 |
| `fmp.py` | 570 | B | ⚠️ 경미 | **타입 어노테이션 오류** (줄 64-65), PEP8 위반 |
| `fmp_fetch_worker.py` | 328 | B+ | ❌ | Ray 통합 양호 |
| `parquet_storage.py` | 679 | B+ | ❌ | 깔끔한 API 설계 |
| `parquet_converter.py` | 549 | B | ⚠️ 경미 | 타입 어노테이션 오류, 하드코딩된 컬럼 리스트 |
| `data_validator.py` | 528 | B+ | ❌ | 포괄적 검증 규칙 |

**fmp.py 타입 어노테이션 오류**:
```python
# 줄 64-65: 타입과 초기값 불일치
self.symbol_list: List[str] = pd.DataFrame()  # ❌ List[str]에 DataFrame 할당
self.current_list: List[str] = pd.DataFrame()  # ❌ 동일 오류
```

---

### 6. `src/tracking/` (추적/리포팅 모듈)

| 파일 | 줄수 | 등급 | 리팩토링 필요 | 핵심 이슈 |
|------|------|------|-------------|-----------|
| `sheets_tracker.py` | 552 | C+ | ⚠️ 권장 | 중복 import, 반복적 try-except, 매직 문자열 |
| `integrated_report.py` | 420 | B | ⚠️ 경미 | **bare `except:` (줄 179)**, 매직 컬러코드 |
| `performance_monitor.py` | 355 | B | ❌ | 경고 과잉 숨김 |
| `config_masker.py` | 278 | A- | ❌ | 깔끔한 재귀 알고리즘 |

**sheets_tracker.py 반복 패턴**:
```python
# 줄 145-200: 유사한 try-except 블록 6회 반복
# 권장: 데코레이터 또는 공통 래퍼 함수로 추출
def _safe_api_call(self, operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except gspread.exceptions.APIError as e:
        self.logger.warning(f"Google Sheets API error: {e}")
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
```

---

### 7. `src/constants/` (상수 모듈)

| 파일 | 줄수 | 등급 | 리팩토링 필요 | 핵심 이슈 |
|------|------|------|-------------|-----------|
| `data_schema.py` | 365 | **A** | ❌ | 우수한 Single Source of Truth 설계 |

---

### 8. `main.py` — 825줄 | 등급: B

| 항목 | 평가 | 상세 |
|------|------|------|
| PEP8 | B | 양호 |
| Docstring | B | 메인 로직 문서화 |
| 가독성 | B- | 반복 패턴, TODO 다수 |
| 에러 처리 | B- | 누락된 파일 에러 처리 부재 |

**코드 내 TODO — 4개 이상**:
- 줄 89-93, 137-138, 208-217: 새 모델 구현 관련

---

## 아키텍처 평가

### 강점

1. **관심사 분리 (A)**
   - `src/data/`, `src/training/`, `src/backtest/`, `src/models/` 등 명확한 모듈 분리
   - `DataSchema` → 단일 진실 원천 (컬럼 정의)
   - `DataProcessor` → 통합 전처리

2. **일원화 아키텍처 (A)**
   - `regressor.py` ↔ `ml_backtest.py` 간 Prediction Cache 공유
   - Fallback 제거로 일원화 강제 (2025-01-17 완료)

3. **설정 관리 (A-)**
   - `conf.yaml` / `secrets.yaml` 분리 (보안)
   - `conf.yaml.template` 제공 (협업)
   - `ConfigLoader` 클래스의 secrets 자동 merge

4. **문서화 (A)**
   - `CLAUDE.md` 51,724 바이트 — 매우 상세한 가이드
   - `docs/codebase-report/` 9개 분석 문서
   - 대부분 파일의 모듈/클래스 docstring 충실

### 약점

1. **God Class 패턴 (C)**
   - `Regressor` 클래스: ~4,300줄 (5개 이상의 책임)
   - `MLBacktest` 클래스: ~1,800줄 (4개 이상의 책임)
   - `MakeMLData` 클래스: ~1,500줄

2. **코드 중복 (C+)**
   - `context_loader.py`의 `create_dir()` 중복
   - `sheets_tracker.py`의 try-except 반복 패턴
   - `regressor.py`와 `ml_backtest.py` 간 일부 잔존 중복

3. **레거시 코드 (B-)**
   - `archive/` 디렉토리에 10개 파일 잔존 (6,309줄)
   - `docs/archive/legacy_code/`에 2개 파일
   - 주석 처리된 코드 블록 (regressor.py, backtest.py)

4. **테스트 커버리지 (B-)**
   - 8개 테스트 파일 존재 (test_data_processor, test_make_mldata, test_ml_backtest 등)
   - 테스트 커버리지 측정 미설정 (pytest-cov)
   - 통합 테스트 부재

---

## 코딩 컨벤션 (PEP8) 종합 평가

### 전체 등급: B-

#### 준수 사항 ✅

| 항목 | 상태 | 비고 |
|------|------|------|
| 네이밍 컨벤션 | ✅ 우수 | snake_case 일관 (함수/변수), PascalCase (클래스) |
| 들여쓰기 | ✅ 양호 | 4 spaces 일관 사용 |
| 빈 줄 | ✅ 양호 | 클래스/함수 간 적절한 간격 |
| 타입 힌트 | ✅ 양호 | 주요 함수에 타입 어노테이션 적용 |
| import 정렬 | ⚠️ 보통 | 대체로 양호, 일부 무질서 |

#### 위반 사항 ❌

| 항목 | 파일 | 상세 |
|------|------|------|
| **라인 길이 > 100자** | ml_backtest.py (59건), backtest.py (79건), regressor.py (다수) | PEP8: 79자 권장, 최대 99자 |
| **bare `except:`** | regressor.py, ml_backtest.py, backtest.py, context_loader.py, optuna_utils.py, mlflow_tracker.py, integrated_report.py | 총 7개 파일에서 발견 |
| **중복 import** | ml_backtest.py (`joblib`), backtest.py (`defaultdict`) | 2개 파일 |
| **미사용 import 의심** | regressor.py (torch.nn, RandomForest, seaborn 등) | 확인 필요 |
| **`print()` 사용** | regressor.py (5곳) | `logging` 사용 필요 |
| **타입 어노테이션 오류** | fmp.py (줄 64-65) | `List[str] = pd.DataFrame()` |

---

## Docstring 종합 평가

### 전체 등급: B+

| 등급 | 파일 수 | 파일 목록 |
|------|---------|-----------|
| A (95%+) | 15 | data_processor, g_variables, file_utils, data_schema, config 등 |
| B (75-94%) | 8 | model_factory, ensemble, fmp_api 등 |
| C (50-74%) | 3 | ml_backtest, sheets_tracker, sector_ensemble |
| D (< 50%) | 1 | **backtest.py (~30%)** |

### Docstring 형식 평가

| 항목 | 상태 | 비고 |
|------|------|------|
| 모듈 docstring | ✅ 양호 | 대부분 파일에 모듈 설명 존재 |
| 클래스 docstring | ✅ 우수 | Attributes, 사용 예시 포함 |
| 메서드 docstring | ⚠️ 보통 | Args/Returns는 있으나 Raises 종종 누락 |
| 인라인 예시 | ⚠️ 보통 | 일부 파일에만 `Examples:` 섹션 존재 |
| 언어 | ⚠️ 혼합 | 한국어/영어 혼용 (통일 권장) |

---

## 가독성 종합 평가

### 전체 등급: C+

#### 초장 메서드 목록 (100줄 이상)

| 파일 | 메서드 | 줄수 | 심각도 |
|------|--------|------|--------|
| `make_mldata.py` | `make_ml_data()` | **807줄** | 🔴 Critical |
| `regressor.py` | `train()` | **~800줄** | 🔴 Critical |
| `ml_backtest.py` | backtest 메인 루프 | **600줄+** | 🔴 Critical |
| `regressor.py` | `dataload()` | **~400줄** | 🟡 High |
| `regressor.py` | `evaluation()` | **~500줄** | 🟡 High |
| `data_processor.py` | `preprocess_training_data()` | **366줄** | 🟡 High |
| `data_processor.py` | `full_pipeline()` | **224줄** | 🟠 Medium |
| `make_mldata.py` | `_export_infinite_removal_details()` | **153줄** | 🟠 Medium |
| `make_mldata.py` | `_filter_extreme_movers()` | **145줄** | 🟠 Medium |
| `data_processor.py` | `log_transform_features()` | **123줄** | 🟠 Medium |

**권장 기준**: 메서드당 최대 80줄 (PEP8 관례), 절대 상한 150줄

#### 매직 넘버 분포

| 파일 | 매직 넘버 | 상세 |
|------|-----------|------|
| `regressor.py` | `92`, `1000`, `1e7` | 임계값, feature 수, 클리핑 값 |
| `ml_backtest.py` | `0.001`, `10`, `2` | 수수료, 거래일 범위, 곱셈 팩터 |
| `backtest.py` | `20480`, `[3,6,9,...]` | 청크 사이즈, 시간 기간 |
| `make_mldata.py` | `12`, `0.25`, `100` | lookback, 롤백, 카운트 |
| `model_factory.py` | `5.0`, `0.8`, `800`, `0.05`, `7` | 하이퍼파라미터 |
| `model_comparator.py` | `0.05` | p-value 임계값 |

---

## 에러 처리 종합 평가

### 전체 등급: B

#### bare `except:` 전체 목록 (즉시 수정 필요)

| # | 파일 | 위치 | 수정 방안 |
|---|------|------|-----------|
| 1 | `regressor.py` | 다수 | `except Exception as e:` |
| 2 | `ml_backtest.py` | 줄 875 | `except Exception as e:` |
| 3 | `ml_backtest.py` | 줄 1908 | `except Exception as e:` |
| 4 | `backtest.py` | 줄 560 | `except Exception as e:` |
| 5 | `context_loader.py` | 줄 403 | `except Exception as e:` |
| 6 | `optuna_utils.py` | 줄 50 | `except (Exception, mlflow.exceptions.MlflowException):` |
| 7 | `mlflow_tracker.py` | 줄 50 | `except (Exception, mlflow.exceptions.MlflowException):` |
| 8 | `integrated_report.py` | 줄 179 | `except Exception as e:` |

**문제점**: `bare except`는 `KeyboardInterrupt`, `SystemExit` 등 시스템 예외까지 잡아 프로그램 종료를 방해

---

## TODO 종합 정리

### 코드 내 TODO 전체 목록

| # | 파일 | 위치 | 내용 | 우선순위 |
|---|------|------|------|---------|
| 1 | `regressor.py` | 줄 56 | 섹터별 예측 PER_SECTOR=True 구현 | P2 |
| 2 | `regressor.py` | 줄 57 | 섹터 매핑을 make_mldata.py로 이동 | P2 |
| 3 | `regressor.py` | 줄 58 | GridSearchCV 코드 마이그레이션/제거 | P3 |
| 4 | `regressor.py` | 줄 269 | `y_col_list` → `DataSchema` 마이그레이션 | P3 |
| 5 | `make_mldata.py` | 줄 211 | TODO 마커 (상세 불명) | P2 |
| 6 | `make_mldata.py` | 줄 567 | TODO 마커 (상세 불명) | P2 |
| 7 | `backtest.py` | 줄 212 | plan 미사용 시 loop/함수 분리 | P2 |
| 8 | `backtest.py` | 줄 286 | multiprocessing 처리 | P3 |
| 9 | `backtest.py` | 줄 379 | pd assign 경고 수정 | P3 |
| 10 | `backtest.py` | 줄 394 | `get_trade_date()` 소속 클래스 결정 | P2 |
| 11 | `backtest.py` | 줄 427 | threshold 관련 | P2 |
| 12 | `main.py` | 줄 89-93 | 새 모델 구현 | P1 |
| 13 | `main.py` | 줄 137-138 | 추가 구현 | P1 |
| 14 | `main.py` | 줄 208-217 | 추가 구현 | P1 |
| 15 | `g_variables.py` | 줄 71-74 | YAML config 마이그레이션 고려 | P3 |
| 16 | `fmp_api.py` | 줄 264, 271 | ETF 심볼 관련 | P3 |
| 17 | `parquet_converter.py` | 줄 34-36 | ParquetStorage 마이그레이션 | P3 |

---

## 리팩토링 우선순위 로드맵

### 🔴 P0: Critical (즉시)

| # | 작업 | 파일 | 효과 |
|---|------|------|------|
| 1 | bare `except:` 전체 수정 (8건) | 7개 파일 | 안정성 ↑, 디버깅 용이 |
| 2 | `make_ml_data()` 807줄 메서드 분리 | make_mldata.py | 가독성 ↑, 유지보수 ↑ |
| 3 | `Regressor` God Class 분리 | regressor.py | 테스트 용이, 재사용성 ↑ |
| 4 | 미사용 import 정리 | regressor.py | 의존성 명확화 |

### 🟡 P1: High (1-2주)

| # | 작업 | 파일 | 효과 |
|---|------|------|------|
| 5 | `MLBacktest` God Class 분리 | ml_backtest.py | 책임 명확화 |
| 6 | `preprocess_training_data()` 366줄 분리 | data_processor.py | 각 단계 독립 테스트 가능 |
| 7 | backtest.py docstring 보강 (30% → 80%+) | backtest.py | 유지보수 ↑ |
| 8 | 중복 import 제거 | ml_backtest.py, backtest.py | PEP8 준수 |
| 9 | `print()` → `logging` 변환 (5건) | regressor.py | 로깅 일관성 |
| 10 | main.py TODO 구현 | main.py | 기능 완성 |

### 🟠 P2: Medium (2-4주)

| # | 작업 | 파일 | 효과 |
|---|------|------|------|
| 11 | 매직 넘버 → 상수/config 추출 | 전체 | 유지보수 ↑ |
| 12 | `create_dir()` 중복 제거 → 공유 유틸리티 | context_loader.py | DRY 원칙 |
| 13 | sheets_tracker.py 반복 try-except 통합 | sheets_tracker.py | 가독성 ↑ |
| 14 | fmp.py 타입 어노테이션 오류 수정 | fmp.py | 타입 안전성 |
| 15 | 주석 처리된 코드 삭제 | regressor.py, backtest.py | 깔끔한 코드 |
| 16 | model_factory.py 하드코딩 → config | model_factory.py | 유연성 ↑ |

### 🟢 P3: Low (유지보수 시)

| # | 작업 | 파일 | 효과 |
|---|------|------|------|
| 17 | Docstring 언어 통일 (한국어 or 영어) | 전체 | 일관성 |
| 18 | archive/ 디렉토리 정리 검토 | archive/ | 코드베이스 정리 |
| 19 | pytest-cov 설정 및 커버리지 측정 | tests/ | 품질 가시화 |
| 20 | 라인 길이 100자 이내로 정리 | 전체 | PEP8 준수 |
| 21 | `_is_enabled()` / `_is_disabled()` → Enum | main.py | 타입 안전성 |
| 22 | 섹터 매핑 make_mldata.py로 이동 | regressor.py | 관심사 분리 |
| 23 | GridSearchCV 레거시 코드 제거 | regressor.py | 코드 정리 |

---

## 파일별 리팩토링 필요 여부 요약

### 총 64개 활성 소스 파일

| 상태 | 파일 수 | 비율 |
|------|---------|------|
| ✅ 리팩토링 불필요 | 40 | 62.5% |
| ⚠️ 경미한 수정 권장 | 16 | 25.0% |
| 🔴 리팩토링 필수 | 8 | 12.5% |

### 리팩토링 필수 파일 (8개)

1. **`regressor.py`** (4,962줄) — God Class 분리, dead code 제거
2. **`make_mldata.py`** (1,983줄) — 807줄 메서드 분리
3. **`ml_backtest.py`** (2,003줄) — God Class 분리, bare except 수정
4. **`backtest.py`** (1,014줄) — docstring 보강, bare except, dead code
5. **`data_processor.py`** (2,823줄) — 초장 메서드 분리
6. **`main.py`** (825줄) — TODO 구현, 에러 처리 보강
7. **`sheets_tracker.py`** (552줄) — 반복 패턴 통합
8. **`fmp.py`** (570줄) — 타입 어노테이션 오류 수정

### 경미한 수정 권장 파일 (16개)

context_loader.py, optuna_utils.py, mlflow_tracker.py, model_factory.py,
ensemble.py, sector_ensemble.py, etf_data_loader.py, rebalance_optimizer.py,
fmp_api.py, parquet_converter.py, integrated_report.py, performance_monitor.py,
base_model.py, catboost_model.py, model_comparator.py, `__init__.py` 파일들

### 리팩토링 불필요 파일 (40개)

config.py, data_schema.py, g_variables.py, logger.py, file_utils.py,
data_quality.py, feature_selector.py, walk_forward.py, time_series_cv.py,
optimizer.py, xgboost_model.py, lightgbm_model.py, config_masker.py,
data_validator.py, parquet_storage.py, fmp_fetch_worker.py,
테스트 파일 8개, 스크립트 12개, 예제 4개

---

## 결론

이 코드베이스는 **명확한 아키텍처 비전과 철저한 문서화**를 갖춘 양호한 프로젝트입니다.
주요 개선점은 **3개 핵심 파일의 God Class 분리**와 **8건의 bare except 수정**으로,
이 두 가지만 해결해도 유지보수성이 크게 향상됩니다.

전체 44,500줄 중 리팩토링이 필수인 파일은 8개(~12,700줄, 28.5%)이며,
나머지 71.5%는 현재 상태로도 충분히 유지 가능합니다.

---

> **다음 문서**: [07_recommendations.md](./07_recommendations.md) (개선 권고와 연계)
