# 코드 품질 및 유지보수성 분석

> **작성일**: 2025-12-17 (최종 업데이트: 2026-02-08)
> **이전 문서**: [04_backtesting.md](./04_backtesting.md)
> **다음 문서**: [06_quant_perspective.md](./06_quant_perspective.md)

---

## 핵심 요약

### 코드 품질 평가: A- (P0/P1 리팩토링 완료)

**강점**:
- 아키텍처 기반 일원화 (regressor ↔ ml_backtest 일관성 100% 보장)
- 통합 아키텍처 (DataSchema, DataProcessor, ModelFactory)
- God Method 전체 분리 완료 (오케스트레이터 패턴)
- bare except 전체 수정 (13건→0건)
- 명확한 문서화 (CLAUDE.md, codebase-report)
- 설정 관리 우수 (conf.yaml, secrets.yaml 분리)

**약점**:
- 단위 테스트 부족
- 일부 레거시 코드 잔존
- 매직 넘버 일부 잔존

**최근 개선 (2026-02-08)**:
- ✅ P0/P1 리팩토링 전체 완료 (6개 커밋, 10개 작업)
- ✅ 8개 God Method 분리 (최대 807줄→119줄, 85% 감소)
- ✅ bare except 13건 전체 수정
- ✅ backtest.py docstring 30%→100%
- ✅ print()→logging 변환 (11건)
- ✅ main.py TODO→명시적 에러 핸들링
- ✅ 중복 import 정리, 미사용 import 주석 처리

---

## 1. 코드 구조 평가

### 강점

#### 모듈 분리 우수

```
src/
├── data_collector/    # 데이터 수집
├── storage/           # 저장
├── constants/         # 상수
├── training/          # 학습
├── models/            # 모델
├── backtest/          # 백테스트
├── validation/        # 검증
└── optimization/      # 최적화
```

**평가**: 관심사 분리 (Separation of Concerns) 우수 (A)

#### 통합 모듈 (Unified Modules)

```python
# Single Source of Truth
DataSchema          # 컬럼 정의 단일화
DataProcessor       # 전처리 파이프라인 통합
ModelFactory        # 모델 생성 팩토리
```

**평가**: 코드 이중화 제거 (A)

### 약점

#### 복잡도 — 대폭 개선 (2026-02-08)

```python
# 파일 규모 (God Method 분리로 관리 가능한 수준)
# regressor.py: 5,184줄 — God Method 5개 분리 완료 (최대 109줄)
# ml_backtest.py: 2,036줄 — run() 356줄→38줄 분리
# data_processor.py: 2,876줄 — preprocess_training_data() 366줄→119줄 분리
# make_mldata.py: 2,139줄 — make_ml_data() 807줄→106줄 분리
```

**개선**: God Method 8개 전체 분리 완료 (오케스트레이터 패턴)
- 최대 메서드 길이: 807줄 → 119줄 (85% 감소)
- 각 서브 메서드는 단일 책임 원칙 준수

**남은 과제**: God Class 자체의 추가 분리 (P2 이후 검토)
```python
# 향후 검토 사항 (현재는 메서드 분리로 충분)
Regressor → DataLoader + ModelTrainer + ModelEvaluator
MLBacktest → BacktestEngine + BacktestReporter
```

---

## 2. 테스트 커버리지

### 현황: D (단위 테스트 없음)

```bash
$ find /home/user/Quant -name "test_*.py" -o -name "*_test.py"
# (empty)
```

### 우선순위 테스트

**1. 핵심 전처리 함수**:
```python
# tests/test_data_processor.py
def test_remove_infinite_values():
    X = pd.DataFrame({'a': [1, np.inf, 3], 'b': [4, 5, -np.inf]})
    X_clean = DataProcessor.remove_infinite_values(X)
    assert not np.isinf(X_clean).any().any()

def test_align_features_to_model():
    X = pd.DataFrame({'feat_a': [1, 2], 'feat_b': [3, 4]})
    model = MockModel(features=['feat_b', 'feat_c'])
    X_aligned = DataProcessor.align_features_to_model(X, model)
    assert 'feat_c' in X_aligned.columns
    assert X_aligned['feat_c'].isnull().all()
```

**2. 백테스트 재현성**:
```python
# tests/test_ml_backtest.py
def test_backtest_reproducibility():
    np.random.seed(42)
    result1 = run_backtest(config)
    
    np.random.seed(42)
    result2 = run_backtest(config)
    
    assert result1.equals(result2)
```

**3. 미래 유출 방지**:
```python
# tests/test_future_leakage.py
def test_no_future_leakage():
    cutoff_date = pd.Timestamp('2024-06-01')
    df = load_data(cutoff_date=cutoff_date)
    
    # 모든 filingDate가 cutoff_date 이전이어야 함
    assert (df['filingDate'] <= cutoff_date).all()
```

---

## 3. 코드 스타일 및 일관성

### PEP 8 준수: B+

```python
# ✅ 좋은 예
def preprocess_training_data(X, y, config, logger):
    """
    통합 전처리 파이프라인
    """
    pass

# ❌ 나쁜 예 (일부 레거시)
def def_model(self):  # "define_model"이 더 명확
    pass
```

**개선 도구**:
- Black (자동 포맷팅)
- flake8 (Linting)
- mypy (Type Checking)

---

## 4. 문서화

### 강점

#### CLAUDE.md (A+)

```markdown
# 시스템 철학
# 2-Stage 아키텍처
# regressor ↔ ml_backtest 일원화
# 미래 유출 방지
```

**평가**: 설계 의도 명확히 문서화

#### Docstring (A-)

**개선 (2026-02-08)**:
- backtest.py: 30% → **100%** (4개 클래스 + 16개 메서드 전체 docstring 추가)
- 신규 서브 메서드 전체에 docstring 추가 (30+ 메서드)
- `cal_price()` misplaced docstring 수정

```python
def preprocess_training_data(X, y, y_cls, config, logger):
    """
    통합 전처리 파이프라인

    Args:
        X: Feature DataFrame
        y: Target (continuous)
        y_cls: Target (binary)
        config: 설정 딕셔너리
        logger: Logger 인스턴스

    Returns:
        X, y, y_cls, selected_features
    """
```

**평가**: 주요 파일 전체 docstring 커버리지 양호

### 약점

#### API 문서 부족

```
# 현재: README.md만 존재
# 필요: Sphinx/MkDocs 기반 API 문서
```

---

## 5. 설정 관리

### 강점: conf.yaml (A)

```yaml
# 명확한 섹션 구분
DATA:
  ...
ML:
  ...
BACKTEST:
  ...

# 주석으로 설명
# ...
```

**평가**: 설정 관리 우수

### 약점: 하드코딩

```python
# ❌ 나쁜 예
threshold = 0.92  # 하드코딩

# ✅ 좋은 예
threshold = config['ML']['CLASSIFIER_THRESHOLD_PERCENTILE']
```

---

## 6. 에러 처리

### 현황: B+ (2026-02-08 개선)

```python
# bare except 13건 전체 수정 완료 (커밋 6be9cbc)
# 이전: except:
# 현재: except Exception as e:
try:
    df = load_parquet(path)
except Exception as e:
    logger.error(f"Failed to load: {e}")
    raise
```

**개선 (2026-02-08)**:
- bare `except:` 13건 → 0건 (7개 파일)
- main.py: silent-fail → `NotImplementedError` / `RuntimeError`

**개선안**:
```python
# 재시도 로직
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=10)
)
def load_parquet_with_retry(path):
    return pd.read_parquet(path)
```

---

## 7. 로깅

### 현황: A- (2026-02-08 개선)

```python
# logger.py로 통합 관리
logger = get_logger('module_name')
logger.info("Processing...")
logger.warning("Warning...")
logger.error("Error occurred")
```

**강점**:
- 통합 로거
- 레벨 관리 (DEBUG, INFO, WARNING, ERROR)
- regressor.py의 print() 11건 → logging 변환 완료 (커밋 `62dd1c7`)

**약점**:
- 로그 분석 도구 없음
- 구조화된 로깅 미사용 (JSON 로깅)

**개선안**:
```python
# 구조화된 로깅 (structlog)
logger.info(
    "backtest_completed",
    period="2020-2021",
    sharpe_ratio=1.2,
    max_drawdown=-0.18
)
# → JSON 출력, 파싱 용이
```

---

## 8. 의존성 관리

### requirements.txt: B

```
pandas==1.5.3
numpy==1.24.0
xgboost==1.7.0
...
```

**강점**:
- 버전 고정

**약점**:
- 그룹핑 없음 (dev/prod 구분 없음)

**개선안**:
```
# requirements/
#   base.txt       # 공통
#   dev.txt        # 개발 (pytest, black, ...)
#   prod.txt       # 프로덕션
```

---

## 9. 코드 복잡도 분석

### Cyclomatic Complexity — 대폭 개선 (2026-02-08)

```python
# ml_backtest.py::run() - 오케스트레이터 패턴 적용 완료 ✅
def run(self):
    price_table = pd.read_parquet(...)
    rebalance_dates = self._generate_rebalance_dates()
    date_pairs = self._adjust_to_trading_days(rebalance_dates, price_table)
    self._execute_walk_forward(date_pairs, price_table)
    results_df, benchmark_df = self._compile_results_and_benchmark(...)
    self._save_backtest_report(results_df, benchmark_df, ...)
    return results_df
# 복잡도: ~5 (이전 ~50에서 대폭 감소)
```

**8개 God Method 전체 분리 완료**:
- `make_ml_data()`: 807줄 → 106줄 (11개 서브 메서드)
- `train()`: 834줄 → 109줄 (4개 서브 메서드)
- `run()`: 356줄 → 38줄 (5개 서브 메서드)
- `dataload()`: 392줄 → 91줄 (4개 서브 메서드)
- `evaluation()`: 499줄 → 83줄 (4개 서브 메서드)
- `preprocess_training_data()`: 366줄 → 119줄 (5개 서브 메서드)
- `latest_prediction()`: 362줄 → 46줄 (3개 서브 메서드)
- `predict_for_date()`: 263줄 → 100줄 (3개 서브 메서드)

---

## 10. 레거시 코드

### archive/ 폴더

```
archive/
├── parquet.py         # 레거시
├── main_ml.py         # 레거시
├── fmp.py             # 레거시
└── regressor.py       # 레거시
```

**우려**:
- 레거시와 신규 코드 혼재
- 혼란 가능성

**권장**:
- 레거시 코드 명확히 표시
- 또는 삭제 (Git 히스토리에 보존)

---

## 11. 아키텍처 기반 일원화 (2026-01-17 추가)

### 구현 내용

regressor.py와 ml_backtest.py 간 일관성을 **아키텍처 자체가 강제**하도록 개선:

```python
# ml_backtest.py
if cache_path.exists():
    self.predictions_cache = joblib.load(cache_path)
else:
    # ✅ 에러 발생 - 일원화 강제
    raise FileNotFoundError(
        "Predictions cache not found!\n"
        "Run regressor.py first, or set USE_CACHED_PREDICTIONS=N"
    )
```

### 일원화 보장 수준 비교

| 접근 방식 | 일원화 보장 | 단점 |
|-----------|------------|------|
| 코드 리뷰 | 사람 의존 | 실수 가능 |
| 유닛테스트 | 테스트 커버리지 의존 | 누락 가능 |
| **아키텍처 강제** | **100% 보장** | 없음 |

---

## 결론

### 코드 품질 평가: A-

| 항목 | 이전 | 현재 | 변화 |
|------|------|------|------|
| 모듈 분리 | A | A | - |
| 통합 아키텍처 | A | A | - |
| 아키텍처 일원화 | A | A | - |
| 복잡도 | C+ | **B+** | ↑ God Method 전체 분리 |
| 테스트 | D | D | - (아직 부족) |
| 문서화 | A- | **A** | ↑ Docstring 100% (backtest.py) |
| 설정 관리 | A | A | - |
| 에러 처리 | C+ | **A-** | ↑ bare except 전체 수정 |
| 로깅 | B+ | **A-** | ↑ print→logging 완료 |

### 개선 우선순위
1. 단위 테스트 추가 (매우 높음) — 유일한 D 등급
2. 매직 넘버 → 상수/config 추출 (P2)
3. 남은 P2/P3 리팩토링 (선택적)

---

**다음 문서**: [06_quant_perspective.md](./06_quant_perspective.md)
