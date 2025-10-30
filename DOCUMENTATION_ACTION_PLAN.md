# 문서화 개선 실행 계획
## Quant-refactoring Python Codebase

**작성일:** October 29, 2025
**상태:** 구현 준비 완료

---

## 빠른 참조 - 수정이 필요한 중요 파일들

### TIER 1: 중요 (먼저 수행) - 총: 10-12시간

| File | Coverage | Issues | Effort | Impact |
|------|----------|--------|--------|--------|
| `/backtest.py` | 5.8% | No module docstring, 4 classes missing docs, 17 methods, 0% type hints, 5 TODO + 1 FIXME | 3-4h | HIGHEST |
| `/training/regressor.py` | 4.0% | No module docstring, class + 7 methods undocumented, 0% type hints, 3 TODO | 2-3h | HIGH |
| `/data_collector/fmp_api.py` | 4.0% | No module docstring, class + 12 methods undocumented, 0% type hints | 2-3h | HIGH |
| `/training/make_mldata.py` | 5.6% | No module docstring, class + 15 methods undocumented, 0% type hints | 2-3h | HIGH |
| `/data_collector/fmp.py` | 6.8% | No module docstring, class undocumented, 4 methods, 0% type hints, 1 FIXME | 1.5-2h | HIGH |
| `/storage/parquet_converter.py` | 4.0% | No module docstring, class + 3 methods undocumented, 0% type hints, 1 TODO | 1-1.5h | MEDIUM |

**소계: 11-17시간** (모든 type hint도 수행하는 경우)

### TIER 2: 중요 (다음 수행) - 총: 3-4시간

| File | Coverage | Issues | Effort | Impact |
|------|----------|--------|--------|--------|
| `/main.py` | 15.6% | 0/9 functions with type hints (critical gap) | 0.5h | HIGH |
| `/models/ensemble.py` | 14.8% | 7/16 functions with type hints (44% coverage) | 1h | MEDIUM |
| `/config/context_loader.py` | 16.0% | 4 functions missing docs, 12/16 type hints | 0.5h | MEDIUM |
| `/models/base_model.py` | 14.9% | 1 function missing docstring, 7/14 type hints | 0.5h | MEDIUM |

**소계: 2.5-3시간**

### TIER 3: 있으면 좋음 (선택사항) - 총: 2-3시간

| File | Coverage | Issues | Effort |
|------|----------|--------|--------|
| `/config/logger.py` | 17.3% | 4 functions missing docstrings, 7/9 type hints | 0.5h |
| `/tools/rank_processing.py` | 4.0% | No module docstring | 0.5h |
| `/config/g_variables.py` | 4.0% | Add section comments to 370-line configuration | 1-1.5h |

**소계: 2-2.5시간**

---

## 실행 로드맵

### Phase 1: 설정 (1시간)
```
[ ] Define docstring standard (Google style recommended)
[ ] Create docstring template for project
[ ] Update CONTRIBUTING.md with documentation guidelines
[ ] Create pull request template with documentation checklist
```

### Phase 2: Critical Documentation (10-12 hours)

**Week 1:**
```
[ ] Document backtest.py (3-4 hours)
    [ ] Add module-level docstring
    [ ] Document Backtest class
    [ ] Document EvaluationHandler class
    [ ] Document PlanHandler class
    [ ] Add docstrings to 17+ missing methods

[ ] Document training/regressor.py (2-3 hours)
    [ ] Add module-level docstring
    [ ] Document Regressor class with initialization
    [ ] Document dataload(), train(), evaluation(), etc.
    [ ] Resolve 3 TODO items
```

**Week 2:**
```
[ ] Document data_collector/fmp_api.py (2-3 hours)
    [ ] Add module-level docstring
    [ ] Document FMPAPI class
    [ ] Document all 12 methods
    [ ] Add type hints to all parameters

[ ] Document training/make_mldata.py (2-3 hours)
    [ ] Add module-level docstring
    [ ] Document AIDataMaker class
    [ ] Document data preparation methods
    [ ] Document feature engineering logic
```

**Week 3:**
```
[ ] Document remaining critical files (2-3 hours)
    [ ] data_collector/fmp.py
    [ ] storage/parquet_converter.py
    [ ] Resolve all TODO/FIXME items (11 total)
```

### Phase 3: Type Hints Addition (2-3 hours)

**Week 4:**
```
[ ] Add type hints to main.py (0.5 hours)
[ ] Add type hints to models/ensemble.py (1 hour)
[ ] Add type hints to config/context_loader.py (0.5 hours)
[ ] Add type hints to models/base_model.py (0.5 hours)
```

### Phase 4: Final Polish (2-3 hours)

```
[ ] Complete config/logger.py docstrings
[ ] Add section comments to config/g_variables.py
[ ] Add module docstring to tools/rank_processing.py
[ ] Code review all documented files
[ ] Verify Args/Returns/Raises completeness
```

### Phase 5: Automation (Optional)

```
[ ] Set up Sphinx documentation generation
[ ] Configure GitHub Pages for auto-generated docs
[ ] Add pydocstyle to CI/CD pipeline
[ ] Create documentation coverage dashboard
```

---

## DETAILED FILE FIXES

### File 1: backtest.py (3-4 hours)

**Current State:**
- 1,700+ lines of code
- 4 classes: Backtest, EvaluationHandler, PlanHandler, (and 1 more)
- 31 methods total
- 0 class docstrings
- 17 methods without docstrings
- 182 inline comments (good foundation!)
- 5 TODO + 1 FIXME items

**What to Add:**
```python
"""
Backtesting Engine for Quantitative Trading Strategies

This module provides the core backtesting functionality for evaluating trading
strategies against historical price and financial data.

Main Components:
    - Backtest: Core backtesting engine that simulates trading over time periods
    - EvaluationHandler: Handles performance evaluation and reporting
    - PlanHandler: Manages trading plan configuration and optimization
"""

class Backtest:
    """
    Core backtesting engine for quantitative trading strategies.
    
    Simulates portfolio performance by:
    1. Loading historical price and financial statement data
    2. Applying trading plan/selection criteria for each rebalance period
    3. Calculating returns based on price movements
    4. Generating evaluation reports
    
    Attributes:
        main_ctx: MainContext instance with configuration
        conf: Configuration dictionary
        plan_handler: PlanHandler instance for plan management
        rebalance_period: Rebalancing frequency in months
        start_year: Starting year for backtest
        end_year: Ending year for backtest
    
    Example:
        >>> bt = Backtest(main_ctx, config, plan_handler, rebalance_period=3)
        >>> # Backtest is executed automatically in __init__
    """
```

**Key Methods to Document:**
- `__init__()` - Constructor and initialization
- `load_bt_table()` - Load historical data tables
- `run()` - Main backtesting loop
- `create_report()` - Generate reports
- And 13+ others

---

### File 2: training/regressor.py (2-3 hours)

**Current State:**
- Legacy ML training code
- Regressor class undocumented
- 7 methods with no docstrings
- 122 inline comments
- 3 TODO items

**Key Methods to Document:**
- `__init__()` - Initialize with data paths
- `dataload()` - Load training data
- `train()` - Train models
- `evaluation()` - Evaluate performance
- `latest_prediction()` - Make predictions on latest data
- Custom model training methods

---

### File 3: data_collector/fmp_api.py (2-3 hours)

**Current State:**
- FMPAPI class handles FMP API endpoint configuration
- 12 methods with 0 docstrings
- Complex URL parsing logic
- No type hints

**Key Methods to Document:**
- `__init__()` - Parse API URL and configuration
- `converted_category` (property) - Get category from API path
- `query_params_str` (property) - Get query string
- Private methods for URL building and API calls

---

### File 4: training/make_mldata.py (2-3 hours)

**Current State:**
- AIDataMaker class for ML data preparation
- 15 methods with only 6 documented
- Complex feature engineering logic
- 104 inline comments mixing Korean and English

**Key Methods to Document:**
- `__init__()` - Initialize with context and config
- Feature calculation methods
- Data preprocessing methods
- Label generation methods

---

### File 5: data_collector/fmp.py (1.5-2 hours)

**Current State:**
- FMP class orchestrates data collection
- 13 methods with 9 documented
- 1 FIXME item
- Clear function structure

**Key Methods to Document:**
- `__init__()` - Initialize FMP collector
- `collect()` - Main collection method
- `__fetch_ticker_list()` - Get available tickers
- `__set_symbol()` - Filter and prepare symbols

---

### File 6: storage/parquet_converter.py (1-1.5 hours)

**Current State:**
- Parquet class for CSV to Parquet conversion
- 3 methods completely undocumented
- 1 TODO item
- Bridge between old CSV and new Parquet storage

**Key Methods to Document:**
- `__init__()` - Initialize converter
- `insert_csv()` - Convert CSV to Parquet
- `rebuild_table_view()` - Rebuild data views

---

## TYPE HINTS ADDITIONS

### main.py - 0.5 hours

Current:
```python
def get_config_path():  # No type hints
def conf_check(config):  # No type hints
def main():  # No type hints
```

Should be:
```python
def get_config_path() -> str:
def conf_check(config: Dict[str, Any]) -> None:
def main() -> None:
```

### models/ensemble.py - 1 hour

Add type hints to all ensemble methods:
```python
def fit(self, X_train: np.ndarray, y_train: np.ndarray, ...) -> 'StackingEnsemble':
def predict(self, X: np.ndarray) -> np.ndarray:
def get_feature_importance(self) -> Dict[str, float]:
```

### config/context_loader.py - 0.5 hours

Add missing type hints to legacy methods.

### models/base_model.py - 0.5 hours

Add type hints to:
- `predict()`
- `evaluate()`
- `save()`
- `load()`

---

## TODO/FIXME RESOLUTION

**backtest.py (5 TODO + 1 FIXME):**
- Identify each TODO item
- Determine if it's a documentation issue or code issue
- Fix or document the reason for not fixing

**training/regressor.py (3 TODO):**
- Review TODO items
- Document or resolve

**main.py (2 TODO):**
- New model training related
- Clarify requirements or implement

**data_collector/fmp.py (1 FIXME):**
- Line 95: "의미 잘 모르겠음" - clarify meaning
- Document the logic

**storage/parquet_converter.py (1 TODO):**
- Identify and resolve

---

## VERIFICATION CHECKLIST

After completing each file, verify:

- [ ] Module has docstring with purpose and brief description
- [ ] All classes have docstrings explaining purpose and key attributes
- [ ] All public methods/functions have docstrings with:
  - [ ] Brief description of what the method does
  - [ ] Args section with parameter descriptions and types
  - [ ] Returns section with return type and value description
  - [ ] Raises section if applicable
  - [ ] Example usage if complex
- [ ] Type hints present for function parameters and returns
- [ ] Inline comments explain complex logic (not obvious from code)
- [ ] TODO/FIXME items are resolved or clearly documented
- [ ] Private methods documented if complex
- [ ] Docstring format is consistent with project standard

---

## RESOURCES & TOOLS

### Docstring Format (Google Style Recommended)
```python
"""
One-line summary.

More detailed explanation if needed. Can span multiple lines.

Args:
    param1 (type): Description of param1
    param2 (type): Description of param2 (optional). Defaults to None.

Returns:
    type: Description of return value

Raises:
    ExceptionType: When this exception is raised

Example:
    >>> result = function(arg1, arg2)
    >>> print(result)
"""
```

### Type Hints
```python
from typing import Dict, List, Optional, Tuple, Union

def function(
    param1: str, 
    param2: int, 
    param3: Optional[List[str]] = None
) -> Dict[str, float]:
    """Function documentation."""
    pass
```

### Tools
- **pydocstyle**: Check docstring format
- **mypy**: Type hint checker
- **Sphinx**: Auto-generate documentation
- **black**: Code formatter

---

## SUCCESS METRICS

After completion:
- 100% of critical files (Tier 1) documented
- All critical files have module docstrings
- All critical classes documented
- All critical methods have docstrings
- 80%+ type hint coverage in all critical files
- All TODO/FIXME items resolved or documented
- Documentation coverage increased from 5.8% to 50%+ for backtest.py
- Documentation coverage increased from 4% to 50%+ for other critical files

---

## TIMELINE SUMMARY

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Setup | 1 hour | Standards and templates |
| Tier 1 Docs | 10-12 hours | Critical files documented |
| Type Hints | 2-3 hours | All type hints added |
| Polish | 2-3 hours | Final review and improvements |
| **Total** | **15-19 hours** | **Fully documented codebase** |

**Estimated calendar time: 1-2 weeks** (working part-time on documentation)

---

## NOTES

- Start with backtest.py as it has the most inline comments to work from
- Use existing comments as basis for docstrings
- Type hints can be added incrementally
- Keep committing as files are completed for easy review
- Consider pair programming for complex files
- Have code review focus on documentation completeness

