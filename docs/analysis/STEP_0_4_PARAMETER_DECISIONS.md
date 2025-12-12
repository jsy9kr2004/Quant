# Parameter Decision Matrix

Based on user responses and historical investigation

---

## 📋 User Decisions Summary

### Core Architecture Decisions

1. **TRAIN/TEST_YEAR vs PERIODS**: **Independent**
   - regressor.py: Uses `TRAIN_START_YEAR`, `TRAIN_END_YEAR`, `TEST_START_YEAR`, `TEST_END_YEAR`
   - ml_backtest.py: Uses `PERIODS` 
   - Same preprocessing/model/parameters, different evaluation periods

2. **USE_OPTUNA**: **Reuse optimized parameters**
   - regressor.py: `USE_OPTUNA=Y` → Run optimization & save
   - ml_backtest.py: `USE_OPTUNA=Y` → Load saved parameters
   - **Critical**: Both read same config value

3. **Sector Naming**: **USE_SECTOR_MODEL** (unified)
   - Remove `PER_SECTOR` from regressor.py
   - Both use `USE_SECTOR_MODEL` from config

4. **Preprocessing**: **Add to config**
   - `USE_WINSORIZATION`: Add to config, both read
   - `USE_FEATURE_SELECTION`: Add to config, both read

---

## 🔍 Parameter Investigation Results

### Category 1: Unused Parameters

#### MIN_VOLUME_PERCENTILE
**Config**: `MIN_VOLUME_PERCENTILE: 50`
**Comment**: "Minimum trading volume (percentile)"
**Usage**: ❌ Not used anywhere
**Purpose**: Filter out low-volume stocks (illiquid)
**Decision**: **IMPLEMENT** - Important for real trading
**Action**: 
- Add to data loading pipeline
- Filter stocks below Nth percentile of trading volume
- Apply in both regressor.py and ml_backtest.py

---

#### SCALER
**Config**: `SCALER: RobustScaler  # RobustScaler or StandardScaler`
**Usage**: ❌ Not used (hardcoded in code)
**Purpose**: Choose normalization method
**Decision**: **REMOVE from config** - Currently not needed
**Reason**: 
- Tree-based models (XGBoost/LightGBM) don't need scaling
- Already removed scaling in recent refactor
- If needed later, can be added back

---

#### MISSING_THRESHOLD
**Config**: `MISSING_THRESHOLD: 0.8  # Remove columns with > 80% missing`
**Usage**: ❌ Not used (DataProcessor has hardcoded 0.5)
**Purpose**: Drop columns with too many missing values
**Current**: DataProcessor uses 0.5 as hardcoded value
**Decision**: **IMPLEMENT** - Read from config
**Action**:
- DataProcessor.drop_sparse_columns should read from config
- Both regressor.py and ml_backtest.py use same threshold

---

#### ROW_MISSING_THRESHOLD
**Config**: `ROW_MISSING_THRESHOLD: 0.6  # Remove rows with > 60% missing`
**Usage**: ❌ Not used
**Purpose**: Drop rows with too many missing values
**Decision**: **IMPLEMENT** or **REMOVE**
**Question**: Do we want row-level filtering? Or just let XGBoost handle NaN?
**Recommendation**: REMOVE - Let XGBoost handle missing values

---

#### SAME_VALUE_THRESHOLD
**Config**: `SAME_VALUE_THRESHOLD: 0.95  # Remove columns with > 95% same value`
**Usage**: ✅ **Used!** (hardcoded in multiple places)
**Found in**:
- `src/training/data_processor.py`: 0.98 (hardcoded)
- `src/training/regressor.py`: 0.95 (hardcoded)
- `src/scripts/debug/deep_infinite_diagnosis.py`: 0.95 (hardcoded)
**Purpose**: Drop columns where 95%+ values are identical (low variance)
**Decision**: **IMPLEMENT** - Read from config
**Action**: Make DataProcessor read from config, remove hardcoded values

---

#### USE_ENSEMBLE
**Config**: `USE_ENSEMBLE: N  # Y = Use stacking ensemble`
**Usage**: ❌ Not implemented
**Purpose**: Use ensemble of multiple models (stacking/voting)
**Decision**: **REMOVE** for now
**Reason**: 
- Not implemented yet
- Can add back when implementing ensemble
- Misleading to have in config

---

#### MEMBER_CNT
**Config**: `MEMBER_CNT: 20  # 실제 매수 종목 수`
**Usage**: ❌ Not used
**Purpose**: Unknown - possibly legacy from old backtest.py
**Decision**: **REMOVE**
**Reason**: 
- `TOP_K_NUM` already specifies number of stocks
- Redundant parameter
- No clear use case

---

#### TOTAL_ASSET
**Config**: `TOTAL_ASSET: 100000`
**Usage**: ❌ Not used
**Purpose**: Total portfolio value (for position sizing)
**Decision**: **REMOVE** for now
**Reason**: 
- Not implemented in current backtest
- Can add back when implementing position sizing
- Currently just uses equal weight

---

### Category 2: Special Parameters

#### THRESHOLD_PERCENTILE
**Config**: `THRESHOLD_PERCENTILE: 92  # Top 8% selection`
**Usage**: ✅ Used in `src/models/config.py`
**Purpose**: Stock selection threshold (select top 8% of predictions)
**Current status**: Defined in models/config.py, not read from main config
**Decision**: **VERIFY** if still needed
**Found**: 
```python
# src/models/config.py
THRESHOLD_PERCENTILE: int = 92  # Select stocks above 92nd percentile
```
**Question**: Is this used by new models? Or only legacy?

---

#### TS_WINDOW
**Config**: `TS_WINDOW: 12  # quarters`
**Usage**: ❌ Not used
**Purpose**: Time series window for rebalancing period experiments
**History**: Added by Claude to support rebalancing period testing
**Decision**: **REMOVE** for now
**Reason**: 
- `REBALANCE_PERIOD` already exists
- Not implemented yet
- Can add back when implementing TS features

---

#### ABSOLUTE_SCORE
**Config**: `ABSOLUTE_SCORE: 500`
**Usage**: main.py only (legacy)
**Purpose**: Legacy parameter from old backtest.py (pre-ML)
**Decision**: **REMOVE**
**Confirmed**: User said it's from old heuristic model, no longer needed

---

### Category 3: Data Parameters

#### STORAGE_TYPE, GET_FMP, MAKE_VIEW
**Usage**: main.py only
**Decision**: **KEEP as-is**
**Reason**: Data ingestion params, don't need to be in regressor/ml_backtest

#### SAVE_DEBUG_CSV
**Usage**: ❌ Not used
**Decision**: **REMOVE**

---

### Category 4: SECTOR_CONFIG

**Config**: Sector-specific model parameters
```yaml
SECTOR_CONFIG:
  Technology:
    model: xgboost
    n_estimators: 200
    max_depth: 8
  Financial:
    model: xgboost
    n_estimators: 150
    max_depth: 6
  # Healthcare, Consumer, Industrial (not used)
```

**Usage**: 
- ml_backtest.py: Reads `SECTOR_CONFIG` when `USE_SECTOR_MODEL=Y`
- regressor.py: ❌ Does not use

**Purpose**: 
- When `USE_SECTOR_MODEL=Y`, allows different hyperparameters per sector
- Example: Tech stocks might benefit from deeper trees, Financial from shallower

**User Question**: "언제 필요하지?"
**Answer**: 
- **When**: `USE_SECTOR_MODEL=Y` (sector-based models enabled)
- **Why**: Each sector has different characteristics
  - Technology: High volatility, complex patterns → deeper trees
  - Financial: Regulated, cyclical → different params
  - Healthcare: Long-term trends → different learning rate
- **How**: ModelFactory reads sector-specific params and creates customized models

**Decision**: **IMPLEMENT** - Make regressor.py also read SECTOR_CONFIG
**Action**:
1. regressor.py should read `SECTOR_CONFIG` when `USE_SECTOR_MODEL=Y`
2. Create sector-specific models with custom hyperparameters
3. Both files use same sector configuration

**Currently used sectors**: Technology, Financial
**Unused sectors**: Healthcare, Consumer, Industrial
**Action**: Keep all 5 sectors in config for future use

---

## 📊 Final Parameter Status

### ✅ KEEP & IMPLEMENT (Read from config in both files)

| Parameter | Status | Action |
|-----------|--------|--------|
| USE_SECTOR_MODEL | One-sided | Add to regressor.py |
| USE_OPTUNA | One-sided | Add to ml_backtest.py (load mode) |
| USE_WINSORIZATION | Hardcoded | Add to config, read in both |
| USE_FEATURE_SELECTION | Hardcoded | Add to config, read in both |
| MIN_VOLUME_PERCENTILE | Unused | Implement filtering |
| MISSING_THRESHOLD | Unused | Implement (DataProcessor) |
| SAME_VALUE_THRESHOLD | Hardcoded | Read from config |
| SECTOR_CONFIG | One-sided | Add to regressor.py |
| TRAIN_START_YEAR | One-sided | Keep (regressor only - OK) |
| TRAIN_END_YEAR | One-sided | Keep (regressor only - OK) |
| TEST_START_YEAR | One-sided | Keep (regressor only - OK) |
| TEST_END_YEAR | One-sided | Keep (regressor only - OK) |

### ❌ REMOVE from config

| Parameter | Reason |
|-----------|--------|
| SCALER | Tree models don't need scaling |
| ROW_MISSING_THRESHOLD | Let XGBoost handle NaN |
| USE_ENSEMBLE | Not implemented |
| MEMBER_CNT | Redundant (use TOP_K_NUM) |
| TOTAL_ASSET | Not implemented |
| TS_WINDOW | Not implemented |
| ABSOLUTE_SCORE | Legacy (pre-ML) |
| SAVE_DEBUG_CSV | Not used |

### ❓ VERIFY

| Parameter | Question |
|-----------|----------|
| THRESHOLD_PERCENTILE | Still used by new models? |

---

## 🎯 Next Steps

See `STEP_0_5_EXECUTION_PLAN.md` for detailed implementation plan.

