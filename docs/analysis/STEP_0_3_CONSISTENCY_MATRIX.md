# Configuration Consistency Matrix

## Executive Summary

**Total Critical Parameters Checked**: 33
**Status Breakdown**:
- ✅ **Used in BOTH** (regressor & ml_backtest): 2 parameters (6%)
- ⚠️  **Regressor ONLY**: 10 parameters (30%)
- ⚠️  **ML Backtest ONLY**: 1 parameter (3%)
- ℹ️  **Main.py ONLY**: 13 parameters (39%)
- ❌ **NOT USED**: 7 parameters (21%)

---

## 🚨 Critical Issues Found

### Issue 1: Training Period Mismatch
**Problem**: regressor.py uses TRAIN/TEST dates, ml_backtest.py doesn't read them

| Parameter | Config | regressor.py | ml_backtest.py | Impact |
|-----------|--------|--------------|----------------|--------|
| TRAIN_START_YEAR | ✅ | ✅ (2 times) | ❌ | **CRITICAL** |
| TRAIN_END_YEAR | ✅ | ✅ (2 times) | ❌ | **CRITICAL** |
| TEST_START_YEAR | ✅ | ✅ (2 times) | ❌ | **CRITICAL** |
| TEST_END_YEAR | ✅ | ✅ (2 times) | ❌ | **CRITICAL** |

**Impact**: Models are evaluated on different time periods!
- regressor.py: Trains on TRAIN_START/END, tests on TEST_START/END
- ml_backtest.py: Uses PERIODS only (different dates!)

---

### Issue 2: Optuna Hyperparameter Tuning
**Problem**: Only regressor.py uses Optuna

| Parameter | Config | regressor.py | ml_backtest.py | Impact |
|-----------|--------|--------------|----------------|--------|
| USE_OPTUNA | ✅ | ✅ (2 times) | ❌ | **CRITICAL** |
| OPTUNA_TRIALS | ✅ | ✅ (1 time) | ❌ | **CRITICAL** |
| OPTUNA_CV_FOLDS | ✅ | ✅ (1 time) | ❌ | **CRITICAL** |
| OPTUNA_TIMEOUT | ✅ | ✅ (1 time) | ❌ | **CRITICAL** |
| OPTUNA_SEARCH_SPACE | ✅ | ✅ (1 time) | ❌ | **CRITICAL** |

**Impact**: 
- regressor.py: Uses optimized hyperparameters
- ml_backtest.py: Uses default parameters
- **Result**: Comparing different models!

---

### Issue 3: Sector Model Strategy
**Problem**: Different naming and usage

| Parameter | Config | regressor.py | ml_backtest.py | Impact |
|-----------|--------|--------------|----------------|--------|
| USE_SECTOR_MODEL | ✅ | ❌ | ✅ (1 time) | **CRITICAL** |
| PER_SECTOR | ❌ (hardcoded) | ✅ (27 times, False) | ❌ | **CRITICAL** |

**Impact**:
- Config has USE_SECTOR_MODEL
- regressor.py ignores it, uses hardcoded PER_SECTOR=False
- ml_backtest.py reads USE_SECTOR_MODEL correctly
- **Result**: Can't sync sector strategy!

---

### Issue 4: Preprocessing Parameters
**Status**: ✅ **These work!** (But are hardcoded, not from config)

| Parameter | Config | regressor.py | ml_backtest.py | Status |
|-----------|--------|--------------|----------------|--------|
| USE_WINSORIZATION | ❌ (not in config) | ✅ (5 times) | ✅ (4 times) | ⚠️  Hardcoded |
| USE_FEATURE_SELECTION | ❌ (not in config) | ✅ (3 times) | ✅ (4 times) | ⚠️  Hardcoded |

**Impact**: Both files use these, but as hardcoded values, not from config!

---

### Issue 5: Unused Parameters
**Problem**: Parameters defined in config but never used

| Parameter | Config | Usage | Impact |
|-----------|--------|-------|--------|
| MIN_VOLUME_PERCENTILE | ✅ | ❌ None | **MEDIUM** |
| MEMBER_CNT | ✅ | ❌ None | **LOW** |
| TOTAL_ASSET | ✅ | ❌ None | **LOW** |
| MISSING_THRESHOLD | ✅ | ❌ None | **MEDIUM** |
| ROW_MISSING_THRESHOLD | ✅ | ❌ None | **MEDIUM** |
| SCALER | ✅ | ❌ None | **MEDIUM** |
| USE_ENSEMBLE | ✅ | ❌ None | **HIGH** |
| THRESHOLD_PERCENTILE | ✅ | ❌ None | **LOW** |
| SAVE_DEBUG_CSV | ✅ | ❌ None | **LOW** |

**Impact**: Config suggests features that don't exist!

---

## 📊 Complete Parameter Matrix

### Category: ML Training

| Parameter | Config | regressor.py | ml_backtest.py | main.py | Consistency |
|-----------|--------|--------------|----------------|---------|-------------|
| USE_NEW_MODELS | ✅ | ❌ | ❌ | ✅ (5) | ⚠️  Main only |
| USE_OPTUNA | ✅ | ✅ (2) | ❌ | ❌ | ❌ Regressor only |
| USE_SECTOR_MODEL | ✅ | ❌ | ✅ (1) | ❌ | ❌ MLB only |
| PER_SECTOR | ❌ | ✅ (27) | ❌ | ❌ | ❌ Hardcoded |
| USE_ENSEMBLE | ✅ | ❌ | ❌ | ❌ | ❌ Unused |
| USE_MLFLOW | ✅ | ❌ | ❌ | ✅ (5) | ⚠️  Main only |
| TRAIN_START_YEAR | ✅ | ✅ (2) | ❌ | ✅ (1) | ❌ Regressor only |
| TRAIN_END_YEAR | ✅ | ✅ (2) | ❌ | ❌ | ❌ Regressor only |
| TEST_START_YEAR | ✅ | ✅ (2) | ❌ | ✅ (2) | ❌ Regressor only |
| TEST_END_YEAR | ✅ | ✅ (2) | ❌ | ✅ (2) | ❌ Regressor only |

### Category: Hyperparameter Tuning

| Parameter | Config | regressor.py | ml_backtest.py | main.py | Consistency |
|-----------|--------|--------------|----------------|---------|-------------|
| OPTUNA_TRIALS | ✅ | ✅ (1) | ❌ | ❌ | ❌ Regressor only |
| OPTUNA_CV_FOLDS | ✅ | ✅ (1) | ❌ | ❌ | ❌ Regressor only |
| OPTUNA_TIMEOUT | ✅ | ✅ (1) | ❌ | ❌ | ❌ Regressor only |
| OPTUNA_SEARCH_SPACE | ✅ | ✅ (1) | ❌ | ❌ | ❌ Regressor only |

### Category: Preprocessing

| Parameter | Config | regressor.py | ml_backtest.py | main.py | Consistency |
|-----------|--------|--------------|----------------|---------|-------------|
| USE_WINSORIZATION | ❌ | ✅ (5) | ✅ (4) | ❌ | ⚠️  Hardcoded both |
| USE_FEATURE_SELECTION | ❌ | ✅ (3) | ✅ (4) | ❌ | ⚠️  Hardcoded both |
| MIN_VOLUME_PERCENTILE | ✅ | ❌ | ❌ | ❌ | ❌ Unused |
| SCALER | ✅ | ❌ | ❌ | ❌ | ❌ Unused |
| MISSING_THRESHOLD | ✅ | ❌ | ❌ | ❌ | ❌ Unused |
| ROW_MISSING_THRESHOLD | ✅ | ❌ | ❌ | ❌ | ❌ Unused |

### Category: Backtest

| Parameter | Config | regressor.py | ml_backtest.py | main.py | Consistency |
|-----------|--------|--------------|----------------|---------|-------------|
| REBALANCE_PERIOD | ✅ | ❌ | ❌ | ✅ (3) | ⚠️  Main only |
| TOP_K_NUM | ✅ | ❌ | ❌ | ✅ (3) | ⚠️  Main only |
| MEMBER_CNT | ✅ | ❌ | ❌ | ❌ | ❌ Unused |
| ABSOLUTE_SCORE | ✅ | ❌ | ❌ | ✅ (1) | ⚠️  Main only |
| TOTAL_ASSET | ✅ | ❌ | ❌ | ❌ | ❌ Unused |

### Category: Data Management

| Parameter | Config | regressor.py | ml_backtest.py | main.py | Consistency |
|-----------|--------|--------------|----------------|---------|-------------|
| STORAGE_TYPE | ✅ | ❌ | ❌ | ✅ (2) | ⚠️  Main only |
| GET_FMP | ✅ | ❌ | ❌ | ✅ (4) | ⚠️  Main only |
| MAKE_VIEW | ✅ | ❌ | ❌ | ✅ (1) | ⚠️  Main only |
| SAVE_DEBUG_CSV | ✅ | ❌ | ❌ | ❌ | ❌ Unused |

### Category: Execution Control

| Parameter | Config | regressor.py | ml_backtest.py | main.py | Consistency |
|-----------|--------|--------------|----------------|---------|-------------|
| RUN_REGRESSION | ✅ | ❌ | ❌ | ✅ (4) | ⚠️  Main only |
| RUN_BACKTEST | ✅ | ❌ | ❌ | ✅ (4) | ⚠️  Main only |
| EXIT_AFTER_ML | ✅ | ❌ | ❌ | ✅ (3) | ⚠️  Main only |

---

## 🎯 Priority Issues

### 🔴 **PRIORITY 1: CRITICAL - Can't Compare Models**

1. **USE_OPTUNA** (regressor only)
   - Impact: Different hyperparameters used
   - Fix: Add Optuna support to ml_backtest.py OR remove from both

2. **USE_SECTOR_MODEL vs PER_SECTOR** (naming mismatch)
   - Impact: Can't sync sector strategy
   - Fix: Unify naming, make both read from config

3. **TRAIN/TEST dates** (regressor only)
   - Impact: Different evaluation periods
   - Fix: Make ml_backtest aware of these dates (separate from PERIODS)

### 🟡 **PRIORITY 2: HIGH - Missing Features**

4. **USE_WINSORIZATION / USE_FEATURE_SELECTION** (hardcoded)
   - Impact: Can't experiment with on/off from config
   - Fix: Add to config, read from config in both files

5. **USE_NEW_MODELS** (main.py only)
   - Impact: ml_backtest doesn't use new models
   - Fix: Make ml_backtest support new models

### 🟢 **PRIORITY 3: MEDIUM - Cleanup**

6. **MIN_VOLUME_PERCENTILE** (unused)
   - Impact: Misleading config
   - Fix: Implement or remove from config

7. **USE_ENSEMBLE** (unused)
   - Impact: Misleading config
   - Fix: Implement or remove from config

---

## 📝 Next Steps

See `STEP_0_4_PROBLEM_ANALYSIS.md` for detailed analysis of each issue.
See `STEP_0_5_EXECUTION_PLAN.md` for step-by-step fix plan.

