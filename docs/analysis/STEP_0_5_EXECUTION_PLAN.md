# Execution Plan: Configuration Unification

Based on Phase 0 analysis (Steps 0.1-0.4)

---

## 📋 Finalized Parameter Decisions

### Pending Items - Resolved (User Confirmed)

1. **SAME_VALUE_THRESHOLD**: **DataProcessor reads from config**
   - Current state: DataProcessor=0.98 (hardcoded), regressor=0.95 (hardcoded)
   - **Decision**: DataProcessor should read from config (not hardcoded)
   - Both regressor.py and ml_backtest.py call DataProcessor methods
   - Config value (0.95) becomes single source of truth
   - User: "DataProcessor 값 사용이 맞는데, 하드코딩이 아닌 config를 바라봐야"

2. **ROW_MISSING_THRESHOLD**: **REMOVE (Option B)**
   - XGBoost handles NaN internally
   - Remove from config for now
   - **Future-proofing**: If needed later, implement unified handling in both regressor and ml_backtest
   - User: "나중에 문제가 생겼을 때에는 regressor와 ml_backtest에서 동일하게 처리할 수 있도록 단일화가 필요"

3. **MIN_VOLUME_PERCENTILE**: **Option A - Data loading stage**
   - Location: `src/data/make_mldata.py`
   - Filter once at data loading time (DRY principle)
   - Both regressor.py and ml_backtest.py inherit filtered data
   - User: "동일한 작업을 두 군데에서 하는 것보다 한군데에서 처리"

---

## 🎯 Implementation Phases

### Phase 1: Configuration Cleanup (Foundational)
**Goal**: Remove unused parameters and standardize naming
**Priority**: HIGH - Foundation for all other work
**Estimated complexity**: LOW

#### Task 1.1: Remove Unused Parameters from config.yaml.template
**Parameters to remove**:
- `SCALER` - Tree models don't need scaling
- `ROW_MISSING_THRESHOLD` - Let XGBoost handle NaN
- `USE_ENSEMBLE` - Not implemented
- `MEMBER_CNT` - Redundant with TOP_K_NUM
- `TOTAL_ASSET` - Not implemented
- `TS_WINDOW` - Not implemented
- `ABSOLUTE_SCORE` - Legacy (pre-ML)
- `SAVE_DEBUG_CSV` - Not used
- `ENSEMBLE_TYPE` - Not implemented

**Action**: Edit config.yaml.template to remove these 9 parameters

#### Task 1.2: Rename PER_SECTOR → USE_SECTOR_MODEL in regressor.py
**File**: `src/training/regressor.py`
**Current**: `PER_SECTOR = False` (hardcoded)
**New**: Read `USE_SECTOR_MODEL` from ml_config
**Lines to modify**: Search for "PER_SECTOR" occurrences

**Change**:
```python
# Before
PER_SECTOR = False

# After
self.use_sector_model = self.ml_config.get('USE_SECTOR_MODEL', 'N') == 'Y'
```

---

### Phase 2: Add Missing Config Parameters (Quick Wins)
**Goal**: Add hardcoded preprocessing flags to config
**Priority**: HIGH - Enables consistent preprocessing
**Estimated complexity**: LOW

#### Task 2.1: Add USE_WINSORIZATION to config
**File**: `config/conf.yaml.template`
**Section**: Under `ML:` or `FEATURES:`
**Add**:
```yaml
FEATURES:
  # Preprocessing flags
  USE_WINSORIZATION: Y  # Y = Apply winsorization to handle outliers
  USE_FEATURE_SELECTION: Y  # Y = Apply feature selection
```

#### Task 2.2: Read USE_WINSORIZATION in both scripts
**Files**: `src/training/regressor.py`, `src/backtest/ml_backtest.py`

**regressor.py changes**:
- Find hardcoded winsorization logic
- Replace with: `use_winsorization = ml_config.get('USE_WINSORIZATION', 'Y') == 'Y'`

**ml_backtest.py changes**:
- Find hardcoded winsorization logic
- Replace with: `self.use_winsorization = ml_config.get('USE_WINSORIZATION', 'Y') == 'Y'`

#### Task 2.3: Read USE_FEATURE_SELECTION in both scripts
**Similar to 2.2**, add feature selection flag reading

---

### Phase 3: Standardize Threshold Parameters (Critical)
**Goal**: Make DataProcessor read thresholds from config
**Priority**: HIGH - Eliminates hardcoded inconsistencies
**Estimated complexity**: MEDIUM

#### Task 3.1: Standardize SAME_VALUE_THRESHOLD (0.95)
**Files to modify**:
1. `src/training/data_processor.py`
   - Line ~same_value_threshold: float = 0.98 → Read from config or default 0.95
2. `src/training/regressor.py`
   - Line ~same_value_threshold = 0.95 → Remove hardcoded, pass from config
3. `src/scripts/debug/deep_infinite_diagnosis.py`
   - Line ~same_value_threshold = 0.95 → Read from config

**DataProcessor change**:
```python
def drop_low_variance_features(
    self,
    X: pd.DataFrame,
    same_value_threshold: Optional[float] = None  # Allow override
) -> Tuple[pd.DataFrame, List[str]]:
    if same_value_threshold is None:
        # Read from config with fallback
        same_value_threshold = float(self.ml_config.get('SAME_VALUE_THRESHOLD', 0.95))
```

#### Task 3.2: Implement MISSING_THRESHOLD in DataProcessor
**File**: `src/training/data_processor.py`
**Method**: `drop_sparse_columns`
**Current**: Hardcoded 0.5
**New**: Read from config (default 0.8)

```python
def drop_sparse_columns(
    self,
    X: pd.DataFrame,
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, List[str]]:
    if threshold is None:
        threshold = float(self.ml_config.get('MISSING_THRESHOLD', 0.8))
```

---

### Phase 4: Implement MIN_VOLUME_PERCENTILE (New Feature)
**Goal**: Filter out illiquid stocks
**Priority**: MEDIUM - Important for real trading but not affecting current models
**Estimated complexity**: MEDIUM

#### Task 4.1: Add volume filtering to make_mldata.py
**File**: `src/data/make_mldata.py`
**Location**: After loading stock data, before feature engineering

**Logic**:
1. Calculate volume percentile per period (quarter)
2. Filter stocks below MIN_VOLUME_PERCENTILE
3. Log number of stocks filtered

**Pseudocode**:
```python
min_volume_pct = ml_config.get('MIN_VOLUME_PERCENTILE', 50)
if min_volume_pct > 0:
    # Calculate volume percentile threshold
    volume_threshold = df.groupby('date')['volume'].transform(
        lambda x: x.quantile(min_volume_pct / 100)
    )
    # Filter
    before_count = len(df)
    df = df[df['volume'] >= volume_threshold]
    after_count = len(df)
    logger.info(f"Volume filter: {before_count} → {after_count} ({before_count - after_count} removed)")
```

#### Task 4.2: Verify filtering works in both regressor and ml_backtest
**Test**: Run both scripts and verify volume filtering is applied

---

### Phase 5: Unify USE_OPTUNA Behavior (Critical Architecture)
**Goal**: ml_backtest should load Optuna-optimized parameters
**Priority**: HIGH - Core to user's comparison goal
**Estimated complexity**: HIGH

#### Task 5.1: Verify Optuna parameter saving in regressor.py
**Check**: When USE_OPTUNA=Y, does regressor.py save parameters?
**Expected location**: `src/models/optuna/` or similar

#### Task 5.2: Implement parameter loading in ml_backtest.py
**File**: `src/backtest/ml_backtest.py`
**Logic**:
```python
if self.use_optuna:
    # Load saved Optuna parameters
    optuna_params = self._load_optuna_params()
    if optuna_params:
        self.logger.info("📦 Loaded Optuna-optimized parameters")
        self.model_params.update(optuna_params)
    else:
        self.logger.warning("⚠️  USE_OPTUNA=Y but no saved params found, using defaults")
```

**Implementation steps**:
1. Find where regressor.py saves Optuna results
2. Create `_load_optuna_params()` method in ml_backtest.py
3. Update ModelFactory to accept dynamic parameters
4. Test with USE_OPTUNA=Y and verify same params used

---

### Phase 6: Implement SECTOR_CONFIG in regressor.py (Medium Priority)
**Goal**: regressor.py uses sector-specific hyperparameters
**Priority**: MEDIUM - Consistency feature, not critical for basic operation
**Estimated complexity**: MEDIUM

#### Task 6.1: Read SECTOR_CONFIG in regressor.py
**File**: `src/training/regressor.py`
**Add** (similar to ml_backtest.py):
```python
self.sector_config = ml_config.get('SECTOR_CONFIG', {}) if self.use_sector_model else {}
```

#### Task 6.2: Use ModelFactory with sector config
**Current**: regressor.py creates models directly
**New**: Use ModelFactory.create_model() with sector parameter

**Example**:
```python
if self.use_sector_model:
    # Create sector-specific models
    for sector in sectors:
        sector_params = self.sector_config.get(sector, {})
        model = ModelFactory.create_model(
            model_type=sector_params.get('model', 'xgboost'),
            params=sector_params,
            logger=self.logger
        )
else:
    # Create unified model
    model = ModelFactory.create_model(
        model_type=self.ml_config.get('MODEL_TYPE', 'xgboost'),
        params=self.ml_config.get('MODEL_PARAMS', {}),
        logger=self.logger
    )
```

---

### Phase 7: Verification & Testing (Critical)
**Goal**: Ensure configuration changes work end-to-end
**Priority**: HIGH - Validate all changes
**Estimated complexity**: MEDIUM

#### Task 7.1: Create test configuration file
**File**: `config/test_config.yaml`
**Purpose**: Test configuration with all new parameters set

#### Task 7.2: Run regressor.py with test config
**Command**: `python src/training/regressor.py --config config/test_config.yaml`
**Verify**:
- USE_WINSORIZATION is read
- USE_FEATURE_SELECTION is read
- SAME_VALUE_THRESHOLD = 0.95 is used
- MISSING_THRESHOLD = 0.8 is used
- MIN_VOLUME_PERCENTILE filtering is applied
- USE_SECTOR_MODEL works with SECTOR_CONFIG
- USE_OPTUNA optimizes and saves parameters

#### Task 7.3: Run ml_backtest.py with same config
**Command**: `python src/backtest/ml_backtest.py --config config/test_config.yaml`
**Verify**:
- All above parameters are read identically
- USE_OPTUNA=Y loads saved parameters from regressor
- Results use same preprocessing, same model, same parameters
- Only difference: evaluation period (TEST_YEAR vs PERIODS)

#### Task 7.4: A/B Test - Change USE_OPTUNA
**Test scenario**:
1. Run with USE_OPTUNA=Y
   - regressor: Optimize + save
   - ml_backtest: Load + backtest
   - Compare: R²/MSE vs Sharpe/MDD
2. Run with USE_OPTUNA=N
   - regressor: Use defaults
   - ml_backtest: Use defaults
   - Compare: Same metrics
3. **Verify**: Changing one parameter affects BOTH systems identically

---

## 📊 Implementation Priority Matrix

| Phase | Priority | Complexity | Dependencies | Est. Time |
|-------|----------|------------|--------------|-----------|
| Phase 1 | HIGH | LOW | None | 30 min |
| Phase 2 | HIGH | LOW | Phase 1 | 45 min |
| Phase 3 | HIGH | MEDIUM | Phase 1 | 1.5 hours |
| Phase 4 | MEDIUM | MEDIUM | Phase 1 | 2 hours |
| Phase 5 | HIGH | HIGH | Phase 1, 2, 3 | 3 hours |
| Phase 6 | MEDIUM | MEDIUM | Phase 1, 2 | 2 hours |
| Phase 7 | HIGH | MEDIUM | All phases | 2 hours |

**Total estimated time**: ~12 hours of focused work

---

## 🔍 Success Criteria

### Functional Requirements
1. ✅ All 12 parameters from "KEEP & IMPLEMENT" list are working
2. ✅ All 8 unused parameters removed from config
3. ✅ No hardcoded values for thresholds (SAME_VALUE, MISSING)
4. ✅ USE_OPTUNA works in both regressor (save) and ml_backtest (load)
5. ✅ MIN_VOLUME_PERCENTILE filters stocks in both scripts
6. ✅ SECTOR_CONFIG works in both scripts when USE_SECTOR_MODEL=Y

### Consistency Requirements
1. ✅ Changing any config parameter affects BOTH regressor and ml_backtest
2. ✅ Same preprocessing pipeline in both scripts
3. ✅ Same model architecture in both scripts
4. ✅ Only intended differences: evaluation period (TRAIN/TEST_YEAR vs PERIODS)

### Testing Requirements
1. ✅ regressor.py runs without errors
2. ✅ ml_backtest.py runs without errors
3. ✅ A/B test shows parameter changes affect both systems
4. ✅ No configuration-related warnings or fallback to defaults

---

## 🚀 Recommended Execution Order

### Week 1: Foundational Cleanup
**Day 1**: Phase 1 (Config cleanup + naming)
**Day 2**: Phase 2 (Add preprocessing flags)
**Day 3**: Phase 3 (Standardize thresholds)

### Week 2: Feature Implementation
**Day 4-5**: Phase 4 (MIN_VOLUME_PERCENTILE)
**Day 6-7**: Phase 5 (USE_OPTUNA unification)

### Week 3: Polish & Verification
**Day 8-9**: Phase 6 (SECTOR_CONFIG in regressor)
**Day 10**: Phase 7 (Testing & verification)

---

## 📝 Implementation Notes

### Critical Files to Modify
1. `config/conf.yaml.template` - Remove 9 params, add 2 params
2. `src/training/regressor.py` - Read 8 new config values
3. `src/backtest/ml_backtest.py` - Read 3 new config values (already has 5)
4. `src/training/data_processor.py` - Read thresholds from config
5. `src/data/make_mldata.py` - Add volume filtering
6. `src/models/model_factory.py` - May need parameter passing updates

### Files to Check (but likely no changes)
1. `src/models/config.py` - Verify THRESHOLD_PERCENTILE usage
2. `src/scripts/debug/deep_infinite_diagnosis.py` - Update SAME_VALUE_THRESHOLD

### Backup Strategy
- Create feature branch for each phase
- Commit after each task completion
- Test before moving to next phase
- Keep Phase 0 analysis as reference

### Rollback Plan
- Each phase is independent (mostly)
- Can roll back to previous phase if issues found
- Git history provides safety net

---

## 🎯 Final Goal Verification

**User's Original Intent**:
> "모든 파라미터는 동일하게 바라보고 있어야 1)과 2)를 함께 보고 평가 자체가 가능함"
> (All parameters must be viewed identically so that 1) and 2) can be evaluated together)

**Success = Ability to**:
1. Edit config.yaml once
2. Run regressor.py → Get prediction accuracy (R², MSE)
3. Run ml_backtest.py → Get investment returns (Sharpe, MDD, profit)
4. **Compare fairly** because both used SAME preprocessing, SAME model, SAME parameters
5. Only difference: regressor tests on fixed TEST_YEAR, ml_backtest tests on walk-forward PERIODS

**Current State**: ❌ 94% inconsistency - Cannot compare fairly
**After Implementation**: ✅ 100% consistency - Fair comparison enabled

---

## 📌 Open Questions (For Future Consideration)

1. **THRESHOLD_PERCENTILE**: Still used by new models? (Investigate during Phase 7)
2. **ENSEMBLE**: When to implement? (Future feature, not current scope)
3. **Position sizing**: When to implement TOTAL_ASSET? (Future feature)
4. **Additional sectors**: When to populate Healthcare, Consumer, Industrial in SECTOR_CONFIG? (When data available)

---

**Next Step**: Begin Phase 1 implementation 🚀
