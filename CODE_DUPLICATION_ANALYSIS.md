# Comprehensive Code Duplication Analysis
## regressor.py vs ml_backtest.py

**Analysis Date:** 2025-12-01
**Files Analyzed:**
- `/home/user/Quant/regressor.py` (1835 lines)
- `/home/user/Quant/Quant-refactoring/src/backtest/ml_backtest.py` (926 lines)

---

## Executive Summary

**Total Duplications Found:** 15 major categories
**Critical Issues:** 2 (different binary thresholds, inconsistent preprocessing)
**High Priority:** 8
**Medium Priority:** 4
**Low Priority:** 1

**Already Refactored:** 3 areas (via DataProcessor and DataSchema)

---

## 1. BINARY TARGET THRESHOLD - CRITICAL INCONSISTENCY ⚠️

### Description
Different thresholds used for converting regression targets to binary classification targets.

### Locations
**regressor.py:**
- Line 803: `y_train_binary = (self.y_train > -0.02).astype(int)`
- Line 804: `y_test_binary = (self.y_test > -0.02).astype(int)`
- Line 861: `cur_y_train_binary = (self.sector_y_train[sec] > -0.02).astype(int)`
- Line 917-918: Same pattern
- Line 1073: Same pattern
- Line 1272: Same pattern
- Line 1461: Same pattern

**ml_backtest.py:**
- Line 282: `y_binary = (y > 0).astype(int)`
- Line 383: `y_binary = (y > 0).astype(int)`

### Similarity
**Different implementations** - This is a critical inconsistency!
- regressor.py uses threshold of **-0.02** (2% loss tolerance)
- ml_backtest.py uses threshold of **0** (strict profit/loss)

### Impact
- Models trained with different thresholds will have different decision boundaries
- Backtest results may not reflect actual model behavior
- Inconsistent evaluation metrics between training and backtesting

### Recommendation
**HIGH PRIORITY** - Create unified method in DataProcessor:
```python
@staticmethod
def create_binary_target(y: pd.Series, threshold: float = -0.02) -> pd.Series:
    """
    Convert regression target to binary classification target.

    Args:
        y: Regression target values
        threshold: Threshold for binary classification (default: -0.02)

    Returns:
        Binary target (1 for above threshold, 0 otherwise)
    """
    return (y > threshold).astype(int)
```

**Estimated Impact:** High - Fixes critical consistency issue

---

## 2. VOLUME-BASED FILTERING - DUPLICATE LOGIC

### Description
Filtering stocks by volume_mul_price metric to select liquid stocks.

### Locations
**regressor.py only:**
- Lines 436-441: `dataload()` - filters top 50% by volume
- Lines 615-618: `last_dataload()` - filters by saved threshold
- Lines 710-713: `val_dataload()` - filters by saved threshold
- Lines 1660-1662: `latest_prediction()` - filters by saved threshold

**ml_backtest.py:**
- NOT PRESENT (potential missing feature)

### Similarity
**Identical logic** repeated 4 times in regressor.py

### Code Pattern
```python
# Pattern 1 (training - compute threshold)
symbol_means = df.groupby('symbol')['volume_mul_price'].mean().reset_index()
top_40_percent_value = symbol_means['volume_mul_price'].quantile(0.50)
top_symbols = symbol_means.nlargest(int(len(symbol_means) * 0.50), 'volume_mul_price')
df = df[df['symbol'].isin(top_symbols['symbol'])]

# Pattern 2 (test/prediction - use saved threshold)
symbol_means = df.groupby('symbol')['volume_mul_price'].mean().reset_index()
top_symbols = symbol_means[symbol_means['volume_mul_price'] >= self.final_top_40_percent_average]
df = df[df['symbol'].isin(top_symbols['symbol'])]
```

### Recommendation
**HIGH PRIORITY** - Create method in DataProcessor:
```python
@staticmethod
def filter_by_liquidity(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    top_pct: float = 0.50
) -> Tuple[pd.DataFrame, float]:
    """
    Filter stocks by volume*price liquidity metric.

    Args:
        df: DataFrame with 'symbol' and 'volume_mul_price' columns
        threshold: Minimum liquidity threshold (if None, compute from top_pct)
        top_pct: Top percentage to keep (default 0.50 for top 50%)

    Returns:
        (filtered_df, threshold_used)
    """
    symbol_means = df.groupby('symbol')['volume_mul_price'].mean().reset_index()

    if threshold is None:
        threshold = symbol_means['volume_mul_price'].quantile(1.0 - top_pct)

    top_symbols = symbol_means[symbol_means['volume_mul_price'] >= threshold]
    filtered_df = df[df['symbol'].isin(top_symbols['symbol'])]

    return filtered_df, threshold
```

**Estimated Impact:** High - Eliminates 4 duplications, ensures ml_backtest uses same filtering

---

## 3. DROP_MANY_NAN_ROW - DUPLICATE METHOD

### Description
Drops rows with excessive NaN values based on threshold.

### Locations
**regressor.py only:**
- Lines 413-424: Method definition
- Line 446: Called with threshold=0.6 (40% NaN tolerance)
- Line 494: Called with threshold=0.95 (5% NaN tolerance)
- Line 623: Called with threshold=0.95
- Line 718: Called with threshold=0.95
- Line 1675: Called with default threshold=0.9

**ml_backtest.py:**
- NOT PRESENT

### Similarity
**Identical method** called 5 times with varying thresholds

### Code
```python
def drop_many_nan_row(self, df, threshold=DROP_NAN_ROW_THRESHOLD):
    # threshold = 0.8 means drop rows with >20% NaN
    df['nan_count_per_row'] = df.isnull().sum(axis=1)
    filtered_row = df['nan_count_per_row'] < int(len(df.columns)*(1-threshold))
    df = df.loc[filtered_row,:]
    return df
```

### Recommendation
**MEDIUM PRIORITY** - Move to DataProcessor:
```python
@staticmethod
def drop_rows_with_excessive_nan(
    df: pd.DataFrame,
    max_nan_ratio: float = 0.1
) -> pd.DataFrame:
    """
    Drop rows where NaN ratio exceeds threshold.

    Args:
        df: Input DataFrame
        max_nan_ratio: Maximum allowed NaN ratio per row (default 0.1 = 10%)

    Returns:
        DataFrame with high-NaN rows removed
    """
    nan_count_per_row = df.isnull().sum(axis=1)
    max_allowed_nans = int(len(df.columns) * max_nan_ratio)

    filtered_df = df[nan_count_per_row <= max_allowed_nans]

    logger.info(f"Dropped {len(df) - len(filtered_df)} rows with >{max_nan_ratio*100}% NaN")

    return filtered_df
```

**Estimated Impact:** Medium - Standardizes NaN handling, adds to ml_backtest

---

## 4. COLUMN DROPPING BY STATISTICS - UNIQUE TO REGRESSOR

### Description
Removes columns with high missing ratios or same-value ratios.

### Locations
**regressor.py only:**
- Lines 460-492: Column dropping logic in `dataload()`
- Saves to `drop_col_list.json`
- Loaded in: lines 607-608, 702-703, 1645-1646

**ml_backtest.py:**
- NOT PRESENT (DataSchema handles column exclusion differently)

### Similarity
**Unique to regressor.py** - Not duplicated but should be evaluated for unification

### Constants Used
```python
missing_threshold = 0.8  # Drop if >80% missing
same_value_threshold = 0.98  # Drop if >98% same value
```

### Code Pattern
```python
columns_to_drop = []
for col in df.columns:
    missing_ratio = df[col].isna().mean()
    if missing_ratio > missing_threshold:
        columns_to_drop.append(col)
    else:
        top_value_ratio = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_value_ratio > same_value_threshold:
            columns_to_drop.append(col)

df = df.drop(columns=columns_to_drop)
```

### Recommendation
**MEDIUM PRIORITY** - Add to DataProcessor as optional feature engineering step:
```python
@staticmethod
def drop_uninformative_columns(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.8,
    max_same_value_ratio: float = 0.98,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with excessive missing or constant values.

    Args:
        df: Input DataFrame
        max_missing_ratio: Max allowed missing ratio (default 0.8)
        max_same_value_ratio: Max allowed same-value ratio (default 0.98)
        exclude_cols: Columns to never drop (e.g., targets)

    Returns:
        (cleaned_df, dropped_columns)
    """
    exclude_cols = exclude_cols or []
    columns_to_drop = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        missing_ratio = df[col].isna().mean()
        if missing_ratio > max_missing_ratio:
            columns_to_drop.append(col)
            continue

        value_counts = df[col].value_counts(normalize=True, dropna=False)
        if len(value_counts) > 0:
            top_value_ratio = value_counts.iloc[0]
            if top_value_ratio > max_same_value_ratio:
                columns_to_drop.append(col)

    cleaned_df = df.drop(columns=columns_to_drop)

    logger.info(f"Dropped {len(columns_to_drop)} uninformative columns")

    return cleaned_df, columns_to_drop
```

**Estimated Impact:** Medium - Adds feature engineering consistency

---

## 5. INFINITE/NaN VALUE CHECKING - DUPLICATE LOGGING

### Description
Extensive diagnostic logging of infinite, NaN, and large values in features.

### Locations
**regressor.py:**
- Lines 530-575: In `dataload()` for training data
- Lines 649-684: In `last_dataload()` - IDENTICAL CODE
- Lines 746-781: In `val_dataload()` - IDENTICAL CODE

**ml_backtest.py:**
- Uses `DataProcessor.remove_infinite_values()` and `DataProcessor.replace_infinite_with_nan()` (already refactored)

### Similarity
**Identical code block** repeated 3 times (45 lines each = 135 lines of duplication)

### Code Pattern
```python
numeric_x_train = self.x_train.select_dtypes(include=[np.number])

column_inf_counts_train = np.isinf(numeric_x_train).sum()
column_nan_counts_train = np.isnan(numeric_x_train).sum()
column_large_counts_train = (numeric_x_train.abs() > 1e9).sum()

row_inf_counts_train = np.isinf(numeric_x_train).sum(axis=1)
row_nan_counts_train = np.isnan(numeric_x_train).sum(axis=1)
row_large_counts_train = (numeric_x_train.abs() > 1e9).sum(axis=1)

# ... filtering and printing logic ...
```

### Recommendation
**HIGH PRIORITY** - Create diagnostic method in DataProcessor:
```python
@staticmethod
def log_data_quality_issues(
    df: pd.DataFrame,
    name: str = "Data",
    large_value_threshold: float = 1e9
) -> Dict[str, Any]:
    """
    Log and return data quality diagnostics.

    Args:
        df: DataFrame to check
        name: Name for logging context
        large_value_threshold: Threshold for "large" values

    Returns:
        Dictionary with diagnostics
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # Column-wise counts
    col_inf = np.isinf(numeric_df).sum()
    col_nan = np.isnan(numeric_df).sum()
    col_large = (numeric_df.abs() > large_value_threshold).sum()

    # Row-wise counts
    row_inf = np.isinf(numeric_df).sum(axis=1)
    row_nan = np.isnan(numeric_df).sum(axis=1)
    row_large = (numeric_df.abs() > large_value_threshold).sum(axis=1)

    logger.info(f"=== {name} - Data Quality Report ===")
    logger.info(f"Columns with inf: {(col_inf > 0).sum()}")
    logger.info(f"Columns with NaN: {(col_nan > 0).sum()}")
    logger.info(f"Columns with large values: {(col_large > 0).sum()}")
    logger.info(f"Rows with issues: {(row_inf > 0).sum() + (row_nan > 0).sum()}")

    return {
        'col_inf': col_inf[col_inf > 0],
        'col_nan': col_nan[col_nan > 0],
        'col_large': col_large[col_large > 0],
        'row_inf': row_inf[row_inf > 0],
        'row_nan': row_nan[row_nan > 0],
        'row_large': row_large[row_large > 0]
    }
```

**Estimated Impact:** High - Eliminates 135 lines of duplication

---

## 6. LARGE VALUE REPLACEMENT - DUPLICATE FUNCTION

### Description
Replaces values >1e9 with XGBOOST_MAX_VALUE to prevent overflow.

### Locations
**regressor.py:**
- Lines 577-582: Function definition in `dataload()`
- Line 582: Applied with `applymap(replace_large_values)`
- Lines 686-691: IDENTICAL function in `last_dataload()`
- Line 691: Applied
- Lines 784-789: IDENTICAL function in `val_dataload()`
- Line 789: Applied

**ml_backtest.py:**
- Uses `DataProcessor.remove_infinite_values()` instead (already refactored)

### Similarity
**Identical function** defined 3 times (18 lines of duplication)

### Code
```python
def replace_large_values(x):
    if isinstance(x, (int, float)):
        return XGBOOST_MAX_VALUE if x > 1e9 else x
    return x

df = df.applymap(replace_large_values)
```

### Recommendation
**HIGH PRIORITY** - Already handled by DataProcessor, remove from regressor.py:
- Replace with `DataProcessor.replace_infinite_with_nan()` or similar
- Or add specific method if needed:

```python
@staticmethod
def clip_large_values(
    df: pd.DataFrame,
    threshold: float = 1e9,
    replacement: Optional[float] = None
) -> pd.DataFrame:
    """
    Replace extremely large values to prevent numerical overflow.

    Args:
        df: Input DataFrame
        threshold: Values above this are considered too large
        replacement: Value to use (default: np.finfo(np.float32).max)

    Returns:
        DataFrame with large values clipped
    """
    if replacement is None:
        replacement = np.finfo(np.float32).max

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mask = df[col].abs() > threshold
        if mask.any():
            df.loc[mask, col] = replacement
            logger.warning(f"Clipped {mask.sum()} large values in {col}")

    return df
```

**Estimated Impact:** High - Eliminates 18 lines, standardizes with ml_backtest

---

## 7. OUTLIER CLIPPING BY QUANTILE - DUPLICATE LOGIC

### Description
Clips features and targets to percentile ranges to handle outliers.

### Locations
**regressor.py only:**

**Feature Clipping (2-98 percentile):**
- Lines 540-549: Clips X features, saves to `train_clip_bounds.json`
- Lines 641-646: Loads and applies clip bounds to `last_x_train`
- Lines 739-744: Loads and applies clip bounds to `x_test`

**Y-value Clipping (1-97 percentile):**
- Lines 518-528: Clips y_train, saves bounds to txt files
- Lines 599-605: Loads bounds from files
- Line 638: Applies clip to `last_y_train`

**ml_backtest.py:**
- NOT PRESENT

### Similarity
**Identical logic** repeated 3 times for features, 2 times for targets

### Code Pattern
```python
# Feature clipping (train)
train_clip_bounds = {}
features = [col for col in self.x_train.columns if col != 'sector']
for col in features:
    lower, upper = self.x_train[col].quantile([0.02, 0.98])
    self.x_train[col] = self.x_train[col].clip(lower, upper)
    train_clip_bounds[col] = (lower, upper)
json.dump(train_clip_bounds, f, indent=2)

# Feature clipping (test - load and apply)
with open("train_clip_bounds.json", "r") as f:
    train_clip_bounds = json.load(f)
for col, (lower, upper) in train_clip_bounds.items():
    x_test[col] = x_test[col].clip(lower, upper)

# Y clipping
lower_bound = np.percentile(self.y_train.to_numpy(), 1)
upper_bound = np.percentile(self.y_train.to_numpy(), 97)
self.y_train = self.y_train.clip(lower=lower_bound, upper=upper_bound)
```

### Recommendation
**HIGH PRIORITY** - Add to DataProcessor:
```python
@staticmethod
def fit_outlier_clipper(
    df: pd.DataFrame,
    lower_percentile: float = 0.02,
    upper_percentile: float = 0.98,
    exclude_cols: Optional[List[str]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute clipping bounds for outlier removal.

    Args:
        df: Training data
        lower_percentile: Lower percentile (default 0.02 = 2nd percentile)
        upper_percentile: Upper percentile (default 0.98 = 98th percentile)
        exclude_cols: Columns to skip (e.g., 'sector')

    Returns:
        Dictionary mapping column names to (lower, upper) bounds
    """
    exclude_cols = exclude_cols or []
    clip_bounds = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in exclude_cols:
            continue

        lower = df[col].quantile(lower_percentile)
        upper = df[col].quantile(upper_percentile)
        clip_bounds[col] = (float(lower), float(upper))

    logger.info(f"Computed clip bounds for {len(clip_bounds)} features")

    return clip_bounds

@staticmethod
def apply_outlier_clipper(
    df: pd.DataFrame,
    clip_bounds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Apply pre-computed clipping bounds.

    Args:
        df: Data to clip
        clip_bounds: Dictionary from fit_outlier_clipper()

    Returns:
        Clipped DataFrame
    """
    df = df.copy()

    for col, (lower, upper) in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower, upper)

    return df
```

**Estimated Impact:** High - Eliminates ~50 lines, adds to ml_backtest

---

## 8. SECTOR COLUMN PREPROCESSING - DUPLICATE PATTERN

### Description
Fills missing sector values and converts to categorical type.

### Locations
**regressor.py:**
- Lines 583-584: In `dataload()` for `x_train`
- Lines 692-693: In `last_dataload()` for `last_x_train`
- Lines 791-792: In `val_dataload()` for `x_test`
- Lines 1064-1065: In `evaluation()` for test data
- Lines 1681-1682: In `latest_prediction()` for latest input

**ml_backtest.py:**
- NOT EXPLICIT (sector handled but not with this pattern)

### Similarity
**Identical 2-line pattern** repeated 5 times

### Code
```python
df['sector'] = df['sector'].fillna("_missing_").astype(str)
df['sector'] = df['sector'].astype('category')
```

### Recommendation
**MEDIUM PRIORITY** - Add utility method:
```python
@staticmethod
def prepare_sector_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare sector column for model training.

    Fills missing values and converts to categorical type for
    efficient handling in tree-based models.

    Args:
        df: DataFrame with 'sector' column

    Returns:
        DataFrame with processed sector column
    """
    if 'sector' not in df.columns:
        logger.warning("No 'sector' column found, skipping sector preprocessing")
        return df

    df = df.copy()
    df['sector'] = df['sector'].fillna("_missing_").astype(str)
    df['sector'] = df['sector'].astype('category')

    logger.debug(f"Prepared sector column: {df['sector'].nunique()} unique sectors")

    return df
```

**Estimated Impact:** Medium - Eliminates 10 lines across multiple methods

---

## 9. THRESHOLD-BASED BINARY CLASSIFICATION - DUPLICATE LOGIC

### Description
Converts probability predictions to binary using percentile-based threshold.

### Locations
**regressor.py only:**
- Lines 1094-1096: In `evaluation()` loop
- Lines 1295-1297: In sector evaluation loop
- Lines 1710-1712: In `latest_prediction()` loop

**ml_backtest.py:**
- NOT PRESENT (uses probability * return for scoring)

### Similarity
**Identical 3-line pattern** repeated 3 times

### Code
```python
threshold = np.percentile(y_probs, THRESHOLD)  # THRESHOLD = 92
y_predict_binary = (y_probs > threshold).astype(int)
logging.info(f"20% positive threshold == {threshold}")  # Actually top 8%
```

### Recommendation
**MEDIUM PRIORITY** - Extract to method:
```python
@staticmethod
def threshold_probabilities(
    probabilities: np.ndarray,
    percentile: float = 92.0
) -> Tuple[np.ndarray, float]:
    """
    Convert probabilities to binary predictions using percentile threshold.

    Args:
        probabilities: Predicted probabilities
        percentile: Percentile for threshold (default 92 = top 8%)

    Returns:
        (binary_predictions, threshold_used)
    """
    threshold = np.percentile(probabilities, percentile)
    binary = (probabilities > threshold).astype(int)

    logger.info(f"Threshold at {percentile}th percentile: {threshold:.4f}")
    logger.info(f"Positive class ratio: {binary.mean():.2%}")

    return binary, threshold
```

**Estimated Impact:** Low - Small duplication but improves clarity

---

## 10. FEATURE IMPORTANCE EXTRACTION - DUPLICATE PATTERN

### Description
Extracts and saves feature importances from trained models.

### Locations
**regressor.py:**
- Lines 848-853: In `train()` for regression models
- Lines 1565-1570: In `latest_prediction()` for classifier
- Lines 1590-1595: In `latest_prediction()` for regressor

**ml_backtest.py:**
- NOT PRESENT (models saved but importance not extracted)

### Similarity
**Similar pattern** repeated 3 times

### Code Pattern
```python
ftr_importances_values = model.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=self.x_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
ftr_importances = ftr_importances.sort_values(ascending=False)
ftr_importances.to_csv(self.MODEL_SAVE_PATH + f'model_{i}_importances.csv')
logging.info(ftr_top20)
```

### Recommendation
**LOW PRIORITY** - Extract to utility method:
```python
@staticmethod
def save_feature_importances(
    model: Any,
    feature_names: List[str],
    output_path: str,
    top_k: int = 20
) -> pd.Series:
    """
    Extract and save feature importances from tree-based model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        output_path: Path to save CSV
        top_k: Number of top features to log

    Returns:
        Series of feature importances (sorted)
    """
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    importances.to_csv(output_path)

    logger.info(f"Top {top_k} features:")
    logger.info(importances.head(top_k))

    return importances
```

**Estimated Impact:** Low - Minor duplication, mostly organizational

---

## 11. TOP-K SELECTION PATTERN - DUPLICATE PATTERN

### Description
Sorts predictions and selects top K stocks.

### Locations
**regressor.py:**
- Line 1176: `top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]`
- Line 1351: Same pattern
- Line 1756: Same pattern

**ml_backtest.py:**
- Lines 567-571: `_select_top_k()` method (similar but method-based)

### Similarity
**Similar inline pattern** vs **dedicated method**

### Recommendation
**LOW PRIORITY** - Already better in ml_backtest with dedicated method. Consider using same approach in regressor.

**Estimated Impact:** Low - Pattern is simple and clear

---

## 12. DATA LOADING FILE FORMATS - DIFFERENT APPROACHES

### Description
Different file formats used for data loading.

### Locations
**regressor.py:**
- Lines 379-401: Loads CSV files
- `pd.read_csv(fpath, low_memory=False)`
- Iterates through years/quarters building file paths

**ml_backtest.py:**
- Lines 143-178: Loads Parquet files
- `pd.read_parquet(file_path)`
- Iterates through years/quarters building file paths

### Similarity
**Different file formats** but similar iteration logic

### Recommendation
**MEDIUM PRIORITY** - Standardize on Parquet for both:
- Parquet is more efficient (compression, faster I/O)
- Already used in refactored codebase
- Update regressor.py to use Parquet files

**Estimated Impact:** Medium - Performance improvement, consistency

---

## 13. FILE I/O FOR METADATA - DUPLICATE PATTERNS

### Description
Saving/loading preprocessing metadata to text and JSON files.

### Locations
**regressor.py:**

**Writes:**
- Lines 451-456: `final_top_40_percent_average.txt`
- Lines 485-488: `drop_col_list.json`
- Lines 520-523: `y_lower_bound.txt`, `y_upper_bound.txt`
- Lines 547-549: `train_clip_bounds.json`

**Reads:**
- Lines 595-608: Loads all 4 files in `last_dataload()`
- Lines 698-703: Loads 2 files in `val_dataload()`
- Lines 1641-1646: Loads 2 files in `latest_prediction()`

**ml_backtest.py:**
- NOT PRESENT (models saved but not preprocessing metadata)

### Similarity
**Repeated file I/O patterns** for same metadata

### Recommendation
**HIGH PRIORITY** - Create unified metadata management:
```python
class PreprocessingMetadata:
    """Manages preprocessing parameters for consistency across train/test."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.metadata_file = self.model_path / "preprocessing_metadata.json"

    def save(self, **kwargs) -> None:
        """Save all preprocessing parameters to single JSON file."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved preprocessing metadata: {list(kwargs.keys())}")

    def load(self) -> Dict[str, Any]:
        """Load preprocessing parameters."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_file}")

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        logger.info(f"Loaded preprocessing metadata from {metadata.get('timestamp')}")

        return metadata

# Usage:
metadata = PreprocessingMetadata(model_path)
metadata.save(
    liquidity_threshold=liquidity_threshold,
    drop_columns=drop_col_list,
    y_clip_bounds=(lower_bound, upper_bound),
    feature_clip_bounds=train_clip_bounds
)

# Later:
params = metadata.load()
```

**Estimated Impact:** High - Simplifies metadata management, prevents file sprawl

---

## 14. CONSTANTS NOT CENTRALIZED - SCATTERED DEFINITIONS

### Description
Magic numbers and thresholds defined inline or at module level.

### Locations
**regressor.py:**
- Line 44: `XGBOOST_MAX_VALUE = np.finfo(np.float32).max`
- Line 45: `DROP_NAN_ROW_THRESHOLD = 0.9`
- Line 43: `THRESHOLD = 92`
- Line 460: `missing_threshold = 0.8`
- Line 461: `same_value_threshold = 0.98`
- Various hardcoded values: `0.6`, `0.95`, `0.02`, `0.98`, `-0.02`, etc.

**ml_backtest.py:**
- Various hardcoded values in DataProcessor calls

### Similarity
**Scattered constants** without centralized configuration

### Recommendation
**MEDIUM PRIORITY** - Create constants module:
```python
# src/constants/preprocessing_constants.py

class PreprocessingConstants:
    """Central repository for preprocessing thresholds and parameters."""

    # Data quality thresholds
    MAX_NAN_RATIO_PER_ROW = 0.1  # 10% max NaN per row
    MAX_MISSING_RATIO_PER_COLUMN = 0.8  # 80% max missing per column
    MAX_SAME_VALUE_RATIO = 0.98  # 98% max same value

    # Numerical handling
    LARGE_VALUE_THRESHOLD = 1e9
    MAX_FLOAT32 = np.finfo(np.float32).max

    # Outlier clipping
    FEATURE_LOWER_PERCENTILE = 0.02  # 2nd percentile
    FEATURE_UPPER_PERCENTILE = 0.98  # 98th percentile
    TARGET_LOWER_PERCENTILE = 0.01  # 1st percentile
    TARGET_UPPER_PERCENTILE = 0.97  # 97th percentile

    # Binary classification
    BINARY_THRESHOLD = -0.02  # For converting regression to binary
    PREDICTION_THRESHOLD_PERCENTILE = 92  # Top 8% positive

    # Liquidity filtering
    LIQUIDITY_TOP_PERCENTAGE = 0.50  # Top 50% by volume*price
```

**Estimated Impact:** Medium - Better maintainability, easier tuning

---

## 15. FILLNA(0) - INCONSISTENT NaN HANDLING

### Description
`fillna(0)` used in some places, DataProcessor in others.

### Locations
**regressor.py:**
- Line 896: `self.x_train_dl = self.x_train_dl.fillna(0)` (DL preprocessing)
- Line 909: `self.x_test_dl = self.x_test_dl.fillna(0)`
- Line 1010, 1070: Similar patterns
- Lines 495, 721: Commented out

**ml_backtest.py:**
- Line 276: Uses `DataProcessor.handle_nan(X, y, method='fillna', fill_value=0)`
- Line 473: `X = X.fillna(0)` in prediction
- Line 533: Same in sector prediction

### Similarity
**Inconsistent** - Sometimes DataProcessor, sometimes inline

### Recommendation
**MEDIUM PRIORITY** - Standardize on DataProcessor:
```python
# Replace all inline fillna(0) with:
X = DataProcessor.handle_nan(X, y=None, method='fillna', fill_value=0)[0]

# Or for just X:
X = X.fillna(0)  # Keep simple for prediction pipeline
```

**Estimated Impact:** Low - Mostly stylistic, but improves consistency

---

## SUMMARY TABLE

| # | Duplication | Priority | Files | Est. Lines Saved | Impact |
|---|-------------|----------|-------|------------------|--------|
| 1 | Binary threshold inconsistency | **CRITICAL** | Both | N/A | **HIGH** - Fixes bug |
| 2 | Volume filtering | HIGH | regressor | ~40 | HIGH - Adds to ml_backtest |
| 3 | drop_many_nan_row | MEDIUM | regressor | ~15 | Medium - Standardizes |
| 4 | Column dropping stats | MEDIUM | regressor | ~30 | Medium - Optional feature |
| 5 | Inf/NaN checking | HIGH | regressor | ~135 | HIGH - Big cleanup |
| 6 | Large value replacement | HIGH | regressor | ~18 | HIGH - Already in DataProcessor |
| 7 | Outlier clipping | HIGH | regressor | ~50 | HIGH - Adds to ml_backtest |
| 8 | Sector preprocessing | MEDIUM | regressor | ~10 | Medium - Simple utility |
| 9 | Threshold classification | MEDIUM | regressor | ~9 | Low - Small duplication |
| 10 | Feature importance | LOW | regressor | ~15 | Low - Organizational |
| 11 | Top-K selection | LOW | Both | ~5 | Low - Pattern OK |
| 12 | File formats | MEDIUM | regressor | N/A | Medium - Performance |
| 13 | Metadata I/O | HIGH | regressor | ~40 | HIGH - Simplifies |
| 14 | Constants | MEDIUM | Both | N/A | Medium - Maintainability |
| 15 | fillna(0) | MEDIUM | Both | ~5 | Low - Consistency |

**Total Estimated Lines to be Eliminated:** ~370+ lines
**Total Duplicated Pattern Instances:** 45+

---

## RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Critical Fixes (Week 1)
1. **Fix binary threshold inconsistency** (#1)
   - Decide on unified threshold (-0.02 or 0)
   - Update ml_backtest.py to match regressor.py OR vice versa
   - Add configuration parameter for threshold

### Phase 2: High-Priority Refactoring (Week 2-3)
2. **Infinite/NaN checking** (#5) - 135 lines saved
3. **Large value replacement** (#6) - 18 lines saved
4. **Outlier clipping** (#7) - 50 lines saved
5. **Volume filtering** (#2) - 40 lines saved
6. **Metadata I/O** (#13) - 40 lines saved

### Phase 3: Medium-Priority Standardization (Week 4)
7. **drop_many_nan_row** (#3)
8. **Column dropping stats** (#4)
9. **Sector preprocessing** (#8)
10. **File format standardization** (#12)
11. **Constants centralization** (#14)
12. **fillna(0) standardization** (#15)

### Phase 4: Low-Priority Cleanup (Week 5)
13. **Threshold classification** (#9)
14. **Feature importance** (#10)
15. **Top-K selection** (#11)

---

## TESTING REQUIREMENTS

After each refactoring:
1. **Unit tests** for new DataProcessor methods
2. **Integration tests** comparing old vs new outputs
3. **Regression tests** ensuring predictions unchanged (unless fixing bugs)
4. **Performance tests** for file I/O changes

---

## NOTES

- **Already Refactored (Good!):**
  - Column exclusion via DataSchema ✅
  - Infinite value handling via DataProcessor ✅
  - Feature scaling via DataProcessor ✅
  - Model creation via ModelFactory ✅

- **Critical Finding:**
  - Different binary thresholds between training and backtesting is a **bug** that could affect results

- **Biggest Wins:**
  - Inf/NaN checking refactoring: 135 lines
  - Outlier clipping: 50 lines
  - Volume filtering: 40 lines
  - Metadata I/O: 40 lines
  - **Total: ~265 lines from top 4 alone**

---

**End of Analysis**
