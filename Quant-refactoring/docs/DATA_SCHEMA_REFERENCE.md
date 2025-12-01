# DataSchema API Reference

**Module:** `src.constants.data_schema`
**Version:** 1.0.0
**Date:** 2025-12-01

---

## Overview

`DataSchema` provides a **single source of truth** for all column definitions used across the ML pipeline. It ensures that `regressor.py`, `ml_backtest.py`, and all other components use identical column names and classifications.

---

## Quick Start

```python
from src.constants.data_schema import DataSchema

# Get columns to exclude from features
excluded_cols = DataSchema.get_excluded_cols()

# Get feature columns from a dataframe
feature_cols = DataSchema.get_feature_cols(df)

# Get regression target name
target_col = DataSchema.REGRESSION_TARGET  # 'price_dev_subavg'

# Separate features and target
X = df[feature_cols]
y = df[target_col]
```

---

## Class: `DataSchema`

### Column Categories

#### 1. Metadata Columns

Columns that identify stocks and company information:

```python
DataSchema.METADATA_COLS = [
    "symbol",              # Stock ticker (e.g., AAPL, TSLA)
    "exchangeShortName",   # Exchange (e.g., NASDAQ, NYSE)
    "type",                # Security type ('stock', 'ETF')
    "delistedDate",        # Delisting date
    "industry",            # Industry classification
    "ipoDate",             # IPO date
    "sector",              # Sector (e.g., Technology, Financial)
]
```

**Usage:**
```python
# Check if column is metadata
if col in DataSchema.METADATA_COLS:
    print(f"{col} is metadata, not a feature")
```

---

#### 2. Date Columns

Temporal information columns:

```python
DataSchema.DATE_COLS = [
    "rebalance_date",      # Portfolio rebalancing date
    "report_date",         # Financial report date
    "fillingDate",         # Filing date (SEC)
    "fillingDate_x",       # Filing date variant
    "acceptedDate",        # Report acceptance date
    "start_date",          # Period start
    "year_period",         # Year period identifier
    "date",                # Generic date
]
```

**Usage:**
```python
# Convert all date columns to datetime
for col in DataSchema.DATE_COLS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
```

---

#### 3. Price/Volume Columns

Market data columns (excluded to prevent future leakage):

```python
DataSchema.PRICE_VOLUME_COLS = [
    "price",               # Stock price
    "volume",              # Trading volume
    "marketCap",           # Market capitalization
    "price_diff",          # Price difference
    "volume_mul_price",    # Volume × Price
]
```

**Why excluded?**
These contain future information or are used only for calculating targets/returns.

---

#### 4. Target Variable Columns

Labels for ML models:

```python
DataSchema.TARGET_COLS = [
    "price_dev",              # Binary target (up/down)
    "price_dev_subavg",       # Main regression target
    "sec_price_dev_subavg",   # Sector-adjusted target
    "price_dev_prediction",   # Model prediction output
    "price_diff_prediction",  # Price diff prediction
    "price_diff_3month",      # 3-month price difference
    "price_diff_6month",      # 6-month price difference
]
```

---

### Target Variable Names

#### Primary Targets

```python
# Main regression target
DataSchema.REGRESSION_TARGET = "price_dev_subavg"

# Binary classification target
DataSchema.CLASSIFICATION_TARGET = "price_dev"

# Sector-specific regression target
DataSchema.SECTOR_REGRESSION_TARGET = "sec_price_dev_subavg"
```

**Usage:**
```python
# Get regression target
y_reg = df[DataSchema.REGRESSION_TARGET]

# Get classification target
y_cls = df[DataSchema.CLASSIFICATION_TARGET]

# Create binary labels
y_binary = (y_reg > 0).astype(int)
```

---

## Class Methods

### `get_excluded_cols()`

Returns all columns that should be **excluded** from model features.

**Signature:**
```python
@classmethod
def get_excluded_cols(cls) -> List[str]
```

**Returns:**
- `List[str]`: Combined list of metadata, date, price/volume, and target columns

**Example:**
```python
excluded = DataSchema.get_excluded_cols()
# Returns: ['symbol', 'exchangeShortName', ..., 'price_dev_subavg']

features = [col for col in df.columns if col not in excluded]
```

**Total Count:** 27 columns excluded

---

### `get_excluded_cols_set()`

Returns excluded columns as a **set** for faster lookup.

**Signature:**
```python
@classmethod
def get_excluded_cols_set(cls) -> Set[str]
```

**Returns:**
- `Set[str]`: Set of all excluded column names

**Example:**
```python
excluded_set = DataSchema.get_excluded_cols_set()

# Fast membership check
if 'symbol' in excluded_set:  # O(1) lookup
    print("Symbol is excluded")
```

---

### `get_feature_cols(df)`

Extract feature column names from a dataframe.

**Signature:**
```python
@classmethod
def get_feature_cols(cls, df: pd.DataFrame) -> List[str]
```

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `List[str]`: List of feature column names

**Example:**
```python
# Automatically filter out excluded columns
feature_cols = DataSchema.get_feature_cols(df)
X = df[feature_cols]

print(f"Using {len(feature_cols)} features")
```

---

### `get_target_column(target_type)`

Get target column name based on task type.

**Signature:**
```python
@classmethod
def get_target_column(cls, target_type: str = 'regression') -> str
```

**Parameters:**
- `target_type` (str): 'regression', 'classification', or 'sector'

**Returns:**
- `str`: Target column name

**Raises:**
- `ValueError`: If `target_type` is invalid

**Example:**
```python
# Regression target
target = DataSchema.get_target_column('regression')
# Returns: 'price_dev_subavg'

# Classification target
target = DataSchema.get_target_column('classification')
# Returns: 'price_dev'

# Sector regression target
target = DataSchema.get_target_column('sector')
# Returns: 'sec_price_dev_subavg'
```

---

### `validate_dataframe(df, require_target)`

Validate that a dataframe contains required columns.

**Signature:**
```python
@classmethod
def validate_dataframe(
    cls,
    df: pd.DataFrame,
    require_target: bool = True
) -> dict
```

**Parameters:**
- `df` (pd.DataFrame): Dataframe to validate
- `require_target` (bool): Check that regression target exists

**Returns:**
```python
{
    'valid': bool,              # Overall validity
    'missing_cols': List[str],  # Required columns not found
    'warnings': List[str]       # Recommended columns not found
}
```

**Example:**
```python
result = DataSchema.validate_dataframe(df, require_target=True)

if not result['valid']:
    print(f"❌ Missing columns: {result['missing_cols']}")
else:
    print("✅ Dataframe is valid")

for warning in result['warnings']:
    print(f"⚠️  {warning}")
```

---

### `summary()`

Get a summary of the data schema.

**Signature:**
```python
@classmethod
def summary(cls) -> str
```

**Returns:**
- `str`: Formatted summary string

**Example:**
```python
print(DataSchema.summary())

# Output:
# Data Schema Summary
# ===================
# Metadata columns: 7
# Date columns: 8
# Price/Volume columns: 5
# Target columns: 7
# Total excluded columns: 27
#
# Primary Targets:
# - Regression: price_dev_subavg
# - Classification: price_dev
# - Sector Regression: sec_price_dev_subavg
```

---

## Backward Compatibility

For gradual migration from legacy code:

```python
# Legacy variable names (deprecated)
from src.constants.data_schema import y_col_list, exclude_cols

# These are aliases to DataSchema.get_excluded_cols()
assert y_col_list == exclude_cols == DataSchema.get_excluded_cols()
```

**Recommendation:** Migrate to `DataSchema` methods instead of using legacy names.

---

## Common Use Cases

### Use Case 1: Feature Engineering

```python
from src.constants.data_schema import DataSchema

# Load data
df = pd.read_parquet('data.parquet')

# Get feature columns
feature_cols = DataSchema.get_feature_cols(df)

# Get target
target_col = DataSchema.REGRESSION_TARGET

# Separate X and y
X = df[feature_cols]
y = df[target_col]

print(f"Features: {len(feature_cols)} columns")
print(f"Target: {target_col}")
```

---

### Use Case 2: Data Validation

```python
from src.constants.data_schema import DataSchema

# Validate training data
result = DataSchema.validate_dataframe(train_df, require_target=True)

if not result['valid']:
    raise ValueError(f"Training data missing: {result['missing_cols']}")

# Validate test data (may not have target)
result = DataSchema.validate_dataframe(test_df, require_target=False)

if result['warnings']:
    logging.warning(f"Test data warnings: {result['warnings']}")
```

---

### Use Case 3: Column Filtering

```python
from src.constants.data_schema import DataSchema

# Remove excluded columns before saving
excluded = DataSchema.get_excluded_cols_set()
df_filtered = df[[col for col in df.columns if col not in excluded]]

# Keep only metadata columns
metadata = DataSchema.METADATA_COLS
df_metadata = df[metadata]

# Keep only features (no metadata, no targets)
features = DataSchema.get_feature_cols(df)
df_features = df[features]
```

---

### Use Case 4: Model Training

```python
from src.constants.data_schema import DataSchema

# Get features and target
X = df[DataSchema.get_feature_cols(df)]
y = df[DataSchema.REGRESSION_TARGET]

# Train model
model.fit(X, y)

# Binary classification
y_binary = df[DataSchema.CLASSIFICATION_TARGET]
clf.fit(X, y_binary)
```

---

## Best Practices

### ✅ DO

```python
# Use DataSchema for all column references
excluded = DataSchema.get_excluded_cols()

# Use constants for target names
target = DataSchema.REGRESSION_TARGET

# Validate dataframes before processing
result = DataSchema.validate_dataframe(df)
```

### ❌ DON'T

```python
# Don't hardcode column lists
excluded = ['symbol', 'price', ...]  # ❌ Will diverge

# Don't hardcode target names
target = 'price_dev_subavg'  # ❌ Not centralized

# Don't skip validation
X = df[features]  # ❌ May fail if features undefined
```

---

## Testing

### Unit Test Example

```python
import pytest
from src.constants.data_schema import DataSchema

def test_excluded_cols_count():
    """Verify total excluded columns."""
    excluded = DataSchema.get_excluded_cols()
    assert len(excluded) == 27

def test_target_columns():
    """Verify target column names."""
    assert DataSchema.REGRESSION_TARGET == 'price_dev_subavg'
    assert DataSchema.CLASSIFICATION_TARGET == 'price_dev'
    assert DataSchema.SECTOR_REGRESSION_TARGET == 'sec_price_dev_subavg'

def test_get_feature_cols():
    """Verify feature extraction."""
    df = pd.DataFrame({
        'symbol': ['AAPL'],
        'sector': ['Technology'],
        'price': [150.0],
        'feature1': [1.0],
        'feature2': [2.0],
        'price_dev_subavg': [0.05]
    })

    features = DataSchema.get_feature_cols(df)
    assert 'feature1' in features
    assert 'feature2' in features
    assert 'symbol' not in features
    assert 'price' not in features
    assert 'price_dev_subavg' not in features
```

---

## Troubleshooting

### Issue 1: Column Not Found

**Problem:**
```python
KeyError: 'price_dev_subavg' not found in dataframe
```

**Solution:**
```python
# Validate dataframe first
result = DataSchema.validate_dataframe(df, require_target=True)
if not result['valid']:
    print(f"Missing: {result['missing_cols']}")
```

---

### Issue 2: Too Many Columns Excluded

**Problem:**
All columns are being excluded as features.

**Solution:**
```python
# Check what's being excluded
excluded = DataSchema.get_excluded_cols()
print(f"Excluded: {len(excluded)} columns")

# Verify your dataframe has feature columns
all_cols = set(df.columns)
excluded_set = set(excluded)
features = all_cols - excluded_set
print(f"Features: {len(features)} columns: {features}")
```

---

## Changelog

### Version 1.0.0 (2025-12-01)
- Initial release
- 27 excluded columns defined
- 3 target variables defined
- 5 class methods implemented
- Full backward compatibility

---

## Related Documentation

- [REFACTORING_GUIDE.md](./REFACTORING_GUIDE.md) - Complete refactoring overview
- [API_REFERENCE.md](./API_REFERENCE.md) - Full system API
- [WORKFLOW_GUIDE.md](./WORKFLOW_GUIDE.md) - Development workflow

---

**Last Updated:** 2025-12-01
**Module:** `src.constants.data_schema`
**Status:** ✅ Production Ready
