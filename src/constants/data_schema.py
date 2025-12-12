"""
Data Schema and Column Definitions

This module provides the SINGLE SOURCE OF TRUTH for all column names,
metadata definitions, and data contracts used across the entire system.

Critical: This ensures regressor.py and ml_backtest.py use identical
column definitions, preventing bugs from column name mismatches.

Author: Quant Trading Team
Date: 2025-12-01
"""

from typing import List, Set


class DataSchema:
    """
    Unified data schema definition.

    This class defines all column names and metadata used in the ML pipeline.

    Purpose:
    --------
    - Single source of truth for column definitions
    - Prevent column name mismatches between regressor.py and ml_backtest.py
    - Enable consistent feature engineering across all modules

    Usage:
    ------
    # Get columns to exclude from features
    excluded = DataSchema.get_excluded_cols()

    # Get feature columns from dataframe
    features = DataSchema.get_feature_cols(df)

    # Get target column for regression
    target = DataSchema.REGRESSION_TARGET
    """

    # ========================================================================
    # Metadata Columns (Stock identifiers and company info)
    # ========================================================================
    METADATA_COLS: List[str] = [
        "symbol",              # Stock ticker symbol (e.g., AAPL, TSLA)
        "exchangeShortName",   # Exchange name (e.g., NASDAQ, NYSE)
        "type",                # Security type (e.g., 'stock', 'ETF')
        "delistedDate",        # Delisting date if applicable
        "industry",            # Industry classification
        "ipoDate",             # IPO date
        "sector",              # Sector classification (e.g., Technology, Financial)
    ]

    # ========================================================================
    # Date Columns (Temporal information)
    # ========================================================================
    DATE_COLS: List[str] = [
        "rebalance_date",      # Portfolio rebalancing date
        "report_date",         # Financial report date
        "fillingDate",         # Filing date (SEC submission)
        "fillingDate_x",       # Filing date variant (from data merge)
        "acceptedDate",        # Report acceptance date
        "start_date",          # Period start date
        "year_period",         # Year period identifier
        "date",                # Generic date column
    ]

    # ========================================================================
    # Price/Volume Columns (Market data - NOT used as features)
    # ========================================================================
    # These are excluded because they contain future information or
    # are used only for calculating targets/returns
    PRICE_VOLUME_COLS: List[str] = [
        "price",               # Stock price (future leakage risk)
        "volume",              # Trading volume (future leakage risk)
        "marketCap",           # Market capitalization (derived from price)
        "price_diff",          # Price difference (target-related)
        "volume_mul_price",    # Volume * Price (used for filtering only)
    ]

    # ========================================================================
    # Target Variable Columns (Labels for ML models)
    # ========================================================================
    TARGET_COLS: List[str] = [
        "price_dev",              # Binary target: price deviation (up/down)
        "price_dev_subavg",       # Main regression target: price deviation - avg
        "sec_price_dev_subavg",   # Sector-adjusted price deviation
        "price_dev_prediction",   # Model prediction output
        "price_diff_prediction",  # Price difference prediction
        "price_diff_3month",      # 3-month price difference
        "price_diff_6month",      # 6-month price difference
    ]

    # ========================================================================
    # Target Variable Names (Single source of truth)
    # ========================================================================
    REGRESSION_TARGET = "price_dev_subavg"        # Main regression target
    CLASSIFICATION_TARGET = "price_dev"           # Binary classification target
    SECTOR_REGRESSION_TARGET = "sec_price_dev_subavg"  # Sector-specific target

    # ========================================================================
    # Class Methods
    # ========================================================================

    @classmethod
    def get_excluded_cols(cls) -> List[str]:
        """
        Get all columns that should be EXCLUDED from model features.

        Returns:
        --------
        List[str]
            Combined list of all metadata, date, price/volume, and target columns

        Usage:
        ------
        excluded = DataSchema.get_excluded_cols()
        features = [col for col in df.columns if col not in excluded]
        """
        return (
            cls.METADATA_COLS +
            cls.DATE_COLS +
            cls.PRICE_VOLUME_COLS +
            cls.TARGET_COLS
        )

    @classmethod
    def get_excluded_cols_set(cls) -> Set[str]:
        """
        Get excluded columns as a set for faster lookup.

        Returns:
        --------
        Set[str]
            Set of all excluded column names
        """
        return set(cls.get_excluded_cols())

    @classmethod
    def get_feature_cols(cls, df) -> List[str]:
        """
        Extract feature column names from a dataframe.

        This method automatically excludes all metadata, dates, price/volume,
        and target columns, returning only the columns that should be used
        as features for ML models.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        List[str]
            List of feature column names

        Usage:
        ------
        feature_cols = DataSchema.get_feature_cols(train_df)
        X = train_df[feature_cols]
        """
        excluded = cls.get_excluded_cols_set()
        return [col for col in df.columns if col not in excluded]

    @classmethod
    def validate_dataframe(cls, df, require_target: bool = True) -> dict:
        """
        Validate that a dataframe contains required columns.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to validate
        require_target : bool
            If True, check that regression target exists

        Returns:
        --------
        dict
            {
                'valid': bool,
                'missing_cols': List[str],
                'warnings': List[str]
            }
        """
        result = {
            'valid': True,
            'missing_cols': [],
            'warnings': []
        }

        # Check for required target
        if require_target and cls.REGRESSION_TARGET not in df.columns:
            result['valid'] = False
            result['missing_cols'].append(cls.REGRESSION_TARGET)

        # Check for metadata columns (optional but recommended)
        for col in ['symbol', 'sector']:
            if col not in df.columns:
                result['warnings'].append(f"Recommended column '{col}' not found")

        return result

    @classmethod
    def get_target_column(cls, target_type: str = 'regression') -> str:
        """
        Get target column name based on task type.

        Parameters:
        -----------
        target_type : str
            'regression', 'classification', or 'sector'

        Returns:
        --------
        str
            Target column name

        Raises:
        -------
        ValueError
            If target_type is invalid
        """
        if target_type == 'regression':
            return cls.REGRESSION_TARGET
        elif target_type == 'classification':
            return cls.CLASSIFICATION_TARGET
        elif target_type == 'sector':
            return cls.SECTOR_REGRESSION_TARGET
        else:
            raise ValueError(
                f"Invalid target_type: {target_type}. "
                f"Must be 'regression', 'classification', or 'sector'"
            )

    @classmethod
    def summary(cls) -> str:
        """
        Get a summary of the data schema.

        Returns:
        --------
        str
            Formatted summary string
        """
        return f"""
Data Schema Summary
===================
Metadata columns: {len(cls.METADATA_COLS)}
Date columns: {len(cls.DATE_COLS)}
Price/Volume columns: {len(cls.PRICE_VOLUME_COLS)}
Target columns: {len(cls.TARGET_COLS)}
Total excluded columns: {len(cls.get_excluded_cols())}

Primary Targets:
- Regression: {cls.REGRESSION_TARGET}
- Classification: {cls.CLASSIFICATION_TARGET}
- Sector Regression: {cls.SECTOR_REGRESSION_TARGET}
"""


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================
# These allow gradual migration from old code

# Old name from regressor.py: y_col_list
# New unified name: DataSchema.get_excluded_cols()
y_col_list = DataSchema.get_excluded_cols()

# Old name from ml_backtest.py: exclude_cols
# New unified name: DataSchema.get_excluded_cols()
exclude_cols = DataSchema.get_excluded_cols()


if __name__ == "__main__":
    # Self-test and documentation
    print(DataSchema.summary())
    print("\nExcluded columns:")
    for i, col in enumerate(DataSchema.get_excluded_cols(), 1):
        print(f"  {i:2d}. {col}")

    print(f"\nTotal: {len(DataSchema.get_excluded_cols())} columns excluded from features")
