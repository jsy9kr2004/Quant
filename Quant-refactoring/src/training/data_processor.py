"""
Unified Data Preprocessing Pipeline

This module provides a centralized DataProcessor that ensures regressor.py
and ml_backtest.py use identical preprocessing steps.

Key Features:
- Sparse row/column removal
- Outlier clipping (quantile-based)
- Feature scaling (RobustScaler or StandardScaler)
- NaN handling
- Feature/target separation
- Consistent preprocessing across training and testing

Critical: This eliminates code duplication and ensures that bugs fixed in
one place automatically apply to both regressor.py and ml_backtest.py.

Author: Quant Trading Team
Date: 2025-12-01
"""

from typing import Tuple, Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.constants.data_schema import DataSchema


class DataProcessor:
    """
    Unified data preprocessing pipeline.

    This class consolidates all data preprocessing logic from regressor.py
    and ml_backtest.py into a single, reusable implementation.

    Purpose:
    --------
    - Single source of truth for preprocessing
    - Ensure identical preprocessing in training and backtesting
    - Enable easy parameter tuning across the entire system

    Usage (Training):
    -----------------
    processor = DataProcessor(config)
    result = processor.full_pipeline(train_df, test_df)
    X_train, y_train = result['X_train'], result['y_train']

    Usage (Backtesting):
    --------------------
    processor = DataProcessor(config)
    processor.fit(train_df)
    X_test = processor.transform(test_df)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize DataProcessor.

        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Configuration dictionary (from conf.yaml)
        logger : Optional[logging.Logger]
            Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger('DataProcessor')

        # State variables (for fit/transform pattern)
        self.scaler = None
        self.clip_bounds = {}  # {col: (lower, upper)}
        self.dropped_cols = []
        self.feature_names = []
        self.is_fitted = False

    # ========================================================================
    # Sparse Data Handling
    # ========================================================================

    @staticmethod
    def drop_sparse_rows(
        df: pd.DataFrame,
        threshold: float = 0.9
    ) -> pd.DataFrame:
        """
        Remove rows with too many NaN values.

        This method removes rows where NaN ratio exceeds threshold.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        threshold : float
            Threshold for NaN tolerance (default: 0.9)
            Example: 0.9 means keep rows with <90% NaN

        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe

        Example:
        --------
        # Remove rows with >90% NaN
        df_clean = DataProcessor.drop_sparse_rows(df, threshold=0.9)
        """
        nan_count_per_row = df.isnull().sum(axis=1)
        max_allowed_nan = int(len(df.columns) * (1 - threshold))
        mask = nan_count_per_row < max_allowed_nan

        dropped_count = len(df) - mask.sum()
        if dropped_count > 0:
            logging.info(f"   Dropped {dropped_count} sparse rows (threshold={threshold})")

        return df[mask].copy()

    def drop_sparse_cols(
        self,
        df: pd.DataFrame,
        missing_threshold: Optional[float] = None,
        same_value_threshold: Optional[float] = None,
        protect_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove columns with too many missing values or same values.

        This method identifies and removes columns that are uninformative:
        1. Columns with >80% missing values
        2. Columns where >98% of values are identical

        **Config-driven**: Reads thresholds from FEATURES config (single source of truth)

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        missing_threshold : Optional[float]
            Maximum allowed missing ratio (default: read from FEATURES.MISSING_THRESHOLD or 0.8)
        same_value_threshold : Optional[float]
            Maximum allowed same-value ratio (default: read from FEATURES.SAME_VALUE_THRESHOLD or 0.95)
        protect_cols : Optional[List[str]]
            Columns to never drop (e.g., metadata)

        Returns:
        --------
        df_clean : pd.DataFrame
            Dataframe with sparse columns removed
        dropped_cols : List[str]
            List of dropped column names

        Example:
        --------
        processor = DataProcessor(config)
        df_clean, dropped = processor.drop_sparse_cols(df)  # Uses config values
        print(f"Dropped {len(dropped)} uninformative columns")
        """
        # Read from config if not provided (same logic for regressor and ml_backtest)
        if missing_threshold is None:
            features_config = self.config.get('FEATURES', {})
            missing_threshold = float(features_config.get('MISSING_THRESHOLD', 0.8))

        if same_value_threshold is None:
            features_config = self.config.get('FEATURES', {})
            same_value_threshold = float(features_config.get('SAME_VALUE_THRESHOLD', 0.95))

        if protect_cols is None:
            protect_cols = DataSchema.get_excluded_cols()

        cols_to_drop = []
        cols_to_drop_missing = []
        cols_to_drop_same = []

        for col in df.columns:
            # Skip protected columns
            if col in protect_cols:
                continue

            # Check missing ratio
            missing_ratio = df[col].isna().mean()
            if missing_ratio > missing_threshold:
                cols_to_drop.append(col)
                cols_to_drop_missing.append(col)
                continue

            # Check same-value ratio
            value_counts = df[col].value_counts(normalize=True, dropna=False)
            if len(value_counts) > 0:
                top_value_ratio = value_counts.iloc[0]
                if top_value_ratio > same_value_threshold:
                    cols_to_drop.append(col)
                    cols_to_drop_same.append(col)

        logging.info(f"   Dropping {len(cols_to_drop)} sparse columns:")
        logging.info(f"     - {len(cols_to_drop_missing)} due to missing values")
        logging.info(f"     - {len(cols_to_drop_same)} due to same values")

        df_clean = df.drop(columns=cols_to_drop)

        return df_clean, cols_to_drop

    # ========================================================================
    # Infinite Value Handling
    # ========================================================================

    @staticmethod
    def remove_infinite_values(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Remove rows with infinite values (XGBoost compatibility).

        XGBoost cannot handle infinite values and will raise an error.
        This method removes all rows containing inf or -inf.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : Optional[pd.Series]
            Target series (will be aligned with X if provided)

        Returns:
        --------
        X_clean : pd.DataFrame
            Dataframe without infinite values
        y_clean : Optional[pd.Series]
            Target series without infinite values (if provided)

        Example:
        --------
        X_clean, y_clean = DataProcessor.remove_infinite_values(X, y)
        """
        # Check for infinite values
        inf_mask = np.isinf(X)
        rows_with_inf = inf_mask.any(axis=1)

        if rows_with_inf.sum() > 0:
            logging.warning(f"⚠️  Found {rows_with_inf.sum()} rows with infinite values, removing...")
            X_clean = X[~rows_with_inf].copy()

            if y is not None:
                y_clean = y[~rows_with_inf].copy()
                logging.info(f"   After infinite removal: {len(X_clean)} rows remaining")
                return X_clean, y_clean
            else:
                logging.info(f"   After infinite removal: {len(X_clean)} rows remaining")
                return X_clean, None

        return X, y

    @staticmethod
    def replace_infinite_with_nan(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Replace infinite values with NaN.

        Sometimes it's better to replace inf with NaN rather than removing rows.
        This allows fillna() to handle them.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : Optional[pd.Series]
            Target series

        Returns:
        --------
        X_clean : pd.DataFrame
            Dataframe with inf replaced by NaN
        y_clean : Optional[pd.Series]
            Target with inf replaced by NaN (if provided)
        """
        X_clean = X.replace([np.inf, -np.inf], np.nan)

        if y is not None:
            y_clean = y.replace([np.inf, -np.inf], np.nan)
            return X_clean, y_clean

        return X_clean, None

    @staticmethod
    def check_duplicate_index(
        df: pd.DataFrame,
        stage_name: str,
        logger: Optional[logging.Logger] = None
    ) -> bool:
        """
        Check for duplicate indices in DataFrame and log detailed information.

        ✨ UNIFIED: Used by both regressor.py and ml_backtest.py for debugging.

        This function helps track down the source of duplicate index errors by:
        - Detecting if duplicates exist
        - Logging which indices are duplicated
        - Showing how many times each duplicate appears
        - Providing actionable debugging information

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to check for duplicate indices
        stage_name : str
            Description of current processing stage (e.g., "after train/test split")
        logger : Optional[logging.Logger]
            Logger instance to use. If None, uses root logger.

        Returns:
        --------
        bool
            True if duplicates found, False otherwise

        Example:
        --------
        >>> if DataProcessor.check_duplicate_index(x_train, "after split", logger):
        >>>     # Handle duplicates
        >>>     x_train = x_train[~x_train.index.duplicated(keep='first')]
        """
        if logger is None:
            logger = logging.getLogger()

        logger.info("=" * 80)
        logger.info(f"🔍 DUPLICATE INDEX CHECK: {stage_name}")

        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.error(f"❌ FOUND {dup_count} DUPLICATE INDICES!")

            # Show first 10 duplicate indices
            dup_indices = df.index[df.index.duplicated()].unique()[:10]
            logger.error(f"   First 10 duplicate indices: {list(dup_indices)}")

            # Show shape
            logger.error(f"   DataFrame shape: {df.shape}")

            # Show most common duplicates
            dup_mask = df.index.duplicated(keep=False)
            dup_values = df.index[dup_mask].value_counts().head(5)
            logger.error(f"   Top 5 most common duplicate values:")
            for idx, count in dup_values.items():
                logger.error(f"      Index {idx} appears {count} times")

            logger.error(f"   ⚠️  This may cause 'cannot reindex on an axis with duplicate labels' error!")
            logger.info("=" * 80)
            return True
        else:
            logger.info(f"✅ No duplicates ({len(df)} rows)")
            logger.info("=" * 80)
            return False

    @staticmethod
    def create_clean_dataframe(
        series: pd.Series,
        columns: pd.Index,
        index: pd.Index
    ) -> pd.DataFrame:
        """
        Create DataFrame from Series with clean index (avoiding index conflicts).

        ✨ UNIFIED: Used by regressor.py in multiple places to avoid
        "cannot reindex on an axis with duplicate labels" errors.

        When creating a DataFrame from a Series that already has an index,
        pandas may try to align the Series' original index with the new index,
        causing conflicts. This method uses .values to strip the original index
        and assign a clean new index.

        Parameters:
        -----------
        series : pd.Series
            Series with data (index will be stripped)
        columns : pd.Index
            Column names for the DataFrame
        index : pd.Index
            New clean index to assign

        Returns:
        --------
        pd.DataFrame
            DataFrame with clean index, no conflicts

        Example:
        --------
        >>> y_clean = pd.Series([1, 2, 3], index=[10, 20, 30])
        >>> new_index = pd.Index([0, 1, 2])
        >>> df = DataProcessor.create_clean_dataframe(y_clean, ['target'], new_index)
        >>> df.index  # [0, 1, 2] not [10, 20, 30]

        Why this matters:
        -----------------
        ❌ WRONG:
        pd.DataFrame({col: series}, index=new_index)
        → pandas tries to align series.index with new_index → conflicts!

        ✅ CORRECT:
        pd.DataFrame(series.values, columns=[col], index=new_index)
        → No alignment, clean assignment
        """
        return pd.DataFrame(series.values, columns=columns, index=index)

    @staticmethod
    def log_transform_features(
        X: pd.DataFrame,
        inverse: bool = False
    ) -> pd.DataFrame:
        """
        Sign-preserving log transformation for extreme value handling.

        ✨ UNIFIED: Replaces hardcoded threshold clipping with adaptive log scaling
        Used by both regressor.py and ml_backtest.py for:
        - Unified model (x_train)
        - Sector models (sector_x_train)

        Why log transform is better than clip_extreme_values(threshold=1e10):
        ========================================================================
        1. ✅ Preserves actual value ordering (Apple $3T > Microsoft $2T)
        2. ✅ Adaptive to any scale (market cap can grow infinitely)
        3. ✅ Natural compression (1e12 → 27.6, 1e10 → 23.0)
        4. ✅ Invertible (can recover original scale if needed)
        5. ✅ No arbitrary threshold (1e10 is too small for large-cap stocks)

        Problems with hard clipping:
        ========================================================================
        - Apple market cap: $3 trillion = 3e12 → clipped to 1e10 (300x compression!)
        - NVIDIA market cap: $2 trillion = 2e12 → clipped to 1e10 (200x compression!)
        - All large-cap stocks become indistinguishable

        Formula:
        ========================================================================
        - Forward: sign(x) * log(1 + |x|)
          → Handles positive, negative, and zero values
          → log1p(x) = log(1 + x) is numerically stable near 0

        - Inverse: sign(x) * (exp(|x|) - 1)
          → Recovers original scale (useful for feature importance interpretation)

        Example transformation:
        ========================================================================
        Original Value    →  Log Transformed  →  Interpretation
        ---------------      ----------------     ---------------
        3e12 (Apple)     →  28.7              →  Preserved distinction
        2e12 (NVIDIA)    →  28.3              →  Preserved distinction
        1e12             →  27.6              →  Preserved distinction
        1e10             →  23.0              →
        1e8              →  18.4              →
        0                →  0                 →  Zero stays zero
        -1e8             →  -18.4             →  Sign preserved

        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
        inverse : bool
            If True, inverse transform (exp) back to original scale
            Default: False (forward transform)

        Returns:
        --------
        X_transformed : pd.DataFrame
            Log-transformed features (same shape and columns as input)

        Usage:
        ------
        # Training (forward transform)
        X_train_log = DataProcessor.log_transform_features(X_train)

        # Prediction (use same transform)
        X_test_log = DataProcessor.log_transform_features(X_test)

        # Interpretation (inverse transform, optional)
        X_original = DataProcessor.log_transform_features(X_train_log, inverse=True)

        Notes:
        ------
        - Applied to ALL features (tree-based models don't care about scale)
        - XGBoost/LightGBM benefit from compressed scale (prevents "value too large" errors)
        - Preserves feature names and DataFrame structure
        """
        if inverse:
            # Inverse: sign(x) * (exp(|x|) - 1)
            # Recovers original scale from log-transformed values
            X_transformed = X.apply(
                lambda col: np.sign(col) * (np.exp(np.abs(col)) - 1)
            )
        else:
            # Forward: sign(x) * log(1 + |x|)
            # Compresses extreme values while preserving order
            X_transformed = X.apply(
                lambda col: np.sign(col) * np.log1p(np.abs(col))
            )

        return X_transformed

    @staticmethod
    def clip_extreme_values(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        threshold: float = 1e10,
        enabled: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], int]:
        """
        Clip extreme values for XGBoost/LightGBM compatibility.

        XGBoost errors with "Input data contains inf or a value too large"
        when values exceed ~1e10, even if not strictly infinite.

        This method provides unified extreme value handling across
        regressor.py and ml_backtest.py.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : Optional[pd.Series]
            Target series (unchanged, only X is clipped)
        threshold : float
            Maximum absolute value allowed (default: 1e10)
        enabled : bool
            Whether to apply clipping (default: True)

        Returns:
        --------
        X_clipped : pd.DataFrame
            Dataframe with extreme values clipped to [-threshold, threshold]
        y : Optional[pd.Series]
            Target series (unchanged)
        n_clipped : int
            Number of values that were clipped

        Example:
        --------
        >>> X_safe, y, n = DataProcessor.clip_extreme_values(X, y, threshold=1e10)
        >>> if n > 0:
        >>>     logging.warning(f"Clipped {n} extreme values")

        Note:
        -----
        - Tree-based models (XGBoost/LightGBM) are robust to outliers
        - BUT they cannot handle values > 1e10 (XGBoost internal limit)
        - Clipping preserves information (direction + relative magnitude)
        - Removing rows would lose valuable data
        """
        if not enabled:
            return X, y, 0

        # Count extreme values before clipping
        if isinstance(X, pd.DataFrame):
            extreme_mask = (X.abs() > threshold)
            n_extreme = extreme_mask.sum().sum()
        else:
            extreme_mask = (np.abs(X) > threshold)
            n_extreme = int(extreme_mask.sum())

        if n_extreme > 0:
            X_clipped = X.clip(-threshold, threshold)
            return X_clipped, y, n_extreme
        else:
            return X, y, 0

    # ========================================================================
    # NaN Handling
    # ========================================================================

    @staticmethod
    def handle_nan(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        method: str = 'fillna',
        fill_value: Any = 0
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Unified NaN handling.

        Provides consistent NaN handling across regressor.py and ml_backtest.py.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : Optional[pd.Series]
            Target series
        method : str
            'fillna' (default), 'drop', or 'forward_fill'
        fill_value : Any
            Value to fill NaN with (default: 0)

        Returns:
        --------
        X_clean : pd.DataFrame
            Dataframe with NaN handled
        y_clean : Optional[pd.Series]
            Target with NaN handled (if provided)

        Example:
        --------
        X, y = DataProcessor.handle_nan(X, y, method='fillna', fill_value=0)
        """
        if method == 'fillna':
            X_clean = X.fillna(fill_value)
            y_clean = y.fillna(fill_value) if y is not None else None
        elif method == 'drop':
            nan_mask = X.isna().any(axis=1)
            X_clean = X[~nan_mask]
            y_clean = y[~nan_mask] if y is not None else None
        elif method == 'forward_fill':
            X_clean = X.fillna(method='ffill')
            y_clean = y.fillna(method='ffill') if y is not None else None
        else:
            raise ValueError(f"Unknown method: {method}. Use 'fillna', 'drop', or 'forward_fill'")

        return X_clean, y_clean

    # ========================================================================
    # Outlier Handling
    # ========================================================================

    def clip_outliers(
        self,
        df: pd.DataFrame,
        lower_percentile: float = 0.02,
        upper_percentile: float = 0.98,
        save_bounds: bool = True,
        apply_saved_bounds: bool = False
    ) -> pd.DataFrame:
        """
        Clip outliers using quantile-based bounds.

        This method clips extreme values to prevent model distortion.

        Critical: This was present in regressor.py but MISSING in ml_backtest.py,
        causing inconsistent preprocessing! Now unified.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        lower_percentile : float
            Lower quantile (default: 0.02 = 2nd percentile)
        upper_percentile : float
            Upper quantile (default: 0.98 = 98th percentile)
        save_bounds : bool
            If True, save bounds for later use on test data
        apply_saved_bounds : bool
            If True, use previously saved bounds (for test data)

        Returns:
        --------
        pd.DataFrame
            Dataframe with clipped values

        Example:
        --------
        # Training: clip and save bounds
        df_train_clipped = processor.clip_outliers(df_train, save_bounds=True)

        # Testing: apply saved bounds
        df_test_clipped = processor.clip_outliers(df_test, apply_saved_bounds=True)
        """
        feature_cols = DataSchema.get_feature_cols(df)
        df_clipped = df.copy()

        # Exclude non-numeric columns (like 'sector')
        numeric_features = [
            col for col in feature_cols
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        if apply_saved_bounds:
            # Use saved bounds (for test data)
            if not self.clip_bounds:
                self.logger.warning("No saved clip bounds found! Skipping clipping.")
                return df_clipped

            for col, (lower, upper) in self.clip_bounds.items():
                if col in df_clipped.columns:
                    df_clipped[col] = df_clipped[col].clip(lower, upper)

            self.logger.info(f"   Applied saved clip bounds to {len(self.clip_bounds)} columns")

        else:
            # Calculate new bounds (for training data)
            for col in numeric_features:
                lower, upper = df[col].quantile([lower_percentile, upper_percentile])
                df_clipped[col] = df[col].clip(lower, upper)

                if save_bounds:
                    self.clip_bounds[col] = (float(lower), float(upper))

            self.logger.info(f"   Clipped {len(numeric_features)} columns (percentiles: {lower_percentile}, {upper_percentile})")

        return df_clipped

    # ========================================================================
    # Feature Scaling
    # ========================================================================

    @staticmethod
    def scale_features(
        X: pd.DataFrame,
        scaler_type: str = 'robust',
        fitted_scaler: Optional[Any] = None
    ) -> Tuple[np.ndarray, Any]:
        """
        Unified feature scaling.

        Provides consistent scaling across regressor.py and ml_backtest.py.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        scaler_type : str
            'robust' (default) or 'standard'
        fitted_scaler : Optional[Any]
            Pre-fitted scaler for transform mode
            If None, creates new scaler and fits (train mode)

        Returns:
        --------
        X_scaled : np.ndarray
            Scaled features
        scaler : Any
            Fitted scaler object (for use in transform mode)

        Example:
        --------
        # Train mode (fit new scaler)
        X_train_scaled, scaler = DataProcessor.scale_features(X_train, 'robust')

        # Test mode (use fitted scaler)
        X_test_scaled, _ = DataProcessor.scale_features(X_test, fitted_scaler=scaler)
        """
        if fitted_scaler is not None:
            # Transform mode (use existing scaler)
            X_scaled = fitted_scaler.transform(X)
            return X_scaled, fitted_scaler
        else:
            # Fit mode (create and fit new scaler)
            if scaler_type == 'robust':
                scaler = RobustScaler()
            elif scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler: {scaler_type}. Use 'robust' or 'standard'")

            X_scaled = scaler.fit_transform(X)
            logging.info(f"   Fitted {scaler_type.upper()} scaler on {X.shape[1]} features")

            return X_scaled, scaler

    # Backward compatibility methods
    def fit_scaler(
        self,
        X: pd.DataFrame,
        scaler_type: str = 'robust'
    ) -> np.ndarray:
        """
        Fit scaler and transform features (backward compatibility).

        Use DataProcessor.scale_features() for new code.
        """
        X_scaled, scaler = self.scale_features(X, scaler_type)
        self.scaler = scaler
        return X_scaled

    def transform_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler (backward compatibility).

        Use DataProcessor.scale_features(X, fitted_scaler=scaler) for new code.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted! Call fit_scaler() first.")

        X_scaled, _ = self.scale_features(X, fitted_scaler=self.scaler)
        return X_scaled

    @staticmethod
    def create_binary_target(
        y: Union[pd.Series, pd.DataFrame],
        threshold: float = 0.0
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Convert regression target to binary classification target.

        This method ensures consistent binary classification across both
        regressor.py and ml_backtest.py.

        Args:
            y: Regression target values (price_dev or price_dev_subavg)
            threshold: Threshold for binary classification
                      - Default: 0.0 (strict profit/loss boundary)
                      - Values > threshold are classified as 1 (up)
                      - Values <= threshold are classified as 0 (down)
                      - Alternative: -0.02 for 2% loss tolerance

        Returns:
            Binary target (1 for above threshold, 0 otherwise)

        Example:
            >>> y = pd.Series([0.05, -0.01, -0.03, 0.02])
            >>> binary = DataProcessor.create_binary_target(y)  # threshold=0
            >>> # Result: [1, 0, 0, 1]
            >>>
            >>> # With tolerance
            >>> binary = DataProcessor.create_binary_target(y, threshold=-0.02)
            >>> # Result: [1, 1, 0, 1] (-0.01 > -0.02, so it's 1)

        Note:
            - threshold=0.0: Strict boundary (default, refactored version)
            - threshold=-0.02: 2% loss tolerance (old version)
            - Use threshold parameter to experiment with different strategies
        """
        if isinstance(y, pd.DataFrame):
            # If DataFrame, apply to first column (typically the target column)
            return (y.iloc[:, 0] > threshold).astype(int).to_frame(name=y.columns[0])
        else:
            return (y > threshold).astype(int)

    @staticmethod
    def fit_outlier_clipper(
        df: pd.DataFrame,
        lower_percentile: float = 0.02,
        upper_percentile: float = 0.98,
        exclude_cols: Optional[List[str]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute clipping bounds for outlier removal based on percentiles.

        This method ensures consistent outlier handling across both
        regressor.py and ml_backtest.py. Extreme values can negatively
        impact model training and predictions.

        Args:
            df: Training data to compute bounds from
            lower_percentile: Lower percentile (default 0.02 = 2nd percentile)
            upper_percentile: Upper percentile (default 0.98 = 98th percentile)
            exclude_cols: Columns to skip (e.g., ['sector'])

        Returns:
            Dictionary mapping column names to (lower, upper) bounds

        Example:
            >>> # Fit on training data
            >>> clip_bounds = DataProcessor.fit_outlier_clipper(X_train)
            >>> # Apply to both train and test
            >>> X_train_clipped = DataProcessor.apply_outlier_clipper(X_train, clip_bounds)
            >>> X_test_clipped = DataProcessor.apply_outlier_clipper(X_test, clip_bounds)

        Note:
            - Default 2-98 percentile removes top/bottom 2% extreme values
            - For targets, use 1-97 percentile (more aggressive)
            - Saves ~50 lines of duplicated clipping code
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

        logging.info(f"Computed clip bounds for {len(clip_bounds)} features "
                    f"(percentiles: {lower_percentile*100:.0f}-{upper_percentile*100:.0f})")

        return clip_bounds

    @staticmethod
    def apply_outlier_clipper(
        df: pd.DataFrame,
        clip_bounds: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Apply pre-computed clipping bounds to remove outliers.

        Args:
            df: Data to clip (train, test, or new data)
            clip_bounds: Dictionary from fit_outlier_clipper()

        Returns:
            Clipped DataFrame (copy, original unchanged)

        Example:
            >>> # First fit on training data
            >>> clip_bounds = DataProcessor.fit_outlier_clipper(X_train)
            >>> # Then apply to test data
            >>> X_test_clipped = DataProcessor.apply_outlier_clipper(X_test, clip_bounds)

        Note:
            - Only clips columns present in both df and clip_bounds
            - Silently skips columns not in df (for flexibility)
        """
        df = df.copy()

        clipped_count = 0
        for col, (lower, upper) in clip_bounds.items():
            if col in df.columns:
                original_values = df[col].copy()
                df[col] = df[col].clip(lower, upper)

                # Count how many values were clipped
                clipped = ((original_values < lower) | (original_values > upper)).sum()
                if clipped > 0:
                    clipped_count += 1

        if clipped_count > 0:
            logging.info(f"Applied outlier clipping to {clipped_count} columns")

        return df

    @staticmethod
    def winsorize_features(
        df: pd.DataFrame,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
        exclude_cols: Optional[List[str]] = None,
        enabled: bool = True
    ) -> pd.DataFrame:
        """
        Apply Winsorization to handle outliers by capping at percentiles.

        Winsorization replaces extreme values with percentile values rather than
        removing them. This is gentler than clipping and preserves information.

        DIFFERENCE from clipping:
        - Clipping: [1, 2, 5, 10, 50, 100, 1000] → [2, 2, 5, 10, 50, 100, 100]
          (cuts off extremes, hard boundary)
        - Winsorization: [1, 2, 5, 10, 50, 100, 1000] → [2, 2, 5, 10, 50, 100, 100]
          (replaces with percentile values, soft cap)

        Args:
            df: Input DataFrame
            lower_percentile: Lower percentile (default 0.01 = 1%)
            upper_percentile: Upper percentile (default 0.99 = 99%)
            exclude_cols: Columns to skip (e.g., ['sector'])
            enabled: If False, returns df unchanged (for easy on/off toggle)

        Returns:
            Winsorized DataFrame

        Example:
            >>> # Enable Winsorization
            >>> df_clean = DataProcessor.winsorize_features(df, enabled=True)
            >>>
            >>> # Disable for comparison
            >>> df_raw = DataProcessor.winsorize_features(df, enabled=False)
            >>>
            >>> # More aggressive (0.5-99.5 percentile)
            >>> df_gentle = DataProcessor.winsorize_features(
            ...     df, lower_percentile=0.005, upper_percentile=0.995
            ... )

        Note:
            - enabled=False: Easy way to disable without changing code
            - Recommended for tree-based models: Try enabled=False first
            - Use Winsorization if raw data has too many extreme outliers
            - Default 1-99% is gentler than clipping's 2-98%
        """
        if not enabled:
            logging.info("Winsorization disabled (enabled=False)")
            return df.copy()

        exclude_cols = exclude_cols or []
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        winsorized_count = 0

        for col in numeric_cols:
            if col in exclude_cols:
                continue

            # Get percentile values
            lower_val = df[col].quantile(lower_percentile)
            upper_val = df[col].quantile(upper_percentile)

            # Count how many will be winsorized
            lower_mask = df[col] < lower_val
            upper_mask = df[col] > upper_val

            if lower_mask.any() or upper_mask.any():
                # Replace extreme values with percentile values
                df.loc[lower_mask, col] = lower_val
                df.loc[upper_mask, col] = upper_val
                winsorized_count += 1

        if winsorized_count > 0:
            logging.info(f"Winsorized {winsorized_count} columns "
                        f"(percentiles: {lower_percentile*100:.1f}-{upper_percentile*100:.1f}%)")

        return df

    @staticmethod
    def select_features_by_importance(
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        n_features: Optional[int] = None,
        top_pct: Optional[float] = None,
        task: str = 'regression',
        enabled: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using model-based importance (LightGBM).

        Uses LightGBM's built-in feature importance to identify the most
        predictive features. This is more robust than correlation-based
        methods and handles multicollinearity automatically.

        STRATEGY: Model-based selection (강력한 방법)
        - Trains a LightGBM model to compute feature importances
        - Selects top N features or top percentage
        - Handles interactions and non-linear relationships
        - No need for feature scaling

        Args:
            X: Feature dataframe
            y: Target variable
            n_features: Number of top features to select (e.g., 1000)
            top_pct: Percentage of top features (e.g., 0.3 for top 30%)
            task: 'regression' or 'classification'
            enabled: If False, returns all features unchanged
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (selected feature dataframe, list of selected feature names)

        Example:
            >>> # Reduce from 4,279 to 1,000 features
            >>> X_selected, selected_cols = DataProcessor.select_features_by_importance(
            ...     X, y, n_features=1000, task='regression'
            ... )
            >>> # Apply same selection to test set
            >>> X_test_selected = X_test[selected_cols]

        Note:
            - Default disabled (enabled=False) - user can test with/without
            - Must specify either n_features or top_pct (not both)
            - Selected feature names must be saved for test set application
            - Target ratio recommendation: samples/features >= 10:1
        """
        if not enabled:
            logging.info("Feature selection disabled (enabled=False)")
            return X.copy(), list(X.columns)

        # Validate parameters
        if n_features is None and top_pct is None:
            raise ValueError("Must specify either n_features or top_pct")
        if n_features is not None and top_pct is not None:
            raise ValueError("Cannot specify both n_features and top_pct")

        # Import LightGBM
        try:
            import lightgbm as lgb
        except ImportError:
            logging.warning("LightGBM not available, returning all features")
            return X.copy(), list(X.columns)

        # Calculate target number of features
        if top_pct is not None:
            n_features = max(1, int(len(X.columns) * top_pct))

        n_features = min(n_features, len(X.columns))

        logging.info(f"Selecting top {n_features} features from {len(X.columns)} "
                    f"using LightGBM importance ({task})")

        # Prepare data
        X_np = X.values
        y_np = y.values.ravel() if hasattr(y, 'values') else y

        # Train LightGBM model for feature importance
        if task == 'regression':
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                verbose=-1,
                force_col_wise=True  # Suppress warning
            )
        else:  # classification
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                verbose=-1,
                force_col_wise=True
            )

        # Fit model
        model.fit(X_np, y_np)

        # Get feature importances (gain-based)
        importances = model.feature_importances_

        # Select top N features by importance
        top_indices = np.argsort(importances)[-n_features:]
        selected_cols = [X.columns[i] for i in top_indices]

        # Create selected dataframe
        X_selected = X[selected_cols].copy()

        # Calculate reduction stats
        reduction_pct = (1 - n_features / len(X.columns)) * 100
        new_ratio = len(X) / n_features

        logging.info(f"✓ Feature selection complete: "
                    f"{len(X.columns)} → {n_features} features "
                    f"({reduction_pct:.1f}% reduction)")
        logging.info(f"  New sample/feature ratio: {new_ratio:.1f}:1 "
                    f"(recommended >= 10:1)")

        return X_selected, selected_cols

    @staticmethod
    def filter_by_liquidity(
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        top_pct: float = 0.50
    ) -> Tuple[pd.DataFrame, float]:
        """
        Filter stocks by volume*price liquidity metric to select liquid stocks.

        This method ensures consistent liquidity filtering across both
        regressor.py and ml_backtest.py. Trading illiquid stocks can lead
        to poor execution and slippage.

        Args:
            df: DataFrame with 'symbol' and 'volume_mul_price' columns
            threshold: Minimum liquidity threshold (if None, compute from top_pct)
            top_pct: Top percentage to keep (default 0.50 for top 50%)

        Returns:
            (filtered_df, threshold_used)

        Example:
            >>> # Training: compute threshold and filter
            >>> df_filtered, threshold = DataProcessor.filter_by_liquidity(train_df)
            >>> # Save threshold for test data
            >>> # Test: use saved threshold
            >>> test_filtered, _ = DataProcessor.filter_by_liquidity(test_df, threshold=threshold)

        Note:
            - Default keeps top 50% most liquid stocks
            - Prevents model from learning on illiquid stocks
            - Eliminates 4 duplications in regressor.py (~40 lines)
        """
        if 'symbol' not in df.columns or 'volume_mul_price' not in df.columns:
            logging.warning("Missing required columns for liquidity filtering, returning original df")
            return df, 0.0

        # Compute mean volume*price per symbol
        symbol_means = df.groupby('symbol')['volume_mul_price'].mean().reset_index()

        if threshold is None:
            # Compute threshold from top_pct
            threshold = symbol_means['volume_mul_price'].quantile(1.0 - top_pct)

        # Filter symbols above threshold
        top_symbols = symbol_means[symbol_means['volume_mul_price'] >= threshold]
        filtered_df = df[df['symbol'].isin(top_symbols['symbol'])].copy()

        logging.info(f"Liquidity filter: {len(symbol_means)} symbols → {len(top_symbols)} symbols "
                    f"(top {top_pct*100:.0f}%, threshold={threshold:.2e})")

        return filtered_df, float(threshold)

    @staticmethod
    def clip_large_values(
        df: pd.DataFrame,
        threshold: float = 1e9,
        replacement: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Replace extremely large values to prevent numerical overflow.

        This method ensures consistent large value handling. Very large
        numbers can cause overflow in XGBoost and other models.

        Args:
            df: Input DataFrame
            threshold: Values above this (absolute) are considered too large
            replacement: Value to use (default: np.finfo(np.float32).max)

        Returns:
            DataFrame with large values clipped

        Example:
            >>> df_clean = DataProcessor.clip_large_values(df, threshold=1e9)

        Note:
            - Prevents XGBoost overflow errors
            - Eliminates 3 duplicate implementations (~18 lines)
        """
        if replacement is None:
            replacement = np.finfo(np.float32).max

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        clipped_cols = 0
        for col in numeric_cols:
            mask = df[col].abs() > threshold
            if mask.any():
                df.loc[mask, col] = np.sign(df.loc[mask, col]) * replacement
                clipped_cols += 1

        if clipped_cols > 0:
            logging.info(f"Clipped large values in {clipped_cols} columns (threshold={threshold:.0e})")

        return df

    @staticmethod
    def drop_many_nan_row(
        df: pd.DataFrame,
        threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Drop rows with excessive NaN values.

        This method ensures consistent NaN row filtering. Rows with too
        many missing values provide little training signal.

        Args:
            df: Input DataFrame
            threshold: Drop rows with NaN ratio > threshold
                      (0.6 = drop if >60% NaN, keep if <=60% NaN)

        Returns:
            DataFrame with excessive-NaN rows removed

        Example:
            >>> # Drop rows with >40% NaN (keep rows with <=40% NaN)
            >>> df_clean = DataProcessor.drop_many_nan_row(df, threshold=0.6)
            >>> # Drop rows with >5% NaN (stricter)
            >>> df_clean = DataProcessor.drop_many_nan_row(df, threshold=0.95)

        Note:
            - threshold=0.6 means "drop if NaN% > 60%", keep if <=60%
            - Eliminates duplicate in regressor.py (~15 lines)
        """
        if df.empty:
            return df

        # Calculate NaN ratio per row
        nan_ratio = df.isna().sum(axis=1) / len(df.columns)

        # Keep rows where NaN ratio <= threshold
        mask = nan_ratio <= threshold
        df_clean = df[mask].copy()

        removed = len(df) - len(df_clean)
        if removed > 0:
            logging.info(f"Removed {removed} rows with >{threshold*100:.0f}% NaN "
                        f"({len(df_clean)}/{len(df)} rows remaining)")

        return df_clean

    # ========================================================================
    # Feature/Target Separation
    # ========================================================================

    def prepare_features_and_target(
        self,
        df: pd.DataFrame,
        target_type: str = 'regression',
        drop_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_type : str
            'regression', 'classification', or 'sector'
        drop_cols : Optional[List[str]]
            Additional columns to drop (e.g., previously identified sparse cols)

        Returns:
        --------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target series

        Raises:
        -------
        KeyError
            If target column not found

        Example:
        --------
        X, y = processor.prepare_features_and_target(df, target_type='regression')
        """
        # Get feature columns
        feature_cols = DataSchema.get_feature_cols(df)

        # Remove additional dropped columns
        if drop_cols:
            feature_cols = [col for col in feature_cols if col not in drop_cols]

        # Extract features
        X = df[feature_cols].copy()

        # Extract target
        target_col = DataSchema.get_target_column(target_type)

        if target_col not in df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in dataframe. "
                f"Available columns: {df.columns.tolist()}"
            )

        y = df[target_col].copy()

        self.logger.info(f"   Separated features ({X.shape[1]} cols) and target ('{target_col}')")

        return X, y

    # ========================================================================
    # Full Pipeline
    # ========================================================================

    def full_pipeline(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        sparse_row_threshold: float = 0.6,
        final_sparse_row_threshold: float = 0.95,
        missing_col_threshold: Optional[float] = None,
        same_value_col_threshold: Optional[float] = None,
        clip_outliers: bool = True,
        clip_percentiles: Tuple[float, float] = (0.02, 0.98),
        scaler_type: str = 'robust',
        target_type: str = 'regression',
        save_artifacts: bool = False,
        artifact_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full preprocessing pipeline.

        This method runs all preprocessing steps in the correct order:
        1. Drop sparse rows (first pass)
        2. Drop sparse columns
        3. Separate features and target
        4. Clip outliers (optional)
        5. Drop sparse rows (final pass)
        6. Scale features

        **Config-driven**: Reads missing/same-value thresholds from FEATURES config

        Parameters:
        -----------
        train_df : pd.DataFrame
            Training dataframe
        test_df : Optional[pd.DataFrame]
            Testing dataframe (will use same preprocessing)
        sparse_row_threshold : float
            Initial sparse row threshold (default: 0.6)
        final_sparse_row_threshold : float
            Final sparse row threshold (default: 0.95)
        missing_col_threshold : Optional[float]
            Column missing value threshold (default: read from FEATURES.MISSING_THRESHOLD or 0.8)
        same_value_col_threshold : Optional[float]
            Column same-value threshold (default: read from FEATURES.SAME_VALUE_THRESHOLD or 0.95)
        clip_outliers : bool
            Whether to clip outliers (default: True)
        clip_percentiles : Tuple[float, float]
            Clipping percentiles (default: (0.02, 0.98))
        scaler_type : str
            'robust' or 'standard' (default: 'robust')
        target_type : str
            'regression', 'classification', or 'sector'
        save_artifacts : bool
            Save preprocessing artifacts (clip bounds, dropped cols, etc.)
        artifact_dir : Optional[str]
            Directory to save artifacts

        Returns:
        --------
        dict
            {
                'X_train': np.ndarray,
                'y_train': np.ndarray,
                'X_test': np.ndarray (if test_df provided),
                'y_test': np.ndarray (if test_df provided),
                'feature_names': List[str],
                'dropped_cols': List[str],
                'clip_bounds': Dict[str, Tuple[float, float]],
                'scaler': StandardScaler or RobustScaler
            }

        Example:
        --------
        processor = DataProcessor()
        result = processor.full_pipeline(train_df, test_df, clip_outliers=True)
        X_train, y_train = result['X_train'], result['y_train']
        """
        self.logger.info("=" * 80)
        self.logger.info("DATA PREPROCESSING PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Train shape: {train_df.shape}")
        if test_df is not None:
            self.logger.info(f"Test shape: {test_df.shape}")

        # Read thresholds from config if not provided (same logic for regressor and ml_backtest)
        if missing_col_threshold is None:
            features_config = self.config.get('FEATURES', {})
            missing_col_threshold = float(features_config.get('MISSING_THRESHOLD', 0.8))

        if same_value_col_threshold is None:
            features_config = self.config.get('FEATURES', {})
            same_value_col_threshold = float(features_config.get('SAME_VALUE_THRESHOLD', 0.95))

        # Step 1: Drop sparse rows (first pass - aggressive)
        self.logger.info("\n[1/7] Dropping sparse rows (first pass)...")
        train_clean = self.drop_sparse_rows(train_df, threshold=sparse_row_threshold)

        # Step 2: Drop sparse columns
        self.logger.info("\n[2/7] Dropping sparse columns...")
        train_clean, dropped_cols = self.drop_sparse_cols(
            train_clean,
            missing_threshold=missing_col_threshold,
            same_value_threshold=same_value_col_threshold
        )
        self.dropped_cols = dropped_cols

        # Step 3: Separate features and target
        self.logger.info("\n[3/7] Separating features and target...")
        X_train, y_train = self.prepare_features_and_target(
            train_clean,
            target_type=target_type,
            drop_cols=dropped_cols
        )

        # Step 4: Clip outliers (optional)
        if clip_outliers:
            self.logger.info("\n[4/7] Clipping outliers...")
            X_train = self.clip_outliers(
                X_train,
                lower_percentile=clip_percentiles[0],
                upper_percentile=clip_percentiles[1],
                save_bounds=True
            )
        else:
            self.logger.info("\n[4/7] Skipping outlier clipping")

        # Step 5: Drop sparse rows (final pass - conservative)
        self.logger.info("\n[5/7] Dropping sparse rows (final pass)...")
        # Recombine for final sparse row check
        temp_df = X_train.copy()
        temp_df['_target_'] = y_train
        temp_df = self.drop_sparse_rows(temp_df, threshold=final_sparse_row_threshold)
        y_train = temp_df['_target_']
        X_train = temp_df.drop(columns=['_target_'])

        # Step 6: Scale features
        self.logger.info("\n[6/7] Scaling features...")
        X_train_scaled = self.fit_scaler(X_train, scaler_type=scaler_type)
        self.feature_names = X_train.columns.tolist()
        self.is_fitted = True

        # Prepare result
        result = {
            'X_train': X_train_scaled,
            'y_train': y_train.fillna(0).values,
            'feature_names': self.feature_names,
            'dropped_cols': self.dropped_cols,
            'clip_bounds': self.clip_bounds,
            'scaler': self.scaler,
            'target_type': target_type
        }

        # Step 7: Process test data (if provided)
        if test_df is not None:
            self.logger.info("\n[7/7] Processing test data...")

            # Apply same column drops
            test_clean = test_df.drop(columns=self.dropped_cols, errors='ignore')

            # Separate features and target
            X_test, y_test = self.prepare_features_and_target(
                test_clean,
                target_type=target_type,
                drop_cols=self.dropped_cols
            )

            # Apply saved clip bounds
            if clip_outliers and self.clip_bounds:
                X_test = self.clip_outliers(X_test, apply_saved_bounds=True)

            # Scale using fitted scaler
            X_test_scaled = self.transform_scaler(X_test)

            result['X_test'] = X_test_scaled
            result['y_test'] = y_test.fillna(0).values

            self.logger.info(f"   Test processed: {X_test_scaled.shape}")

        # Save artifacts (optional)
        if save_artifacts and artifact_dir:
            self._save_artifacts(artifact_dir, result)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PREPROCESSING COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Final train shape: {X_train_scaled.shape}")
        self.logger.info(f"Feature count: {len(self.feature_names)}")
        self.logger.info(f"Dropped columns: {len(self.dropped_cols)}")

        return result

    # ========================================================================
    # Artifact Management
    # ========================================================================

    def _save_artifacts(self, artifact_dir: str, result: Dict[str, Any]):
        """Save preprocessing artifacts for reproducibility."""
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)

        # Save dropped columns
        with open(artifact_path / 'dropped_cols.json', 'w') as f:
            json.dump(self.dropped_cols, f, indent=2)

        # Save clip bounds
        if self.clip_bounds:
            with open(artifact_path / 'clip_bounds.json', 'w') as f:
                json.dump(self.clip_bounds, f, indent=2)

        # Save feature names
        with open(artifact_path / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)

        self.logger.info(f"   Saved artifacts to {artifact_path}")

    def load_artifacts(self, artifact_dir: str):
        """Load preprocessing artifacts from directory."""
        artifact_path = Path(artifact_dir)

        # Load dropped columns
        with open(artifact_path / 'dropped_cols.json', 'r') as f:
            self.dropped_cols = json.load(f)

        # Load clip bounds
        clip_bounds_file = artifact_path / 'clip_bounds.json'
        if clip_bounds_file.exists():
            with open(clip_bounds_file, 'r') as f:
                self.clip_bounds = json.load(f)

        # Load feature names
        with open(artifact_path / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)

        self.logger.info(f"   Loaded artifacts from {artifact_path}")


if __name__ == "__main__":
    # Self-test
    print("DataProcessor module loaded successfully")
    print("\nKey features:")
    print("  - Sparse row/column removal")
    print("  - Outlier clipping")
    print("  - Feature scaling (Robust/Standard)")
    print("  - NaN handling")
    print("  - Artifact saving/loading")
