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

from typing import Tuple, Optional, Dict, Any, List
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

    @staticmethod
    def drop_sparse_cols(
        df: pd.DataFrame,
        missing_threshold: float = 0.8,
        same_value_threshold: float = 0.98,
        protect_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove columns with too many missing values or same values.

        This method identifies and removes columns that are uninformative:
        1. Columns with >80% missing values
        2. Columns where >98% of values are identical

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        missing_threshold : float
            Maximum allowed missing ratio (default: 0.8)
        same_value_threshold : float
            Maximum allowed same-value ratio (default: 0.98)
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
        df_clean, dropped = DataProcessor.drop_sparse_cols(df)
        print(f"Dropped {len(dropped)} uninformative columns")
        """
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
        missing_col_threshold: float = 0.8,
        same_value_col_threshold: float = 0.98,
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
        missing_col_threshold : float
            Column missing value threshold (default: 0.8)
        same_value_col_threshold : float
            Column same-value threshold (default: 0.98)
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
