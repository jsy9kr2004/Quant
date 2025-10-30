"""LightGBM model wrapper for gradient boosting with Microsoft's LightGBM.

This module provides a wrapper class for LightGBM (Light Gradient Boosting Machine),
supporting both classification and regression tasks with GPU acceleration.

LightGBM is a gradient boosting framework developed by Microsoft that uses
tree-based learning algorithms. Key advantages include:
- Fast training speed and high efficiency
- Lower memory usage
- Better accuracy than many other frameworks
- GPU acceleration support
- Handles large-scale data well
- Categorical feature support

The LightGBMModel class extends BaseModel and provides:
- Pre-configured settings optimized for trading
- Automatic feature name cleaning for LightGBM compatibility
- Support for custom hyperparameters
- GPU-accelerated training
- Early stopping with callbacks
- Easy integration with the training pipeline

Usage Example:
    Basic classification:
        from models.lightgbm_model import LightGBMModel

        # Create and build model
        model = LightGBMModel(task='classification', config_name='default')
        model.build_model()

        # Train with early stopping
        model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)

        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    Custom configuration:
        # Use custom hyperparameters
        custom_params = {
            'max_depth': 10,
            'learning_rate': 0.05,
            'n_estimators': 1500,
            'num_leaves': 64
        }
        model = LightGBMModel(task='classification')
        model.build_model(custom_params)

    Feature name cleaning:
        # LightGBM requires valid feature names (no special characters)
        # The model automatically cleans feature names
        df_cleaned = LightGBMModel.clean_feature_names(df)

    Regression:
        model = LightGBMModel(task='regression')
        model.build_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

Attributes:
    model_type (str): Always 'lightgbm'
    task (str): Task type ('classification' or 'regression')
    config_name (str): Name of the configuration preset
    default_params (Dict): Default hyperparameters for the model
    model: LightGBM classifier or regressor instance

Note:
    LightGBM has strict requirements for feature names. They must not contain
    special characters like [, ], <, >. The clean_feature_names() method
    automatically handles this by removing invalid characters.
"""

import lightgbm as lgb
import re
from .base_model import BaseModel
from .config import LIGHTGBM_CLASSIFIER_CONFIGS, LIGHTGBM_REGRESSOR_CONFIGS
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np


class LightGBMModel(BaseModel):
    """LightGBM model wrapper with GPU acceleration and automatic feature cleaning.

    This class wraps LightGBM classifiers and regressors, providing a consistent
    interface compatible with the BaseModel API. It includes automatic feature name
    cleaning to comply with LightGBM's naming requirements.

    LightGBM features:
    - Leaf-wise tree growth (vs level-wise in XGBoost)
    - Histogram-based algorithm for faster training
    - GPU acceleration via device='gpu'
    - Native categorical feature support
    - Lower memory usage than XGBoost

    Attributes:
        model_type (str): Type identifier, always 'lightgbm'.
        task (str): Task type, either 'classification' or 'regression'.
        config_name (str): Name of the configuration preset being used.
        default_params (Dict[str, Any]): Dictionary of default hyperparameters.
        model (lgb.LGBMClassifier or lgb.LGBMRegressor): The underlying LightGBM model.
        feature_names (Optional[List[str]]): List of feature names (cleaned).
        is_trained (bool): Flag indicating if model has been trained.

    Example:
        >>> # Classification with default settings
        >>> model = LightGBMModel(task='classification')
        >>> model.build_model()
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>>
        >>> # Clean feature names before training
        >>> X_train_clean = LightGBMModel.clean_feature_names(X_train)
        >>> X_test_clean = LightGBMModel.clean_feature_names(X_test)
        >>>
        >>> # Get feature importance
        >>> importance_df = model.get_feature_importance(top_n=10)
    """

    def __init__(
        self,
        task: str = 'classification',
        config_name: str = 'default'
    ) -> None:
        """Initialize LightGBM model with specified configuration.

        Args:
            task (str, optional): Task type, either 'classification' or 'regression'.
                Defaults to 'classification'.
            config_name (str, optional): Name of the configuration preset to use.
                Currently only 'default' is available for both tasks.
                Defaults to 'default'.

        Example:
            >>> # Default classification model
            >>> model = LightGBMModel(task='classification')
            >>>
            >>> # Regression model
            >>> model = LightGBMModel(task='regression')
        """
        super().__init__(model_type='lightgbm', task=task)
        self.config_name = config_name

        # Load default configuration based on task type
        if task == 'classification':
            self.default_params = LIGHTGBM_CLASSIFIER_CONFIGS.get(
                config_name,
                LIGHTGBM_CLASSIFIER_CONFIGS['default']
            )
        else:
            self.default_params = LIGHTGBM_REGRESSOR_CONFIGS.get(
                config_name,
                LIGHTGBM_REGRESSOR_CONFIGS['default']
            )

    def build_model(self, params: Optional[Dict[str, Any]] = None) -> 'LightGBMModel':
        """Build LightGBM model with specified or default parameters.

        Creates a LightGBM classifier or regressor based on the task type.
        If custom parameters are provided, they are merged with the default
        parameters, with custom parameters taking precedence.

        Args:
            params (Optional[Dict[str, Any]], optional): Custom hyperparameters.
                If None, uses default parameters from the configuration.
                If provided, merges with defaults (custom params override defaults).
                Common parameters:
                - max_depth (int): Maximum tree depth
                - learning_rate (float): Boosting learning rate
                - n_estimators (int): Number of boosting iterations
                - num_leaves (int): Maximum number of leaves per tree
                - subsample (float): Fraction of samples for training
                - colsample_bytree (float): Fraction of features for training
                - min_child_samples (int): Minimum samples per leaf
                Defaults to None.

        Returns:
            LightGBMModel: Self for method chaining.

        Example:
            >>> # Use default parameters
            >>> model.build_model()
            >>>
            >>> # Use custom parameters
            >>> custom_params = {
            ...     'max_depth': 12,
            ...     'learning_rate': 0.05,
            ...     'n_estimators': 2000,
            ...     'num_leaves': 64
            ... }
            >>> model.build_model(custom_params)
            >>>
            >>> # Partial override
            >>> model.build_model({'num_leaves': 128})
        """
        if params is None:
            params = self.default_params
        else:
            # Merge custom parameters with defaults
            merged_params = self.default_params.copy()
            merged_params.update(params)
            params = merged_params

        # Create appropriate model type based on task
        if self.task == 'classification':
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

        return self

    @staticmethod
    def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
        """Clean feature names to comply with LightGBM requirements.

        LightGBM requires feature names to contain only alphanumeric characters
        and underscores. This method removes invalid characters (like brackets,
        special symbols) and handles duplicate names by appending indices.

        Args:
            df (pd.DataFrame): DataFrame with potentially invalid feature names.

        Returns:
            pd.DataFrame: DataFrame with cleaned feature names.

        Example:
            >>> # DataFrame with special characters in column names
            >>> df = pd.DataFrame({
            ...     'feature[1]': [1, 2, 3],
            ...     'feature<2>': [4, 5, 6],
            ...     'normal_feature': [7, 8, 9]
            ... })
            >>> df_clean = LightGBMModel.clean_feature_names(df)
            >>> print(df_clean.columns)
            Index(['feature1', 'feature2', 'normal_feature'], dtype='object')
            >>>
            >>> # Handles duplicate names after cleaning
            >>> df = pd.DataFrame({
            ...     'feature[1]': [1, 2, 3],
            ...     'feature(1)': [4, 5, 6]
            ... })
            >>> df_clean = LightGBMModel.clean_feature_names(df)
            >>> print(df_clean.columns)
            Index(['feature1', 'feature1_1'], dtype='object')

        Note:
            - Removes all characters except A-Z, a-z, 0-9, and underscore
            - Preserves column order
            - Automatically handles duplicate names after cleaning
        """
        # Remove all non-alphanumeric characters (except underscore)
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
        new_n_list = list(new_names.values())

        # Handle duplicate names by appending index
        # If a cleaned name appears multiple times, add _1, _2, etc.
        new_names = {
            col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }

        df = df.rename(columns=new_names)
        return df

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 100
    ) -> 'LightGBMModel':
        """Train LightGBM model with automatic feature cleaning and early stopping.

        Trains the LightGBM model on the provided data. Automatically cleans
        feature names to comply with LightGBM requirements. If validation data
        is provided, enables early stopping using LightGBM callbacks.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Training features.
                If DataFrame, feature names will be automatically cleaned.
            y_train (Union[pd.Series, np.ndarray]): Training labels.
            X_val (Optional[Union[pd.DataFrame, np.ndarray]], optional): Validation
                features for early stopping. Will be cleaned if DataFrame.
                Defaults to None.
            y_val (Optional[Union[pd.Series, np.ndarray]], optional): Validation
                labels for early stopping. Defaults to None.
            early_stopping_rounds (int, optional): Number of rounds with no
                improvement after which training will be stopped. Only used if
                validation data is provided. Defaults to 50.
            verbose (int, optional): Logging frequency. Prints metrics every
                'verbose' iterations. Set to 0 for silent training. Defaults to 100.

        Returns:
            LightGBMModel: Self for method chaining.

        Raises:
            ValueError: If model has not been built.

        Example:
            >>> # Train without early stopping
            >>> model.fit(X_train, y_train, verbose=100)
            >>>
            >>> # Train with early stopping
            >>> model.fit(
            ...     X_train, y_train,
            ...     X_val, y_val,
            ...     early_stopping_rounds=100,
            ...     verbose=50
            ... )
            >>>
            >>> # Silent training
            >>> model.fit(X_train, y_train, verbose=0)

        Note:
            - Feature names are automatically cleaned if input is DataFrame
            - Early stopping uses lgb.early_stopping and lgb.log_evaluation callbacks
            - Validation metric depends on task (binary log loss for classification,
              RMSE for regression)
        """
        # Clean feature names if input is DataFrame (LightGBM requirement)
        if hasattr(X_train, 'columns'):
            X_train = self.clean_feature_names(X_train)
            if X_val is not None:
                X_val = self.clean_feature_names(X_val)

        kwargs = {}

        # Add callbacks for early stopping and logging if validation data provided
        if X_val is not None and y_val is not None:
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=verbose)
            ]
            kwargs['callbacks'] = callbacks

        # Call parent fit method with prepared kwargs
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
