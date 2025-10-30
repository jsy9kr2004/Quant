"""XGBoost model wrapper for gradient boosting.

This module provides a wrapper class for XGBoost (Extreme Gradient Boosting) models,
supporting both classification and regression tasks with GPU acceleration.

XGBoost is a powerful gradient boosting framework that uses decision trees as base
learners. It's known for:
- High performance and speed
- GPU acceleration support
- Built-in regularization to prevent overfitting
- Feature importance scores
- Early stopping capabilities

The XGBoostModel class extends BaseModel and provides:
- Pre-configured settings for common use cases
- Support for custom hyperparameters
- GPU-accelerated training
- Early stopping support
- Easy integration with the training pipeline

Usage Example:
    Basic classification:
        from models.xgboost_model import XGBoostModel

        # Create and build model with default configuration
        model = XGBoostModel(task='classification', config_name='default')
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
            'n_estimators': 1000,
            'subsample': 0.9
        }
        model = XGBoostModel(task='classification')
        model.build_model(custom_params)

    Different depth configurations:
        # Use pre-configured deeper model
        model = XGBoostModel(task='classification', config_name='depth_10')
        model.build_model()

    Regression:
        model = XGBoostModel(task='regression', config_name='deep')
        model.build_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

Attributes:
    model_type (str): Always 'xgboost'
    task (str): Task type ('classification' or 'regression')
    config_name (str): Name of the configuration preset
    default_params (Dict): Default hyperparameters for the model
    model: XGBoost classifier or regressor instance
"""

import xgboost as xgb
from .base_model import BaseModel
from .config import XGBOOST_CLASSIFIER_CONFIGS, XGBOOST_REGRESSOR_CONFIGS
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np


class XGBoostModel(BaseModel):
    """XGBoost model wrapper with GPU acceleration support.

    This class wraps XGBoost classifiers and regressors, providing a consistent
    interface compatible with the BaseModel API. It includes pre-configured
    settings optimized for different scenarios and supports GPU acceleration.

    XGBoost features:
    - Tree-based gradient boosting
    - GPU acceleration via tree_method='gpu_hist'
    - Regularization (L1 and L2)
    - Column and row subsampling
    - Early stopping support

    Attributes:
        model_type (str): Type identifier, always 'xgboost'.
        task (str): Task type, either 'classification' or 'regression'.
        config_name (str): Name of the configuration preset being used.
        default_params (Dict[str, Any]): Dictionary of default hyperparameters.
        model (xgb.XGBClassifier or xgb.XGBRegressor): The underlying XGBoost model.
        feature_names (Optional[List[str]]): List of feature names.
        is_trained (bool): Flag indicating if model has been trained.

    Example:
        >>> # Classification with default settings
        >>> model = XGBoostModel(task='classification', config_name='default')
        >>> model.build_model()
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>>
        >>> # Get feature importance
        >>> importance_df = model.get_feature_importance(top_n=10)
        >>> print(importance_df)
        >>>
        >>> # Save and load
        >>> model.save('model.pkl')
        >>> loaded_model = XGBoostModel()
        >>> loaded_model.load('model.pkl')
    """

    def __init__(
        self,
        task: str = 'classification',
        config_name: str = 'default'
    ) -> None:
        """Initialize XGBoost model with specified configuration.

        Args:
            task (str, optional): Task type, either 'classification' or 'regression'.
                Defaults to 'classification'.
            config_name (str, optional): Name of the configuration preset to use.
                For classification: 'default', 'depth_9', 'depth_10'
                For regression: 'default', 'deep'
                Defaults to 'default'.

        Example:
            >>> # Default classification model
            >>> model = XGBoostModel(task='classification')
            >>>
            >>> # Deeper classification model
            >>> model = XGBoostModel(task='classification', config_name='depth_10')
            >>>
            >>> # Regression model
            >>> model = XGBoostModel(task='regression', config_name='deep')
        """
        super().__init__(model_type='xgboost', task=task)
        self.config_name = config_name

        # Load default configuration based on task type
        if task == 'classification':
            self.default_params = XGBOOST_CLASSIFIER_CONFIGS.get(
                config_name,
                XGBOOST_CLASSIFIER_CONFIGS['default']
            )
        else:
            self.default_params = XGBOOST_REGRESSOR_CONFIGS.get(
                config_name,
                XGBOOST_REGRESSOR_CONFIGS['default']
            )

    def build_model(self, params: Optional[Dict[str, Any]] = None) -> 'XGBoostModel':
        """Build XGBoost model with specified or default parameters.

        Creates an XGBoost classifier or regressor based on the task type.
        If custom parameters are provided, they are merged with the default
        parameters, with custom parameters taking precedence.

        Args:
            params (Optional[Dict[str, Any]], optional): Custom hyperparameters.
                If None, uses default parameters from the configuration.
                If provided, merges with defaults (custom params override defaults).
                Common parameters:
                - max_depth (int): Maximum tree depth
                - learning_rate (float): Step size shrinkage
                - n_estimators (int): Number of boosting rounds
                - subsample (float): Fraction of samples used per tree
                - colsample_bytree (float): Fraction of features used per tree
                - gamma (float): Minimum loss reduction for split
                Defaults to None.

        Returns:
            XGBoostModel: Self for method chaining.

        Example:
            >>> # Use default parameters
            >>> model.build_model()
            >>>
            >>> # Use custom parameters
            >>> custom_params = {
            ...     'max_depth': 12,
            ...     'learning_rate': 0.05,
            ...     'n_estimators': 2000,
            ...     'gamma': 1.0
            ... }
            >>> model.build_model(custom_params)
            >>>
            >>> # Partial override (other params use defaults)
            >>> model.build_model({'max_depth': 10})
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
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)

        return self

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> 'XGBoostModel':
        """Train XGBoost model with optional early stopping.

        Trains the XGBoost model on the provided data. If validation data is
        provided, enables early stopping to prevent overfitting by monitoring
        the validation metric.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Training features.
            y_train (Union[pd.Series, np.ndarray]): Training labels.
            X_val (Optional[Union[pd.DataFrame, np.ndarray]], optional): Validation
                features for early stopping. Defaults to None.
            y_val (Optional[Union[pd.Series, np.ndarray]], optional): Validation
                labels for early stopping. Defaults to None.
            early_stopping_rounds (int, optional): Number of rounds with no
                improvement after which training will be stopped. Only used if
                validation data is provided. Defaults to 50.
            verbose (bool, optional): Whether to print training progress.
                Defaults to True.

        Returns:
            XGBoostModel: Self for method chaining.

        Raises:
            ValueError: If model has not been built.

        Example:
            >>> # Train without early stopping
            >>> model.fit(X_train, y_train, verbose=True)
            >>>
            >>> # Train with early stopping
            >>> model.fit(
            ...     X_train, y_train,
            ...     X_val, y_val,
            ...     early_stopping_rounds=100,
            ...     verbose=True
            ... )
            >>>
            >>> # Silent training
            >>> model.fit(X_train, y_train, verbose=False)

        Note:
            Early stopping monitors the evaluation metric specified in the model
            configuration (e.g., 'logloss' for classification, 'rmse' for regression).
        """
        kwargs = {
            'verbose': verbose
        }

        # Add early stopping parameters if validation data is provided
        if X_val is not None and y_val is not None:
            kwargs['early_stopping_rounds'] = early_stopping_rounds
            kwargs['eval_metric'] = self.model.eval_metric

        # Call parent fit method with prepared kwargs
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
