"""CatBoost model wrapper for gradient boosting with Yandex's CatBoost.

This module provides a wrapper class for CatBoost (Categorical Boosting), supporting
both classification and regression tasks with GPU acceleration.

CatBoost is a gradient boosting framework developed by Yandex that excels at handling
categorical features and provides robust performance out of the box. Key advantages:
- Robust to overfitting (critical for trading models)
- Handles categorical features automatically without preprocessing
- Ordered boosting reduces prediction bias
- Fast GPU training with minimal memory usage
- Built-in handling of missing values
- Lower hyperparameter tuning requirements
- Symmetric tree structure for faster prediction

The CatBoostModel class extends BaseModel and provides:
- Pre-configured settings optimized for trading (default and deep)
- Native categorical feature support
- Support for custom hyperparameters
- GPU-accelerated training
- Early stopping support
- Multiple feature importance types
- Easy integration with the training pipeline

Usage Example:
    Basic classification:
        from models.catboost_model import CatBoostModel

        # Create and build model
        model = CatBoostModel(task='classification', config_name='default')
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
            'depth': 10,
            'learning_rate': 0.05,
            'iterations': 2000,
            'l2_leaf_reg': 5
        }
        model = CatBoostModel(task='classification')
        model.build_model(custom_params)

    Deep model for complex patterns:
        # Use pre-configured deeper model
        model = CatBoostModel(task='classification', config_name='deep')
        model.build_model()

    Regression:
        model = CatBoostModel(task='regression')
        model.build_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    Feature importance:
        # CatBoost supports multiple importance types
        importance_df = model.get_feature_importance(top_n=10)
        # Uses PredictionValuesChange by default

Attributes:
    model_type (str): Always 'catboost'
    task (str): Task type ('classification' or 'regression')
    config_name (str): Name of the configuration preset
    default_params (Dict): Default hyperparameters for the model
    model: CatBoost classifier or regressor instance

Note:
    CatBoost is particularly recommended when:
    - Overfitting is a concern (uses symmetric trees and ordered boosting)
    - Working with categorical features
    - Need robust performance with minimal tuning
    - GPU memory is limited
"""

from catboost import CatBoostClassifier, CatBoostRegressor
from .base_model import BaseModel
from .config import CATBOOST_CLASSIFIER_CONFIGS, CATBOOST_REGRESSOR_CONFIGS
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np


class CatBoostModel(BaseModel):
    """CatBoost model wrapper with GPU acceleration and robust overfitting prevention.

    This class wraps CatBoost classifiers and regressors, providing a consistent
    interface compatible with the BaseModel API. CatBoost is particularly well-suited
    for trading applications due to its robustness to overfitting.

    CatBoost features:
    - Ordered boosting: Reduces prediction shift and overfitting
    - Symmetric trees: Faster prediction, balanced tree structure
    - Native categorical feature support (no encoding needed)
    - GPU acceleration via task_type='GPU'
    - Built-in regularization (L2 leaf regularization)
    - Automatic handling of missing values
    - Multiple feature importance types

    Advantages for trading:
    - Robust to overfitting (critical for financial data)
    - Handles market regime changes better
    - Less sensitive to hyperparameter choices
    - Excellent out-of-the-box performance

    Attributes:
        model_type (str): Type identifier, always 'catboost'.
        task (str): Task type, either 'classification' or 'regression'.
        config_name (str): Name of the configuration preset being used.
        default_params (Dict[str, Any]): Dictionary of default hyperparameters.
        model (CatBoostClassifier or CatBoostRegressor): The underlying CatBoost model.
        feature_names (Optional[List[str]]): List of feature names.
        is_trained (bool): Flag indicating if model has been trained.

    Example:
        >>> # Classification with default settings
        >>> model = CatBoostModel(task='classification')
        >>> model.build_model()
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>>
        >>> # Deep model for complex patterns
        >>> model = CatBoostModel(task='classification', config_name='deep')
        >>> model.build_model()
        >>>
        >>> # Get feature importance
        >>> importance_df = model.get_feature_importance(top_n=10)
    """

    def __init__(
        self,
        task: str = 'classification',
        config_name: str = 'default'
    ) -> None:
        """Initialize CatBoost model with specified configuration.

        Args:
            task (str, optional): Task type, either 'classification' or 'regression'.
                Defaults to 'classification'.
            config_name (str, optional): Name of the configuration preset to use.
                For both tasks: 'default', 'deep'
                - 'default': Depth 8, good for most cases
                - 'deep': Depth 10, for complex patterns
                Defaults to 'default'.

        Example:
            >>> # Default classification model
            >>> model = CatBoostModel(task='classification')
            >>>
            >>> # Deeper classification model for complex patterns
            >>> model = CatBoostModel(task='classification', config_name='deep')
            >>>
            >>> # Regression model
            >>> model = CatBoostModel(task='regression')
        """
        super().__init__(model_type='catboost', task=task)
        self.config_name = config_name

        # Load default configuration based on task type
        if task == 'classification':
            self.default_params = CATBOOST_CLASSIFIER_CONFIGS.get(
                config_name,
                CATBOOST_CLASSIFIER_CONFIGS['default']
            )
        else:
            self.default_params = CATBOOST_REGRESSOR_CONFIGS.get(
                config_name,
                CATBOOST_REGRESSOR_CONFIGS['default']
            )

    def build_model(self, params: Optional[Dict[str, Any]] = None) -> 'CatBoostModel':
        """Build CatBoost model with specified or default parameters.

        Creates a CatBoost classifier or regressor based on the task type.
        If custom parameters are provided, they are merged with the default
        parameters, with custom parameters taking precedence.

        Args:
            params (Optional[Dict[str, Any]], optional): Custom hyperparameters.
                If None, uses default parameters from the configuration.
                If provided, merges with defaults (custom params override defaults).
                Common parameters:
                - depth (int): Tree depth (CatBoost uses 'depth' not 'max_depth')
                - learning_rate (float): Learning rate
                - iterations (int): Number of boosting iterations
                - l2_leaf_reg (float): L2 regularization coefficient
                - subsample (float): Sample rate for bagging
                - border_count (int): Number of splits for numerical features
                - bootstrap_type (str): Bootstrap type ('Bayesian', 'Bernoulli', 'MVS')
                Defaults to None.

        Returns:
            CatBoostModel: Self for method chaining.

        Example:
            >>> # Use default parameters
            >>> model.build_model()
            >>>
            >>> # Use custom parameters
            >>> custom_params = {
            ...     'depth': 12,
            ...     'learning_rate': 0.03,
            ...     'iterations': 2000,
            ...     'l2_leaf_reg': 10
            ... }
            >>> model.build_model(custom_params)
            >>>
            >>> # Partial override (other params use defaults)
            >>> model.build_model({'depth': 10, 'iterations': 1500})
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
            self.model = CatBoostClassifier(**params)
        else:
            self.model = CatBoostRegressor(**params)

        return self

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> 'CatBoostModel':
        """Train CatBoost model with optional early stopping.

        Trains the CatBoost model on the provided data. If validation data is
        provided, enables early stopping to prevent overfitting by monitoring
        the validation metric.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Training features.
                Can contain categorical features (will be handled automatically).
            y_train (Union[pd.Series, np.ndarray]): Training labels.
            X_val (Optional[Union[pd.DataFrame, np.ndarray]], optional): Validation
                features for early stopping. Defaults to None.
            y_val (Optional[Union[pd.Series, np.ndarray]], optional): Validation
                labels for early stopping. Defaults to None.
            early_stopping_rounds (int, optional): Number of rounds with no
                improvement after which training will be stopped. Only used if
                validation data is provided. Defaults to 50.
            verbose (bool, optional): Whether to print training progress.
                If True, prints metrics every 'verbose' iterations (from config).
                Defaults to True.

        Returns:
            CatBoostModel: Self for method chaining.

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
            - CatBoost automatically detects and handles categorical features
            - Early stopping monitors the evaluation metric specified in config
              (e.g., 'AUC' for classification, 'RMSE' for regression)
            - Ordered boosting is used by default for better generalization
        """
        kwargs = {
            'verbose': verbose
        }

        # Add early stopping parameters if validation data is provided
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            kwargs['eval_set'] = eval_set
            kwargs['early_stopping_rounds'] = early_stopping_rounds

        # Call parent fit method with prepared kwargs
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)

    def get_feature_importance(self, top_n: Optional[int] = 20) -> pd.DataFrame:
        """Get feature importance scores from the trained CatBoost model.

        CatBoost provides multiple types of feature importance. This method uses
        'PredictionValuesChange' by default, which measures the average change in
        prediction when the feature value is changed.

        Available importance types in CatBoost:
        - PredictionValuesChange: Average prediction change (default)
        - LossFunctionChange: Impact on loss function
        - FeatureImportance: Split-based importance

        Args:
            top_n (Optional[int], optional): Number of top features to return.
                If None, returns all features. Defaults to 20.

        Returns:
            pd.DataFrame: DataFrame with columns 'feature' and 'importance',
                sorted by importance in descending order.

        Raises:
            ValueError: If model has not been trained.

        Example:
            >>> # Get top 10 most important features
            >>> importance_df = model.get_feature_importance(top_n=10)
            >>> print(importance_df)
            >>>
            >>> # Plot feature importance
            >>> importance_df.plot(x='feature', y='importance', kind='barh')
            >>>
            >>> # Get all features
            >>> importance_df = model.get_feature_importance(top_n=None)

        Note:
            CatBoost's feature importance is more robust than tree-based methods
            because it accounts for the ordered boosting scheme used during training.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        # Get feature importance using PredictionValuesChange (default)
        importances = self.model.get_feature_importance()

        # Use stored feature names or generate generic names
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names

        # Create DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance in descending order
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Return top N features if specified
        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df
