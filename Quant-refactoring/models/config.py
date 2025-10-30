"""Model configuration settings for machine learning algorithms.

This module contains pre-configured hyperparameter settings for XGBoost, LightGBM,
and CatBoost models, optimized for quantitative trading applications.

The configurations are organized by:
1. Model type (XGBoost, LightGBM, CatBoost)
2. Task type (Classification, Regression)
3. Configuration variants (default, deep, etc.)

All configurations use GPU acceleration for faster training. The parameters are
tuned based on extensive experimentation with financial time series data.

Configuration Structure:
    - XGBOOST_CLASSIFIER_CONFIGS: XGBoost classification settings
    - XGBOOST_REGRESSOR_CONFIGS: XGBoost regression settings
    - LIGHTGBM_CLASSIFIER_CONFIGS: LightGBM classification settings
    - LIGHTGBM_REGRESSOR_CONFIGS: LightGBM regression settings
    - CATBOOST_CLASSIFIER_CONFIGS: CatBoost classification settings
    - CATBOOST_REGRESSOR_CONFIGS: CatBoost regression settings
    - OPTUNA_SEARCH_SPACE: Hyperparameter search ranges for optimization
    - ENSEMBLE_CONFIG: Ensemble method configurations
    - THRESHOLD_PERCENTILE: Stock selection threshold

Usage Example:
    from models.config import XGBOOST_CLASSIFIER_CONFIGS

    # Get default XGBoost classifier configuration
    default_config = XGBOOST_CLASSIFIER_CONFIGS['default']

    # Get deeper XGBoost classifier configuration
    deep_config = XGBOOST_CLASSIFIER_CONFIGS['depth_10']

    # Create custom configuration by modifying defaults
    custom_config = XGBOOST_CLASSIFIER_CONFIGS['default'].copy()
    custom_config['learning_rate'] = 0.05
    custom_config['max_depth'] = 10

Note:
    - All GPU-based configs require CUDA-compatible GPU
    - tree_method='gpu_hist' is used for XGBoost GPU acceleration
    - device='gpu' is used for LightGBM GPU acceleration
    - task_type='GPU' is used for CatBoost GPU acceleration
"""

from typing import Dict, Any, Tuple

# =============================================================================
# XGBoost Configurations
# =============================================================================

XGBOOST_CLASSIFIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'tree_method': 'gpu_hist',      # GPU-accelerated histogram algorithm
        'gpu_id': 0,                     # GPU device ID (0 for first GPU)
        'n_estimators': 500,             # Number of boosting rounds
        'learning_rate': 0.1,            # Step size shrinkage (eta)
        'gamma': 0,                      # Minimum loss reduction for split
        'subsample': 0.8,                # Fraction of samples per tree (0.8 = 80%)
        'colsample_bytree': 0.8,         # Fraction of features per tree (0.8 = 80%)
        'max_depth': 8,                  # Maximum tree depth
        'objective': 'binary:logistic',  # Binary classification objective
        'eval_metric': 'logloss'         # Evaluation metric (log loss)
    },
    'depth_9': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 500,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 9,                  # Deeper trees for complex patterns
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    },
    'depth_10': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 500,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 10,                 # Even deeper trees for very complex patterns
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
}

XGBOOST_REGRESSOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 1000,            # More iterations for regression (1000)
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 8,
        'objective': 'reg:squarederror',  # Squared error regression objective
        'eval_metric': 'rmse'             # Root mean squared error
    },
    'deep': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 10,                  # Deeper trees for regression
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
}

# =============================================================================
# LightGBM Configurations
# =============================================================================

LIGHTGBM_CLASSIFIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'boosting_type': 'gbdt',          # Gradient Boosting Decision Tree
        'objective': 'binary',            # Binary classification
        'n_estimators': 1000,             # Number of boosting iterations
        'max_depth': 8,                   # Maximum tree depth
        'learning_rate': 0.1,             # Learning rate (eta)
        'device': 'gpu',                  # Use GPU for training
        'boost_from_average': False       # Don't boost from average (better for imbalanced data)
    }
}

LIGHTGBM_REGRESSOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'boosting_type': 'gbdt',          # Gradient Boosting Decision Tree
        'objective': 'regression',        # Regression objective
        'n_estimators': 1000,             # Number of boosting iterations
        'max_depth': 8,                   # Maximum tree depth
        'learning_rate': 0.1,             # Learning rate
        'subsample': 0.8,                 # Fraction of samples per tree
        'colsample_bytree': 0.8,          # Fraction of features per tree
        'device': 'gpu'                   # Use GPU for training
    }
}

# =============================================================================
# CatBoost Configurations
# =============================================================================

CATBOOST_CLASSIFIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'iterations': 1000,               # Number of boosting iterations
        'learning_rate': 0.1,             # Learning rate
        'depth': 8,                       # Tree depth (CatBoost uses 'depth' not 'max_depth')
        'task_type': 'GPU',               # Use GPU for training
        'loss_function': 'Logloss',       # Log loss for binary classification
        'eval_metric': 'AUC',             # Area Under ROC Curve for evaluation
        'bootstrap_type': 'Bernoulli',    # Bernoulli bootstrap (random sampling)
        'subsample': 0.8,                 # Sample rate for Bernoulli bootstrap
        'verbose': 100                    # Print metrics every 100 iterations
    },
    'deep': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 10,                      # Deeper trees for complex patterns
        'task_type': 'GPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    }
}

CATBOOST_REGRESSOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 8,
        'task_type': 'GPU',
        'loss_function': 'RMSE',          # Root Mean Squared Error
        'eval_metric': 'RMSE',            # RMSE for evaluation
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    },
    'deep': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 10,                      # Deeper trees for regression
        'task_type': 'GPU',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    }
}

# =============================================================================
# Optuna Hyperparameter Search Spaces
# =============================================================================

# Define search ranges for hyperparameter optimization using Optuna
# Each tuple represents (min_value, max_value) for the parameter
OPTUNA_SEARCH_SPACE: Dict[str, Dict[str, Tuple[int, int] | Tuple[float, float]]] = {
    'xgboost': {
        'max_depth': (6, 12),              # Tree depth range
        'learning_rate': (0.01, 0.3),      # Learning rate range
        'n_estimators': (500, 2000),       # Number of trees range
        'subsample': (0.6, 1.0),           # Sample rate range
        'colsample_bytree': (0.6, 1.0),    # Feature fraction range
        'gamma': (0, 5),                   # Min loss reduction range
        'min_child_weight': (1, 10)        # Min sum of instance weight range
    },
    'lightgbm': {
        'max_depth': (6, 12),              # Tree depth range
        'learning_rate': (0.01, 0.3),      # Learning rate range
        'n_estimators': (500, 2000),       # Number of trees range
        'num_leaves': (20, 100),           # Max leaves per tree (LightGBM specific)
        'min_child_samples': (10, 50),     # Min samples per leaf
        'subsample': (0.6, 1.0),           # Sample rate range
        'colsample_bytree': (0.6, 1.0)     # Feature fraction range
    },
    'catboost': {
        'depth': (6, 12),                  # Tree depth range
        'learning_rate': (0.01, 0.3),      # Learning rate range
        'iterations': (500, 2000),         # Number of iterations range
        'l2_leaf_reg': (1, 10),            # L2 regularization coefficient
        'subsample': (0.6, 1.0)            # Sample rate range
    }
}

# =============================================================================
# Ensemble Configurations
# =============================================================================

ENSEMBLE_CONFIG: Dict[str, Dict[str, Any]] = {
    'stacking': {
        'cv_folds': 5,                     # Cross-validation folds for meta-learner training
        'meta_learner': 'ridge',           # Meta-learner: 'ridge', 'lasso', or 'elasticnet'
        'meta_learner_params': {
            'alpha': 1.0                   # Regularization strength for Ridge regression
        }
    },
    'voting': {
        'voting': 'soft',                  # Voting type: 'hard' (majority) or 'soft' (probability average)
        'weights': None                    # Model weights: None for equal weights, or list of floats
    }
}

# =============================================================================
# Model Selection Configuration
# =============================================================================

# Threshold percentile for stock selection
# 92nd percentile means selecting top 8% of stocks based on model predictions
# Higher percentile = more selective (fewer stocks, potentially higher quality)
# Lower percentile = less selective (more stocks, better diversification)
THRESHOLD_PERCENTILE: int = 92  # Select stocks with predictions above 92nd percentile (top 8%)
