"""Machine learning models package for quantitative trading.

This package provides a unified interface for training, evaluating, and deploying
machine learning models for stock market prediction and trading strategy development.

The package includes:
1. Base model architecture (BaseModel)
2. Gradient boosting implementations:
   - XGBoost: Fast and efficient gradient boosting
   - LightGBM: Microsoft's gradient boosting with low memory usage
   - CatBoost: Yandex's gradient boosting with categorical feature support
3. Ensemble methods:
   - StackingEnsemble: Meta-learning based ensemble
   - VotingEnsemble: Voting/averaging based ensemble
4. Pre-configured settings optimized for financial data

Key Features:
- Consistent API across all model types
- GPU acceleration support for all models
- Built-in cross-validation and evaluation
- Feature importance analysis
- Model persistence (save/load)
- Early stopping to prevent overfitting
- Time-series aware cross-validation

Architecture:
    BaseModel (abstract)
    ├── XGBoostModel
    ├── LightGBMModel
    └── CatBoostModel

    Ensembles
    ├── StackingEnsemble
    └── VotingEnsemble

Usage Example:
    Basic model training:
        from models import XGBoostModel

        # Create and train model
        model = XGBoostModel(task='classification', config_name='default')
        model.build_model()
        model.fit(X_train, y_train, X_val, y_val)

        # Make predictions
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)

        # Save model
        model.save('trained_model.pkl')

    Ensemble training:
        from models import StackingEnsemble, XGBoostModel, LightGBMModel, CatBoostModel

        # Train base models
        xgb = XGBoostModel(task='classification').build_model()
        lgb = LightGBMModel(task='classification').build_model()
        cat = CatBoostModel(task='classification').build_model()

        xgb.fit(X_train, y_train, X_val, y_val)
        lgb.fit(X_train, y_train, X_val, y_val)
        cat.fit(X_train, y_train, X_val, y_val)

        # Create ensemble
        base_models = [
            ('xgboost', xgb.model),
            ('lightgbm', lgb.model),
            ('catboost', cat.model)
        ]

        ensemble = StackingEnsemble(
            base_models=base_models,
            task='classification',
            meta_learner='ridge'
        )
        ensemble.build_ensemble()
        ensemble.fit(X_train, y_train)

        # Make predictions
        predictions = ensemble.predict(X_test)

    Cross-validation:
        from models import CatBoostModel

        model = CatBoostModel(task='classification').build_model()

        # Cross-validate and train
        avg_scores, fold_scores = model.fit_with_cv(
            X, y,
            dates=date_series,  # For time-series CV
            cv_splits=5
        )

        print(f"CV Accuracy: {avg_scores['accuracy']:.4f}")

Available Models:
    - BaseModel: Abstract base class for all models
    - XGBoostModel: XGBoost classifier/regressor
    - LightGBMModel: LightGBM classifier/regressor
    - CatBoostModel: CatBoost classifier/regressor
    - StackingEnsemble: Stacking ensemble combiner
    - VotingEnsemble: Voting ensemble combiner (not exported by default)

Configuration:
    All models use pre-configured settings from config.py, optimized for
    financial time series data. Configurations can be customized by:
    1. Using different config_name presets
    2. Passing custom parameters to build_model()
    3. Modifying config.py directly

Model Selection Guide:
    - XGBoost: Fast training, good general performance, mature library
    - LightGBM: Fastest training, lowest memory, great for large datasets
    - CatBoost: Best for overfitting prevention, handles categorical features,
                excellent out-of-the-box performance (recommended for trading)
    - Ensemble: Best overall performance, combines strengths of multiple models

Performance Tips:
    - Use GPU acceleration (enabled by default in configs)
    - Enable early stopping to prevent overfitting
    - Use cross-validation to assess generalization
    - Start with CatBoost for robust baseline
    - Use ensembles for production systems

Note:
    - All models require GPU with CUDA support for GPU acceleration
    - Time-series cross-validation is recommended for financial data
    - Feature engineering is critical for model performance
    - Always validate on out-of-sample data

Author: Quantitative Trading Team
License: Proprietary
"""

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel
from .ensemble import StackingEnsemble

__all__ = [
    'BaseModel',
    'XGBoostModel',
    'LightGBMModel',
    'CatBoostModel',
    'StackingEnsemble'
]

__version__ = '1.0.0'

# Package metadata
__author__ = 'Quantitative Trading Team'
__description__ = 'Machine learning models for quantitative trading'
