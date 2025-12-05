"""
Unified Model Factory for consistent model creation across the system.

This module provides a centralized ModelFactory that ensures regressor.py
and ml_backtest.py use identical models with identical parameters.

Key Features:
- Single source of truth for model parameters
- Optuna hyperparameter optimization support
- Ensemble model creation (multiple models with different hyperparameters)
- Sector-specific model creation
- Consistent model configuration across training and backtesting

Author: Quant Trading Team
Date: 2025-11-28
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import xgboost
import lightgbm as lgb
from .config import (
    XGBOOST_CLASSIFIER_CONFIGS,
    XGBOOST_REGRESSOR_CONFIGS,
    LIGHTGBM_CLASSIFIER_CONFIGS,
    LIGHTGBM_REGRESSOR_CONFIGS
)


class ModelFactory:
    """
    Factory for creating ML models with consistent parameters.

    This factory ensures that regressor.py and ml_backtest.py use the exact same
    models, preventing inconsistencies between training and backtesting.

    Usage (regressor.py - ensemble mode):
        factory = ModelFactory(config, optuna_params=optuna_best_params)
        classifiers, regressors = factory.create_ensemble_models()

    Usage (ml_backtest.py - single model mode):
        factory = ModelFactory(config)
        classifier, regressor = factory.create_single_models()

    Usage (sector models):
        factory = ModelFactory(config)
        sector_models = factory.create_sector_models(sector_list)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        optuna_params: Optional[Dict[str, Any]] = None,
        use_ensemble: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ModelFactory.

        Parameters:
        ----------
        config : Dict[str, Any]
            Configuration dictionary from conf.yaml
        optuna_params : Optional[Dict[str, Any]]
            Optuna-optimized parameters to apply to the first classifier
        use_ensemble : bool
            If True, create ensemble models (regressor.py mode)
            If False, create single models (ml_backtest.py mode)
        logger : Optional[logging.Logger]
            Logger instance for logging
        """
        self.config = config
        self.optuna_params = optuna_params
        self.use_ensemble = use_ensemble
        self.logger = logger or logging.getLogger('ModelFactory')

        # Get ML configuration
        self.ml_config = config.get('ML', {})
        self.use_sector_model = self.ml_config.get('USE_SECTOR_MODEL', 'N') == 'Y'
        self.sector_config = self.ml_config.get('SECTOR_CONFIG', {}) if self.use_sector_model else {}

    def create_ensemble_models(self) -> Tuple[List[Any], List[Any]]:
        """
        Create ensemble models (regressor.py mode).

        Creates multiple models with different hyperparameters for ensemble prediction:
        - 4 Classifiers: XGB (depth 8,9,10) + LGBM
        - 2 Regressors: XGB (depth 8,10)

        Returns:
        -------
        classifiers : List[Any]
            List of 4 classification models [clsmodel_0, clsmodel_1, clsmodel_2, clsmodel_3]
        regressors : List[Any]
            List of 2 regression models [model_0, model_1]
        """
        classifiers = []
        regressors = []

        # ===== Classifiers =====
        # Classifier 0: XGBoost depth=8 (with Optuna params if available)
        if self.optuna_params:
            self.logger.info(f" Using Optuna-optimized params for classifier_0: {self.optuna_params}")
            clf_0 = xgboost.XGBClassifier(
                tree_method='hist',
                device='cpu',  # Use CPU for compatibility
                n_estimators=self.optuna_params.get('n_estimators', 500),
                learning_rate=self.optuna_params.get('learning_rate', 0.1),
                gamma=self.optuna_params.get('gamma', 0),
                subsample=self.optuna_params.get('subsample', 0.8),
                colsample_bytree=self.optuna_params.get('colsample_bytree', 0.8),
                max_depth=self.optuna_params.get('max_depth', 8),
                objective='binary:logistic',
                eval_metric='logloss',
                missing=np.nan  # XGBoost handles NaN automatically (use np.nan, not None)
            )
        else:
            clf_config = XGBOOST_CLASSIFIER_CONFIGS['default'].copy()
            clf_config['device'] = 'cpu'  # Override to CPU
            clf_0 = xgboost.XGBClassifier(**clf_config, missing=np.nan)

        classifiers.append(clf_0)

        # Classifier 1: XGBoost depth=9
        clf_config_9 = XGBOOST_CLASSIFIER_CONFIGS['depth_9'].copy()
        clf_config_9['device'] = 'cpu'
        clf_1 = xgboost.XGBClassifier(**clf_config_9, missing=np.nan)
        classifiers.append(clf_1)

        # Classifier 2: XGBoost depth=10
        clf_config_10 = XGBOOST_CLASSIFIER_CONFIGS['depth_10'].copy()
        clf_config_10['device'] = 'cpu'
        clf_2 = xgboost.XGBClassifier(**clf_config_10, missing=np.nan)
        classifiers.append(clf_2)

        # Classifier 3: LightGBM
        lgb_clf_config = LIGHTGBM_CLASSIFIER_CONFIGS['default'].copy()
        lgb_clf_config['device'] = 'cpu'
        # LightGBM uses different parameter structure
        clf_3 = lgb.LGBMClassifier(
            boosting_type=lgb_clf_config.get('boosting_type', 'gbdt'),
            objective=lgb_clf_config.get('objective', 'binary'),
            n_estimators=lgb_clf_config.get('n_estimators', 1000),
            max_depth=lgb_clf_config.get('max_depth', 8),
            learning_rate=lgb_clf_config.get('learning_rate', 0.1),
            device='cpu',
            boost_from_average=False
        )
        classifiers.append(clf_3)

        # ===== Regressors =====
        # Regressor 0: XGBoost depth=8
        reg_config_8 = XGBOOST_REGRESSOR_CONFIGS['default'].copy()
        reg_config_8['device'] = 'cpu'
        reg_0 = xgboost.XGBRegressor(**reg_config_8, missing=np.nan)
        regressors.append(reg_0)

        # Regressor 1: XGBoost depth=10
        reg_config_10 = XGBOOST_REGRESSOR_CONFIGS['depth_10'].copy()
        reg_config_10['device'] = 'cpu'
        reg_1 = xgboost.XGBRegressor(**reg_config_10, missing=np.nan)
        regressors.append(reg_1)

        self.logger.info(f" Created ensemble models: {len(classifiers)} classifiers, {len(regressors)} regressors")

        return classifiers, regressors

    def create_single_models(self, use_gpu: bool = False) -> Tuple[Any, Any]:
        """
        Create single models (ml_backtest.py mode).

        Creates simple single models for walk-forward backtesting:
        - 1 Classifier: XGBoost depth=8
        - 1 Regressor: XGBoost depth=8

        Note: To maintain consistency, we use the SAME parameters as ensemble's first model.

        Parameters:
        ----------
        use_gpu : bool
            Whether to use GPU acceleration (default: False)

        Returns:
        -------
        classifier : Any
            Single classification model
        regressor : Any
            Single regression model
        """
        device = 'cuda:0' if use_gpu else 'cpu'
        tree_method = 'gpu_hist' if use_gpu else 'hist'

        # Use the SAME config as ensemble's first models to ensure consistency
        if self.optuna_params:
            self.logger.info(f" Using Optuna-optimized params for single models: {self.optuna_params}")
            classifier = xgboost.XGBClassifier(
                tree_method=tree_method,
                device=device,
                n_estimators=self.optuna_params.get('n_estimators', 500),
                learning_rate=self.optuna_params.get('learning_rate', 0.1),
                gamma=self.optuna_params.get('gamma', 0),
                subsample=self.optuna_params.get('subsample', 0.8),
                colsample_bytree=self.optuna_params.get('colsample_bytree', 0.8),
                max_depth=self.optuna_params.get('max_depth', 8),
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                missing=np.nan  # ✅ CRITICAL FIX: Use np.nan, not None
            )
        else:
            clf_config = XGBOOST_CLASSIFIER_CONFIGS['default'].copy()
            clf_config['device'] = device
            clf_config['tree_method'] = tree_method
            classifier = xgboost.XGBClassifier(**clf_config, random_state=42, missing=np.nan)

        # Regressor: Use same config as ensemble's first regressor
        reg_config = XGBOOST_REGRESSOR_CONFIGS['default'].copy()
        reg_config['device'] = device
        reg_config['tree_method'] = tree_method
        regressor = xgboost.XGBRegressor(**reg_config, random_state=42, missing=np.nan)

        self.logger.info(f" Created single models (device={device})")

        return classifier, regressor

    def create_sector_models(
        self,
        sector_list: List[str],
        num_variants: int = 2,
        sector_optuna_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[Tuple[str, int], Any]:
        """
        Create sector-specific models.

        Creates separate regression models for each sector with slightly different
        hyperparameters to capture sector-specific patterns.

        **Parameter Priority**: Optuna > SECTOR_CONFIG > Default

        Parameters:
        ----------
        sector_list : List[str]
            List of sector names (e.g., ['Technology', 'Financial', ...])
        num_variants : int
            Number of model variants per sector (default: 2)
        sector_optuna_params : Optional[Dict[str, Dict[str, Any]]]
            Optuna-optimized parameters per sector
            Format: {'Technology': {...}, 'Financial': {...}, ...}

        Returns:
        -------
        sector_models : Dict[Tuple[str, int], Any]
            Dictionary mapping (sector, variant_idx) to model
            Example: {('Technology', 0): model, ('Technology', 1): model, ...}
        """
        sector_models = {}

        for sector in sector_list:
            # Get sector-specific config if available
            sector_cfg = self.sector_config.get(sector, {})

            # Get Optuna-optimized params if available
            optuna_cfg = sector_optuna_params.get(sector, {}) if sector_optuna_params else {}

            # Default parameters (matching regressor.py sector models)
            # Priority: Optuna > SECTOR_CONFIG > Default
            default_params = {
                'tree_method': 'hist',
                'device': 'cpu',
                'n_estimators': optuna_cfg.get('n_estimators') or sector_cfg.get('n_estimators', 1000),
                'learning_rate': optuna_cfg.get('learning_rate') or sector_cfg.get('learning_rate', 0.05),
                'gamma': optuna_cfg.get('gamma', 0.01),
                'subsample': optuna_cfg.get('subsample', 0.8),
                'colsample_bytree': optuna_cfg.get('colsample_bytree', 0.7),
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'missing': None
            }

            # Create variants with different max_depth
            for variant_idx in range(num_variants):
                params = default_params.copy()
                # max_depth: Use Optuna if available, else increment from base
                base_depth = optuna_cfg.get('max_depth', 7)
                params['max_depth'] = base_depth + variant_idx  # variant 0: depth, variant 1: depth+1

                # ✅ CRITICAL FIX: Add missing=np.nan for NaN handling
                model = xgboost.XGBRegressor(**params, missing=np.nan)
                sector_models[(sector, variant_idx)] = model

                param_source = "Optuna" if sector in (sector_optuna_params or {}) else "SECTOR_CONFIG"
                self.logger.debug(f"Created sector model: {sector} variant {variant_idx}, max_depth={params['max_depth']} ({param_source})")

        optuna_count = len([s for s in sector_list if s in (sector_optuna_params or {})])
        self.logger.info(f"✅ Created sector models: {len(sector_list)} sectors x {num_variants} variants = {len(sector_models)} models")
        if optuna_count > 0:
            self.logger.info(f"   {optuna_count}/{len(sector_list)} sectors using Optuna-optimized params")

        return sector_models

    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available for training.

        Returns:
        -------
        bool
            True if GPU is available, False otherwise
        """
        try:
            import cupy
            return True
        except:
            return False


# ============================================================================
# Convenience Functions
# ============================================================================

def create_models_for_regressor(
    config: Dict[str, Any],
    optuna_params: Optional[Dict[str, Any]] = None,
    sector_list: Optional[List[str]] = None,
    use_sector_model: bool = False,
    sector_optuna_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[List[Any], List[Any], Dict[Tuple[str, int], Any]]:
    """
    Create all models for regressor.py (ensemble + sectors).

    Parameters:
    ----------
    config : Dict[str, Any]
        Configuration from conf.yaml
    optuna_params : Optional[Dict[str, Any]]
        Optuna-optimized parameters for classifier
    sector_list : Optional[List[str]]
        List of sectors for sector models
    use_sector_model : bool
        Whether to create sector models
    sector_optuna_params : Optional[Dict[str, Dict[str, Any]]]
        Optuna-optimized parameters per sector
        Format: {'Technology': {...}, 'Financial': {...}, ...}

    Returns:
    -------
    classifiers : List[Any]
        4 classification models
    regressors : List[Any]
        2 regression models
    sector_models : Dict[Tuple[str, int], Any]
        Sector-specific models (empty if use_sector_model=False)
    """
    factory = ModelFactory(config, optuna_params=optuna_params, use_ensemble=True)

    classifiers, regressors = factory.create_ensemble_models()

    sector_models = {}
    if use_sector_model and sector_list:
        sector_models = factory.create_sector_models(sector_list, sector_optuna_params=sector_optuna_params)

    return classifiers, regressors, sector_models


def create_models_for_backtest(
    config: Dict[str, Any],
    optuna_params: Optional[Dict[str, Any]] = None,
    use_gpu: bool = False
) -> Tuple[Any, Any]:
    """
    Create models for ml_backtest.py (single models).

    **Logic Unification**: Uses same Optuna-optimized parameters as regressor.py

    Parameters:
    ----------
    config : Dict[str, Any]
        Configuration from conf.yaml
    optuna_params : Optional[Dict[str, Any]]
        Optuna-optimized parameters (loaded from regressor.py results)
    use_gpu : bool
        Whether to use GPU acceleration

    Returns:
    -------
    classifier : Any
        Single classification model
    regressor : Any
        Single regression model
    """
    factory = ModelFactory(config, optuna_params=optuna_params, use_ensemble=False)
    return factory.create_single_models(use_gpu=use_gpu)
