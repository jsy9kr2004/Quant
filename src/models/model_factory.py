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
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False
from .config import (
    XGBOOST_CLASSIFIER_CONFIGS,
    XGBOOST_REGRESSOR_CONFIGS,
    LIGHTGBM_CLASSIFIER_CONFIGS,
    LIGHTGBM_REGRESSOR_CONFIGS,
    CATBOOST_CLASSIFIER_CONFIGS,
    CATBOOST_REGRESSOR_CONFIGS
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
        self.use_classifier = self.ml_config.get('USE_CLASSIFIER', 'Y') == 'Y'
        self.use_sector_model = self.ml_config.get('USE_SECTOR_MODEL', 'N') == 'Y'

        # Load sector configuration (원본 섹터별 설정)
        self.sector_config = self.ml_config.get('SECTOR_CONFIG', {}) if self.use_sector_model else {}

        # Load sector categorization configuration (카테고리 통합 설정)
        self.sector_categorization = self.ml_config.get('SECTOR_CATEGORIZATION', {})
        self.use_categorization = self.sector_categorization.get('ENABLED', 'N') == 'Y'

        # Determine which config to use for sector models
        if self.use_sector_model and self.use_categorization:
            # ENABLED=Y: Use category-based model configs
            self.effective_sector_config = self._extract_category_configs()
            self.logger.info("🔧 Using category-based model configs (CATEGORIZATION.ENABLED=Y)")
        else:
            # ENABLED=N: Use original sector-based model configs
            self.effective_sector_config = self.sector_config
            if self.use_sector_model:
                self.logger.info("🔧 Using original sector-based model configs (CATEGORIZATION.ENABLED=N)")

    def _extract_category_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract model_config from each category in SECTOR_CATEGORIZATION.

        Returns:
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping category name to model_config
            Example: {'Financial': {'model': 'xgboost', 'n_estimators': 200, ...}, ...}
        """
        categories = self.sector_categorization.get('CATEGORIES', {})
        category_configs = {}

        for category_name, category_info in categories.items():
            model_config = category_info.get('model_config', {})
            if model_config:
                category_configs[category_name] = model_config

        return category_configs

    def _build_catboost_config(
        self,
        base_config: Dict[str, Any],
        role: str,
        is_sector: bool = False,
        sector_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """CatBoost 설정을 conf.yaml의 CATBOOST_CONFIG에서 읽어 구성합니다.

        Args:
            base_config: config.py의 기본 설정 (CATBOOST_*_CONFIGS['default'])
            role: "classifier" 또는 "regressor"
            is_sector: 섹터별 모델 여부
            sector_cfg: 섹터별 SECTOR_CONFIG (있으면 learning_rate/depth/iterations 우선 적용)

        Returns:
            CatBoost 생성자에 전달할 설정 dict
        """
        cb_config = base_config.copy()
        cat_cfg = self.ml_config.get('CATBOOST_CONFIG', {})
        role_cfg = cat_cfg.get(role.upper(), {})

        # 고정 실행 설정
        cb_config.update({
            'task_type': 'CPU',
            'verbose': False,
            'random_seed': 42,
            'allow_writing_files': False,
        })

        # 섹터 config 우선 적용 (CatBoost 파라미터명으로 매핑)
        if sector_cfg:
            if 'learning_rate' in sector_cfg:
                cb_config['learning_rate'] = sector_cfg['learning_rate']
            if 'max_depth' in sector_cfg:
                cb_config['depth'] = int(sector_cfg['max_depth'])
            if 'n_estimators' in sector_cfg:
                cb_config['iterations'] = int(sector_cfg['n_estimators'])

        # CATBOOST_CONFIG에서 공통 파라미터 setdefault
        cb_config.setdefault('l2_leaf_reg', cat_cfg.get('l2_leaf_reg', 5.0))
        cb_config.setdefault('rsm', cat_cfg.get('rsm', 0.8))
        cb_config.setdefault('learning_rate', cat_cfg.get('learning_rate', 0.05))

        # role별 파라미터 (iterations, depth)
        if is_sector:
            default_iterations = role_cfg.get('sector_iterations',
                                              role_cfg.get('iterations', 800))
        else:
            default_iterations = role_cfg.get('iterations', 800)
        cb_config.setdefault('iterations', default_iterations)
        cb_config.setdefault('depth', role_cfg.get('depth', 7))

        return cb_config

    def create_ensemble_models(self) -> Tuple[List[Any], List[Any]]:
        """
        Create ensemble models (regressor.py mode).

        Creates multiple models with different hyperparameters for ensemble prediction.

        When USE_CLASSIFIER=Y:
        - 4 Classifiers: XGB (depth 8,9,10) + CatBoost
        - 2 Regressors: XGB (depth 8) + CatBoost

        When USE_CLASSIFIER=N:
        - 0 Classifiers (empty list)
        - 2 Regressors: XGB (depth 8) + CatBoost

        Returns:
        -------
        classifiers : List[Any]
            List of classification models (empty if USE_CLASSIFIER=N)
        regressors : List[Any]
            List of 2 regression models
        """
        classifiers = []
        regressors = []

        # ===== Classifiers (only if USE_CLASSIFIER=Y) =====
        if self.use_classifier:
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

            # Classifier 3: CatBoost
            if CATBOOST_AVAILABLE:
                # CatBoost (robust baseline, handles noisy features well)
                cb_clf_config = self._build_catboost_config(
                    CATBOOST_CLASSIFIER_CONFIGS['default'],
                    role='classifier', is_sector=False,
                )
                clf_3 = CatBoostClassifier(**cb_clf_config)
            else:
                # Fallback: LightGBM
                lgb_clf_config = LIGHTGBM_CLASSIFIER_CONFIGS['default'].copy()
                lgb_clf_config['device'] = 'cpu'
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
            self.logger.info(f" Created {len(classifiers)} classifiers")
        else:
            self.logger.info(" USE_CLASSIFIER=N: Skipping classifier creation")

        # ===== Regressors =====
        # Regressor 0: XGBoost depth=8
        reg_config_8 = XGBOOST_REGRESSOR_CONFIGS['default'].copy()
        reg_config_8['device'] = 'cpu'
        reg_0 = xgboost.XGBRegressor(**reg_config_8, missing=np.nan)
        regressors.append(reg_0)

        # Regressor 1: CatBoost (additional model diversity vs XGB-only)
        if CATBOOST_AVAILABLE:
            cb_reg_config = self._build_catboost_config(
                CATBOOST_REGRESSOR_CONFIGS['default'],
                role='regressor', is_sector=False,
            )
            reg_1 = CatBoostRegressor(**cb_reg_config)
            regressors.append(reg_1)
        else:
            # Fallback: XGBoost depth=10
            reg_config_10 = XGBOOST_REGRESSOR_CONFIGS['depth_10'].copy()
            reg_config_10['device'] = 'cpu'
            reg_1 = xgboost.XGBRegressor(**reg_config_10, missing=np.nan)
            regressors.append(reg_1)

        self.logger.info(f" Created ensemble models: {len(classifiers)} classifiers, {len(regressors)} regressors")

        return classifiers, regressors

    def create_single_models(self, use_gpu: bool = False) -> Tuple[Any, Any]:
        """
        Create single models (ml_backtest.py mode).

        Creates simple single models for walk-forward backtesting.

        When USE_CLASSIFIER=Y:
        - 1 Classifier: XGBoost depth=8
        - 1 Regressor: XGBoost depth=8

        When USE_CLASSIFIER=N:
        - Classifier: None
        - 1 Regressor: XGBoost depth=8

        Note: To maintain consistency, we use the SAME parameters as ensemble's first model.

        Parameters:
        ----------
        use_gpu : bool
            Whether to use GPU acceleration (default: False)

        Returns:
        -------
        classifier : Any
            Single classification model (None if USE_CLASSIFIER=N)
        regressor : Any
            Single regression model
        """
        device = 'cuda:0' if use_gpu else 'cpu'
        # XGBoost 2.0+: Use device parameter for GPU, tree_method='hist' for all cases
        tree_method = 'hist'

        # ===== Classifier (only if USE_CLASSIFIER=Y) =====
        classifier = None
        if self.use_classifier:
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
                    missing=np.nan
                )
            else:
                clf_config = XGBOOST_CLASSIFIER_CONFIGS['default'].copy()
                clf_config['device'] = device
                clf_config['tree_method'] = tree_method
                classifier = xgboost.XGBClassifier(**clf_config, random_state=42, missing=np.nan)
    
            self.logger.info(f" Created single classifier (device={device})")
        else:
            self.logger.info(" USE_CLASSIFIER=N: Skipping single classifier creation")

        # ===== Regressor =====
        reg_config = XGBOOST_REGRESSOR_CONFIGS['default'].copy()
        reg_config['device'] = device
        reg_config['tree_method'] = tree_method
        regressor = xgboost.XGBRegressor(**reg_config, random_state=42, missing=np.nan)

        self.logger.info(f" Created single regressor (device={device})")

        return classifier, regressor

    def create_sector_models(
        self,
        sector_list: List[str],
        num_regressor_variants: int = 2,
        sector_optuna_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[Dict[Tuple[str, int], Any], Dict[Tuple[str, int], Any]]:
        """
        Create sector-specific models.

        Creates separate models for each sector with slightly different
        hyperparameters to capture sector-specific patterns.

        When USE_CLASSIFIER=Y:
        - Per sector: 4 classifiers (XGB depth 8/9/10 + CatBoost) + 2 regressors

        When USE_CLASSIFIER=N:
        - Per sector: 0 classifiers + 2 regressors

        **Parameter Priority**: Optuna > SECTOR_CONFIG > Default

        Parameters:
        ----------
        sector_list : List[str]
            List of sector names (e.g., ['Technology', 'Financial', ...])
        num_regressor_variants : int
            Number of regressor variants per sector (default: 2)
        sector_optuna_params : Optional[Dict[str, Dict[str, Any]]]
            Optuna-optimized parameters per sector
            Format: {'Technology': {...}, 'Financial': {...}, ...}

        Returns:
        -------
        sector_classifiers : Dict[Tuple[str, int], Any]
            Dictionary mapping (sector, variant_idx) to classifier
            Empty dict if USE_CLASSIFIER=N
        sector_regressors : Dict[Tuple[str, int], Any]
            Dictionary mapping (sector, variant_idx) to regressor
            Example: {('Technology', 0): model, ('Technology', 1): model, ...}
        """
        sector_classifiers = {}
        sector_regressors = {}

        for sector in sector_list:
            # Get sector-specific or category-specific config
            # When CATEGORIZATION.ENABLED=Y: sector = category name (e.g., "Financial")
            # When CATEGORIZATION.ENABLED=N: sector = original sector name (e.g., "Financials")
            sector_cfg = self.effective_sector_config.get(sector, {})

            # Get Optuna-optimized params if available
            optuna_cfg = sector_optuna_params.get(sector, {}) if sector_optuna_params else {}

            # ===== Classifiers (only if USE_CLASSIFIER=Y) =====
            if self.use_classifier:
                # Classifier 0: XGBoost depth=8 (with Optuna params if available)
                if optuna_cfg:
                    clf_0 = xgboost.XGBClassifier(
                        tree_method='hist',
                        device='cpu',
                        n_estimators=optuna_cfg.get('n_estimators', 500),
                        learning_rate=optuna_cfg.get('learning_rate', 0.1),
                        gamma=optuna_cfg.get('gamma', 0),
                        subsample=optuna_cfg.get('subsample', 0.8),
                        colsample_bytree=optuna_cfg.get('colsample_bytree', 0.8),
                        max_depth=optuna_cfg.get('max_depth', 8),
                        objective='binary:logistic',
                        eval_metric='logloss',
                        missing=np.nan
                    )
                else:
                    clf_config = XGBOOST_CLASSIFIER_CONFIGS['default'].copy()
                    clf_config['device'] = 'cpu'
                    clf_0 = xgboost.XGBClassifier(**clf_config, missing=np.nan)
                sector_classifiers[(sector, 0)] = clf_0

                # Classifier 1: XGBoost depth=9
                clf_config_9 = XGBOOST_CLASSIFIER_CONFIGS['depth_9'].copy()
                clf_config_9['device'] = 'cpu'
                clf_1 = xgboost.XGBClassifier(**clf_config_9, missing=np.nan)
                sector_classifiers[(sector, 1)] = clf_1

                # Classifier 2: XGBoost depth=10
                clf_config_10 = XGBOOST_CLASSIFIER_CONFIGS['depth_10'].copy()
                clf_config_10['device'] = 'cpu'
                clf_2 = xgboost.XGBClassifier(**clf_config_10, missing=np.nan)
                sector_classifiers[(sector, 2)] = clf_2

                # Classifier 3: CatBoost (fallback to LightGBM if CatBoost unavailable)
                if CATBOOST_AVAILABLE:
                    cb_clf_config = self._build_catboost_config(
                        CATBOOST_CLASSIFIER_CONFIGS['default'],
                        role='classifier', is_sector=True, sector_cfg=sector_cfg,
                    )
                    sector_classifiers[(sector, 3)] = CatBoostClassifier(**cb_clf_config)
                else:
                    lgb_clf_config = LIGHTGBM_CLASSIFIER_CONFIGS['default'].copy()
                    lgb_clf_config['device'] = 'cpu'
                    sector_classifiers[(sector, 3)] = lgb.LGBMClassifier(
                        boosting_type=lgb_clf_config.get('boosting_type', 'gbdt'),
                        objective=lgb_clf_config.get('objective', 'binary'),
                        n_estimators=lgb_clf_config.get('n_estimators', 1000),
                        max_depth=lgb_clf_config.get('max_depth', 8),
                        learning_rate=lgb_clf_config.get('learning_rate', 0.1),
                        device='cpu',
                        boost_from_average=False
                    )

            # ===== Regressors =====
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
                'missing': np.nan  # ✅ CRITICAL FIX: Set to np.nan for NaN handling
            }

            # Variant 0: XGBoost regressor (sector-config / optuna aware)
            params_0 = default_params.copy()
            base_depth = optuna_cfg.get('max_depth', 7)
            params_0['max_depth'] = base_depth
            sector_regressors[(sector, 0)] = xgboost.XGBRegressor(**params_0)

            # Variant 1: CatBoost regressor (fallback to XGBoost depth+1 if CatBoost unavailable)
            if CATBOOST_AVAILABLE:
                cb_reg_config = self._build_catboost_config(
                    CATBOOST_REGRESSOR_CONFIGS['default'],
                    role='regressor', is_sector=True, sector_cfg=sector_cfg,
                )
                sector_regressors[(sector, 1)] = CatBoostRegressor(**cb_reg_config)
            else:
                params_1 = default_params.copy()
                params_1['max_depth'] = base_depth + 1
                sector_regressors[(sector, 1)] = xgboost.XGBRegressor(**params_1)

            param_source = "Optuna" if sector in (sector_optuna_params or {}) else "SECTOR_CONFIG"
            self.logger.debug(
                f"Created sector models: {sector} "
                f"(0)=XGB depth={params_0['max_depth']} ({param_source}), "
                f"(1)={'CatBoost' if CATBOOST_AVAILABLE else 'XGB'}"
            )

        optuna_count = len([s for s in sector_list if s in (sector_optuna_params or {})])

        if self.use_classifier:
            num_clf_per_sector = 4
            self.logger.info(f"✅ Created sector models: {len(sector_list)} sectors")
            self.logger.info(f"   Classifiers: {len(sector_list)} sectors x {num_clf_per_sector} variants = {len(sector_classifiers)} models")
            self.logger.info(f"   Regressors: {len(sector_list)} sectors x {num_regressor_variants} variants = {len(sector_regressors)} models")
        else:
            self.logger.info(f"✅ Created sector models: {len(sector_list)} sectors x {num_regressor_variants} regressors = {len(sector_regressors)} models")
            self.logger.info(f"   USE_CLASSIFIER=N: No sector classifiers created")

        if optuna_count > 0:
            self.logger.info(f"   {optuna_count}/{len(sector_list)} sectors using Optuna-optimized params")

        return sector_classifiers, sector_regressors

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
        except ImportError:
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
) -> Tuple[List[Any], List[Any], Dict[Tuple[str, int], Any], Dict[Tuple[str, int], Any]]:
    """
    Create all models for regressor.py (ensemble + sectors).

    Parameters:
    ----------
    config : Dict[str, Any]
        Configuration from conf.yaml
    optuna_params : Optional[Dict[str, Any]]
        Optuna-optimized parameters for ensemble classifier
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
        Ensemble classification models (empty if USE_CLASSIFIER=N)
    regressors : List[Any]
        Ensemble regression models
    sector_classifiers : Dict[Tuple[str, int], Any]
        Sector-specific classifiers (empty if use_sector_model=False or USE_CLASSIFIER=N)
    sector_regressors : Dict[Tuple[str, int], Any]
        Sector-specific regressors (empty if use_sector_model=False)
    """
    factory = ModelFactory(config, optuna_params=optuna_params, use_ensemble=True)

    classifiers, regressors = factory.create_ensemble_models()

    sector_classifiers = {}
    sector_regressors = {}
    if use_sector_model and sector_list:
        sector_classifiers, sector_regressors = factory.create_sector_models(
            sector_list, sector_optuna_params=sector_optuna_params
        )

    return classifiers, regressors, sector_classifiers, sector_regressors


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
