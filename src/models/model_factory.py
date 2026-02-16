"""
시스템 전체에서 일관된 모델 생성을 위한 통합 Model Factory.

이 모듈은 regressor.py와 ml_backtest.py가 동일한 파라미터로
동일한 모델을 사용하도록 보장하는 중앙화된 ModelFactory를 제공합니다.

주요 기능:
- 모델 파라미터의 단일 진실 공급원 (Single Source of Truth)
- Optuna 하이퍼파라미터 최적화 지원
- 앙상블 모델 생성 (서로 다른 하이퍼파라미터를 가진 복수 모델)
- 섹터별 모델 생성
- 학습과 백테스팅 간 일관된 모델 설정

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
    일관된 파라미터로 ML 모델을 생성하는 팩토리.

    이 팩토리는 regressor.py와 ml_backtest.py가 정확히 동일한 모델을 사용하도록
    보장하여 학습과 백테스팅 간 불일치를 방지합니다.

    사용 예시 (regressor.py - 앙상블 모드):
        factory = ModelFactory(config, optuna_params=optuna_best_params)
        classifiers, regressors = factory.create_ensemble_models()

    사용 예시 (ml_backtest.py - 단일 모델 모드):
        factory = ModelFactory(config)
        classifier, regressor = factory.create_single_models()

    사용 예시 (섹터 모델):
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
        ModelFactory를 초기화합니다.

        Args:
            config: conf.yaml에서 로드한 설정 딕셔너리
            optuna_params: 첫 번째 분류기에 적용할 Optuna 최적화 파라미터
            use_ensemble: True이면 앙상블 모델 생성 (regressor.py 모드),
                False이면 단일 모델 생성 (ml_backtest.py 모드)
            logger: 로깅을 위한 Logger 인스턴스
        """
        self.config = config
        self.optuna_params = optuna_params
        self.use_ensemble = use_ensemble
        self.logger = logger or logging.getLogger('ModelFactory')

        # ML 설정 가져오기
        self.ml_config = config.get('ML', {})
        self.use_classifier = self.ml_config.get('USE_CLASSIFIER', 'Y') == 'Y'
        self.use_sector_model = self.ml_config.get('USE_SECTOR_MODEL', 'N') == 'Y'

        # 섹터 설정 로드 (원본 섹터별 설정)
        self.sector_config = self.ml_config.get('SECTOR_CONFIG', {}) if self.use_sector_model else {}

        # 섹터 카테고리화 설정 로드 (카테고리 통합 설정)
        self.sector_categorization = self.ml_config.get('SECTOR_CATEGORIZATION', {})
        self.use_categorization = self.sector_categorization.get('ENABLED', 'N') == 'Y'

        # 섹터 모델에 사용할 설정 결정
        if self.use_sector_model and self.use_categorization:
            # ENABLED=Y: 카테고리 기반 모델 설정 사용
            self.effective_sector_config = self._extract_category_configs()
            self.logger.info("🔧 Using category-based model configs (CATEGORIZATION.ENABLED=Y)")
        else:
            # ENABLED=N: 원본 섹터 기반 모델 설정 사용
            self.effective_sector_config = self.sector_config
            if self.use_sector_model:
                self.logger.info("🔧 Using original sector-based model configs (CATEGORIZATION.ENABLED=N)")

    def _extract_category_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        SECTOR_CATEGORIZATION의 각 카테고리에서 model_config를 추출합니다.

        Returns:
            카테고리 이름을 model_config에 매핑하는 딕셔너리.
            예시: {'Financial': {'model': 'xgboost', 'n_estimators': 200, ...}, ...}
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
        앙상블 모델을 생성합니다 (regressor.py 모드).

        앙상블 예측을 위해 서로 다른 하이퍼파라미터를 가진 복수 모델을 생성합니다.

        USE_CLASSIFIER=Y일 때:
        - 4개 분류기: XGB (depth 8,9,10) + CatBoost
        - 2개 회귀기: XGB (depth 8) + CatBoost

        USE_CLASSIFIER=N일 때:
        - 0개 분류기 (빈 리스트)
        - 2개 회귀기: XGB (depth 8) + CatBoost

        Returns:
            classifiers: 분류 모델 리스트 (USE_CLASSIFIER=N이면 빈 리스트)
            regressors: 2개의 회귀 모델 리스트
        """
        classifiers = []
        regressors = []

        # ===== 분류기 (USE_CLASSIFIER=Y일 때만) =====
        if self.use_classifier:
            # 분류기 0: XGBoost depth=8 (Optuna 파라미터가 있으면 적용)
            if self.optuna_params:
                self.logger.info(f" Using Optuna-optimized params for classifier_0: {self.optuna_params}")
                clf_0 = xgboost.XGBClassifier(
                    tree_method='hist',
                    device='cpu',  # 호환성을 위해 CPU 사용
                    n_estimators=self.optuna_params.get('n_estimators', 500),
                    learning_rate=self.optuna_params.get('learning_rate', 0.1),
                    gamma=self.optuna_params.get('gamma', 0),
                    subsample=self.optuna_params.get('subsample', 0.8),
                    colsample_bytree=self.optuna_params.get('colsample_bytree', 0.8),
                    max_depth=self.optuna_params.get('max_depth', 8),
                    objective='binary:logistic',
                    eval_metric='logloss',
                    missing=np.nan  # XGBoost가 NaN을 자동 처리 (None이 아닌 np.nan 사용)
                )
            else:
                clf_config = XGBOOST_CLASSIFIER_CONFIGS['default'].copy()
                clf_config['device'] = 'cpu'  # CPU로 오버라이드
                clf_0 = xgboost.XGBClassifier(**clf_config, missing=np.nan)

            classifiers.append(clf_0)

            # 분류기 1: XGBoost depth=9
            clf_config_9 = XGBOOST_CLASSIFIER_CONFIGS['depth_9'].copy()
            clf_config_9['device'] = 'cpu'
            clf_1 = xgboost.XGBClassifier(**clf_config_9, missing=np.nan)
            classifiers.append(clf_1)

            # 분류기 2: XGBoost depth=10
            clf_config_10 = XGBOOST_CLASSIFIER_CONFIGS['depth_10'].copy()
            clf_config_10['device'] = 'cpu'
            clf_2 = xgboost.XGBClassifier(**clf_config_10, missing=np.nan)
            classifiers.append(clf_2)

            # 분류기 3: CatBoost
            if CATBOOST_AVAILABLE:
                # CatBoost (견고한 기준선, 노이즈가 많은 feature를 잘 처리)
                cb_clf_config = self._build_catboost_config(
                    CATBOOST_CLASSIFIER_CONFIGS['default'],
                    role='classifier', is_sector=False,
                )
                clf_3 = CatBoostClassifier(**cb_clf_config)
            else:
                # 대체: LightGBM
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

        # ===== 회귀기 =====
        # 회귀기 0: XGBoost depth=8
        reg_config_8 = XGBOOST_REGRESSOR_CONFIGS['default'].copy()
        reg_config_8['device'] = 'cpu'
        reg_0 = xgboost.XGBRegressor(**reg_config_8, missing=np.nan)
        regressors.append(reg_0)

        # 회귀기 1: CatBoost (XGB 단독 대비 모델 다양성 추가)
        if CATBOOST_AVAILABLE:
            cb_reg_config = self._build_catboost_config(
                CATBOOST_REGRESSOR_CONFIGS['default'],
                role='regressor', is_sector=False,
            )
            reg_1 = CatBoostRegressor(**cb_reg_config)
            regressors.append(reg_1)
        else:
            # 대체: XGBoost depth=10
            reg_config_10 = XGBOOST_REGRESSOR_CONFIGS['depth_10'].copy()
            reg_config_10['device'] = 'cpu'
            reg_1 = xgboost.XGBRegressor(**reg_config_10, missing=np.nan)
            regressors.append(reg_1)

        self.logger.info(f" Created ensemble models: {len(classifiers)} classifiers, {len(regressors)} regressors")

        return classifiers, regressors

    def create_single_models(self, use_gpu: bool = False) -> Tuple[Any, Any]:
        """
        단일 모델을 생성합니다 (ml_backtest.py 모드).

        Walk-forward 백테스팅을 위한 단순한 단일 모델을 생성합니다.

        USE_CLASSIFIER=Y일 때:
        - 1개 분류기: XGBoost depth=8
        - 1개 회귀기: XGBoost depth=8

        USE_CLASSIFIER=N일 때:
        - 분류기: None
        - 1개 회귀기: XGBoost depth=8

        Note: 일관성 유지를 위해 앙상블의 첫 번째 모델과 동일한 파라미터를 사용합니다.

        Args:
            use_gpu: GPU 가속 사용 여부 (기본값: False)

        Returns:
            classifier: 단일 분류 모델 (USE_CLASSIFIER=N이면 None)
            regressor: 단일 회귀 모델
        """
        device = 'cuda:0' if use_gpu else 'cpu'
        # XGBoost 2.0+: GPU는 device 파라미터 사용, 모든 경우에 tree_method='hist'
        tree_method = 'hist'

        # ===== 분류기 (USE_CLASSIFIER=Y일 때만) =====
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

        # ===== 회귀기 =====
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
        섹터별 모델을 생성합니다.

        섹터별 고유 패턴을 포착하기 위해 약간 다른 하이퍼파라미터를 가진
        각 섹터별 독립 모델을 생성합니다.

        USE_CLASSIFIER=Y일 때:
        - 섹터당: 4개 분류기 (XGB depth 8/9/10 + CatBoost) + 2개 회귀기

        USE_CLASSIFIER=N일 때:
        - 섹터당: 0개 분류기 + 2개 회귀기

        **파라미터 우선순위**: Optuna > SECTOR_CONFIG > Default

        Args:
            sector_list: 섹터 이름 리스트 (예: ['Technology', 'Financial', ...])
            num_regressor_variants: 섹터당 회귀기 변형 수 (기본값: 2)
            sector_optuna_params: 섹터별 Optuna 최적화 파라미터.
                형식: {'Technology': {...}, 'Financial': {...}, ...}

        Returns:
            sector_classifiers: (섹터, 변형_인덱스)를 분류기에 매핑하는 딕셔너리.
                USE_CLASSIFIER=N이면 빈 딕셔너리
            sector_regressors: (섹터, 변형_인덱스)를 회귀기에 매핑하는 딕셔너리.
                예시: {('Technology', 0): model, ('Technology', 1): model, ...}
        """
        sector_classifiers = {}
        sector_regressors = {}

        for sector in sector_list:
            # 섹터별 또는 카테고리별 설정 가져오기
            # CATEGORIZATION.ENABLED=Y일 때: sector = 카테고리 이름 (예: "Financial")
            # CATEGORIZATION.ENABLED=N일 때: sector = 원본 섹터 이름 (예: "Financials")
            sector_cfg = self.effective_sector_config.get(sector, {})

            # Optuna 최적화 파라미터가 있으면 가져오기
            optuna_cfg = sector_optuna_params.get(sector, {}) if sector_optuna_params else {}

            # ===== 분류기 (USE_CLASSIFIER=Y일 때만) =====
            if self.use_classifier:
                # 분류기 0: XGBoost depth=8 (Optuna 파라미터가 있으면 적용)
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

                # 분류기 1: XGBoost depth=9
                clf_config_9 = XGBOOST_CLASSIFIER_CONFIGS['depth_9'].copy()
                clf_config_9['device'] = 'cpu'
                clf_1 = xgboost.XGBClassifier(**clf_config_9, missing=np.nan)
                sector_classifiers[(sector, 1)] = clf_1

                # 분류기 2: XGBoost depth=10
                clf_config_10 = XGBOOST_CLASSIFIER_CONFIGS['depth_10'].copy()
                clf_config_10['device'] = 'cpu'
                clf_2 = xgboost.XGBClassifier(**clf_config_10, missing=np.nan)
                sector_classifiers[(sector, 2)] = clf_2

                # 분류기 3: CatBoost (CatBoost 사용 불가 시 LightGBM으로 대체)
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

            # ===== 회귀기 =====
            # 기본 파라미터 (regressor.py 섹터 모델과 일치)
            # 우선순위: Optuna > SECTOR_CONFIG > Default
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
                'missing': np.nan  # ✅ 핵심 수정: NaN 처리를 위해 np.nan으로 설정
            }

            # 변형 0: XGBoost 회귀기 (섹터 설정 / Optuna 인식)
            params_0 = default_params.copy()
            base_depth = optuna_cfg.get('max_depth', 7)
            params_0['max_depth'] = base_depth
            sector_regressors[(sector, 0)] = xgboost.XGBRegressor(**params_0)

            # 변형 1: CatBoost 회귀기 (CatBoost 사용 불가 시 XGBoost depth+1로 대체)
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
        학습을 위한 GPU 사용 가능 여부를 확인합니다.

        Returns:
            GPU를 사용할 수 있으면 True, 그렇지 않으면 False
        """
        try:
            import cupy
            return True
        except ImportError:
            return False


# ============================================================================
# 편의 함수
# ============================================================================

def create_models_for_regressor(
    config: Dict[str, Any],
    optuna_params: Optional[Dict[str, Any]] = None,
    sector_list: Optional[List[str]] = None,
    use_sector_model: bool = False,
    sector_optuna_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[List[Any], List[Any], Dict[Tuple[str, int], Any], Dict[Tuple[str, int], Any]]:
    """
    regressor.py용 모든 모델을 생성합니다 (앙상블 + 섹터).

    Args:
        config: conf.yaml에서 로드한 설정
        optuna_params: 앙상블 분류기용 Optuna 최적화 파라미터
        sector_list: 섹터 모델을 위한 섹터 리스트
        use_sector_model: 섹터 모델 생성 여부
        sector_optuna_params: 섹터별 Optuna 최적화 파라미터.
            형식: {'Technology': {...}, 'Financial': {...}, ...}

    Returns:
        classifiers: 앙상블 분류 모델 (USE_CLASSIFIER=N이면 빈 리스트)
        regressors: 앙상블 회귀 모델
        sector_classifiers: 섹터별 분류기 (use_sector_model=False 또는 USE_CLASSIFIER=N이면 빈 딕셔너리)
        sector_regressors: 섹터별 회귀기 (use_sector_model=False이면 빈 딕셔너리)
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
    ml_backtest.py용 모델을 생성합니다 (단일 모델).

    **로직 일원화**: regressor.py와 동일한 Optuna 최적화 파라미터를 사용합니다.

    Args:
        config: conf.yaml에서 로드한 설정
        optuna_params: Optuna 최적화 파라미터 (regressor.py 결과에서 로드)
        use_gpu: GPU 가속 사용 여부

    Returns:
        classifier: 단일 분류 모델
        regressor: 단일 회귀 모델
    """
    factory = ModelFactory(config, optuna_params=optuna_params, use_ensemble=False)
    return factory.create_single_models(use_gpu=use_gpu)
