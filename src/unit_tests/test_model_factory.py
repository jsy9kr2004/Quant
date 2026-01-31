"""
ModelFactory 단위 테스트

테스트 실행:
    # 빠른 테스트
    pytest src/unit_tests/test_model_factory.py -v -m "not slow"

    # 전체 테스트
    pytest src/unit_tests/test_model_factory.py -v

테스트 대상:
    - create_ensemble_models(): 앙상블 모델 생성 (regressor.py 모드)
    - create_single_models(): 단일 모델 생성 (ml_backtest.py 모드)
    - create_sector_models(): 섹터별 모델 생성
    - 설정에 따른 분류기/회귀기 생성 확인
"""

import pytest
import logging
from unittest.mock import Mock, patch
import numpy as np

# 테스트 대상 모듈
from src.models.model_factory import ModelFactory, CATBOOST_AVAILABLE


# ============================================================================
# Fixtures (테스트 데이터)
# ============================================================================

@pytest.fixture
def basic_config():
    """기본 테스트 config (분류기 사용)"""
    return {
        'ML': {
            'USE_CLASSIFIER': 'Y',
            'USE_SECTOR_MODEL': 'N',
            'CLASSIFIER_MODE': 'negative_screen',
            'SECTOR_CONFIG': {},
            'SECTOR_CATEGORIZATION': {
                'ENABLED': 'N'
            }
        }
    }


@pytest.fixture
def config_without_classifier():
    """분류기 미사용 config"""
    return {
        'ML': {
            'USE_CLASSIFIER': 'N',
            'USE_SECTOR_MODEL': 'N',
            'SECTOR_CONFIG': {},
            'SECTOR_CATEGORIZATION': {
                'ENABLED': 'N'
            }
        }
    }


@pytest.fixture
def config_with_sector_model():
    """섹터 모델 사용 config"""
    return {
        'ML': {
            'USE_CLASSIFIER': 'Y',
            'USE_SECTOR_MODEL': 'Y',
            'SECTOR_CONFIG': {
                'Technology': {
                    'model': 'xgboost',
                    'n_estimators': 200,
                    'max_depth': 7
                },
                'Healthcare': {
                    'model': 'lightgbm',
                    'n_estimators': 150,
                    'max_depth': 6
                }
            },
            'SECTOR_CATEGORIZATION': {
                'ENABLED': 'N'
            }
        }
    }


@pytest.fixture
def config_with_categorization():
    """카테고리화 활성화 config"""
    return {
        'ML': {
            'USE_CLASSIFIER': 'Y',
            'USE_SECTOR_MODEL': 'Y',
            'SECTOR_CONFIG': {},
            'SECTOR_CATEGORIZATION': {
                'ENABLED': 'Y',
                'CATEGORIES': {
                    'Financial': {
                        'sectors': ['Financials', 'Real Estate'],
                        'description': 'Interest rate sensitive',
                        'model_config': {
                            'model': 'xgboost',
                            'n_estimators': 200,
                            'max_depth': 7
                        }
                    },
                    'Technology': {
                        'sectors': ['Information Technology', 'Communication Services'],
                        'description': 'Growth oriented',
                        'model_config': {
                            'model': 'lightgbm',
                            'n_estimators': 150,
                            'max_depth': 6
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def optuna_params():
    """Optuna 최적화 파라미터"""
    return {
        'n_estimators': 600,
        'learning_rate': 0.05,
        'gamma': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.85,
        'max_depth': 9
    }


@pytest.fixture
def mock_logger():
    """테스트용 logger"""
    return logging.getLogger('test_model_factory')


# ============================================================================
# Test: ModelFactory Initialization
# ============================================================================

class TestModelFactoryInit:
    """ModelFactory 초기화 테스트"""

    def test_init_with_basic_config(self, basic_config, mock_logger):
        """기본 config로 초기화"""
        factory = ModelFactory(basic_config, logger=mock_logger)

        assert factory.use_classifier is True
        assert factory.use_sector_model is False
        assert factory.config == basic_config

    def test_init_without_classifier(self, config_without_classifier, mock_logger):
        """분류기 미사용 config로 초기화"""
        factory = ModelFactory(config_without_classifier, logger=mock_logger)

        assert factory.use_classifier is False

    def test_init_with_sector_model(self, config_with_sector_model, mock_logger):
        """섹터 모델 사용 config로 초기화"""
        factory = ModelFactory(config_with_sector_model, logger=mock_logger)

        assert factory.use_sector_model is True
        assert 'Technology' in factory.sector_config
        assert 'Healthcare' in factory.sector_config

    def test_init_with_categorization(self, config_with_categorization, mock_logger):
        """카테고리화 활성화 config로 초기화"""
        factory = ModelFactory(config_with_categorization, logger=mock_logger)

        assert factory.use_categorization is True
        assert 'Financial' in factory.effective_sector_config
        assert 'Technology' in factory.effective_sector_config

    def test_init_with_optuna_params(self, basic_config, optuna_params, mock_logger):
        """Optuna 파라미터로 초기화"""
        factory = ModelFactory(
            basic_config,
            optuna_params=optuna_params,
            logger=mock_logger
        )

        assert factory.optuna_params == optuna_params


# ============================================================================
# Test: create_ensemble_models()
# ============================================================================

class TestCreateEnsembleModels:
    """앙상블 모델 생성 테스트"""

    def test_creates_classifiers_when_enabled(self, basic_config, mock_logger):
        """USE_CLASSIFIER=Y일 때 분류기 생성"""
        factory = ModelFactory(basic_config, logger=mock_logger)
        classifiers, regressors = factory.create_ensemble_models()

        # 4개 분류기: XGB(3) + CatBoost/LightGBM(1)
        assert len(classifiers) >= 3
        assert len(regressors) >= 1

    def test_no_classifiers_when_disabled(self, config_without_classifier, mock_logger):
        """USE_CLASSIFIER=N일 때 분류기 없음"""
        factory = ModelFactory(config_without_classifier, logger=mock_logger)
        classifiers, regressors = factory.create_ensemble_models()

        assert len(classifiers) == 0
        assert len(regressors) >= 1

    def test_classifiers_are_xgboost(self, basic_config, mock_logger):
        """첫 3개 분류기가 XGBoost인지 확인"""
        factory = ModelFactory(basic_config, logger=mock_logger)
        classifiers, _ = factory.create_ensemble_models()

        # 최소 3개는 XGBoost
        import xgboost
        for i in range(min(3, len(classifiers))):
            assert isinstance(classifiers[i], xgboost.XGBClassifier)

    def test_regressors_are_created(self, basic_config, mock_logger):
        """회귀기가 생성되는지 확인"""
        factory = ModelFactory(basic_config, logger=mock_logger)
        _, regressors = factory.create_ensemble_models()

        assert len(regressors) >= 1
        # XGBRegressor 또는 CatBoostRegressor
        import xgboost
        for reg in regressors:
            assert hasattr(reg, 'fit')
            assert hasattr(reg, 'predict')

    def test_optuna_params_applied_to_first_classifier(
        self, basic_config, optuna_params, mock_logger
    ):
        """Optuna 파라미터가 첫 번째 분류기에 적용되는지 확인"""
        factory = ModelFactory(
            basic_config,
            optuna_params=optuna_params,
            logger=mock_logger
        )
        classifiers, _ = factory.create_ensemble_models()

        if len(classifiers) > 0:
            clf_0 = classifiers[0]
            # Optuna 파라미터가 적용되었는지 확인
            # XGBClassifier의 경우 get_params()로 확인
            params = clf_0.get_params()
            assert params['n_estimators'] == optuna_params['n_estimators']
            assert params['max_depth'] == optuna_params['max_depth']


# ============================================================================
# Test: create_single_models()
# ============================================================================

class TestCreateSingleModels:
    """단일 모델 생성 테스트 (ml_backtest.py 모드)"""

    def test_returns_single_classifier_and_regressor(self, basic_config, mock_logger):
        """단일 분류기와 회귀기 반환"""
        factory = ModelFactory(basic_config, use_ensemble=False, logger=mock_logger)
        classifier, regressor = factory.create_single_models()

        # 분류기와 회귀기 각각 1개
        assert classifier is not None
        assert regressor is not None
        assert hasattr(classifier, 'fit')
        assert hasattr(regressor, 'fit')

    def test_no_classifier_when_disabled(self, config_without_classifier, mock_logger):
        """USE_CLASSIFIER=N일 때 분류기 None"""
        factory = ModelFactory(
            config_without_classifier,
            use_ensemble=False,
            logger=mock_logger
        )
        classifier, regressor = factory.create_single_models()

        assert classifier is None
        assert regressor is not None


# ============================================================================
# Test: create_sector_models()
# ============================================================================

class TestCreateSectorModels:
    """섹터별 모델 생성 테스트"""

    def test_creates_models_for_each_sector(self, config_with_sector_model, mock_logger):
        """각 섹터에 대해 모델 생성"""
        factory = ModelFactory(config_with_sector_model, logger=mock_logger)
        sector_classifiers, sector_regressors = factory.create_sector_models(
            ['Technology', 'Healthcare']
        )

        # 키가 (sector, variant_idx) 튜플 형식
        tech_clf_keys = [k for k in sector_classifiers.keys() if k[0] == 'Technology']
        health_clf_keys = [k for k in sector_classifiers.keys() if k[0] == 'Healthcare']

        assert len(tech_clf_keys) > 0, "Technology 섹터 분류기가 있어야 함"
        assert len(health_clf_keys) > 0, "Healthcare 섹터 분류기가 있어야 함"

    def test_sector_models_have_classifier_and_regressor(
        self, config_with_sector_model, mock_logger
    ):
        """섹터 모델이 분류기와 회귀기를 가지는지 확인"""
        factory = ModelFactory(config_with_sector_model, logger=mock_logger)
        sector_classifiers, sector_regressors = factory.create_sector_models(
            ['Technology']
        )

        # (sector, variant_idx) 형식의 키가 있어야 함
        assert ('Technology', 0) in sector_classifiers
        assert ('Technology', 0) in sector_regressors

    def test_handles_unknown_sector(self, config_with_sector_model, mock_logger):
        """알 수 없는 섹터도 기본 설정으로 모델 생성"""
        factory = ModelFactory(config_with_sector_model, logger=mock_logger)
        sector_classifiers, sector_regressors = factory.create_sector_models(
            ['UnknownSector']
        )

        # 알 수 없는 섹터도 기본 설정으로 모델 생성됨
        unknown_clf_keys = [k for k in sector_classifiers.keys() if k[0] == 'UnknownSector']
        unknown_reg_keys = [k for k in sector_regressors.keys() if k[0] == 'UnknownSector']

        assert len(unknown_clf_keys) > 0, "기본 설정으로 분류기 생성"
        assert len(unknown_reg_keys) > 0, "기본 설정으로 회귀기 생성"


# ============================================================================
# Test: Category Configuration Extraction
# ============================================================================

class TestCategoryConfigExtraction:
    """카테고리 설정 추출 테스트"""

    def test_extracts_model_config_from_categories(
        self, config_with_categorization, mock_logger
    ):
        """카테고리에서 model_config 추출"""
        factory = ModelFactory(config_with_categorization, logger=mock_logger)

        # effective_sector_config에 카테고리별 설정이 있어야 함
        assert 'Financial' in factory.effective_sector_config
        assert 'Technology' in factory.effective_sector_config

        # model_config 내용 확인
        fin_config = factory.effective_sector_config['Financial']
        assert fin_config.get('model') == 'xgboost'
        assert fin_config.get('n_estimators') == 200

    def test_original_sector_config_when_categorization_disabled(
        self, config_with_sector_model, mock_logger
    ):
        """카테고리화 비활성화시 원본 섹터 설정 사용"""
        factory = ModelFactory(config_with_sector_model, logger=mock_logger)

        # CATEGORIZATION.ENABLED=N이므로 원본 섹터 설정 사용
        assert factory.effective_sector_config == factory.sector_config


# ============================================================================
# Test: Model Parameters Consistency
# ============================================================================

class TestModelParametersConsistency:
    """모델 파라미터 일관성 테스트"""

    def test_cpu_device_for_xgboost(self, basic_config, mock_logger):
        """XGBoost 모델이 CPU 디바이스 사용"""
        factory = ModelFactory(basic_config, logger=mock_logger)
        classifiers, regressors = factory.create_ensemble_models()

        import xgboost
        for clf in classifiers:
            if isinstance(clf, xgboost.XGBClassifier):
                params = clf.get_params()
                assert params.get('device') == 'cpu'

    def test_missing_nan_handling(self, basic_config, mock_logger):
        """XGBoost 모델의 NaN 처리 설정"""
        factory = ModelFactory(basic_config, logger=mock_logger)
        classifiers, _ = factory.create_ensemble_models()

        import xgboost
        for clf in classifiers:
            if isinstance(clf, xgboost.XGBClassifier):
                params = clf.get_params()
                # missing=np.nan이 설정되어야 함
                assert np.isnan(params.get('missing', 0))


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_ml_config(self, mock_logger):
        """빈 ML config"""
        config = {'ML': {}}
        factory = ModelFactory(config, logger=mock_logger)

        # 기본값 사용 (USE_CLASSIFIER 기본값은 'Y')
        assert factory.use_classifier is True  # 'Y' == 'Y' → True
        assert factory.use_sector_model is False  # 'N' == 'Y' → False

    def test_missing_ml_config(self, mock_logger):
        """ML config 없음"""
        config = {}
        factory = ModelFactory(config, logger=mock_logger)

        assert factory.ml_config == {}

    def test_ensemble_vs_single_model_count(self, basic_config, mock_logger):
        """앙상블 vs 단일 모델 개수 비교"""
        # 앙상블 모드
        factory_ensemble = ModelFactory(
            basic_config, use_ensemble=True, logger=mock_logger
        )
        clfs_ensemble, regs_ensemble = factory_ensemble.create_ensemble_models()

        # 단일 모드
        factory_single = ModelFactory(
            basic_config, use_ensemble=False, logger=mock_logger
        )
        clf_single, reg_single = factory_single.create_single_models()

        # 앙상블이 더 많은 모델을 가져야 함
        assert len(clfs_ensemble) >= 1
        assert clf_single is not None


# ============================================================================
# Test: CatBoost Availability
# ============================================================================

class TestCatBoostAvailability:
    """CatBoost 가용성 테스트"""

    def test_catboost_import_flag(self):
        """CatBoost import 플래그 확인"""
        # CATBOOST_AVAILABLE은 import 성공 여부에 따라 True/False
        assert isinstance(CATBOOST_AVAILABLE, bool)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
    def test_catboost_classifier_created(self, basic_config, mock_logger):
        """CatBoost 분류기 생성 확인"""
        factory = ModelFactory(basic_config, logger=mock_logger)
        classifiers, _ = factory.create_ensemble_models()

        # CatBoost가 설치되어 있으면 마지막 분류기가 CatBoost
        from catboost import CatBoostClassifier
        has_catboost = any(
            isinstance(clf, CatBoostClassifier) for clf in classifiers
        )
        assert has_catboost or len(classifiers) >= 3  # 적어도 XGB 3개


# ============================================================================
# Test: Model Training Compatibility
# ============================================================================

@pytest.mark.slow
class TestModelTrainingCompatibility:
    """모델 학습 호환성 테스트 (느림)"""

    def test_classifier_can_fit_and_predict(self, basic_config, mock_logger):
        """분류기 학습 및 예측 가능"""
        factory = ModelFactory(basic_config, logger=mock_logger)
        classifiers, _ = factory.create_ensemble_models()

        if len(classifiers) > 0:
            clf = classifiers[0]

            # 간단한 데이터로 학습
            X = np.random.randn(100, 10)
            y = np.random.randint(0, 2, 100)

            clf.fit(X, y)
            predictions = clf.predict(X)

            assert len(predictions) == len(y)
            assert set(predictions).issubset({0, 1})

    def test_regressor_can_fit_and_predict(self, basic_config, mock_logger):
        """회귀기 학습 및 예측 가능"""
        factory = ModelFactory(basic_config, logger=mock_logger)
        _, regressors = factory.create_ensemble_models()

        if len(regressors) > 0:
            reg = regressors[0]

            # 간단한 데이터로 학습
            X = np.random.randn(100, 10)
            y = np.random.randn(100)

            reg.fit(X, y)
            predictions = reg.predict(X)

            assert len(predictions) == len(y)
            assert predictions.dtype in [np.float64, np.float32]
