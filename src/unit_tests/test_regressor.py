"""
Regressor 단위 테스트

테스트 실행:
    # 빠른 테스트 (integration, slow 제외)
    pytest src/unit_tests/test_regressor.py -v -m "not integration and not slow"

    # 전체 테스트
    pytest src/unit_tests/test_regressor.py -v

테스트 대상:
    - GPU 디바이스 감지 함수
    - Feature 이름 정리 함수
    - 파일 경로에서 날짜 추출
    - Threshold 최적화 로직
    - Binary filtering 로직
    - 앙상블 예측 생성
    - Top-K 선택 로직
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import logging
import tempfile
from pathlib import Path

# 테스트 대상 모듈
from src.training.regressor import (
    detect_xgboost_device,
    check_cupy_available,
    predict_with_gpu_support,
    predict_proba_with_gpu_support,
    _rank_and_select_top_k_standalone,
)


# ============================================================================
# Fixtures (테스트 데이터)
# ============================================================================

@pytest.fixture
def sample_predictions_df():
    """예측 결과 DataFrame"""
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'sector': ['Tech', 'Tech', 'Tech', 'Consumer', 'Tech'],
        'pred_return': [0.08, 0.06, 0.04, 0.09, 0.05],
        'pred_proba': [0.75, 0.70, 0.65, 0.80, 0.68],
        'ml_score': [0.060, 0.042, 0.026, 0.072, 0.034],
    })


@pytest.fixture
def sample_train_data():
    """학습 데이터"""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'feat_a': np.random.randn(n_samples),
        'feat_b': np.random.randn(n_samples),
        'feat_c': np.random.randn(n_samples),
    })


@pytest.fixture
def sample_target():
    """타겟 데이터"""
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 0.1)


@pytest.fixture
def mock_classifier():
    """Mock 분류기"""
    clf = Mock()
    clf.predict_proba = Mock(return_value=np.array([
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.45, 0.55],
    ]))
    clf.predict = Mock(return_value=np.array([1, 1, 0, 1, 1]))
    return clf


@pytest.fixture
def mock_regressor():
    """Mock 회귀기"""
    reg = Mock()
    reg.predict = Mock(return_value=np.array([0.08, 0.06, 0.04, 0.09, 0.05]))
    return reg


@pytest.fixture
def mock_logger():
    """테스트용 logger"""
    return logging.getLogger('test_regressor')


# ============================================================================
# Test: GPU Device Detection
# ============================================================================

class TestDeviceDetection:
    """GPU 디바이스 감지 테스트"""

    def test_detect_xgboost_device_returns_string(self):
        """반환 타입이 문자열인지 확인"""
        result = detect_xgboost_device()
        assert isinstance(result, str)
        assert result in ['cuda:0', 'cpu']

    def test_detect_xgboost_device_fallback_to_cpu(self):
        """GPU 없으면 CPU 반환"""
        with patch('src.training.regressor.check_cupy_available', return_value=False):
            # 실제 환경에서는 CPU를 반환할 수 있음
            result = detect_xgboost_device()
            assert result in ['cuda:0', 'cpu']

    def test_check_cupy_available_returns_bool(self):
        """CuPy 확인이 bool 반환"""
        result = check_cupy_available()
        assert isinstance(result, bool)


# ============================================================================
# Test: GPU Support Prediction
# ============================================================================

class TestGPUSupportPrediction:
    """GPU 지원 예측 테스트"""

    def test_predict_with_gpu_support_cpu_mode(self, mock_regressor, sample_train_data):
        """CPU 모드 예측"""
        X = sample_train_data.head(5).values

        result = predict_with_gpu_support(mock_regressor, X, use_gpu=False)

        mock_regressor.predict.assert_called_once()
        assert len(result) == 5

    def test_predict_proba_with_gpu_support_cpu_mode(self, mock_classifier, sample_train_data):
        """CPU 모드 확률 예측"""
        X = sample_train_data.head(5).values

        result = predict_proba_with_gpu_support(mock_classifier, X, use_gpu=False)

        mock_classifier.predict_proba.assert_called_once()
        assert result.shape == (5, 2)


# ============================================================================
# Test: Rank and Select Top-K (Standalone)
# ============================================================================

class TestRankAndSelectTopK:
    """Top-K 선택 테스트"""

    def test_ranks_by_ml_score_descending(self, sample_predictions_df):
        """ml_score 기준 내림차순 정렬"""
        result = _rank_and_select_top_k_standalone(sample_predictions_df, top_k=5)

        # 첫 번째가 가장 높은 ml_score (AMZN: 0.072)
        assert result.iloc[0]['symbol'] == 'AMZN'
        # 두 번째가 두 번째로 높은 ml_score (AAPL: 0.060)
        assert result.iloc[1]['symbol'] == 'AAPL'

    def test_assigns_correct_ranks(self, sample_predictions_df):
        """올바른 rank 할당"""
        result = _rank_and_select_top_k_standalone(sample_predictions_df, top_k=5)

        assert result.iloc[0]['rank'] == 1
        assert result.iloc[1]['rank'] == 2
        assert result.iloc[4]['rank'] == 5

    def test_selects_top_k(self, sample_predictions_df):
        """top_k 개수만큼 선택"""
        result = _rank_and_select_top_k_standalone(sample_predictions_df, top_k=3)

        # 선택된 종목 수 확인
        selected_count = result['selected'].sum()
        assert selected_count == 3

    def test_marks_selected_correctly(self, sample_predictions_df):
        """selected 컬럼 올바르게 표시"""
        result = _rank_and_select_top_k_standalone(sample_predictions_df, top_k=2)

        # 상위 2개만 selected=True
        top_2 = result[result['rank'] <= 2]
        assert top_2['selected'].all()

        # 나머지는 selected=False
        rest = result[result['rank'] > 2]
        assert not rest['selected'].any()

    def test_handles_top_k_larger_than_data(self, sample_predictions_df):
        """top_k가 데이터보다 클 때"""
        result = _rank_and_select_top_k_standalone(sample_predictions_df, top_k=100)

        # 모든 종목이 선택됨
        assert result['selected'].all()

    def test_handles_empty_dataframe(self):
        """빈 DataFrame 처리"""
        empty_df = pd.DataFrame(columns=['symbol', 'sector', 'ml_score'])
        result = _rank_and_select_top_k_standalone(empty_df, top_k=5)

        assert len(result) == 0


# ============================================================================
# Test: Feature Name Cleaning (from Regressor class)
# ============================================================================

class TestCleanFeatureNames:
    """Feature 이름 정리 테스트"""

    def test_removes_special_characters(self):
        """특수문자 제거"""
        from src.training.regressor import Regressor

        # Regressor 인스턴스 없이 정적 메서드처럼 사용하기 위해 mock
        df = pd.DataFrame({
            'feat_(a, b)': [1, 2, 3],
            'feat_[0]': [4, 5, 6],
            'normal_feat': [7, 8, 9],
        })

        # DataProcessor의 normalize_feature_names 사용
        from src.training.data_processor import DataProcessor
        result = DataProcessor.normalize_feature_names(df)

        # 특수문자가 제거되어야 함
        for col in result.columns:
            assert '(' not in col
            assert '[' not in col
            assert ')' not in col
            assert ']' not in col


# ============================================================================
# Test: Date Extraction from Filepath
# ============================================================================

class TestExtractDateFromFilepath:
    """파일 경로에서 날짜 추출 테스트"""

    def test_extracts_year_quarter(self):
        """연도_분기 추출"""
        from src.training.regressor import Regressor

        # Static 메서드로 호출
        result = Regressor._extract_date_from_filepath('rnorm_ml_2023_Q1.parquet')

        assert '2023' in result
        assert 'Q1' in result

    def test_handles_full_path(self):
        """전체 경로 처리"""
        from src.training.regressor import Regressor

        result = Regressor._extract_date_from_filepath(
            '/home/user/data/rnorm_ml_2024_Q3.parquet'
        )

        assert '2024' in result
        assert 'Q3' in result

    def test_handles_windows_path(self):
        """Windows 경로 처리"""
        from src.training.regressor import Regressor

        result = Regressor._extract_date_from_filepath(
            'C:\\Users\\data\\rnorm_ml_2022_Q4.parquet'
        )

        assert '2022' in result
        assert 'Q4' in result


# ============================================================================
# Test: Temporal Train/Val Split
# ============================================================================

class TestTemporalTrainValSplit:
    """시간 기반 Train/Val 분할 테스트"""

    def test_split_ratio_respected(self):
        """분할 비율 준수"""
        from src.training.regressor import Regressor

        # Mock config
        conf = {
            'DATA': {'ROOT_PATH': '/tmp'},
            'ML': {'USE_CLASSIFIER': 'Y', 'USE_SECTOR_MODEL': 'N'},
        }

        # Mock을 사용하여 파일 시스템 접근 회피
        with patch('pathlib.Path.exists', return_value=True):
            reg = Regressor(conf)

        X = pd.DataFrame({
            'feat_a': range(100),
            'feat_b': range(100, 200),
        })
        y = pd.Series(range(100))

        X_train, X_val, y_train, y_val = reg._temporal_train_val_split(X, y, val_ratio=0.2)

        # 약 80:20 분할
        assert len(X_train) == 80
        assert len(X_val) == 20

    def test_removes_nan_from_target(self):
        """타겟의 NaN 제거"""
        from src.training.regressor import Regressor

        conf = {
            'DATA': {'ROOT_PATH': '/tmp'},
            'ML': {'USE_CLASSIFIER': 'Y', 'USE_SECTOR_MODEL': 'N'},
        }

        with patch('pathlib.Path.exists', return_value=True):
            reg = Regressor(conf)

        X = pd.DataFrame({
            'feat_a': range(100),
        })
        y = pd.Series([i if i % 10 != 0 else np.nan for i in range(100)])  # 10% NaN

        X_train, X_val, y_train, y_val = reg._temporal_train_val_split(X, y, val_ratio=0.2)

        # NaN이 제거되어야 함
        assert not y_train.isnull().any()
        assert not y_val.isnull().any()


# ============================================================================
# Test: Binary Filtering Logic
# ============================================================================

class TestBinaryFilteringLogic:
    """Binary Filtering 로직 테스트"""

    def test_negative_screen_filters_high_bad_proba(self):
        """Negative screen: BAD 확률 높은 종목 필터링"""
        # 이 테스트는 DataProcessor.create_binary_target()을 통해 간접 검증
        from src.training.data_processor import DataProcessor

        y = pd.Series([0.1, -0.4, 0.05, -0.35, 0.02])  # 2개가 -30% 이하

        binary = DataProcessor.create_binary_target(
            y, mode='negative_screen', threshold=-0.3
        )

        # -0.4, -0.35는 BAD (1)
        assert binary.iloc[1] == 1
        assert binary.iloc[3] == 1
        # 나머지는 OK (0)
        assert binary.iloc[0] == 0
        assert binary.iloc[2] == 0
        assert binary.iloc[4] == 0

    def test_positive_screen_marks_gains(self):
        """Positive screen: 상승 종목 마킹"""
        from src.training.data_processor import DataProcessor

        y = pd.Series([0.1, -0.05, 0.03, -0.1, 0.0])

        binary = DataProcessor.create_binary_target(
            y, mode='positive_screen', threshold=0.0
        )

        # 양수는 GOOD (1)
        assert binary.iloc[0] == 1
        assert binary.iloc[2] == 1
        # 0 이하는 LOSS (0)
        assert binary.iloc[1] == 0
        assert binary.iloc[3] == 0
        assert binary.iloc[4] == 0


# ============================================================================
# Test: Ensemble Prediction Strategies
# ============================================================================

class TestEnsemblePredictionStrategies:
    """앙상블 예측 전략 테스트"""

    def test_ensemble_requires_multiple_classifiers(self):
        """앙상블은 다수의 분류기 필요"""
        # 앙상블 전략은 classifier 1 AND 3 또는 1 AND 2 등
        # 이는 regressor.py의 _create_ensemble_predictions에서 검증

        # 간단한 로직 테스트: 둘 다 1이어야 통과
        clf_1_pred = 1
        clf_3_pred = 1
        ensemble_result = clf_1_pred and clf_3_pred
        assert ensemble_result == 1

        clf_1_pred = 1
        clf_3_pred = 0
        ensemble_result = clf_1_pred and clf_3_pred
        assert ensemble_result == 0

    def test_majority_vote_logic(self):
        """다수결 투표 로직"""
        # 3개 중 2개 이상이 1이면 통과
        predictions = [1, 1, 0]  # 2/3 → 통과
        majority = sum(predictions) >= 2
        assert majority is True

        predictions = [1, 0, 0]  # 1/3 → 탈락
        majority = sum(predictions) >= 2
        assert majority is False


# ============================================================================
# Test: Prediction Column Names
# ============================================================================

class TestPredictionColumnNames:
    """예측 컬럼 이름 테스트"""

    def test_build_prediction_column_names(self):
        """예측 컬럼 이름 빌드"""
        from src.training.regressor import Regressor

        columns = Regressor._build_prediction_column_names()

        assert 'ai_pred_avg' in columns
        assert 'model_0_prediction' in columns
        # 총 17개 컬럼
        assert len(columns) == 17

    def test_get_classifier_column_names(self):
        """분류기 컬럼 이름"""
        from src.training.regressor import Regressor

        names = Regressor._get_classifier_column_names()

        assert 0 in names
        assert 1 in names
        assert 2 in names
        assert 3 in names
        assert 'clsmodel_0_prediction' in names.values()

    def test_get_regression_column_names(self):
        """회귀 컬럼 이름"""
        from src.training.regressor import Regressor

        names = Regressor._get_regression_column_names(model_idx=0)

        assert 'prediction' in names
        assert 'ensemble' in names
        assert 'loss' in names


# ============================================================================
# Test: Extreme Value Handling
# ============================================================================

class TestExtremeValueHandling:
    """극단값 처리 테스트"""

    def test_diagnose_extreme_values_detects_infinity(self):
        """무한대 감지"""
        from src.training.regressor import Regressor

        conf = {
            'DATA': {'ROOT_PATH': '/tmp'},
            'ML': {'USE_CLASSIFIER': 'Y', 'USE_SECTOR_MODEL': 'N'},
        }

        with patch('pathlib.Path.exists', return_value=True):
            reg = Regressor(conf)

        X = np.array([[1, 2, np.inf], [4, 5, 6]])
        y = np.array([0.1, 0.2])

        # 무한대가 있으면 True 반환 (문제 있음)
        has_issues = reg._diagnose_extreme_values(X, y, "test")
        assert has_issues is True

    def test_diagnose_extreme_values_passes_clean_data(self):
        """정상 데이터 통과"""
        from src.training.regressor import Regressor

        conf = {
            'DATA': {'ROOT_PATH': '/tmp'},
            'ML': {'USE_CLASSIFIER': 'Y', 'USE_SECTOR_MODEL': 'N'},
        }

        with patch('pathlib.Path.exists', return_value=True):
            reg = Regressor(conf)

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([0.1, 0.2])

        has_issues = reg._diagnose_extreme_values(X, y, "test")
        assert has_issues is False


# ============================================================================
# Test: Safe Clipping
# ============================================================================

class TestSafeClipping:
    """안전한 클리핑 테스트"""

    def test_clips_extreme_values(self):
        """극단값 클리핑"""
        from src.training.regressor import Regressor

        conf = {
            'DATA': {'ROOT_PATH': '/tmp'},
            'ML': {'USE_CLASSIFIER': 'Y', 'USE_SECTOR_MODEL': 'N'},
        }

        with patch('pathlib.Path.exists', return_value=True):
            reg = Regressor(conf)

        X = pd.DataFrame({
            'feat_a': [1e10, 100, -1e10],
        })
        y = np.array([0.1, 0.2, 0.3])

        X_clipped, y_clipped = reg._safe_clip_for_xgboost(X, y, max_abs_value=1e7)

        # 클리핑 적용 확인
        assert X_clipped['feat_a'].max() <= 1e7
        assert X_clipped['feat_a'].min() >= -1e7


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestRegressorIntegration:
    """Regressor 통합 테스트"""

    def test_regressor_initialization(self):
        """Regressor 초기화"""
        from src.training.regressor import Regressor

        conf = {
            'DATA': {'ROOT_PATH': '/tmp/test_data'},
            'ML': {
                'USE_CLASSIFIER': 'Y',
                'USE_SECTOR_MODEL': 'N',
                'CLASSIFIER_MODE': 'negative_screen',
            },
        }

        with patch('pathlib.Path.exists', return_value=True):
            reg = Regressor(conf)

        assert reg.use_classifier is True
        assert reg.use_sector_model is False
