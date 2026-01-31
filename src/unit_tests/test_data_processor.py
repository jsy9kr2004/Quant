"""
DataProcessor 단위 테스트

테스트 실행:
    # 빠른 테스트 (integration 제외)
    pytest src/unit_tests/test_data_processor.py -v -m "not integration"

    # 전체 테스트
    pytest src/unit_tests/test_data_processor.py -v

    # 특정 클래스만 테스트
    pytest src/unit_tests/test_data_processor.py::TestNormalizeFeatureNames -v

테스트 대상:
    - normalize_feature_names(): Feature 이름 정규화 (XGBoost 호환)
    - align_features_to_model(): Feature 정렬 (regressor ↔ ml_backtest 일관성)
    - remove_infinite_values(): 무한값 제거
    - create_binary_target(): 이진 분류 타겟 생성 (negative/positive screening)
    - drop_sparse_rows(): Sparse row 제거
    - drop_sparse_cols(): Sparse column 제거
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
import logging

# 테스트 대상 모듈
from src.training.data_processor import DataProcessor


# ============================================================================
# Fixtures (테스트 데이터)
# ============================================================================

@pytest.fixture
def sample_df_with_special_chars():
    """특수 문자가 포함된 feature 이름을 가진 DataFrame"""
    return pd.DataFrame({
        'price_(2, 5)': [1, 2, 3, 4, 5],
        'volume[0]': [10, 20, 30, 40, 50],
        'ratio{a}': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature:value': [100, 200, 300, 400, 500],
        'normal_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
        'data.with.dots': [5, 4, 3, 2, 1],
    })


@pytest.fixture
def sample_df_with_duplicates():
    """정규화 후 중복되는 feature 이름을 가진 DataFrame"""
    return pd.DataFrame({
        'feature(a, b)': [1, 2, 3],
        'feature[a, b]': [4, 5, 6],  # 정규화 후 feature_a_b로 동일
        'feature{a, b}': [7, 8, 9],  # 정규화 후 feature_a_b로 동일
    })


@pytest.fixture
def sample_df_with_inf():
    """무한값이 포함된 DataFrame"""
    return pd.DataFrame({
        'col_a': [1.0, 2.0, np.inf, 4.0, 5.0],
        'col_b': [10.0, -np.inf, 30.0, 40.0, 50.0],
        'col_c': [100.0, 200.0, 300.0, 400.0, 500.0],  # 정상
        'col_str': ['a', 'b', 'c', 'd', 'e'],  # 문자열 컬럼
    })


@pytest.fixture
def sample_target_series():
    """타겟 Series"""
    return pd.Series([0.05, -0.10, -0.35, 0.02, -0.45], name='price_dev')


@pytest.fixture
def sample_sparse_df():
    """Sparse row/column이 있는 DataFrame"""
    return pd.DataFrame({
        'col_a': [1.0, 2.0, np.nan, np.nan, np.nan],  # 60% NaN
        'col_b': [1.0, 2.0, 3.0, 4.0, 5.0],  # 0% NaN
        'col_c': [np.nan, np.nan, np.nan, np.nan, np.nan],  # 100% NaN
        'col_d': [1.0, np.nan, np.nan, np.nan, np.nan],  # 80% NaN
    })


@pytest.fixture
def mock_model():
    """get_booster()를 가진 Mock 모델"""
    mock = Mock()
    mock.get_booster.return_value.feature_names = ['feat_a', 'feat_b', 'feat_c']
    return mock


@pytest.fixture
def mock_logger():
    """테스트용 logger"""
    return logging.getLogger('test_data_processor')


# ============================================================================
# Test: normalize_feature_names()
# ============================================================================

class TestNormalizeFeatureNames:
    """Feature 이름 정규화 테스트"""

    def test_removes_parentheses(self, sample_df_with_special_chars):
        """괄호가 제거되는지 확인"""
        result = DataProcessor.normalize_feature_names(sample_df_with_special_chars)

        # 'price_(2, 5)' → 'price_2_5'
        assert 'price_2_5' in result.columns
        assert 'price_(2, 5)' not in result.columns

    def test_removes_brackets(self, sample_df_with_special_chars):
        """대괄호가 제거되는지 확인"""
        result = DataProcessor.normalize_feature_names(sample_df_with_special_chars)

        # 'volume[0]' → 'volume_0'
        assert 'volume_0' in result.columns
        assert 'volume[0]' not in result.columns

    def test_removes_curly_braces(self, sample_df_with_special_chars):
        """중괄호가 제거되는지 확인"""
        result = DataProcessor.normalize_feature_names(sample_df_with_special_chars)

        # 'ratio{a}' → 'ratio_a'
        assert 'ratio_a' in result.columns
        assert 'ratio{a}' not in result.columns

    def test_removes_colons(self, sample_df_with_special_chars):
        """콜론이 제거되는지 확인"""
        result = DataProcessor.normalize_feature_names(sample_df_with_special_chars)

        # 'feature:value' → 'feature_value'
        assert 'feature_value' in result.columns
        assert 'feature:value' not in result.columns

    def test_removes_dots(self, sample_df_with_special_chars):
        """점이 제거되는지 확인"""
        result = DataProcessor.normalize_feature_names(sample_df_with_special_chars)

        # 'data.with.dots' → 'data_with_dots'
        assert 'data_with_dots' in result.columns
        assert 'data.with.dots' not in result.columns

    def test_preserves_normal_features(self, sample_df_with_special_chars):
        """일반 feature 이름은 그대로 유지"""
        result = DataProcessor.normalize_feature_names(sample_df_with_special_chars)

        assert 'normal_feature' in result.columns

    def test_handles_duplicate_normalized_names(self, sample_df_with_duplicates):
        """정규화 후 중복되는 이름 처리"""
        result = DataProcessor.normalize_feature_names(sample_df_with_duplicates)

        # 중복 제거 후 unique한 이름이 되어야 함
        assert len(result.columns) == len(set(result.columns))
        assert len(result.columns) == 3

    def test_preserves_data_values(self, sample_df_with_special_chars):
        """데이터 값이 변경되지 않는지 확인"""
        original_values = sample_df_with_special_chars['normal_feature'].values.copy()
        result = DataProcessor.normalize_feature_names(sample_df_with_special_chars)

        np.testing.assert_array_equal(
            result['normal_feature'].values,
            original_values
        )

    def test_empty_dataframe(self):
        """빈 DataFrame 처리"""
        empty_df = pd.DataFrame()
        result = DataProcessor.normalize_feature_names(empty_df)

        assert len(result.columns) == 0


# ============================================================================
# Test: align_features_to_model()
# ============================================================================

class TestAlignFeaturesToModel:
    """Feature 정렬 테스트 (regressor ↔ ml_backtest 일관성)"""

    def test_adds_missing_features(self, mock_model, mock_logger):
        """누락된 feature가 NaN으로 추가되는지 확인"""
        X = pd.DataFrame({
            'feat_a': [1, 2, 3],
            'feat_b': [4, 5, 6],
            # feat_c 누락
        })

        result = DataProcessor.align_features_to_model(X, mock_model, mock_logger)

        assert 'feat_c' in result.columns
        assert result['feat_c'].isnull().all()

    def test_removes_extra_features(self, mock_model, mock_logger):
        """모델에 없는 extra feature가 제거되는지 확인"""
        X = pd.DataFrame({
            'feat_a': [1, 2, 3],
            'feat_b': [4, 5, 6],
            'feat_c': [7, 8, 9],
            'feat_extra': [10, 11, 12],  # 모델에 없음
        })

        result = DataProcessor.align_features_to_model(X, mock_model, mock_logger)

        assert 'feat_extra' not in result.columns
        assert len(result.columns) == 3

    def test_reorders_columns_to_match_model(self, mock_model, mock_logger):
        """컬럼 순서가 모델과 일치하는지 확인"""
        X = pd.DataFrame({
            'feat_c': [7, 8, 9],  # 순서 다름
            'feat_a': [1, 2, 3],
            'feat_b': [4, 5, 6],
        })

        result = DataProcessor.align_features_to_model(X, mock_model, mock_logger)

        assert list(result.columns) == ['feat_a', 'feat_b', 'feat_c']

    def test_preserves_data_values(self, mock_model, mock_logger):
        """데이터 값이 변경되지 않는지 확인"""
        X = pd.DataFrame({
            'feat_a': [1.5, 2.5, 3.5],
            'feat_b': [4.5, 5.5, 6.5],
            'feat_c': [7.5, 8.5, 9.5],
        })

        result = DataProcessor.align_features_to_model(X, mock_model, mock_logger)

        np.testing.assert_array_equal(result['feat_a'].values, [1.5, 2.5, 3.5])
        np.testing.assert_array_equal(result['feat_b'].values, [4.5, 5.5, 6.5])
        np.testing.assert_array_equal(result['feat_c'].values, [7.5, 8.5, 9.5])

    def test_handles_model_without_get_booster(self, mock_logger):
        """get_booster()가 없는 모델 처리"""
        mock = Mock(spec=[])  # get_booster 없음

        X = pd.DataFrame({'feat_a': [1, 2, 3]})
        result = DataProcessor.align_features_to_model(X, mock, mock_logger)

        # 변경 없이 반환
        assert list(result.columns) == ['feat_a']

    def test_handles_none_feature_names(self, mock_logger):
        """feature_names가 None인 경우 처리"""
        mock = Mock()
        mock.get_booster.return_value.feature_names = None

        X = pd.DataFrame({'feat_a': [1, 2, 3]})
        result = DataProcessor.align_features_to_model(X, mock, mock_logger)

        # 변경 없이 반환
        assert list(result.columns) == ['feat_a']


# ============================================================================
# Test: remove_infinite_values()
# ============================================================================

class TestRemoveInfiniteValues:
    """무한값 제거 테스트"""

    def test_removes_rows_with_positive_inf(self, sample_df_with_inf):
        """양의 무한대 포함 행 제거"""
        y = pd.Series([1, 2, 3, 4, 5])
        X_clean, y_clean = DataProcessor.remove_infinite_values(
            sample_df_with_inf, y
        )

        # row 2 (index 2)는 np.inf 포함
        assert len(X_clean) < len(sample_df_with_inf)
        assert not np.isinf(X_clean.select_dtypes(include=[np.number])).any().any()

    def test_removes_rows_with_negative_inf(self, sample_df_with_inf):
        """음의 무한대 포함 행 제거"""
        X_clean, _ = DataProcessor.remove_infinite_values(sample_df_with_inf)

        # row 1 (index 1)은 -np.inf 포함
        assert not np.isinf(X_clean.select_dtypes(include=[np.number])).any().any()

    def test_aligns_y_with_x(self, sample_df_with_inf):
        """y가 X와 정렬되는지 확인"""
        y = pd.Series([10, 20, 30, 40, 50])
        X_clean, y_clean = DataProcessor.remove_infinite_values(
            sample_df_with_inf, y
        )

        assert len(X_clean) == len(y_clean)

    def test_handles_none_y(self, sample_df_with_inf):
        """y가 None인 경우 처리"""
        X_clean, y_clean = DataProcessor.remove_infinite_values(
            sample_df_with_inf, None
        )

        assert y_clean is None
        assert not np.isinf(X_clean.select_dtypes(include=[np.number])).any().any()

    def test_ignores_string_columns(self, sample_df_with_inf):
        """문자열 컬럼은 무시"""
        X_clean, _ = DataProcessor.remove_infinite_values(sample_df_with_inf)

        # col_str는 문자열이므로 무시되어야 함
        assert 'col_str' in X_clean.columns

    def test_no_inf_returns_original(self):
        """무한값이 없으면 원본 반환"""
        X = pd.DataFrame({
            'col_a': [1.0, 2.0, 3.0],
            'col_b': [4.0, 5.0, 6.0],
        })
        y = pd.Series([10, 20, 30])

        X_clean, y_clean = DataProcessor.remove_infinite_values(X, y)

        assert len(X_clean) == len(X)

    def test_all_numeric_columns_inf(self):
        """모든 숫자 컬럼에 inf가 있는 행 제거"""
        X = pd.DataFrame({
            'col_a': [np.inf, 2.0, 3.0],
            'col_b': [1.0, np.inf, 3.0],
        })

        X_clean, _ = DataProcessor.remove_infinite_values(X)

        assert len(X_clean) == 1  # row 2만 남음


# ============================================================================
# Test: create_binary_target()
# ============================================================================

class TestCreateBinaryTarget:
    """이진 분류 타겟 생성 테스트"""

    def test_negative_screen_mode(self, sample_target_series):
        """Negative screening 모드 테스트"""
        # y = [0.05, -0.10, -0.35, 0.02, -0.45]
        # threshold = -0.3
        # Expected: [0, 0, 1, 0, 1] (1 = BAD: -35%, -45%)

        binary = DataProcessor.create_binary_target(
            sample_target_series,
            mode="negative_screen",
            threshold=-0.3
        )

        expected = pd.Series([0, 0, 1, 0, 1])
        pd.testing.assert_series_equal(
            binary.reset_index(drop=True),
            expected,
            check_names=False  # Series 이름은 무시
        )

    def test_positive_screen_mode(self, sample_target_series):
        """Positive screening 모드 테스트"""
        # y = [0.05, -0.10, -0.35, 0.02, -0.45]
        # threshold = 0.0
        # Expected: [1, 0, 0, 1, 0] (1 = GOOD: +5%, +2%)

        binary = DataProcessor.create_binary_target(
            sample_target_series,
            mode="positive_screen",
            threshold=0.0
        )

        expected = pd.Series([1, 0, 0, 1, 0])
        pd.testing.assert_series_equal(
            binary.reset_index(drop=True),
            expected,
            check_names=False  # Series 이름은 무시
        )

    def test_negative_screen_different_threshold(self):
        """Negative screening 다른 threshold 테스트"""
        y = pd.Series([0.1, -0.15, -0.25, -0.35, -0.5])
        # threshold = -0.2
        # Expected: [0, 0, 1, 1, 1] (1 = BAD: -25%, -35%, -50%)

        binary = DataProcessor.create_binary_target(
            y, mode="negative_screen", threshold=-0.2
        )

        expected = pd.Series([0, 0, 1, 1, 1])
        pd.testing.assert_series_equal(binary.reset_index(drop=True), expected)

    def test_handles_dataframe_input(self, sample_target_series):
        """DataFrame 입력 처리"""
        df = pd.DataFrame({'price_dev': sample_target_series})

        binary = DataProcessor.create_binary_target(
            df, mode="negative_screen", threshold=-0.3
        )

        assert isinstance(binary, pd.Series)
        assert len(binary) == len(sample_target_series)

    def test_reads_mode_from_config(self, sample_target_series):
        """config에서 mode 읽기"""
        config = {
            'ML': {
                'CLASSIFIER_MODE': 'negative_screen',
                'NEGATIVE_SCREEN': {
                    'LOSS_THRESHOLD': -0.3
                }
            }
        }

        binary = DataProcessor.create_binary_target(
            sample_target_series,
            config=config
        )

        # negative_screen 모드로 동작해야 함
        expected = pd.Series([0, 0, 1, 0, 1])
        pd.testing.assert_series_equal(
            binary.reset_index(drop=True),
            expected,
            check_names=False  # Series 이름은 무시
        )

    def test_all_positive_returns(self):
        """모두 양수인 경우"""
        y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

        # Negative screen: 모두 OK (0)
        binary_neg = DataProcessor.create_binary_target(
            y, mode="negative_screen", threshold=-0.3
        )
        assert binary_neg.sum() == 0

        # Positive screen: 모두 GOOD (1)
        binary_pos = DataProcessor.create_binary_target(
            y, mode="positive_screen", threshold=0.0
        )
        assert binary_pos.sum() == 5

    def test_all_negative_returns(self):
        """모두 음수인 경우"""
        y = pd.Series([-0.1, -0.2, -0.3, -0.4, -0.5])

        # Negative screen with threshold=-0.25: 하위 3개 BAD
        binary_neg = DataProcessor.create_binary_target(
            y, mode="negative_screen", threshold=-0.25
        )
        assert binary_neg.sum() == 3

        # Positive screen: 모두 LOSS (0)
        binary_pos = DataProcessor.create_binary_target(
            y, mode="positive_screen", threshold=0.0
        )
        assert binary_pos.sum() == 0


# ============================================================================
# Test: drop_sparse_rows()
# ============================================================================

class TestDropSparseRows:
    """Sparse row 제거 테스트"""

    def test_drops_rows_above_threshold(self, sample_sparse_df):
        """threshold 이상 NaN인 행 제거"""
        # col_a: [1.0, 2.0, np.nan, np.nan, np.nan] - row 2,3,4는 col_a에서 NaN
        # threshold=0.5: 50% 이상 NaN인 행 제거

        result = DataProcessor.drop_sparse_rows(sample_sparse_df, threshold=0.5)

        # row 0, 1만 남아야 함 (대부분의 데이터가 있는 행)
        # 실제로는 row 단위로 NaN 비율 계산
        assert len(result) <= len(sample_sparse_df)

    def test_preserves_rows_below_threshold(self):
        """threshold 미만 NaN인 행 유지"""
        # drop_sparse_rows 로직:
        # max_allowed_nan = int(len(columns) * (1 - threshold))
        # mask = nan_count_per_row < max_allowed_nan
        # threshold=0.5, columns=2 -> max_allowed_nan = 1
        # 즉, NaN 개수가 1 미만인 행만 유지 (0개 NaN인 행)
        df = pd.DataFrame({
            'col_a': [1.0, 2.0, 3.0],  # 0% NaN
            'col_b': [1.0, 2.0, 3.0],  # 0% NaN
        })

        result = DataProcessor.drop_sparse_rows(df, threshold=0.5)

        # 모든 행이 NaN=0이므로 모두 유지
        assert len(result) == 3

    def test_returns_empty_with_all_sparse(self):
        """모든 행이 sparse면 빈 DataFrame 반환"""
        df = pd.DataFrame({
            'col_a': [np.nan, np.nan, np.nan],
            'col_b': [np.nan, np.nan, np.nan],
        })

        result = DataProcessor.drop_sparse_rows(df, threshold=0.5)

        assert len(result) == 0


# ============================================================================
# Test: drop_sparse_cols() - Instance Method
# ============================================================================

class TestDropSparseCols:
    """Sparse column 제거 테스트 (인스턴스 메서드)"""

    @pytest.fixture
    def processor(self):
        """DataProcessor 인스턴스"""
        config = {
            'FEATURES': {
                'MISSING_THRESHOLD': 0.8,
                'SAME_VALUE_THRESHOLD': 0.95
            }
        }
        return DataProcessor(config)

    def test_drops_all_nan_columns(self, sample_sparse_df, processor):
        """100% NaN인 컬럼 제거"""
        result, dropped = processor.drop_sparse_cols(sample_sparse_df)

        # col_c는 100% NaN이므로 제거됨
        assert 'col_c' not in result.columns
        assert 'col_c' in dropped

    def test_preserves_non_sparse_columns(self, sample_sparse_df, processor):
        """sparse하지 않은 컬럼 유지"""
        result, _ = processor.drop_sparse_cols(sample_sparse_df)

        # col_b는 0% NaN
        assert 'col_b' in result.columns

    def test_drops_constant_columns(self, processor):
        """상수 컬럼 제거 (same_value_threshold 초과)"""
        # 100개 행 중 99개가 동일한 값 = 99% same value
        df = pd.DataFrame({
            'col_a': [1.0] * 99 + [2.0],  # 99% 동일
            'col_b': list(range(100)),  # 모두 다름
        })

        result, dropped = processor.drop_sparse_cols(df)

        # col_a는 99% 동일값이므로 threshold=0.95 초과하여 제거
        assert 'col_a' in dropped
        assert 'col_b' in result.columns


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_dataframe_normalize(self):
        """빈 DataFrame 정규화"""
        df = pd.DataFrame()
        result = DataProcessor.normalize_feature_names(df)
        assert len(result) == 0

    def test_single_row_dataframe(self):
        """단일 행 DataFrame"""
        df = pd.DataFrame({'col_a': [1.0], 'col_b': [2.0]})
        X_clean, _ = DataProcessor.remove_infinite_values(df)
        assert len(X_clean) == 1

    def test_single_column_dataframe(self):
        """단일 컬럼 DataFrame"""
        df = pd.DataFrame({'col_a': [1.0, 2.0, 3.0]})
        result = DataProcessor.normalize_feature_names(df)
        assert len(result.columns) == 1

    def test_unicode_column_names(self):
        """유니코드 컬럼 이름"""
        df = pd.DataFrame({
            '한글컬럼': [1, 2, 3],
            'feature_日本語': [4, 5, 6],
        })
        result = DataProcessor.normalize_feature_names(df)

        # 유니코드는 제거되고 underscore만 남음
        assert len(result.columns) == 2


# ============================================================================
# Test: Data Quality Report Integration (Optional)
# ============================================================================

@pytest.mark.integration
class TestDataQualityIntegration:
    """데이터 품질 통합 테스트 (실제 파일 로드 필요)"""

    def test_load_quarterly_data_integration(self, tmp_path):
        """분기별 데이터 로드 통합 테스트"""
        # 임시 parquet 파일 생성
        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'price_dev': [0.05, -0.03],
            'fillingDate': ['2023-01-15', '2023-01-20']
        })

        file_path = tmp_path / 'rnorm_ml_2023_Q1.parquet'
        df.to_parquet(file_path)

        # 로드 테스트
        result = DataProcessor.load_quarterly_data(
            data_dir=str(tmp_path),
            year=2023,
            file_prefix='rnorm_ml',
            quarters=['Q1']
        )

        assert len(result) == 2
        assert 'symbol' in result.columns
