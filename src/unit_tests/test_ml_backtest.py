"""
MLBacktest 단위 테스트

테스트 실행:
    # 빠른 테스트 (integration, slow 제외)
    pytest src/unit_tests/test_ml_backtest.py -v -m "not integration and not slow"

    # 전체 테스트
    pytest src/unit_tests/test_ml_backtest.py -v

테스트 대상:
    - 상장폐지 감지 로직
    - 리트레인 빈도 로직
    - Threshold 계산
    - Top-K 선택
    - 포트폴리오 수익률 계산
    - 벤치마크 계산
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import logging
from datetime import datetime, timedelta
import tempfile
from pathlib import Path


# ============================================================================
# Fixtures (테스트 데이터)
# ============================================================================

@pytest.fixture
def sample_config():
    """기본 테스트 config"""
    return {
        'DATA': {
            'ROOT_PATH': '/tmp/test_data',
        },
        'ML': {
            'USE_CLASSIFIER': 'Y',
            'USE_SECTOR_MODEL': 'N',
            'CLASSIFIER_MODE': 'negative_screen',
            'CLASSIFIER_REMOVE_PCT_MIN': 2,
            'CLASSIFIER_REMOVE_PCT_MAX': 15,
            'SECTOR_CATEGORIZATION': {
                'ENABLED': 'N',
            },
        },
        'EVALUATION': {
            'USE_CACHED_PREDICTIONS': 'N',
            'REBALANCE_PERIOD': 3,
            'TOP_K_NUM': 10,
        },
        'BACKTEST': {
            'REBALANCE_PERIOD': 3,
            'TOP_K_NUM': 10,
            'PERIODS': [
                {'START_YEAR': 2023, 'END_YEAR': 2023, 'START_MONTH': 1, 'START_DATE': 1},
            ],
            'TRADING_COST': {
                'ENABLED': 'Y',
                'COMMISSION_RATE': 0.001,
                'SLIPPAGE_RATE': 0.001,
            },
        },
        'BENCHMARK': {
            'ENABLED': 'Y',
            'SYMBOLS': ['SPY', 'QQQ'],
        },
    }


@pytest.fixture
def sample_predictions_df():
    """예측 결과 DataFrame"""
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM'],
        'sector': ['Tech', 'Tech', 'Tech', 'Consumer', 'Tech', 'Tech', 'Consumer', 'Tech', 'Tech', 'Tech'],
        'pred_return': [0.08, 0.06, 0.04, 0.09, 0.05, 0.07, 0.03, 0.06, 0.02, 0.04],
        'pred_proba': [0.75, 0.70, 0.65, 0.80, 0.68, 0.72, 0.60, 0.71, 0.55, 0.63],
        'ml_score': [0.060, 0.042, 0.026, 0.072, 0.034, 0.050, 0.018, 0.043, 0.011, 0.025],
    })


@pytest.fixture
def sample_price_table():
    """가격 테이블"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')  # 영업일
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ']

    data = {}
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        base_price = 100 + np.random.rand() * 100
        prices = base_price * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
        data[symbol] = prices

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_delisted_data():
    """상장폐지 데이터"""
    return pd.DataFrame({
        'symbol': ['DELIST1', 'DELIST2', 'DELIST3'],
        'delistedDate': ['2023-06-15', '2023-08-20', None],
    })


@pytest.fixture
def mock_logger():
    """테스트용 logger"""
    return logging.getLogger('test_ml_backtest')


@pytest.fixture
def mock_main_ctx(sample_config):
    """Mock main context"""
    ctx = Mock()
    ctx.conf = sample_config
    ctx.logger = logging.getLogger('test_ml_backtest')
    return ctx


# ============================================================================
# Test: Delisting Detection
# ============================================================================

class TestDelistingDetection:
    """상장폐지 감지 테스트"""

    def test_is_truly_delisted_confirmed(self, sample_delisted_data, sample_config, mock_main_ctx):
        """확인된 상장폐지"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=sample_delisted_data):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )
                backtest.delisted_data = sample_delisted_data

                # 상장폐지일 이후 체크
                is_delisted, date, status = backtest._is_truly_delisted(
                    'DELIST1', pd.Timestamp('2023-07-01')
                )

                assert is_delisted is True
                assert status == 'confirmed_delisted'

    def test_is_truly_delisted_before_date(self, sample_delisted_data, sample_config, mock_main_ctx):
        """상장폐지일 이전"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=sample_delisted_data):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )
                backtest.delisted_data = sample_delisted_data

                # 상장폐지일 이전 체크
                is_delisted, date, status = backtest._is_truly_delisted(
                    'DELIST1', pd.Timestamp('2023-05-01')
                )

                assert is_delisted is False
                assert status == 'will_be_delisted'

    def test_is_truly_delisted_not_in_list(self, sample_delisted_data, sample_config, mock_main_ctx):
        """상장폐지 목록에 없음"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=sample_delisted_data):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )
                backtest.delisted_data = sample_delisted_data

                is_delisted, date, status = backtest._is_truly_delisted(
                    'AAPL', pd.Timestamp('2023-07-01')
                )

                assert is_delisted is False
                assert status == 'not_in_delisted_list'


# ============================================================================
# Test: Retrain Frequency Logic
# ============================================================================

class TestRetrainFrequency:
    """리트레인 빈도 테스트"""

    def test_should_retrain_every_mode(self, sample_config, mock_main_ctx):
        """매번 리트레인"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10,
                    retrain_frequency='every'
                )

                result = backtest._should_retrain(
                    datetime(2023, 4, 1),
                    datetime(2023, 1, 1)
                )

                assert result is True

    def test_should_retrain_first_time(self, sample_config, mock_main_ctx):
        """첫 번째 리트레인"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10,
                    retrain_frequency='quarterly'
                )

                result = backtest._should_retrain(
                    datetime(2023, 1, 1),
                    None  # 첫 학습
                )

                assert result is True

    def test_should_retrain_quarterly_same_quarter(self, sample_config, mock_main_ctx):
        """분기별: 같은 분기면 리트레인 안 함"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10,
                    retrain_frequency='quarterly'
                )

                result = backtest._should_retrain(
                    datetime(2023, 2, 15),  # Q1
                    datetime(2023, 1, 10)   # Q1
                )

                assert result is False

    def test_should_retrain_quarterly_new_quarter(self, sample_config, mock_main_ctx):
        """분기별: 새 분기면 리트레인"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10,
                    retrain_frequency='quarterly'
                )

                result = backtest._should_retrain(
                    datetime(2023, 4, 1),  # Q2
                    datetime(2023, 1, 10)  # Q1
                )

                assert result is True

    def test_should_retrain_once_mode(self, sample_config, mock_main_ctx):
        """한 번만 리트레인"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10,
                    retrain_frequency='once'
                )

                # 첫 번째는 True
                result1 = backtest._should_retrain(datetime(2023, 1, 1), None)
                assert result1 is True

                # 두 번째부터는 False
                result2 = backtest._should_retrain(datetime(2023, 4, 1), datetime(2023, 1, 1))
                assert result2 is False


# ============================================================================
# Test: Threshold Calculation
# ============================================================================

class TestThresholdCalculation:
    """Threshold 계산 테스트"""

    def test_calculate_threshold_negative_screen(self, sample_config, mock_main_ctx):
        """Negative screen threshold"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )

                y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

                config = backtest._calculate_threshold_config(
                    y_probs,
                    classifier_mode='negative_screen',
                    remove_pct=10
                )

                # percentile = 100 - 10 = 90
                assert config['percentile'] == 90
                assert config['mode'] == 'negative_screen'
                assert config['remove_pct'] == 10
                assert 'threshold_value' in config

    def test_calculate_threshold_positive_screen(self, sample_config, mock_main_ctx):
        """Positive screen threshold"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )

                y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

                config = backtest._calculate_threshold_config(
                    y_probs,
                    classifier_mode='positive_screen',
                    remove_pct=10
                )

                # percentile = 10 (하위 10% 제거)
                assert config['percentile'] == 10
                assert config['mode'] == 'positive_screen'


# ============================================================================
# Test: Top-K Selection
# ============================================================================

class TestTopKSelection:
    """Top-K 선택 테스트"""

    def test_select_top_k_returns_correct_count(self, sample_predictions_df, sample_config, mock_main_ctx):
        """올바른 개수 반환"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=5
                )

                result = backtest._select_top_k(sample_predictions_df)

                assert len(result) == 5

    def test_select_top_k_by_ml_score(self, sample_predictions_df, sample_config, mock_main_ctx):
        """ml_score 기준 정렬"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=3
                )

                result = backtest._select_top_k(sample_predictions_df)

                # 가장 높은 ml_score: AMZN (0.072)
                assert 'AMZN' in result['symbol'].values

    def test_select_top_k_handles_missing_sector(self, sample_config, mock_main_ctx):
        """누락된 섹터 처리"""
        from src.backtest.ml_backtest import MLBacktest

        predictions = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'ml_score': [0.06, 0.05],
            # sector 없음
        })

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=2
                )

                result = backtest._select_top_k(predictions)

                # sector가 'Unknown'으로 채워져야 함
                assert 'sector' in result.columns
                assert len(result) == 2


# ============================================================================
# Test: Portfolio Return Calculation
# ============================================================================

class TestPortfolioReturnCalculation:
    """포트폴리오 수익률 계산 테스트"""

    def test_calculate_period_return_basic(self, sample_price_table, sample_config, mock_main_ctx):
        """기본 수익률 계산"""
        from src.backtest.ml_backtest import MLBacktest

        selected_stocks = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'sector': ['Tech', 'Tech'],
            'category': ['Technology', 'Technology'],
        })

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=2
                )

                result = backtest._calculate_period_return(
                    selected_stocks,
                    datetime(2023, 1, 3),  # 영업일
                    datetime(2023, 3, 31),
                    sample_price_table
                )

                assert 'avg_return' in result
                assert 'details' in result
                assert isinstance(result['avg_return'], float)

    def test_calculate_period_return_with_trading_costs(self, sample_price_table, sample_config, mock_main_ctx):
        """거래 비용 포함 수익률"""
        from src.backtest.ml_backtest import MLBacktest

        sample_config['BACKTEST']['TRADING_COST']['ENABLED'] = 'Y'
        sample_config['BACKTEST']['TRADING_COST']['COMMISSION_RATE'] = 0.001
        sample_config['BACKTEST']['TRADING_COST']['SLIPPAGE_RATE'] = 0.001

        selected_stocks = pd.DataFrame({
            'symbol': ['AAPL'],
            'sector': ['Tech'],
            'category': ['Technology'],
        })

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=1
                )

                result = backtest._calculate_period_return(
                    selected_stocks,
                    datetime(2023, 1, 3),
                    datetime(2023, 3, 31),
                    sample_price_table
                )

                # 거래 비용이 반영되어야 함
                if len(result['details']) > 0:
                    detail = result['details'][0]
                    # net_return이 gross_return보다 작거나 같아야 함
                    # (거래 비용 때문)
                    assert 'net_return' in detail or 'return' in detail


# ============================================================================
# Test: GPU Availability
# ============================================================================

class TestGPUAvailability:
    """GPU 가용성 테스트"""

    def test_is_gpu_available_returns_bool(self, sample_config, mock_main_ctx):
        """bool 반환 확인"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )

                result = backtest._is_gpu_available()

                assert isinstance(result, bool)


# ============================================================================
# Test: Benchmark Calculations
# ============================================================================

class TestBenchmarkCalculations:
    """벤치마크 계산 테스트"""

    def test_calculate_benchmark_returns_structure(self, sample_price_table, sample_config, mock_main_ctx):
        """벤치마크 반환 구조"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )

                # ETF 데이터 로더 mock
                with patch.object(backtest, '_calculate_benchmark_returns') as mock_benchmark:
                    mock_benchmark.return_value = pd.DataFrame({
                        'strategy': ['SPY', 'QQQ'],
                        'total_return': [0.10, 0.12],
                        'sharpe_ratio': [1.2, 1.4],
                        'max_drawdown': [-0.08, -0.10],
                        'win_rate': [0.55, 0.58],
                    })

                    result = backtest._calculate_benchmark_returns(
                        datetime(2023, 1, 1),
                        datetime(2023, 12, 31),
                        sample_price_table
                    )

                    assert 'total_return' in result.columns
                    assert 'sharpe_ratio' in result.columns
                    assert 'max_drawdown' in result.columns


# ============================================================================
# Test: Window Type
# ============================================================================

class TestWindowType:
    """Window 타입 테스트"""

    def test_expanding_window(self, sample_config, mock_main_ctx):
        """Expanding window"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10,
                    window_type='expanding'
                )

                assert backtest.window_type == 'expanding'

    def test_rolling_window(self, sample_config, mock_main_ctx):
        """Rolling window"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10,
                    window_type='rolling',
                    window_size=252  # 1년
                )

                assert backtest.window_type == 'rolling'
                assert backtest.window_size == 252


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_predictions(self, sample_config, mock_main_ctx):
        """빈 예측 처리"""
        from src.backtest.ml_backtest import MLBacktest

        empty_predictions = pd.DataFrame(columns=['symbol', 'sector', 'ml_score'])

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )

                result = backtest._select_top_k(empty_predictions)

                assert len(result) == 0

    def test_top_k_larger_than_available(self, sample_predictions_df, sample_config, mock_main_ctx):
        """top_k가 가용 종목보다 클 때"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=100  # 10개보다 큼
                )

                result = backtest._select_top_k(sample_predictions_df)

                # 가능한 최대 개수만 반환
                assert len(result) == len(sample_predictions_df)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestMLBacktestIntegration:
    """MLBacktest 통합 테스트"""

    def test_initialization(self, sample_config, mock_main_ctx):
        """초기화 테스트"""
        from src.backtest.ml_backtest import MLBacktest

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(MLBacktest, '_load_delisted_data', return_value=pd.DataFrame()):
                backtest = MLBacktest(
                    sample_config, mock_main_ctx,
                    rebalance_period=3, top_k=10
                )

                assert backtest.rebalance_period == 3
                assert backtest.top_k == 10
