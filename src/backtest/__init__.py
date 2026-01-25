"""Backtesting module for quantitative trading strategies.

This package provides backtesting infrastructure for testing trading strategies
on historical data.

Main Components:
    MLBacktest: ML-based Walk-Forward backtesting engine (RECOMMENDED)
    Backtest: Legacy plan-based backtesting engine (deprecated)
    PlanHandler: Legacy strategy plan execution and stock scoring (deprecated)

Example:
    ML Walk-Forward backtesting workflow (RECOMMENDED)::

        from src.backtest import MLBacktest
        from src.infra.context_loader import load_config, MainContext

        # Load configuration
        config = load_config('config/conf.yaml')
        main_ctx = MainContext(config)

        # Run ML-based walk-forward backtest
        ml_bt = MLBacktest(
            config=config,
            main_ctx=main_ctx,
            rebalance_period=3,
            top_k=20,
            retrain_frequency='quarterly',
            window_type='expanding'
        )

        # Execute backtest
        results = ml_bt.run()

        # Results are automatically saved to reports directory

See Also:
    - docs/ML_BACKTEST_GUIDE.md: Comprehensive ML backtesting guide
    - docs/WORKFLOW_GUIDE.md: System workflow documentation
"""

from .backtest import Backtest, PlanHandler
from .ml_backtest import MLBacktest

__all__ = ['MLBacktest', 'Backtest', 'PlanHandler']

__version__ = '1.0.0'
__author__ = 'Quant Trading Team'
__description__ = 'Backtesting framework for quantitative trading'
