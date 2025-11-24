"""Backtesting module for quantitative trading strategies.

This package provides backtesting infrastructure for testing trading strategies
on historical data. It includes plan-based scoring, performance evaluation,
and comprehensive reporting capabilities.

Main Components:
    Backtest: Main backtesting engine
    PlanHandler: Strategy plan execution and stock scoring

Example:
    Basic backtesting workflow::

        from backtest import Backtest, PlanHandler
        from config.context_loader import load_config, MainContext

        # Load configuration
        config = load_config('config/conf.yaml')
        main_ctx = MainContext(config)

        # Setup plan handler with scoring strategy
        plan_handler = PlanHandler(
            top_k_num=100,
            absolute_score=500,
            main_ctx=main_ctx
        )

        # Run backtest
        bt = Backtest(
            main_ctx=main_ctx,
            conf=config,
            plan_handler=plan_handler,
            rebalance_period=3  # months
        )

        # Results are automatically saved to reports directory

See Also:
    - ml_backtest: Machine learning based backtesting
    - docs/WORKFLOW_GUIDE.md: Detailed backtesting documentation
"""

from .backtest import Backtest, PlanHandler

__all__ = ['Backtest', 'PlanHandler']

__version__ = '1.0.0'
__author__ = 'Quant Trading Team'
__description__ = 'Backtesting framework for quantitative trading'
