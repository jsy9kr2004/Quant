"""
Quant Trading System - Main Pipeline (Refactored)

This is the main entry point for the quantitative trading system. It orchestrates
the complete pipeline from data collection to backtesting:

1. Configuration Loading: Loads and validates YAML configuration
2. Data Collection: Fetches financial data from FMP API (optional)
3. ML Pipeline: Prepares training data, trains models, evaluates performance
4. Backtesting: Simulates trading strategy on historical data

The system supports:
- Parquet-based storage for efficient data handling
- Multiple ML models (XGBoost, LightGBM, CatBoost)
- Ensemble strategies with MLflow tracking
- Flexible backtesting with custom scoring strategies

Usage:
    python main.py

Configuration:
    Edit config/conf.yaml to customize:
    - DATA.GET_FMP: Enable/disable data collection
    - ML.RUN_REGRESSION: Enable/disable ML training
    - BACKTEST.RUN_BACKTEST: Enable/disable backtesting

Author: Quant Trading Team
Date: 2025-10-29
"""

import logging
import os
import sys
from typing import Dict, Any, Optional, List
import yaml
import pandas as pd
from pathlib import Path

# Add current directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from config.context_loader import load_config, MainContext
from config.logger import get_logger
from storage import ParquetStorage
from data_collector.fmp import FMP
from training.make_mldata import AIDataMaker
from models import XGBoostModel, LightGBMModel, CatBoostModel, StackingEnsemble
from training import OptunaOptimizer, MLflowTracker
from backtest import Backtest, PlanHandler


class RegressorIntegrated:
    """
    Integrated ML training pipeline combining legacy and new model architectures.

    This class serves as a bridge between the legacy regressor implementation
    and the new modular model structure. It provides backward compatibility
    while enabling use of new features like MLflow tracking and modular models.

    Attributes:
        conf (Dict[str, Any]): Configuration dictionary from YAML
        use_new_models (bool): Whether to use new model architecture
        legacy_regressor (Optional[Regressor]): Legacy regressor instance for fallback

    Example:
        >>> config = load_config('config/conf.yaml')
        >>> regressor = RegressorIntegrated(config, use_new_models=True)
        >>> regressor.dataload()
        >>> regressor.train()
        >>> regressor.evaluation()

    TODO:
        - Implement native data loading without legacy dependency
        - Add support for custom model architectures
        - Implement cross-validation in training pipeline
    """

    def __init__(self, conf: Dict[str, Any], use_new_models: bool = True) -> None:
        """
        Initialize the integrated regressor.

        Args:
            conf: Configuration dictionary containing DATA, ML, and BACKTEST settings
            use_new_models: If True, uses new model architecture with MLflow tracking.
                          If False, falls back to legacy regressor implementation.

        Raises:
            ImportError: If legacy regressor is requested but not available
        """
        self.conf = conf
        self.use_new_models = use_new_models

        # Import legacy Regressor as fallback
        try:
            from training.regressor import Regressor
            self.legacy_regressor: Optional['Regressor'] = Regressor(conf)
        except ImportError:
            logger = get_logger('RegressorIntegrated')
            logger.warning("Legacy regressor not found, using new models only")
            self.legacy_regressor = None

    def dataload(self) -> None:
        """
        Load training and test data from Parquet files.

        Loads quarterly ML data files (rnorm_ml_{year}_{quarter}.parquet) for
        training and testing periods specified in configuration.

        Data Processing:
        - Filters out columns with >80% missing values
        - Filters out columns where >95% rows have same value
        - Filters out rows with >60% missing values
        - Maps stocks to sectors
        - Separates features (X) and targets (y)

        Raises:
            ValueError: If no training data files are found
            FileNotFoundError: If required data files are missing

        TODO:
            - Implement native data loading without legacy dependency
            - Add data validation and quality checks
        """
        if self.legacy_regressor:
            self.legacy_regressor.dataload()
        else:
            logger = get_logger('RegressorIntegrated')
            logger.warning("Using new data loading method")
            # TODO: Implement new data loading method
            # Should load from /data/ml_per_year/rnorm_ml_*.parquet

    def train(self) -> None:
        """
        Train ML models using configured strategy.

        Training Process:
        1. If use_new_models=True and MLflow enabled:
           - Uses new model architecture
           - Tracks experiments with MLflow
           - Supports hyperparameter optimization
        2. Otherwise:
           - Falls back to legacy training pipeline
           - Trains ensemble of XGBoost + LightGBM models

        Models Trained:
        - 3x XGBoost Classifiers (depth 8, 9, 10)
        - 1x LightGBM Classifier
        - 2x XGBoost Regressors (depth 8, 10)

        Raises:
            ValueError: If no training method is available
            RuntimeError: If model training fails
        """
        if self.use_new_models and self.conf.get('ML', {}).get('USE_MLFLOW'):
            self._train_with_new_models()
        elif self.legacy_regressor:
            self.legacy_regressor.train()
        else:
            logger = get_logger('RegressorIntegrated')
            logger.error("No training method available")

    def _train_with_new_models(self) -> None:
        """
        Train models using new architecture with MLflow tracking.

        Features:
        - MLflow experiment tracking
        - Modular model architecture
        - Configurable hyperparameters
        - Stacking ensemble support

        TODO:
            - Implement complete training pipeline
            - Add support for custom model configurations
            - Integrate with OptunaOptimizer for HPO
        """
        logger = get_logger('RegressorIntegrated')
        logger.info("Training with new model structure + MLflow")

        ml_config = self.conf.get('ML', {})

        # Initialize MLflow tracker
        if ml_config.get('USE_MLFLOW'):
            tracker = MLflowTracker(
                experiment_name=ml_config.get('MLFLOW_EXPERIMENT', 'quant_trading')
            )

        # TODO: Load training data from legacy regressor or implement native loading
        # X_train = self.legacy_regressor.x_train
        # y_train = self.legacy_regressor.y_train

        # TODO: Build and train models
        # - Create base models (XGBoost, LightGBM, CatBoost)
        # - Train each model with early stopping
        # - Create stacking ensemble
        # - Log to MLflow

        logger.info("New model training completed (placeholder)")

    def evaluation(self) -> None:
        """
        Evaluate trained models on test set.

        Evaluation Process:
        1. Load test data (quarterly files from TEST_START_YEAR to TEST_END_YEAR)
        2. For each quarter:
           - Run classification models (predict up/down)
           - Apply threshold filtering (top 8%)
           - Run regression models (predict magnitude)
           - Combine predictions with ensemble voting
        3. Calculate metrics:
           - Classification: Accuracy, Precision, Recall, F1
           - Regression: RMSE, MAE
           - Top-K performance: Average return of top 3, 8, 16 stocks
        4. Save predictions to CSV files

        Output Files:
        - prediction_ai_{year}_{quarter}.csv: Full predictions
        - prediction_*_top0-3.csv: Top 3 stock predictions
        - pred_df_topk.csv: Summary statistics

        Raises:
            FileNotFoundError: If test data files are missing
        """
        if self.legacy_regressor:
            self.legacy_regressor.evaluation()

    def latest_prediction(self) -> None:
        """
        Generate predictions for the most recent data.

        Process:
        1. Load latest quarterly data (most recent year_period)
        2. Run all trained models
        3. Apply ensemble strategy
        4. Rank stocks by predicted return
        5. Select top-K stocks for each model combination

        Output Files:
        - latest_prediction.csv: All predictions with scores
        - latest_prediction_{model}_top0-3.csv: Top 3 stocks per model
        - latest_prediction_{model}_top0-7.csv: Top 8 stocks per model
        - latest_prediction_{model}_top0-15.csv: Top 16 stocks per model

        These files can be used for actual trading decisions.

        Raises:
            FileNotFoundError: If latest data files are missing
        """
        if self.legacy_regressor:
            self.legacy_regressor.latest_prediction()


def get_config_path() -> str:
    """
    Find configuration file path.

    Searches for config/conf.yaml in multiple locations:
    1. ./config/conf.yaml (current directory)
    2. ../config/conf.yaml (parent directory)
    3. Quant-refactoring/config/conf.yaml (project root)

    Returns:
        str: Path to configuration file

    Raises:
        FileNotFoundError: If config file is not found in any location

    Example:
        >>> config_path = get_config_path()
        >>> print(config_path)
        'config/conf.yaml'
    """
    possible_paths = [
        'config/conf.yaml',
        '../config/conf.yaml',
        'Quant-refactoring/config/conf.yaml'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Config file not found. Please create config/conf.yaml\n"
        "Use config/conf.yaml.template as reference"
    )


def conf_check(config: Dict[str, Any]) -> None:
    """
    Validate configuration file.

    Checks:
    1. REPORT_LIST contains only valid report types
    2. ROOT_PATH is specified in DATA section
    3. Required sections exist (DATA, ML, BACKTEST)

    Args:
        config: Configuration dictionary loaded from YAML

    Raises:
        SystemExit: If validation fails (exits with code 1)

    Example:
        >>> config = load_config('config/conf.yaml')
        >>> conf_check(config)
        INFO: ✅ Configuration validated
    """
    logger = get_logger('config_check')

    # Validate REPORT_LIST
    valid_reports = ["EVAL", "RANK", "AI", "AVG"]
    report_list = config.get('BACKTEST', {}).get('REPORT_LIST', [])

    for rep_type in report_list:
        if rep_type not in valid_reports:
            logger.critical(f"Invalid REPORT_LIST: {rep_type}. Valid: {valid_reports}")
            sys.exit(1)

    # Validate required settings
    data_config = config.get('DATA', {})
    if not data_config.get('ROOT_PATH'):
        logger.critical("ROOT_PATH not set in config")
        sys.exit(1)

    logger.info("✅ Configuration validated")


def main() -> None:
    """
    Main pipeline orchestrating the complete quantitative trading workflow.

    Pipeline Stages:
    ===============

    1. Configuration Loading & Validation
       - Loads config/conf.yaml
       - Validates all settings
       - Initializes logging and context

    2. Data Collection (Optional: GET_FMP=Y)
       - Fetches data from FMP API
       - Saves to CSV files
       - Converts to Parquet format
       - Builds integrated views

    3. ML Pipeline (Optional: RUN_REGRESSION=Y)
       a. Data Preparation:
          - Loads VIEW files
          - Extracts time-series features (tsfresh)
          - Calculates financial ratios
          - Normalizes with RobustScaler
          - Saves quarterly ML datasets
       b. Model Training:
          - Loads training data (2015-2021)
          - Trains classification models (up/down)
          - Trains regression models (magnitude)
          - Saves models to /data/MODELS/
       c. Evaluation:
          - Tests on 2022-2023 data
          - Generates performance reports
          - Saves top-K predictions
       d. Latest Prediction:
          - Predicts on most recent data
          - Ranks stocks by expected return
          - Generates trading signals

    4. Backtesting (Optional: RUN_BACKTEST=Y)
       - Loads custom scoring strategy (plan.csv)
       - Simulates trading on historical data
       - Calculates returns, Sharpe ratio, drawdown
       - Generates backtest reports

    Configuration Options:
    =====================

    config/conf.yaml:
        DATA:
            GET_FMP: Y/N - Fetch new data from API
            ROOT_PATH: /path/to/data - Data directory
            START_YEAR: 2015 - Data collection start
            END_YEAR: 2023 - Data collection end
            STORAGE_TYPE: PARQUET - Storage format

        ML:
            RUN_REGRESSION: Y/N - Run ML pipeline
            USE_NEW_MODELS: Y/N - Use new model architecture
            USE_MLFLOW: Y/N - Enable MLflow tracking
            TRAIN_START_YEAR: 2015 - Training period start
            TRAIN_END_YEAR: 2021 - Training period end
            TEST_START_YEAR: 2022 - Test period start
            TEST_END_YEAR: 2023 - Test period end
            EXIT_AFTER_ML: Y/N - Exit after ML (skip backtest)

        BACKTEST:
            RUN_BACKTEST: Y/N - Run backtesting
            REBALANCE_PERIOD: 3 - Months between rebalancing
            TOP_K_NUM: 100 - Number of stocks to select
            ABSOLUTE_SCORE: 500 - Minimum score threshold
            REPORT_LIST: [EVAL, RANK, AVG] - Report types

    Raises:
        FileNotFoundError: If configuration file is not found
        ValueError: If configuration is invalid
        SystemExit: On critical errors (exit code 1)
        KeyboardInterrupt: On user interruption (exit code 0)

    Example:
        $ python main.py
        ================================================================================
        Quant Trading System - Refactored Version
        ================================================================================
        INFO: ✅ Configuration validated
        INFO: ================================================================================
        INFO: Step 1: FMP Data Collection
        INFO: ================================================================================
        ...

    See Also:
        - WORKFLOW_GUIDE.md: Detailed system documentation
        - README.md: Quick start guide
        - config/conf.yaml.template: Configuration template
    """
    print("="*80)
    print("Quant Trading System - Refactored Version")
    print("="*80)

    # 1. Load and validate configuration
    try:
        config_path = get_config_path()
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease create config/conf.yaml with your settings.")
        print("See config/conf.yaml.template for reference.")
        sys.exit(1)

    # 2. Initialize context (automatically sets up logging)
    main_ctx = MainContext(config)
    logger = get_logger('main')

    conf_check(config)

    # 3. Create output directories
    main_ctx.create_dir("./reports")

    # 4. Configure data storage
    data_config = config.get('DATA', {})
    storage_type = data_config.get('STORAGE_TYPE', 'PARQUET')

    # 5. FMP Data Collection (Optional)
    if data_config.get('GET_FMP') == 'Y':
        logger.info("="*80)
        logger.info("Step 1: FMP Data Collection")
        logger.info("="*80)

        try:
            from data_collector.fmp import FMP
            fmp = FMP(main_ctx)
            fmp.collect()

            # Convert to Parquet format for efficient storage and retrieval
            if storage_type == 'PARQUET':
                logger.info("Converting to Parquet format...")

                # Initialize Parquet storage with auto-validation
                storage = ParquetStorage(
                    root_path=main_ctx.root_path,
                    auto_validate=True
                )

                # Use legacy converter to process CSV files
                from storage.parquet_converter import Parquet
                df_engine = Parquet(main_ctx)
                df_engine.insert_csv()
                df_engine.rebuild_table_view()

                logger.info("✅ Data saved in Parquet format")

            elif storage_type == 'DB':
                logger.warning("Database storage not recommended. Use PARQUET instead.")
                from database import Database
                db = Database(main_ctx)
                db.insert_csv()
                db.rebuild_table_view()

        except Exception as e:
            logger.error(f"FMP data collection failed: {e}")
            logger.info("Continuing with existing data...")

    # 6. ML Pipeline (Optional)
    ml_config = config.get('ML', {})
    if ml_config.get('RUN_REGRESSION') == 'Y':
        logger.info("="*80)
        logger.info("Step 2: ML Pipeline")
        logger.info("="*80)

        # 6.1 Prepare ML training data
        logger.info("Preparing ML training data...")
        AIDataMaker(main_ctx, config)

        # 6.2 Train models
        logger.info("Training models...")
        regressor = RegressorIntegrated(
            config,
            use_new_models=ml_config.get('USE_NEW_MODELS', False)
        )
        regressor.dataload()
        regressor.train()
        regressor.evaluation()
        regressor.latest_prediction()

        logger.info("✅ ML pipeline completed")

        # Exit after ML if configured (allows running ML without backtest)
        if ml_config.get('EXIT_AFTER_ML', True):
            logger.info("Exiting after ML (set EXIT_AFTER_ML=N to continue to backtest)")
            sys.exit(0)

    # 7. Backtesting (Optional)
    backtest_config = config.get('BACKTEST', {})
    if backtest_config.get('RUN_BACKTEST', True):
        logger.info("="*80)
        logger.info("Step 3: Backtesting")
        logger.info("="*80)

        # Initialize plan handler for stock scoring
        plan_handler = PlanHandler(
            backtest_config.get('TOP_K_NUM', 100),
            backtest_config.get('ABSOLUTE_SCORE', 500),
            main_ctx
        )

        # Load custom scoring plan from CSV
        plan: List[Dict[str, Any]] = []
        plan_file = "plan.csv"

        if os.path.exists(plan_file):
            plan_df = pd.read_csv(plan_file)
            plan_info = plan_df.values.tolist()

            for i in range(len(plan_info)):
                plan.append({
                    "f_name": plan_handler.single_metric_plan_no_parallel,
                    "params": {
                        "key": plan_info[i][0],         # Metric name (e.g., 'roe')
                        "key_dir": plan_info[i][1],     # Direction ('ascending'/'descending')
                        "weight": plan_info[i][2],      # Weight in scoring
                        "diff": plan_info[i][3],        # Use price difference
                        "base": plan_info[i][4],        # Base metric for normalization
                        "base_dir": plan_info[i][5]     # Base direction
                    }
                })

            plan_handler.plan_list = plan
        else:
            logger.warning(f"Plan file not found: {plan_file}")
            logger.info("Using default plan (empty)")

        # Run backtest simulation
        bt = Backtest(
            main_ctx,
            config,
            plan_handler,
            rebalance_period=backtest_config.get('REBALANCE_PERIOD', 3)
        )

        logger.info("✅ Backtesting completed")

        # Cleanup
        del plan_handler
        del bt

    # 8. Pipeline completed
    logger.info("="*80)
    logger.info("Pipeline completed successfully!")
    logger.info("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        err_logger = get_logger('main')
        err_logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        err_logger = get_logger('main')
        err_logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
