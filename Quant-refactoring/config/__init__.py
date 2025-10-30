"""Configuration module for the Quant Trading System.

This package provides configuration management, logging infrastructure, and global
variables for the Quant Trading System. It serves as the central hub for all
application-wide settings and utilities.

The config package includes:

1. **Configuration Loading** (context_loader.py):
   - ConfigLoader: Modern YAML configuration loader with dot notation
   - MainContext: Main application context with integrated logging
   - ContextLoader: Legacy configuration loader for backward compatibility
   - load_config: Convenience function for loading configuration files

2. **Unified Logging System** (logger.py):
   - setup_logging: Initialize centralized logging with multiprocessing support
   - get_logger: Get logger instances throughout the application
   - Colored console output with automatic rotation
   - Thread-safe and multiprocessing-safe logging

3. **Global Variables** (g_variables.py):
   - Feature column lists for machine learning
   - Sector classification mappings
   - Data quality indicators
   - Financial metric definitions

Primary Exports:
    - load_config: Load configuration from YAML file
    - MainContext: Main application context with logging and settings

Usage:
    Basic application setup::

        from config import load_config, MainContext

        # Load configuration file
        config = load_config('config/conf.yaml')

        # Create main application context
        # This automatically sets up logging and loads all settings
        context = MainContext(config)

        # Get a logger for your module
        logger = context.get_logger(__name__)
        logger.info("Application initialized")

        # Access configuration settings
        print(f"Data path: {context.root_path}")
        print(f"Processing {context.start_year} to {context.end_year}")

    Using the modern config loader::

        from config.context_loader import ConfigLoader

        # Load and access configuration
        config = ConfigLoader('config/conf.yaml')

        # Access nested values with dot notation
        start_year = config.get('DATA.START_YEAR', 2015)
        api_key = config.get('DATA.API_KEY')

        # Get configuration sections
        ml_config = config.get_ml_config()
        backtest_config = config.get_backtest_config()

    Using the logging system directly::

        from config.logger import setup_logging, get_logger

        # Setup logging (call once at application start)
        setup_logging(
            log_level='INFO',
            log_file='trading.log',
            log_dir='logs',
            use_colors=True
        )

        # Get logger in any module
        logger = get_logger(__name__)
        logger.info("Processing data")
        logger.error("Failed to connect", extra={'host': 'api.example.com'})

    Using global variables::

        from config.g_variables import ratio_col_list, sector_map

        # Select financial ratio features
        ratio_features = df[ratio_col_list]

        # Map industry to sector
        sector = sector_map.get(industry, 'Unknown')

Module Structure:
    config/
    ├── __init__.py           # This file - package initialization
    ├── context_loader.py     # Configuration loading and context management
    ├── logger.py            # Unified logging system
    ├── g_variables.py       # Global variables and constants
    └── conf.yaml            # Configuration file (not in Python package)

Configuration File Format (conf.yaml):
    The configuration file should contain the following sections::

        DATA:
          START_YEAR: 2015
          END_YEAR: 2023
          ROOT_PATH: /home/user/Quant/data
          API_KEY: your_api_key_here
          FMP_URL: https://financialmodelingprep.com

        ML:
          MODEL_TYPE: xgboost
          TRAIN_START_YEAR: 2015
          TEST_START_YEAR: 2020

        BACKTEST:
          INITIAL_CAPITAL: 1000000
          REBALANCE_PERIOD: monthly

        LOG_LVL: 20  # 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL

Design Principles:
    - **Centralized Configuration**: All settings in one place (conf.yaml)
    - **Type Safety**: Type hints throughout for better IDE support
    - **Backward Compatibility**: Legacy ContextLoader maintained for old code
    - **Modern Patterns**: New code uses ConfigLoader and MainContext
    - **Multiprocessing Support**: Logging system is multiprocessing-safe
    - **Easy Migration**: Existing code can gradually migrate to new patterns

Migration Guide:
    From legacy ContextLoader to modern approach::

        # Old code
        from config.context_loader import ContextLoader
        context = ContextLoader()
        logger = context.get_logger('MyModule')

        # New code (recommended)
        from config import load_config, MainContext
        config = load_config('config/conf.yaml')
        context = MainContext(config)
        logger = context.get_logger('MyModule')

See Also:
    - context_loader: For detailed configuration loading documentation
    - logger: For logging system architecture and features
    - g_variables: For available feature lists and sector mappings

Notes:
    - The unified logging system is automatically configured when MainContext
      is initialized
    - Configuration files are loaded using safe YAML parsing (yaml.safe_load)
    - All loggers share the same configuration for consistency
    - Global variables are constants and should not be modified at runtime

Examples:
    Complete application initialization::

        from config import load_config, MainContext
        from config.logger import get_logger

        def main():
            # Load configuration
            config = load_config('config/conf.yaml')

            # Initialize main context (sets up logging automatically)
            context = MainContext(config)

            # Get logger for main module
            logger = get_logger(__name__)
            logger.info("="*80)
            logger.info("Application Starting")
            logger.info("="*80)

            # Create required directories
            context.create_dir(context.root_path)

            # Your application logic here
            logger.info(f"Processing data from {context.start_year}")

            logger.info("Application completed successfully")

        if __name__ == '__main__':
            main()

    Multiprocessing application::

        from config import load_config, MainContext
        from config.logger import setup_logger_for_multiprocessing, get_logger
        import multiprocessing as mp

        def worker_function(symbol, config):
            # Configure logging for child process
            setup_logger_for_multiprocessing()
            logger = get_logger(__name__)

            logger.info(f"Processing {symbol}")
            # Worker logic here
            logger.info(f"Completed {symbol}")

        def main():
            # Setup in main process
            config = load_config('config/conf.yaml')
            context = MainContext(config)
            logger = get_logger(__name__)

            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

            # Create worker pool
            with mp.Pool(4) as pool:
                pool.starmap(worker_function, [(s, config) for s in symbols])

            logger.info("All workers completed")

        if __name__ == '__main__':
            main()
"""

from .context_loader import load_config, MainContext

__all__ = ['load_config', 'MainContext']
