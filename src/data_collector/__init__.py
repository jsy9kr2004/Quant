"""Data collection module for financial data from FMP API.

This package provides tools for collecting financial data from the
Financial Modeling Prep (FMP) API. It handles API requests, rate limiting,
data caching, and storage in both CSV and Parquet formats.

Main Components:
    FMP: Main data collector for FMP API
    FMPApi: Low-level API client
    FetchWorker: Worker for parallel data fetching

Example:
    Basic data collection workflow::

        from data_collector import FMP
        from config.context_loader import load_config, MainContext

        # Load configuration
        config = load_config('config/conf.yaml')
        main_ctx = MainContext(config)

        # Collect data from FMP API
        fmp = FMP(main_ctx)
        fmp.collect()

        # Data is automatically saved to ROOT_PATH/fmp_raw/

Features:
    - Automatic rate limiting and retry logic
    - Parallel fetching with worker pool
    - CSV and Parquet output formats
    - Incremental updates (only fetch new data)
    - Symbol list caching
    - Progress tracking with tqdm

Data Categories:
    - Income statements
    - Balance sheets
    - Cash flow statements
    - Key metrics
    - Financial ratios
    - Stock prices
    - Available indexes

See Also:
    - fmp_api: Low-level API client
    - fmp_fetch_worker: Parallel fetch worker
    - docs/WORKFLOW_GUIDE.md: Data collection documentation
"""

from .fmp import FMP

__all__ = ['FMP']

__version__ = '1.0.0'
__author__ = 'Quant Trading Team'
__description__ = 'Financial data collection from FMP API'
