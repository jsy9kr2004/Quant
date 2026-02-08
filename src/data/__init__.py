"""Data module for financial data collection, storage and validation.

This package provides a comprehensive suite of tools for:
1. Collecting financial data from FMP API
2. Storing data in Parquet format with compression
3. Validating data quality

Main Components:
    FMP: Main data collector for FMP API
    FMPAPI: Low-level API client
    FetchWorker: Worker for parallel data fetching
    ParquetStorage: Modern storage manager with validation and compression
    DataValidator: Comprehensive data quality validator
    Parquet: Legacy converter for consolidating multiple data sources

Example:
    Data collection and storage workflow::

        from src.data import FMP, ParquetStorage
        from src.infra import load_config, MainContext

        # Load configuration
        config = load_config('config/conf.yaml')
        main_ctx = MainContext(config)

        # Collect data from FMP API
        fmp = FMP(main_ctx)
        fmp.collect()

        # Use storage for data management
        storage = ParquetStorage(root_path="/data/financial")
        df = storage.load_parquet('stock_prices')
"""

from .fmp import FMP
from .fmp_api import FMPAPI
from .parquet_storage import ParquetStorage
from .data_validator import DataValidator
from .parquet_converter import Parquet

__all__ = [
    'FMP',
    'FMPAPI',
    'ParquetStorage',
    'DataValidator',
    'Parquet',
]

__version__ = '1.0.0'
__author__ = 'Quant Trading Team'
__description__ = 'Financial data collection, storage and validation'
