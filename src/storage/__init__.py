"""Storage module for Parquet data management with validation.

This package provides a comprehensive suite of tools for managing financial data
stored in Parquet format. It includes storage management, data conversion,
validation, and quality assurance functionality.

The storage module is designed for financial data pipelines that need:
    - Efficient Parquet file storage with compression
    - Automatic data validation and quality checks
    - Legacy data format conversion and consolidation
    - Sample generation for quick data inspection
    - Metadata tracking without loading full datasets

Main Components:
    ParquetStorage: Modern storage manager with validation and compression
    DataValidator: Comprehensive data quality validator
    Parquet: Legacy converter for consolidating multiple data sources

Example:
    Using the storage module for a complete workflow::

        from storage import ParquetStorage, DataValidator, Parquet
        import pandas as pd

        # 1. Modern storage with validation
        storage = ParquetStorage(
            root_path="/data/financial",
            auto_validate=True
        )

        # Save data with automatic validation
        df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0],
            'date': pd.date_range('2024-01-01', periods=2)
        })

        storage.save_parquet(
            df=df,
            name='stock_prices',
            compression='zstd',
            save_sample=True
        )

        # Load with filtering
        df_loaded = storage.load_parquet(
            'stock_prices',
            columns=['symbol', 'price']
        )

        # 2. Direct validation
        validator = DataValidator()
        results = validator.validate_all('/data/financial/parquet/')
        report_path = validator.generate_report(results)

        # 3. Legacy data conversion (if needed)
        converter = Parquet(main_ctx)
        converter.insert_csv()
        converter.rebuild_table_view()

Module Structure:
    storage/
    ├── __init__.py           # This file - package exports
    ├── parquet_storage.py    # Modern storage with validation
    ├── data_validator.py     # Data quality validation
    └── parquet_converter.py  # Legacy data conversion

Usage Patterns:
    For new projects, use ParquetStorage for all storage operations::

        from storage import ParquetStorage

        storage = ParquetStorage("/data")
        storage.save_parquet(df, "table_name")
        df = storage.load_parquet("table_name")

    For data validation::

        from storage import DataValidator

        validator = DataValidator()
        results = validator.validate_all("/data/parquet/")

    For legacy data migration::

        from storage import Parquet

        converter = Parquet(main_ctx)
        converter.insert_csv()
        converter.rebuild_table_view()

Dependencies:
    - pandas: DataFrame operations and Parquet I/O
    - pyarrow: Efficient Parquet file format support
    - pathlib: Modern file path handling
    - logging: Comprehensive operation logging
    - tqdm: Progress bars for long operations

See Also:
    - ParquetStorage: Main storage class documentation
    - DataValidator: Validation rules and methods
    - Parquet: Legacy converter documentation
"""

from .data_validator import DataValidator
from .parquet_storage import ParquetStorage
from .parquet_converter import Parquet

# Define public API
__all__ = ['DataValidator', 'ParquetStorage', 'Parquet']

# Version information
__version__ = '1.0.0'
__author__ = 'Quant Team'
__description__ = 'Parquet storage and validation for financial data'
