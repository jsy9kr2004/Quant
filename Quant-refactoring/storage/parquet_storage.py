"""자동 검증 및 샘플 생성 기능이 있는 Parquet 저장소 모듈입니다.

이 모듈은 내장 검증, 샘플 생성, 메타데이터 추적 기능이 있는 Parquet 파일 관리를 위한
포괄적인 ParquetStorage 클래스를 제공합니다. 자동 압축, 파티셔닝, 데이터 품질 검사를
지원합니다.

ParquetStorage 클래스는 PyArrow Parquet 작업을 래핑하고 다음을 추가합니다:
    - 저장 후 자동 데이터 검증
    - 빠른 검사를 위한 샘플 CSV 생성
    - 압축 통계 및 리포팅
    - 전체 데이터셋 로드 없이 메타데이터 쿼리
    - 저장된 모든 테이블의 배치 검증

Example:
    자동 검증을 사용한 기본 사용법::

        from storage import ParquetStorage
        import pandas as pd

        # 자동 검증이 활성화된 저장소 초기화
        storage = ParquetStorage(
            root_path="/data/stocks",
            auto_validate=True
        )

        # 압축과 함께 DataFrame 저장
        df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'price': [150.0, 2800.0, 300.0]
        })

        success = storage.save_parquet(
            df=df,
            name='stock_prices',
            compression='snappy',
            save_sample=True
        )

        # 컬럼 필터링으로 데이터 로드
        df_loaded = storage.load_parquet(
            name='stock_prices',
            columns=['symbol', 'price']
        )

        # 데이터 로드 없이 파일 정보 가져오기
        info = storage.get_info('stock_prices')
        print(f"Rows: {info['rows']}, Size: {info['size_mb']:.2f} MB")

Attributes:
    Module-level exports: ParquetStorage, DataValidator
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from .data_validator import DataValidator


class ParquetStorage:
    """Enhanced Parquet storage manager with validation and sample generation.

    This class provides a high-level interface for storing and retrieving Parquet
    files with automatic data validation, sample CSV generation, and comprehensive
    metadata tracking. It handles directory structure creation, compression,
    partitioning, and quality checks.

    The storage creates three main directories:
        - parquet/: Main storage for Parquet files
        - VIEW/: View tables for processed data
        - samples/: CSV samples for quick inspection

    Attributes:
        root_path (Path): Root directory for all storage operations.
        parquet_path (Path): Directory where Parquet files are stored.
        view_path (Path): Directory for view tables.
        sample_path (Path): Directory for sample CSV files.
        auto_validate (bool): Whether to automatically validate data after saving.
        validator (DataValidator): Validator instance for data quality checks.

    Example:
        Initialize storage and save data with partitioning::

            storage = ParquetStorage(
                root_path="/data/financial",
                auto_validate=True
            )

            # Save with year partitioning
            df = pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=1000),
                'symbol': ['AAPL'] * 1000,
                'price': range(1000),
                'year': [2020] * 1000
            })

            storage.save_parquet(
                df=df,
                name='prices',
                partition_cols=['year'],
                compression='zstd'
            )

            # List all stored tables
            tables = storage.list_tables()
            print(f"Stored tables: {tables}")

            # Validate all tables
            results = storage.validate_all_tables()
    """

    def __init__(self, root_path: str, auto_validate: bool = True) -> None:
        """Initialize ParquetStorage with directory structure.

        Creates necessary directories for Parquet storage, views, and samples.
        Initializes the data validator for quality checks.

        Args:
            root_path (str): Root directory path for all data storage. Will be
                created if it doesn't exist.
            auto_validate (bool, optional): If True, automatically validates data
                after saving. Defaults to True.

        Example:
            Initialize storage with custom validation settings::

                # With auto-validation
                storage = ParquetStorage(
                    root_path="/data/stocks",
                    auto_validate=True
                )

                # Without auto-validation (faster for trusted data)
                storage_fast = ParquetStorage(
                    root_path="/data/stocks",
                    auto_validate=False
                )
        """
        self.root_path = Path(root_path)
        self.parquet_path = self.root_path / "parquet"
        self.view_path = self.root_path / "VIEW"
        self.sample_path = self.root_path / "samples"

        # Create directory structure
        self.parquet_path.mkdir(parents=True, exist_ok=True)
        self.view_path.mkdir(parents=True, exist_ok=True)
        self.sample_path.mkdir(parents=True, exist_ok=True)

        self.auto_validate = auto_validate
        self.validator = DataValidator()

        logging.info(f"ParquetStorage initialized at: {self.root_path}")

    def save_parquet(self,
                    df: pd.DataFrame,
                    name: str,
                    compression: str = 'snappy',
                    save_sample: bool = True,
                    sample_size: int = 100,
                    partition_cols: Optional[List[str]] = None) -> bool:
        """Save DataFrame as Parquet with validation and sample generation.

        Saves a DataFrame to Parquet format with optional compression, partitioning,
        and automatic validation. Also generates a sample CSV for quick inspection.
        Logs detailed statistics including file size, compression ratio, and
        validation results.

        Args:
            df (pd.DataFrame): DataFrame to save. Must not be empty.
            name (str): Table name (without .parquet extension). Used as filename
                or directory name if partitioned.
            compression (str, optional): Compression algorithm. Supported values:
                'snappy' (fast, default), 'gzip' (smaller), 'zstd' (balanced),
                'brotli' (smallest). Defaults to 'snappy'.
            save_sample (bool, optional): If True, saves a sample CSV for quick
                inspection. Defaults to True.
            sample_size (int, optional): Number of rows in sample CSV. Defaults to 100.
            partition_cols (Optional[List[str]], optional): Column names for
                partitioning. Creates directory structure with partition values.
                Example: ['year'] creates year=2020/, year=2021/ subdirectories.
                Defaults to None.

        Returns:
            bool: True if save and validation succeeded, False otherwise.

        Raises:
            Exception: Logs exception if save operation fails, returns False.

        Example:
            Save with different compression algorithms::

                # Fast compression (default)
                storage.save_parquet(df, 'fast_data', compression='snappy')

                # Maximum compression
                storage.save_parquet(df, 'archive_data', compression='brotli')

                # With partitioning by year
                storage.save_parquet(
                    df=df,
                    name='historical_prices',
                    partition_cols=['year'],
                    compression='zstd'
                )

                # Save without sample generation
                storage.save_parquet(
                    df=large_df,
                    name='huge_dataset',
                    save_sample=False
                )

        Note:
            - Partitioned datasets skip automatic validation
            - Compression ratio is calculated as memory_size / file_size
            - Validation errors are logged but don't raise exceptions
        """
        file_path = self.parquet_path / f"{name}.parquet"

        try:
            # 1. Save to Parquet format
            logging.info(f"Saving {name}.parquet...")

            if partition_cols:
                # Partitioned save (directory structure)
                df.to_parquet(
                    self.parquet_path / name,
                    engine='pyarrow',
                    compression=compression,
                    partition_cols=partition_cols,
                    index=False
                )
                file_path = self.parquet_path / name
            else:
                # Single file save
                df.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression=compression,
                    index=False
                )

            # 2. Calculate and log storage statistics
            if file_path.is_file():
                file_size = file_path.stat().st_size / 1024**2
            else:
                # For partitioned datasets, calculate total directory size
                file_size = sum(f.stat().st_size for f in file_path.rglob('*.parquet')) / 1024**2

            memory_size = df.memory_usage(deep=True).sum() / 1024**2
            compression_ratio = memory_size / file_size if file_size > 0 else 0

            logging.info(f"✅ Saved: {name}.parquet")
            logging.info(f"   📊 Rows: {len(df):,} | Columns: {len(df.columns)}")
            logging.info(f"   📁 File size: {file_size:.2f} MB")
            logging.info(f"   💾 Memory size: {memory_size:.2f} MB")
            logging.info(f"   🗜️  Compression ratio: {compression_ratio:.1f}x")

            # 3. Generate sample CSV for quick inspection
            if save_sample:
                sample_file = self.sample_path / f"{name}_sample.csv"
                df.head(sample_size).to_csv(sample_file, index=False)
                sample_size_kb = sample_file.stat().st_size / 1024
                logging.info(f"   📄 Sample saved: {sample_file.name} ({sample_size_kb:.1f} KB)")

            # 4. Automatic validation (skip for partitioned datasets)
            if self.auto_validate and not partition_cols:
                logging.info(f"   🔍 Validating {name}...")
                result = self.validator.validate_file(str(file_path), name)

                if not result['passed']:
                    logging.error(f"   ❌ Validation failed for {name}")
                    for error in result['errors']:
                        logging.error(f"      {error}")
                    return False
                else:
                    logging.info(f"   ✅ Validation passed")

                    # Log warnings if present
                    for warning in result['warnings']:
                        logging.warning(f"      ⚠️  {warning}")

            return True

        except Exception as e:
            logging.error(f"❌ Failed to save {name}: {e}")
            return False

    def load_parquet(self,
                    name: str,
                    columns: Optional[List[str]] = None,
                    filters: Optional[List] = None) -> pd.DataFrame:
        """Load Parquet file or partitioned dataset into DataFrame.

        Efficiently loads Parquet data with optional column selection and row
        filtering. Supports both single-file and partitioned datasets. Uses
        PyArrow for fast I/O operations.

        Args:
            name (str): Table name (without .parquet extension).
            columns (Optional[List[str]], optional): List of columns to load.
                If None, loads all columns. Column filtering reduces memory usage.
                Defaults to None.
            filters (Optional[List], optional): PyArrow filter expressions for
                row filtering. Format: [('column', 'operator', value)].
                Example: [('year', '=', 2020), ('price', '>', 100)].
                Defaults to None.

        Returns:
            pd.DataFrame: Loaded DataFrame with specified columns and filters applied.

        Raises:
            FileNotFoundError: If the specified Parquet file or partition doesn't exist.

        Example:
            Load with various filtering options::

                # Load all data
                df = storage.load_parquet('stock_prices')

                # Load specific columns only
                df = storage.load_parquet(
                    'stock_prices',
                    columns=['symbol', 'close', 'volume']
                )

                # Load with row filtering
                df = storage.load_parquet(
                    'stock_prices',
                    filters=[('date', '>=', '2020-01-01')]
                )

                # Combine column selection and filtering
                df = storage.load_parquet(
                    'stock_prices',
                    columns=['symbol', 'close'],
                    filters=[('symbol', 'in', ['AAPL', 'GOOGL'])]
                )

        Note:
            - Column filtering happens at read time, reducing I/O
            - PyArrow filters support: =, !=, <, <=, >, >=, in, not in
            - For partitioned datasets, filters can use partition columns
        """
        file_path = self.parquet_path / f"{name}.parquet"

        if not file_path.exists():
            # Check if it's a partitioned directory
            partition_path = self.parquet_path / name
            if partition_path.exists() and partition_path.is_dir():
                file_path = partition_path
            else:
                raise FileNotFoundError(f"Parquet file not found: {file_path}")

        logging.info(f"Loading {name}.parquet...")

        df = pd.read_parquet(
            file_path,
            columns=columns,
            filters=filters,
            engine='pyarrow'
        )

        logging.info(f"✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")
        return df

    def get_info(self, name: str) -> Dict[str, Any]:
        """Get Parquet file metadata without loading data.

        Retrieves file metadata including schema, row count, and file size without
        reading the actual data. This is much faster than loading the full dataset.
        Uses PyArrow's metadata reading capabilities.

        Args:
            name (str): Table name (without .parquet extension).

        Returns:
            Dict[str, Any]: Dictionary containing metadata with keys:
                - name (str): Table name
                - rows (int): Number of rows
                - row_groups (int): Number of row groups
                - columns (int): Number of columns
                - column_names (List[str]): List of column names
                - dtypes (Dict[str, str]): Mapping of column names to data types
                - size_mb (float): File size in megabytes
                - created (Optional[str]): Creation metadata if available

                Returns {'error': 'File not found'} if file doesn't exist.

        Example:
            Query metadata without loading data::

                # Get basic info
                info = storage.get_info('stock_prices')
                print(f"Rows: {info['rows']:,}")
                print(f"Columns: {info['column_names']}")
                print(f"Size: {info['size_mb']:.2f} MB")

                # Check schema before loading
                info = storage.get_info('financial_data')
                if 'revenue' in info['column_names']:
                    df = storage.load_parquet('financial_data')

                # Get all table sizes
                for table in storage.list_tables():
                    info = storage.get_info(table)
                    print(f"{table}: {info['size_mb']:.2f} MB")

        Note:
            - Only reads file metadata, not actual data
            - Much faster than loading the full DataFrame
            - Does not work with partitioned datasets (directory structure)
        """
        file_path = self.parquet_path / f"{name}.parquet"

        if not file_path.exists():
            return {'error': 'File not found'}

        # Read metadata only using PyArrow (fast operation)
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema_arrow

        return {
            'name': name,
            'rows': metadata.num_rows,
            'row_groups': metadata.num_row_groups,
            'columns': len(schema),
            'column_names': schema.names,
            'dtypes': {field.name: str(field.type) for field in schema},
            'size_mb': file_path.stat().st_size / 1024**2,
            'created': metadata.created_by if hasattr(metadata, 'created_by') else None
        }

    def list_tables(self) -> List[str]:
        """List all stored table names.

        Scans the Parquet directory for both single-file tables and partitioned
        datasets (directories containing Parquet files). Returns sorted list of
        table names without file extensions.

        Returns:
            List[str]: Sorted list of table names (without .parquet extension).

        Example:
            List and iterate over all tables::

                # Get all table names
                tables = storage.list_tables()
                print(f"Found {len(tables)} tables: {tables}")

                # Process all tables
                for table_name in storage.list_tables():
                    df = storage.load_parquet(table_name)
                    print(f"{table_name}: {len(df)} rows")

                # Check if table exists
                if 'stock_prices' in storage.list_tables():
                    df = storage.load_parquet('stock_prices')
        """
        tables = []

        # Find single Parquet files
        for file_path in self.parquet_path.glob("*.parquet"):
            tables.append(file_path.stem)

        # Find partitioned directories
        for dir_path in self.parquet_path.iterdir():
            if dir_path.is_dir() and list(dir_path.glob("*.parquet")):
                tables.append(dir_path.name)

        return sorted(tables)

    def delete_table(self, name: str) -> bool:
        """Delete a table and its associated files.

        Removes the Parquet file or partitioned directory and any associated
        sample CSV files. Handles both single-file and partitioned datasets.

        Args:
            name (str): Table name to delete (without .parquet extension).

        Returns:
            bool: True if deletion succeeded, False if table not found or error occurred.

        Example:
            Delete tables with confirmation::

                # Delete a single table
                if storage.delete_table('old_data'):
                    print("Table deleted successfully")

                # Delete multiple tables
                tables_to_delete = ['temp1', 'temp2', 'test_data']
                for table in tables_to_delete:
                    storage.delete_table(table)

                # Delete with existence check
                if 'deprecated_table' in storage.list_tables():
                    storage.delete_table('deprecated_table')

        Note:
            - Also deletes associated sample CSV files
            - For partitioned datasets, removes entire directory tree
            - Logs warnings if table not found
        """
        file_path = self.parquet_path / f"{name}.parquet"
        partition_path = self.parquet_path / name

        try:
            if file_path.exists():
                file_path.unlink()
                logging.info(f"🗑️  Deleted: {name}.parquet")
            elif partition_path.exists() and partition_path.is_dir():
                import shutil
                shutil.rmtree(partition_path)
                logging.info(f"🗑️  Deleted: {name}/ (partitioned)")
            else:
                logging.warning(f"Table not found: {name}")
                return False

            # Delete associated sample file
            sample_file = self.sample_path / f"{name}_sample.csv"
            if sample_file.exists():
                sample_file.unlink()

            return True

        except Exception as e:
            logging.error(f"Failed to delete {name}: {e}")
            return False

    def validate_all_tables(self) -> Dict[str, Dict[str, Any]]:
        """Validate all stored Parquet tables.

        Runs comprehensive validation on all tables in the storage directory.
        Checks data quality, schema consistency, required columns, and data types.
        Uses the DataValidator for detailed validation rules.

        Returns:
            Dict[str, Dict[str, Any]]: Nested dictionary mapping table names to
                validation results. Each result contains:
                - passed (bool): Overall validation status
                - errors (List[str]): List of validation errors
                - warnings (List[str]): List of warnings
                - info (Dict): Metadata and statistics

        Example:
            Validate all tables and check results::

                # Run validation on all tables
                results = storage.validate_all_tables()

                # Check which tables passed
                for table_name, result in results.items():
                    if result['passed']:
                        print(f"✅ {table_name}: OK")
                    else:
                        print(f"❌ {table_name}: FAILED")
                        for error in result['errors']:
                            print(f"   - {error}")

                # Count validation issues
                failed = sum(1 for r in results.values() if not r['passed'])
                print(f"{failed} tables failed validation")

        See Also:
            generate_validation_report: Generate a text report from validation results.
        """
        return self.validator.validate_all(str(self.parquet_path))

    def generate_validation_report(self, output_file: str = "validation_report.txt") -> str:
        """Generate a text report of validation results.

        Validates all tables and writes a detailed report to a text file.
        The report includes validation status, errors, warnings, and statistics
        for each table in a human-readable format.

        Args:
            output_file (str, optional): Path to output report file. Can be
                absolute or relative path. Defaults to "validation_report.txt".

        Returns:
            str: Path to the generated report file.

        Example:
            Generate and review validation report::

                # Generate report with default name
                report_path = storage.generate_validation_report()
                print(f"Report saved to: {report_path}")

                # Generate with custom name
                report_path = storage.generate_validation_report(
                    output_file="/reports/parquet_validation_2024.txt"
                )

                # Read and print report
                with open(report_path, 'r') as f:
                    print(f.read())

        Note:
            - Report includes summary statistics for all tables
            - Lists all errors and warnings in detail
            - Output file is overwritten if it exists
        """
        results = self.validate_all_tables()
        return self.validator.generate_report(results, output_file)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire storage.

        Aggregates information about all stored tables including total size,
        row counts, and per-table statistics. Useful for monitoring storage
        usage and data volume.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_tables (int): Number of stored tables
                - total_size_mb (float): Total storage size in MB
                - total_rows (int): Total number of rows across all tables
                - tables (List[Dict]): Per-table statistics with keys:
                    - name (str): Table name
                    - rows (int): Number of rows
                    - size_mb (float): File size in MB

        Example:
            Monitor storage usage::

                # Get overall statistics
                stats = storage.get_statistics()
                print(f"Total tables: {stats['total_tables']}")
                print(f"Total size: {stats['total_size_mb']:.2f} MB")
                print(f"Total rows: {stats['total_rows']:,}")

                # Find largest tables
                tables = sorted(
                    stats['tables'],
                    key=lambda x: x['size_mb'],
                    reverse=True
                )
                print("\nTop 5 largest tables:")
                for table in tables[:5]:
                    print(f"  {table['name']}: {table['size_mb']:.2f} MB")

                # Calculate average table size
                avg_size = stats['total_size_mb'] / stats['total_tables']
                print(f"Average table size: {avg_size:.2f} MB")

        Note:
            - Statistics calculation may be slow for many large tables
            - Skips tables that cannot be read (logs warning)
            - Does not include sample CSV files in size calculations
        """
        tables = self.list_tables()
        total_size = 0
        total_rows = 0

        stats = {
            'total_tables': len(tables),
            'tables': []
        }

        for table in tables:
            try:
                info = self.get_info(table)
                total_size += info.get('size_mb', 0)
                total_rows += info.get('rows', 0)
                stats['tables'].append({
                    'name': table,
                    'rows': info.get('rows', 0),
                    'size_mb': info.get('size_mb', 0)
                })
            except Exception as e:
                logging.warning(f"Failed to get info for {table}: {e}")

        stats['total_size_mb'] = total_size
        stats['total_rows'] = total_rows

        return stats
