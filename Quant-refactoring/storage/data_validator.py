"""Parquet 파일을 위한 데이터 검증 모듈입니다.

이 모듈은 금융 데이터 파이프라인에서 사용되는 Parquet 파일에 대한 포괄적인
데이터 품질 검증을 제공합니다. 스키마 준수, 데이터 타입, null 값, 데이터 품질 메트릭,
도메인 특정 비즈니스 규칙을 검증합니다.

DataValidator 클래스는 다양한 테이블 유형에 대한 설정 가능한 검증 규칙을 구현하고
오류, 경고, 통계를 포함한 상세한 검증 보고서를 생성합니다.

검증 항목:
    - 필수 컬럼 존재 여부
    - 중요 컬럼의 null 값 검증
    - 최소 행 수 임계값
    - 데이터 타입 검증 (특히 datetime 컬럼)
    - 중복 감지
    - 데이터 품질 메트릭 (null 비율 등)
    - 날짜 범위 검증

Example:
    기본 검증 워크플로우::

        from storage import DataValidator
        import pandas as pd

        # 검증기 초기화
        validator = DataValidator()

        # 단일 파일 검증
        result = validator.validate_file(
            file_path='/data/parquet/stock_prices.parquet',
            table_name='price'
        )

        if result['passed']:
            print("Validation passed!")
            print(f"Rows: {result['info']['rows']:,}")
        else:
            print("Validation failed:")
            for error in result['errors']:
                print(f"  - {error}")

        # 디렉토리의 모든 파일 검증
        all_results = validator.validate_all('/data/parquet/')

        # 상세 보고서 생성
        report_path = validator.generate_report(
            all_results,
            output_file='validation_report.txt'
        )

Attributes:
    Module-level exports: DataValidator
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class DataValidator:
    """Parquet file data quality validator.

    This class provides comprehensive validation for Parquet files containing
    financial data. It checks schema compliance, data types, null values,
    and data quality metrics based on configurable rules.

    Each table type has its own validation rules defining:
        - Required columns that must be present
        - Columns that cannot contain null values
        - Minimum row count thresholds
        - Description of the table's purpose

    Attributes:
        validation_rules (Dict[str, Dict[str, Any]]): Dictionary mapping table names
            to their validation rules. Each rule set contains:
            - required_columns (List[str]): Columns that must exist
            - not_null_columns (List[str]): Columns that cannot have nulls
            - min_rows (int): Minimum expected row count
            - description (str): Table description

    Example:
        Initialize and validate with custom rules::

            validator = DataValidator()

            # Add custom validation rules
            validator.validation_rules['my_table'] = {
                'required_columns': ['id', 'date', 'value'],
                'not_null_columns': ['id', 'date'],
                'min_rows': 1000,
                'description': 'My custom financial table'
            }

            # Run validation
            result = validator.validate_file('my_table.parquet', 'my_table')

    Note:
        The validator comes pre-configured with rules for common financial
        tables: symbol_list, price, financial_statement, and metrics.
    """

    def __init__(self) -> None:
        """Initialize DataValidator with default validation rules.

        Sets up validation rules for standard financial data tables including
        symbol lists, price data, financial statements, and metrics.

        Example:
            Create validator and inspect rules::

                validator = DataValidator()

                # View available table rules
                print("Configured tables:", validator.validation_rules.keys())

                # View specific table rules
                price_rules = validator.validation_rules['price']
                print(f"Price table requires: {price_rules['required_columns']}")
        """
        # Define validation rules for each table type
        self.validation_rules = {
            'symbol_list': {
                'required_columns': ['symbol', 'exchangeShortName', 'type', 'ipoDate'],
                'not_null_columns': ['symbol', 'exchangeShortName'],
                'min_rows': 1000,
                'description': 'Stock symbol list with exchange and IPO information'
            },
            'price': {
                'required_columns': ['date', 'symbol', 'close', 'volume', 'marketCap'],
                'not_null_columns': ['date', 'symbol', 'close'],
                'min_rows': 100000,
                'description': 'Historical price data with market cap'
            },
            'financial_statement': {
                'required_columns': ['date', 'symbol', 'revenue', 'netIncome'],
                'not_null_columns': ['date', 'symbol'],
                'min_rows': 10000,
                'description': 'Financial statements (income, balance, cashflow)'
            },
            'metrics': {
                'required_columns': ['date', 'symbol', 'peRatio', 'marketCap'],
                'not_null_columns': ['date', 'symbol'],
                'min_rows': 10000,
                'description': 'Key financial metrics and ratios'
            }
        }

    def validate_file(self, file_path: str, table_name: str) -> Dict[str, Any]:
        """Validate a single Parquet file against defined rules.

        Performs comprehensive validation checks on a Parquet file including
        schema validation, data type checks, null value detection, duplicate
        checking, and data quality metrics. Returns detailed results with
        errors, warnings, and informational statistics.

        Validation steps:
            1. Load file and get basic info (rows, columns, size)
            2. Check for required columns
            3. Verify minimum row count
            4. Validate null values in critical columns
            5. Check for duplicates (symbol_list only)
            6. Validate date column types
            7. Compute data quality metrics

        Args:
            file_path (str): Absolute path to the Parquet file to validate.
            table_name (str): Name of the table type for rule lookup. Should
                match a key in validation_rules dictionary.

        Returns:
            Dict[str, Any]: Validation results dictionary containing:
                - file (str): File path that was validated
                - table (str): Table name
                - passed (bool): Overall validation status (True if no errors)
                - errors (List[str]): List of validation errors (blocking issues)
                - warnings (List[str]): List of warnings (non-blocking issues)
                - info (Dict[str, Any]): Metadata and statistics:
                    - rows (int): Number of rows
                    - columns (int): Number of columns
                    - size_mb (float): File size in megabytes
                    - memory_mb (float): Memory usage in megabytes
                    - date_range (Tuple[str, str]): Min and max dates (if date column exists)
                    - unique_symbols (int): Count of unique symbols (if symbol column exists)

        Example:
            Validate and analyze results::

                validator = DataValidator()

                # Validate a price file
                result = validator.validate_file(
                    '/data/parquet/price.parquet',
                    'price'
                )

                # Check overall status
                if result['passed']:
                    print("✅ Validation passed")
                    info = result['info']
                    print(f"Data: {info['rows']:,} rows, {info['size_mb']:.2f} MB")
                    print(f"Date range: {info['date_range'][0]} to {info['date_range'][1]}")
                else:
                    print("❌ Validation failed")

                # Review errors
                if result['errors']:
                    print("Errors:")
                    for error in result['errors']:
                        print(f"  - {error}")

                # Review warnings
                if result['warnings']:
                    print("Warnings:")
                    for warning in result['warnings']:
                        print(f"  - {warning}")

        Note:
            - If no validation rules exist for the table_name, basic checks are still performed
            - Files that cannot be read will have passed=False with error message
            - Warnings don't cause validation failure but indicate potential issues
        """
        # Initialize results structure
        results = {
            'file': file_path,
            'table': table_name,
            'passed': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        try:
            # Load Parquet file
            df = pd.read_parquet(file_path)
            rules = self.validation_rules.get(table_name, {})

            # Collect basic file information
            results['info']['rows'] = len(df)
            results['info']['columns'] = len(df.columns)
            results['info']['size_mb'] = Path(file_path).stat().st_size / 1024**2
            results['info']['memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2

            # 1. Validate required columns are present
            if 'required_columns' in rules:
                missing_cols = set(rules['required_columns']) - set(df.columns)
                if missing_cols:
                    results['errors'].append(f"Missing required columns: {missing_cols}")
                    results['passed'] = False

            # 2. Check minimum row count threshold
            if 'min_rows' in rules:
                if len(df) < rules['min_rows']:
                    results['warnings'].append(
                        f"Row count ({len(df):,}) below minimum ({rules['min_rows']:,})"
                    )

            # 3. Validate null values in critical columns
            if 'not_null_columns' in rules:
                for col in rules['not_null_columns']:
                    if col in df.columns:
                        null_count = df[col].isnull().sum()
                        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
                        if null_count > 0:
                            results['errors'].append(
                                f"Column '{col}' has {null_count:,} null values ({null_pct:.1f}%)"
                            )
                            results['passed'] = False

            # 4. Check for duplicate symbols (symbol_list table only)
            if table_name == 'symbol_list' and 'symbol' in df.columns:
                dup_count = df['symbol'].duplicated().sum()
                if dup_count > 0:
                    results['warnings'].append(f"Duplicate symbols: {dup_count}")

            # 5. Validate date column data type
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    results['errors'].append("'date' column is not datetime type")
                    results['passed'] = False
                else:
                    # Extract date range for valid datetime columns
                    results['info']['date_range'] = (
                        df['date'].min().strftime('%Y-%m-%d'),
                        df['date'].max().strftime('%Y-%m-%d')
                    )

            # 6. Analyze data quality - identify columns with high null percentages
            null_summary = df.isnull().sum()
            high_null_cols = null_summary[null_summary > len(df) * 0.5].to_dict()
            if high_null_cols:
                results['warnings'].append(
                    f"Columns with >50% null values: {list(high_null_cols.keys())[:5]}"
                )

            # 7. Calculate unique symbol count if symbol column exists
            if 'symbol' in df.columns:
                results['info']['unique_symbols'] = df['symbol'].nunique()

        except Exception as e:
            # Handle file read errors or other exceptions
            results['passed'] = False
            results['errors'].append(f"Failed to read file: {str(e)}")

        return results

    def validate_all(self, parquet_dir: str) -> Dict[str, Dict[str, Any]]:
        """Validate all Parquet files in a directory.

        Scans a directory for Parquet files and validates each one, logging
        results and generating a summary report. This is useful for batch
        validation of all tables in a storage location.

        Args:
            parquet_dir (str): Path to directory containing Parquet files.
                All *.parquet files in this directory will be validated.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping table names to their
                validation results. Each value is a result dictionary from
                validate_file() containing passed status, errors, warnings,
                and info.

        Example:
            Validate entire directory and summarize::

                validator = DataValidator()

                # Validate all files
                results = validator.validate_all('/data/parquet/')

                # Count passes and failures
                passed = sum(1 for r in results.values() if r['passed'])
                failed = len(results) - passed
                print(f"Results: {passed} passed, {failed} failed")

                # List failed tables
                failed_tables = [
                    name for name, result in results.items()
                    if not result['passed']
                ]
                if failed_tables:
                    print(f"Failed tables: {failed_tables}")

                # Get total data volume
                total_mb = sum(
                    r['info'].get('size_mb', 0)
                    for r in results.values()
                )
                print(f"Total data size: {total_mb:.2f} MB")

        Note:
            - Logs detailed information for each file during validation
            - Displays progress with info and error level logs
            - Returns empty dict if no Parquet files found
            - Prints summary statistics at completion
        """
        parquet_path = Path(parquet_dir)
        all_results = {}

        logging.info("="*80)
        logging.info("Starting Parquet Data Validation")
        logging.info("="*80)

        # Find all Parquet files in directory
        parquet_files = list(parquet_path.glob("*.parquet"))
        if not parquet_files:
            logging.warning(f"No parquet files found in {parquet_dir}")
            return all_results

        # Validate each file
        for file_path in parquet_files:
            table_name = file_path.stem
            logging.info(f"\nValidating: {table_name}.parquet")

            # Run validation
            result = self.validate_file(str(file_path), table_name)
            all_results[table_name] = result

            # Log validation result
            if result['passed']:
                logging.info(f"✅ {table_name}: PASSED")
            else:
                logging.error(f"❌ {table_name}: FAILED")

            # Log basic information
            info = result['info']
            logging.info(f"   📊 Rows: {info.get('rows', 0):,}")
            logging.info(f"   📁 Size: {info.get('size_mb', 0):.2f} MB")
            logging.info(f"   💾 Memory: {info.get('memory_mb', 0):.2f} MB")

            # Log date range if available
            if 'date_range' in info:
                logging.info(f"   📅 Date range: {info['date_range'][0]} ~ {info['date_range'][1]}")

            # Log symbol count if available
            if 'unique_symbols' in info:
                logging.info(f"   🏢 Unique symbols: {info['unique_symbols']:,}")

            # Log all errors
            for error in result['errors']:
                logging.error(f"   🔴 {error}")

            # Log all warnings
            for warning in result['warnings']:
                logging.warning(f"   🟡 {warning}")

        # Print summary statistics
        logging.info("\n" + "="*80)
        passed = sum(1 for r in all_results.values() if r['passed'])
        total = len(all_results)
        logging.info(f"Validation Summary: {passed}/{total} passed")

        if passed == total:
            logging.info("✅ All validations passed!")
        else:
            logging.warning(f"⚠️  {total - passed} validation(s) failed")

        logging.info("="*80)

        return all_results

    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_file: str = "validation_report.txt"
    ) -> str:
        """Generate a detailed text report from validation results.

        Creates a comprehensive validation report in text format with formatted
        sections for each table including status, metadata, errors, and warnings.
        The report is suitable for sharing or archiving validation results.

        Args:
            results (Dict[str, Dict[str, Any]]): Validation results from validate_all()
                or a manually constructed dict with same structure.
            output_file (str, optional): Path to output report file. Can be absolute
                or relative path. Defaults to "validation_report.txt".

        Returns:
            str: Path to the generated report file.

        Example:
            Generate and review reports::

                validator = DataValidator()

                # Validate and generate report
                results = validator.validate_all('/data/parquet/')
                report_path = validator.generate_report(
                    results,
                    output_file='/reports/validation_2024.txt'
                )

                # Read and print report
                with open(report_path, 'r', encoding='utf-8') as f:
                    print(f.read())

                # Generate report for specific tables only
                subset_results = {
                    k: v for k, v in results.items()
                    if k in ['price', 'symbol_list']
                }
                validator.generate_report(
                    subset_results,
                    'subset_validation.txt'
                )

        Note:
            - Report includes overall summary at the top
            - Each table has its own section with full details
            - File is written with UTF-8 encoding
            - Existing file at output_file path will be overwritten
            - Report path is logged after generation
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write report header
            f.write("="*80 + "\n")
            f.write("Parquet Data Validation Report\n")
            f.write("="*80 + "\n\n")

            # Write overall summary
            passed = sum(1 for r in results.values() if r['passed'])
            total = len(results)
            f.write(f"Overall: {passed}/{total} tables passed validation\n\n")

            # Write detailed section for each table
            for table_name, result in results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Table: {table_name}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}\n")
                f.write(f"File: {result['file']}\n\n")

                # Write basic information section
                f.write("Basic Information:\n")
                info = result['info']
                f.write(f"  - Rows: {info.get('rows', 0):,}\n")
                f.write(f"  - Columns: {info.get('columns', 0)}\n")
                f.write(f"  - File Size: {info.get('size_mb', 0):.2f} MB\n")
                f.write(f"  - Memory Usage: {info.get('memory_mb', 0):.2f} MB\n")

                # Write date range if available
                if 'date_range' in info:
                    f.write(f"  - Date Range: {info['date_range'][0]} ~ {info['date_range'][1]}\n")

                # Write symbol count if available
                if 'unique_symbols' in info:
                    f.write(f"  - Unique Symbols: {info['unique_symbols']:,}\n")

                # Write errors section
                if result['errors']:
                    f.write("\n❌ Errors:\n")
                    for error in result['errors']:
                        f.write(f"  - {error}\n")

                # Write warnings section
                if result['warnings']:
                    f.write("\n⚠️  Warnings:\n")
                    for warning in result['warnings']:
                        f.write(f"  - {warning}\n")

                # Write success message if no issues
                if not result['errors'] and not result['warnings']:
                    f.write("\n✅ No issues found\n")

                f.write("\n")

        logging.info(f"📄 Validation report saved to: {output_file}")
        return output_file
