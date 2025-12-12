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

사용 예시:
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
    """Parquet 파일 데이터 품질 검증기입니다.

    이 클래스는 금융 데이터를 포함하는 Parquet 파일에 대한 포괄적인 검증을 제공합니다.
    설정 가능한 규칙을 기반으로 스키마 준수, 데이터 타입, null 값, 데이터 품질 메트릭을
    검사합니다.

    각 테이블 유형은 다음을 정의하는 자체 검증 규칙을 가집니다:
        - 반드시 존재해야 하는 필수 컬럼
        - null 값을 포함할 수 없는 컬럼
        - 최소 행 수 임계값
        - 테이블 목적에 대한 설명

    Attributes:
        validation_rules (Dict[str, Dict[str, Any]]): 테이블 이름을 검증 규칙에
            매핑하는 딕셔너리. 각 규칙 세트는 다음을 포함합니다:
            - required_columns (List[str]): 반드시 존재해야 하는 컬럼
            - not_null_columns (List[str]): null을 가질 수 없는 컬럼
            - min_rows (int): 최소 예상 행 수
            - description (str): 테이블 설명

    사용 예시:
        커스텀 규칙으로 초기화하고 검증::

            validator = DataValidator()

            # 커스텀 검증 규칙 추가
            validator.validation_rules['my_table'] = {
                'required_columns': ['id', 'date', 'value'],
                'not_null_columns': ['id', 'date'],
                'min_rows': 1000,
                'description': 'My custom financial table'
            }

            # 검증 실행
            result = validator.validate_file('my_table.parquet', 'my_table')

    Note:
        검증기는 일반적인 금융 테이블에 대한 규칙이 사전 설정되어 있습니다:
        symbol_list, price, financial_statement, metrics.
    """

    def __init__(self) -> None:
        """기본 검증 규칙으로 DataValidator를 초기화합니다.

        심볼 리스트, 가격 데이터, 재무제표, 메트릭을 포함한 표준 금융 데이터
        테이블에 대한 검증 규칙을 설정합니다.

        사용 예시:
            검증기 생성 및 규칙 확인::

                validator = DataValidator()

                # 사용 가능한 테이블 규칙 보기
                print("Configured tables:", validator.validation_rules.keys())

                # 특정 테이블 규칙 보기
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
        """정의된 규칙에 대해 단일 Parquet 파일을 검증합니다.

        Parquet 파일에 대해 스키마 검증, 데이터 타입 검사, null 값 감지, 중복 검사,
        데이터 품질 메트릭을 포함한 포괄적인 검증을 수행합니다. 오류, 경고,
        정보 통계가 포함된 상세 결과를 반환합니다.

        검증 단계:
            1. 파일을 로드하고 기본 정보 가져오기 (행, 컬럼, 크기)
            2. 필수 컬럼 확인
            3. 최소 행 수 검증
            4. 중요 컬럼의 null 값 검증
            5. 중복 검사 (symbol_list만 해당)
            6. 날짜 컬럼 타입 검증
            7. 데이터 품질 메트릭 계산

        Args:
            file_path (str): 검증할 Parquet 파일의 절대 경로.
            table_name (str): 규칙 조회를 위한 테이블 유형 이름.
                validation_rules 딕셔너리의 키와 일치해야 합니다.

        Returns:
            Dict[str, Any]: 다음을 포함하는 검증 결과 딕셔너리:
                - file (str): 검증된 파일 경로
                - table (str): 테이블 이름
                - passed (bool): 전체 검증 상태 (오류가 없으면 True)
                - errors (List[str]): 검증 오류 리스트 (차단 문제)
                - warnings (List[str]): 경고 리스트 (비차단 문제)
                - info (Dict[str, Any]): 메타데이터 및 통계:
                    - rows (int): 행 수
                    - columns (int): 컬럼 수
                    - size_mb (float): 파일 크기 (MB)
                    - memory_mb (float): 메모리 사용량 (MB)
                    - date_range (Tuple[str, str]): 최소 및 최대 날짜 (date 컬럼이 있는 경우)
                    - unique_symbols (int): 고유 심볼 수 (symbol 컬럼이 있는 경우)

        사용 예시:
            검증 및 결과 분석::

                validator = DataValidator()

                # 가격 파일 검증
                result = validator.validate_file(
                    '/data/parquet/price.parquet',
                    'price'
                )

                # 전체 상태 확인
                if result['passed']:
                    print("✅ Validation passed")
                    info = result['info']
                    print(f"Data: {info['rows']:,} rows, {info['size_mb']:.2f} MB")
                    print(f"Date range: {info['date_range'][0]} to {info['date_range'][1]}")
                else:
                    print("❌ Validation failed")

                # 오류 검토
                if result['errors']:
                    print("Errors:")
                    for error in result['errors']:
                        print(f"  - {error}")

                # 경고 검토
                if result['warnings']:
                    print("Warnings:")
                    for warning in result['warnings']:
                        print(f"  - {warning}")

        Note:
            - table_name에 대한 검증 규칙이 없는 경우에도 기본 검사는 수행됩니다
            - 읽을 수 없는 파일은 passed=False 및 오류 메시지를 가집니다
            - 경고는 검증 실패를 야기하지 않지만 잠재적 문제를 나타냅니다
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
        """디렉토리의 모든 Parquet 파일을 검증합니다.

        디렉토리에서 Parquet 파일을 스캔하고 각각을 검증하여 결과를 로깅하고
        요약 보고서를 생성합니다. 저장소 위치의 모든 테이블에 대한 배치 검증에
        유용합니다.

        Args:
            parquet_dir (str): Parquet 파일이 있는 디렉토리 경로.
                이 디렉토리의 모든 *.parquet 파일이 검증됩니다.

        Returns:
            Dict[str, Dict[str, Any]]: 테이블 이름을 검증 결과에 매핑하는 딕셔너리.
                각 값은 passed 상태, errors, warnings, info를 포함하는
                validate_file()의 결과 딕셔너리입니다.

        사용 예시:
            전체 디렉토리 검증 및 요약::

                validator = DataValidator()

                # 모든 파일 검증
                results = validator.validate_all('/data/parquet/')

                # 성공 및 실패 수 계산
                passed = sum(1 for r in results.values() if r['passed'])
                failed = len(results) - passed
                print(f"Results: {passed} passed, {failed} failed")

                # 실패한 테이블 나열
                failed_tables = [
                    name for name, result in results.items()
                    if not result['passed']
                ]
                if failed_tables:
                    print(f"Failed tables: {failed_tables}")

                # 전체 데이터 볼륨 가져오기
                total_mb = sum(
                    r['info'].get('size_mb', 0)
                    for r in results.values()
                )
                print(f"Total data size: {total_mb:.2f} MB")

        Note:
            - 검증 중 각 파일에 대한 상세 정보를 로깅합니다
            - info 및 error 레벨 로그로 진행 상황을 표시합니다
            - Parquet 파일을 찾지 못한 경우 빈 딕셔너리를 반환합니다
            - 완료 시 요약 통계를 출력합니다
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
        """검증 결과로부터 상세 텍스트 보고서를 생성합니다.

        각 테이블의 상태, 메타데이터, 오류, 경고를 포함하는 형식화된 섹션이 있는
        텍스트 형식의 포괄적인 검증 보고서를 생성합니다. 보고서는 검증 결과를
        공유하거나 보관하는 데 적합합니다.

        Args:
            results (Dict[str, Dict[str, Any]]): validate_all()의 검증 결과 또는
                동일한 구조로 수동으로 구성된 딕셔너리.
            output_file (str, optional): 출력 보고서 파일 경로. 절대 또는
                상대 경로일 수 있습니다. 기본값은 "validation_report.txt".

        Returns:
            str: 생성된 보고서 파일 경로.

        사용 예시:
            보고서 생성 및 검토::

                validator = DataValidator()

                # 검증 및 보고서 생성
                results = validator.validate_all('/data/parquet/')
                report_path = validator.generate_report(
                    results,
                    output_file='/reports/validation_2024.txt'
                )

                # 보고서 읽기 및 출력
                with open(report_path, 'r', encoding='utf-8') as f:
                    print(f.read())

                # 특정 테이블에 대해서만 보고서 생성
                subset_results = {
                    k: v for k, v in results.items()
                    if k in ['price', 'symbol_list']
                }
                validator.generate_report(
                    subset_results,
                    'subset_validation.txt'
                )

        Note:
            - 보고서는 상단에 전체 요약을 포함합니다
            - 각 테이블은 전체 세부 정보가 있는 자체 섹션을 가집니다
            - 파일은 UTF-8 인코딩으로 작성됩니다
            - output_file 경로의 기존 파일은 덮어씌워집니다
            - 보고서 경로는 생성 후 로깅됩니다
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
