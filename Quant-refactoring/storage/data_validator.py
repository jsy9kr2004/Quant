"""
Data validation module for Parquet files
Validates data integrity, schema, and quality
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


class DataValidator:
    """Parquet 데이터 검증 클래스"""

    def __init__(self):
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

    def validate_file(self, file_path: str, table_name: str) -> Dict:
        """
        단일 Parquet 파일 검증

        Args:
            file_path: Parquet 파일 경로
            table_name: 테이블 이름

        Returns:
            검증 결과 딕셔너리
        """
        results = {
            'file': file_path,
            'table': table_name,
            'passed': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        try:
            df = pd.read_parquet(file_path)
            rules = self.validation_rules.get(table_name, {})

            # 기본 정보
            results['info']['rows'] = len(df)
            results['info']['columns'] = len(df.columns)
            results['info']['size_mb'] = Path(file_path).stat().st_size / 1024**2
            results['info']['memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2

            # 1. 필수 컬럼 체크
            if 'required_columns' in rules:
                missing_cols = set(rules['required_columns']) - set(df.columns)
                if missing_cols:
                    results['errors'].append(f"Missing required columns: {missing_cols}")
                    results['passed'] = False

            # 2. 최소 행 수 체크
            if 'min_rows' in rules:
                if len(df) < rules['min_rows']:
                    results['warnings'].append(
                        f"Row count ({len(df):,}) below minimum ({rules['min_rows']:,})"
                    )

            # 3. Null 값 체크 (필수 컬럼)
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

            # 4. 중복 체크 (symbol_list)
            if table_name == 'symbol_list' and 'symbol' in df.columns:
                dup_count = df['symbol'].duplicated().sum()
                if dup_count > 0:
                    results['warnings'].append(f"Duplicate symbols: {dup_count}")

            # 5. 날짜 타입 체크
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    results['errors'].append("'date' column is not datetime type")
                    results['passed'] = False
                else:
                    results['info']['date_range'] = (
                        df['date'].min().strftime('%Y-%m-%d'),
                        df['date'].max().strftime('%Y-%m-%d')
                    )

            # 6. 데이터 품질 체크
            null_summary = df.isnull().sum()
            high_null_cols = null_summary[null_summary > len(df) * 0.5].to_dict()
            if high_null_cols:
                results['warnings'].append(
                    f"Columns with >50% null values: {list(high_null_cols.keys())[:5]}"
                )

            # 7. 기본 통계
            if 'symbol' in df.columns:
                results['info']['unique_symbols'] = df['symbol'].nunique()

        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Failed to read file: {str(e)}")

        return results

    def validate_all(self, parquet_dir: str) -> Dict:
        """
        모든 Parquet 파일 검증

        Args:
            parquet_dir: Parquet 파일들이 있는 디렉토리

        Returns:
            전체 검증 결과 딕셔너리
        """
        parquet_path = Path(parquet_dir)
        all_results = {}

        logging.info("="*80)
        logging.info("Starting Parquet Data Validation")
        logging.info("="*80)

        parquet_files = list(parquet_path.glob("*.parquet"))
        if not parquet_files:
            logging.warning(f"No parquet files found in {parquet_dir}")
            return all_results

        for file_path in parquet_files:
            table_name = file_path.stem
            logging.info(f"\nValidating: {table_name}.parquet")

            result = self.validate_file(str(file_path), table_name)
            all_results[table_name] = result

            # 결과 출력
            if result['passed']:
                logging.info(f"✅ {table_name}: PASSED")
            else:
                logging.error(f"❌ {table_name}: FAILED")

            # 정보 출력
            info = result['info']
            logging.info(f"   📊 Rows: {info.get('rows', 0):,}")
            logging.info(f"   📁 Size: {info.get('size_mb', 0):.2f} MB")
            logging.info(f"   💾 Memory: {info.get('memory_mb', 0):.2f} MB")

            if 'date_range' in info:
                logging.info(f"   📅 Date range: {info['date_range'][0]} ~ {info['date_range'][1]}")

            if 'unique_symbols' in info:
                logging.info(f"   🏢 Unique symbols: {info['unique_symbols']:,}")

            # 에러 출력
            for error in result['errors']:
                logging.error(f"   🔴 {error}")

            # 경고 출력
            for warning in result['warnings']:
                logging.warning(f"   🟡 {warning}")

        # 전체 요약
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

    def generate_report(self, results: Dict, output_file: str = "validation_report.txt"):
        """
        검증 결과 리포트 생성

        Args:
            results: validate_all()의 결과
            output_file: 출력 파일 경로
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Parquet Data Validation Report\n")
            f.write("="*80 + "\n\n")

            # 전체 요약
            passed = sum(1 for r in results.values() if r['passed'])
            total = len(results)
            f.write(f"Overall: {passed}/{total} tables passed validation\n\n")

            for table_name, result in results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Table: {table_name}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}\n")
                f.write(f"File: {result['file']}\n\n")

                # 기본 정보
                f.write("Basic Information:\n")
                info = result['info']
                f.write(f"  - Rows: {info.get('rows', 0):,}\n")
                f.write(f"  - Columns: {info.get('columns', 0)}\n")
                f.write(f"  - File Size: {info.get('size_mb', 0):.2f} MB\n")
                f.write(f"  - Memory Usage: {info.get('memory_mb', 0):.2f} MB\n")

                if 'date_range' in info:
                    f.write(f"  - Date Range: {info['date_range'][0]} ~ {info['date_range'][1]}\n")

                if 'unique_symbols' in info:
                    f.write(f"  - Unique Symbols: {info['unique_symbols']:,}\n")

                # 에러
                if result['errors']:
                    f.write("\n❌ Errors:\n")
                    for error in result['errors']:
                        f.write(f"  - {error}\n")

                # 경고
                if result['warnings']:
                    f.write("\n⚠️  Warnings:\n")
                    for warning in result['warnings']:
                        f.write(f"  - {warning}\n")

                if not result['errors'] and not result['warnings']:
                    f.write("\n✅ No issues found\n")

                f.write("\n")

        logging.info(f"📄 Validation report saved to: {output_file}")
        return output_file
