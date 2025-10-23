#!/usr/bin/env python3
"""
Parquet file viewer CLI tool
Usage: python parquet_viewer.py <file> [options]
"""

import sys
import pandas as pd
import argparse
from pathlib import Path


def view_parquet(file_path: str,
                head: int = 10,
                tail: int = 0,
                info: bool = False,
                describe: bool = False,
                columns: str = None,
                sample: int = 0,
                query: str = None):
    """
    Parquet 파일 내용 확인 도구

    Args:
        file_path: Parquet 파일 경로
        head: 처음 N개 행
        tail: 마지막 N개 행
        info: 컬럼 정보 표시
        describe: 통계 정보 표시
        columns: 특정 컬럼만 표시 (쉼표 구분)
        sample: 랜덤 샘플 N개
        query: pandas query 표현식
    """
    try:
        # 파일 존재 확인
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)

        df = pd.read_parquet(file_path)

        print(f"\n{'='*80}")
        print(f"📁 File: {file_path}")
        print(f"{'='*80}\n")

        # 기본 정보
        print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        file_size_mb = file_path_obj.stat().st_size / 1024**2
        print(f"💾 Memory: {memory_mb:.2f} MB")
        print(f"📁 File size: {file_size_mb:.2f} MB")
        print(f"🗜️  Compression ratio: {memory_mb / file_size_mb:.1f}x\n")

        # 컬럼 정보
        if info or not any([head, tail, describe, sample]):
            print("📋 Column Info:")
            print(df.dtypes.to_string())
            print()

        # 통계 정보
        if describe:
            print("📈 Statistics:")
            print(df.describe())
            print()

        # 특정 컬럼만 보기
        if columns:
            col_list = [col.strip() for col in columns.split(',')]
            # 존재하는 컬럼만 필터링
            valid_cols = [col for col in col_list if col in df.columns]
            invalid_cols = [col for col in col_list if col not in df.columns]

            if invalid_cols:
                print(f"⚠️  Invalid columns (ignored): {invalid_cols}\n")

            if valid_cols:
                df = df[valid_cols]
                print(f"🔍 Showing columns: {valid_cols}\n")
            else:
                print("❌ No valid columns specified")
                sys.exit(1)

        # Query 필터링
        if query:
            try:
                original_len = len(df)
                df = df.query(query)
                print(f"🔍 Query: {query}")
                print(f"   Filtered: {original_len:,} → {len(df):,} rows\n")
            except Exception as e:
                print(f"❌ Query error: {e}")
                sys.exit(1)

        # 데이터 미리보기
        if sample > 0:
            print(f"🎲 Random sample ({sample} rows):")
            print(df.sample(min(sample, len(df))))
            print()
        elif tail > 0:
            print(f"👁️  Last {tail} rows:")
            print(df.tail(tail))
            print()
        elif head > 0:
            print(f"👁️  First {head} rows:")
            print(df.head(head))
            print()

        # Null 체크
        null_counts = df.isnull().sum()
        if null_counts.any():
            print("⚠️  Null values detected:")
            null_info = null_counts[null_counts > 0]
            for col, count in null_info.items():
                pct = (count / len(df)) * 100 if len(df) > 0 else 0
                print(f"   - {col}: {count:,} ({pct:.1f}%)")
        else:
            print("✅ No null values")

        # 날짜 범위 (date 컬럼이 있으면)
        if 'date' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['date']):
                print(f"\n📅 Date range: {df['date'].min()} ~ {df['date'].max()}")

        # Symbol 통계 (symbol 컬럼이 있으면)
        if 'symbol' in df.columns:
            print(f"\n🏢 Unique symbols: {df['symbol'].nunique():,}")

        print()

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Parquet file viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (first 10 rows)
  python parquet_viewer.py data.parquet

  # Show detailed info
  python parquet_viewer.py data.parquet -a

  # Show specific columns
  python parquet_viewer.py data.parquet -c "symbol,date,close"

  # Show 100 rows
  python parquet_viewer.py data.parquet -n 100

  # Random sample
  python parquet_viewer.py data.parquet -s 50

  # Query filter
  python parquet_viewer.py data.parquet -q "close > 100"

  # Last 20 rows
  python parquet_viewer.py data.parquet -t 20
        """
    )

    parser.add_argument('file', help='Parquet file path')
    parser.add_argument('-n', '--head', type=int, default=10,
                       help='Number of first rows to show (default: 10)')
    parser.add_argument('-t', '--tail', type=int, default=0,
                       help='Number of last rows to show')
    parser.add_argument('-i', '--info', action='store_true',
                       help='Show column info')
    parser.add_argument('-d', '--describe', action='store_true',
                       help='Show statistics')
    parser.add_argument('-c', '--columns',
                       help='Comma-separated column names to display')
    parser.add_argument('-s', '--sample', type=int, default=0,
                       help='Show random sample of N rows')
    parser.add_argument('-q', '--query',
                       help='Pandas query expression (e.g., "close > 100")')
    parser.add_argument('-a', '--all', action='store_true',
                       help='Show all info (equivalent to -i -d)')

    args = parser.parse_args()

    if args.all:
        args.info = True
        args.describe = True

    view_parquet(
        args.file,
        head=args.head,
        tail=args.tail,
        info=args.info,
        describe=args.describe,
        columns=args.columns,
        sample=args.sample,
        query=args.query
    )


if __name__ == "__main__":
    main()
