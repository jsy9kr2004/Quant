"""
Parquet 파일의 infinite 값 분석 스크립트
원본 parquet 파일에 infinite가 있는지 확인
"""
import pandas as pd
import numpy as np
import os
import glob

def check_infinite_in_parquet(file_path):
    """parquet 파일에서 infinite 값 확인"""
    if not os.path.exists(file_path):
        return None

    df = pd.read_parquet(file_path, engine='pyarrow')

    # infinite 값 체크
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_stats = {}

    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        pos_inf = np.isposinf(df[col]).sum()
        neg_inf = np.isneginf(df[col]).sum()

        # 매우 큰 값도 체크 (XGBoost가 문제 삼을 수 있음)
        very_large = (np.abs(df[col]) > 1e10).sum()

        if inf_count > 0 or very_large > 0:
            inf_stats[col] = {
                'total_inf': int(inf_count),
                'pos_inf': int(pos_inf),
                'neg_inf': int(neg_inf),
                'very_large': int(very_large),
                'max': float(df[col].max()) if not np.isinf(df[col].max()) else 'inf',
                'min': float(df[col].min()) if not np.isinf(df[col].min()) else 'inf',
            }

    return {
        'file': file_path,
        'shape': df.shape,
        'total_rows': len(df),
        'has_infinite': len(inf_stats) > 0,
        'infinite_columns': inf_stats
    }

def main():
    # 학습 파일들 체크
    data_dir = "../data_parquet/ml_per_year/"

    print("=" * 80)
    print("📊 Checking parquet files for infinite values")
    print("=" * 80)

    # 2015~2021년 학습 파일 체크
    train_years = range(2015, 2022)
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']

    files_checked = 0
    files_with_inf = 0

    for year in train_years:
        for Q in quarters:
            file_path = f"{data_dir}rnorm_ml_{year}_{Q}.parquet"
            result = check_infinite_in_parquet(file_path)

            if result is None:
                print(f"❌ {year}_{Q}: File not found")
                continue

            files_checked += 1

            if result['has_infinite']:
                files_with_inf += 1
                print(f"\n🔴 {year}_{Q}: INFINITE VALUES FOUND!")
                print(f"   Shape: {result['shape']}")
                print(f"   Columns with infinite/very large values:")
                for col, stats in result['infinite_columns'].items():
                    print(f"      - {col}:")
                    print(f"         Total inf: {stats['total_inf']}")
                    print(f"         Pos inf: {stats['pos_inf']}")
                    print(f"         Neg inf: {stats['neg_inf']}")
                    print(f"         Very large (>1e10): {stats['very_large']}")
                    print(f"         Range: [{stats['min']}, {stats['max']}]")
            else:
                print(f"✅ {year}_{Q}: No infinite values (shape: {result['shape']})")

    print("\n" + "=" * 80)
    print(f"📈 Summary:")
    print(f"   Files checked: {files_checked}")
    print(f"   Files with infinite/very large values: {files_with_inf}")
    print("=" * 80)

if __name__ == "__main__":
    main()
