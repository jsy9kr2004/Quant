"""
Deep diagnostic script for infinite value generation in regressor.py
Identifies EXACTLY where infinite values are created during load_data()
"""
import pandas as pd
import numpy as np
import os
import sys

def check_infinite(df, stage_name):
    """Check and report infinite values"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols])
    rows_with_inf = inf_mask.any(axis=1)
    cols_with_inf = inf_mask.any(axis=0)

    if rows_with_inf.sum() > 0:
        print(f"\n🔴 [{stage_name}] INFINITE VALUES FOUND!")
        print(f"   Rows with inf: {rows_with_inf.sum()} / {len(df)} ({rows_with_inf.sum()/len(df)*100:.2f}%)")
        print(f"   Columns with inf: {cols_with_inf.sum()}")

        # Find which columns have infinite
        inf_cols = cols_with_inf[cols_with_inf].index.tolist()
        print(f"   Columns: {inf_cols[:10]}")  # First 10

        # Sample rows with infinite
        sample_rows = df[rows_with_inf].head(3)
        print(f"\n   Sample rows with infinite:")
        for idx, row in sample_rows.iterrows():
            inf_in_row = inf_mask.loc[idx][inf_mask.loc[idx]].index.tolist()
            print(f"      Row {idx}: infinite in {inf_in_row[:5]}")

        return True, rows_with_inf
    else:
        print(f"✅ [{stage_name}] No infinite values ({len(df)} rows, {len(numeric_cols)} numeric cols)")
        return False, None

def diagnose_load_data_infinite():
    """
    Simulate load_data() step by step to find where infinite is created
    """
    print("=" * 80)
    print("INFINITE VALUE DIAGNOSTIC - STEP BY STEP")
    print("=" * 80)

    # Configuration
    data_dir = "../data_parquet/ml_per_year/"
    files = [
        f"{data_dir}rnorm_ml_2024_Q2.parquet",
        f"{data_dir}rnorm_ml_2024_Q3.parquet",
        f"{data_dir}rnorm_ml_2024_Q4.parquet",
    ]

    train_df = pd.DataFrame()

    # STEP 1: Load parquet files
    print("\n" + "=" * 80)
    print("STEP 1: Loading parquet files")
    print("=" * 80)

    for fpath in files:
        if not os.path.exists(fpath):
            print(f"⚠️  File not found: {fpath}")
            continue

        print(f"\n📂 Loading: {os.path.basename(fpath)}")
        df = pd.read_parquet(fpath, engine='pyarrow')
        print(f"   Shape: {df.shape}")

        # Check immediately after load
        has_inf, _ = check_infinite(df, "Immediately after parquet load")

        # Dropna on price_diff
        df = df.dropna(axis=0, subset=['price_diff'])
        has_inf, _ = check_infinite(df, "After dropna(price_diff)")

        train_df = pd.concat([train_df, df], axis=0)

    print(f"\n📊 Combined train_df shape: {train_df.shape}")
    check_infinite(train_df, "After concat all files")

    # STEP 2: Column filtering (missing ratio and value_counts)
    print("\n" + "=" * 80)
    print("STEP 2: Column filtering (THIS IS SUSPICIOUS)")
    print("=" * 80)

    missing_threshold = 0.8
    same_value_threshold = 0.95
    columns_to_drop = []

    y_col_list = ['rebalance_date', 'symbol', 'industry', 'price_diff',
                  'price_dev', 'price_dev_subavg', 'sec_price_dev_subavg']

    problematic_cols = []

    for i, col in enumerate(train_df.columns):
        if i % 1000 == 0:
            print(f"   Processing column {i}/{len(train_df.columns)}...")

        try:
            missing_ratio = train_df[col].isna().mean()
            if missing_ratio > missing_threshold:
                columns_to_drop.append(col)
            else:
                # THIS IS WHERE INFINITE MIGHT BE CREATED
                value_counts_result = train_df[col].value_counts(normalize=True, dropna=False)

                if len(value_counts_result) > 0:
                    top_value_ratio = value_counts_result.iloc[0]

                    # Check if top_value_ratio is infinite
                    if np.isinf(top_value_ratio):
                        print(f"\n⚠️  FOUND IT! Column '{col}' produces infinite in value_counts()!")
                        print(f"      Value counts result: {value_counts_result.head()}")
                        print(f"      Column sample: {train_df[col].head(10).tolist()}")
                        problematic_cols.append(col)

                    if top_value_ratio > same_value_threshold:
                        columns_to_drop.append(col)
        except Exception as e:
            print(f"\n❌ Error processing column '{col}': {e}")
            problematic_cols.append(col)

    if problematic_cols:
        print(f"\n🔴 Problematic columns found: {len(problematic_cols)}")
        print(f"   {problematic_cols[:20]}")

    columns_to_drop = [col for col in columns_to_drop if col not in y_col_list]

    print(f"\n   Columns to drop: {len(columns_to_drop)}")
    check_infinite(train_df, "Before dropping columns")

    train_df = train_df.drop(columns=columns_to_drop)
    check_infinite(train_df, "After dropping columns")

    # STEP 3: Row filtering (NaN count)
    print("\n" + "=" * 80)
    print("STEP 3: Row filtering by NaN count")
    print("=" * 80)

    train_df['nan_count_per_row'] = train_df.isnull().sum(axis=1)
    check_infinite(train_df, "After calculating nan_count_per_row")

    filtered_row = train_df['nan_count_per_row'] < int(len(train_df.columns)*0.6)
    train_df = train_df.loc[filtered_row,:]
    check_infinite(train_df, "After filtering rows by NaN count")

    # STEP 4: Sector calculation (VERY SUSPICIOUS)
    print("\n" + "=" * 80)
    print("STEP 4: Sector calculation (MOST SUSPICIOUS)")
    print("=" * 80)

    # Import sector_map (you'll need to adjust this)
    from src.training.sector import sector_map

    train_df["sector"] = train_df["industry"].map(sector_map)
    check_infinite(train_df, "After mapping industry to sector")

    sector_list = list(train_df['sector'].unique())
    sector_list = [x for x in sector_list if str(x) != 'nan']

    print(f"\n   Processing {len(sector_list)} sectors...")

    for sec in sector_list:
        sec_mask = train_df['sector'] == sec
        sec_count = sec_mask.sum()

        # Check price_dev column
        price_dev_in_sector = train_df.loc[sec_mask, 'price_dev']
        if np.isinf(price_dev_in_sector).any():
            print(f"\n⚠️  Sector '{sec}': price_dev already has infinite!")

        sec_mean = train_df.loc[sec_mask, 'price_dev'].mean()

        if np.isinf(sec_mean):
            print(f"\n🔴 Sector '{sec}': sec_mean is INFINITE!")
            print(f"      Sector count: {sec_count}")
            print(f"      price_dev sample: {price_dev_in_sector.head(10).tolist()}")
            print(f"      price_dev stats: min={price_dev_in_sector.min()}, max={price_dev_in_sector.max()}")

        # This subtraction might create infinite
        train_df.loc[sec_mask, 'sec_price_dev_subavg'] = train_df.loc[sec_mask, 'price_dev'] - sec_mean

    check_infinite(train_df, "After sector calculation (CRITICAL CHECK)")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

    # Final report
    numeric_cols_final = train_df.select_dtypes(include=[np.number]).columns
    inf_mask_final = np.isinf(train_df[numeric_cols_final])

    if inf_mask_final.any().any():
        print("\n🔴 INFINITE VALUES PRESENT IN FINAL DATAFRAME")
        print("   This will cause XGBoost error!")

        # Detailed analysis
        cols_with_inf = inf_mask_final.any(axis=0)
        inf_cols = cols_with_inf[cols_with_inf].index.tolist()

        print(f"\n   Columns with infinite ({len(inf_cols)} total):")
        for col in inf_cols[:20]:
            inf_count = inf_mask_final[col].sum()
            print(f"      - {col}: {inf_count} rows")
    else:
        print("\n✅ NO INFINITE VALUES - Something else is wrong!")

if __name__ == "__main__":
    diagnose_load_data_infinite()
