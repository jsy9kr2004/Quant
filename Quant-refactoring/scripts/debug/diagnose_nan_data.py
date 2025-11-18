"""
NaN 데이터 진단 스크립트

목적:
1. 어떤 컬럼에 NaN이 많은지 확인
2. NaN 패턴 분석 (시간별, 기업별)
3. long format 변환 후 실제 손실량 예측
4. 샘플 데이터 확인

실행: python3 diagnose_nan_data.py
"""

import sys
import os
sys.path.append('Quant-refactoring')

import pandas as pd
import numpy as np
from config.g_variables import cal_timefeature_col_list
from config.context_loader import load_config, MainContext

def diagnose_nan_patterns():
    """NaN 패턴 진단"""

    print("=" * 80)
    print("🔍 NaN 데이터 진단")
    print("=" * 80)

    # 설정 로드
    config = load_config('Quant-refactoring/config/conf.yaml')
    main_ctx = MainContext(config)

    # VIEW 데이터 로드
    print("\n📂 VIEW 데이터 로딩 중...")

    # 2024년 데이터로 테스트 (문제가 있는 연도)
    test_year = 2024

    try:
        fs_file = f"{main_ctx.root_path}/VIEW/financial_statement_{test_year}.parquet"
        metrics_file = f"{main_ctx.root_path}/VIEW/metrics_{test_year}.parquet"

        if not os.path.exists(fs_file):
            print(f"❌ 파일 없음: {fs_file}")
            return
        if not os.path.exists(metrics_file):
            print(f"❌ 파일 없음: {metrics_file}")
            return

        fs_table = pd.read_parquet(fs_file)
        metrics_table = pd.read_parquet(metrics_file)

        print(f"✅ 데이터 로드 완료")
        print(f"   재무제표: {fs_table.shape}")
        print(f"   메트릭: {metrics_table.shape}")

    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # fs와 metrics 합치기 (실제 코드와 동일)
    print("\n🔄 데이터 병합 중...")

    # 간단한 병합 (실제 코드의 로직 재현)
    fs_metrics = pd.merge(
        fs_table,
        metrics_table,
        on=['symbol', 'date', 'period', 'calendarYear'],
        how='outer'
    )

    print(f"✅ 병합 완료: {fs_metrics.shape}")

    # 29개 컬럼 분석
    print("\n" + "=" * 80)
    print("📊 29개 컬럼별 NaN 분석")
    print("=" * 80)

    total_rows = len(fs_metrics)
    print(f"\n총 rows: {total_rows:,}")

    nan_stats = []

    for col in cal_timefeature_col_list:
        if col in fs_metrics.columns:
            nan_count = fs_metrics[col].isna().sum()
            nan_ratio = nan_count / total_rows * 100
            inf_count = np.isinf(fs_metrics[col]).sum() if fs_metrics[col].dtype in [np.float64, np.float32] else 0

            nan_stats.append({
                'column': col,
                'nan_count': nan_count,
                'nan_ratio': nan_ratio,
                'inf_count': inf_count,
                'exists': True
            })
        else:
            nan_stats.append({
                'column': col,
                'nan_count': 0,
                'nan_ratio': 0,
                'inf_count': 0,
                'exists': False
            })

    # 정렬: NaN 비율 높은 순
    nan_stats_df = pd.DataFrame(nan_stats).sort_values('nan_ratio', ascending=False)

    print("\n컬럼별 NaN 상태 (상위 15개):")
    print("-" * 80)
    print(f"{'컬럼명':<35} {'존재':<6} {'NaN 개수':<12} {'NaN 비율':<12} {'무한대':<8}")
    print("-" * 80)

    for _, row in nan_stats_df.head(15).iterrows():
        exists = "✅" if row['exists'] else "❌"
        col_name = row['column'][:33]
        print(f"{col_name:<35} {exists:<6} {row['nan_count']:<12,.0f} {row['nan_ratio']:<11.2f}% {row['inf_count']:<8,.0f}")

    # 통계 요약
    print("\n" + "=" * 80)
    print("📈 통계 요약")
    print("=" * 80)

    exists_cols = nan_stats_df[nan_stats_df['exists'] == True]

    print(f"\n존재하는 컬럼: {len(exists_cols)}/29")
    print(f"존재하지 않는 컬럼: {len(nan_stats_df) - len(exists_cols)}/29")

    if len(exists_cols) > 0:
        print(f"\nNaN 비율 < 30%: {len(exists_cols[exists_cols['nan_ratio'] < 30])}")
        print(f"NaN 비율 >= 30%: {len(exists_cols[exists_cols['nan_ratio'] >= 30])}")
        print(f"NaN 비율 = 0% (완전히 깨끗): {len(exists_cols[exists_cols['nan_ratio'] == 0])}")

        avg_nan = exists_cols['nan_ratio'].mean()
        print(f"\n평균 NaN 비율: {avg_nan:.2f}%")

    # 샘플 데이터 확인
    print("\n" + "=" * 80)
    print("🔬 샘플 데이터 확인 (상위 5개 컬럼)")
    print("=" * 80)

    for col in nan_stats_df.head(5)['column']:
        if col in fs_metrics.columns:
            print(f"\n[{col}]")
            sample = fs_metrics[['symbol', 'date', col]].head(10)
            print(sample.to_string(index=False))

    # long format 변환 시뮬레이션
    print("\n" + "=" * 80)
    print("🧪 Long Format 변환 시뮬레이션")
    print("=" * 80)

    # 샘플로 첫 100개 row만 사용
    sample_data = fs_metrics.head(100)

    # 존재하는 컬럼만 선택
    valid_cols = [col for col in cal_timefeature_col_list if col in sample_data.columns]

    print(f"\n샘플 데이터: {len(sample_data)} rows")
    print(f"선택된 컬럼: {len(valid_cols)}")

    # long format으로 변환
    long_data = []
    for col in valid_cols:
        for idx, row in sample_data.iterrows():
            long_data.append({
                'symbol': row.get('symbol', 'UNKNOWN'),
                'kind': col,
                'value': row[col]
            })

    long_df = pd.DataFrame(long_data)

    print(f"\nLong format 총 rows: {len(long_df)}")
    print(f"NaN 포함 rows: {long_df['value'].isna().sum()}")
    print(f"NaN 비율: {long_df['value'].isna().sum() / len(long_df) * 100:.2f}%")

    # dropna 후 예상
    clean_df = long_df.dropna(subset=['value'])
    print(f"\ndropna 후 남은 rows: {len(clean_df)}")
    print(f"손실된 rows: {len(long_df) - len(clean_df)}")
    print(f"손실 비율: {(len(long_df) - len(clean_df)) / len(long_df) * 100:.2f}%")

    # 심볼별 데이터 포인트 체크
    symbol_counts = clean_df.groupby(['symbol', 'kind']).size()
    print(f"\n심볼-컬럼별 평균 데이터 포인트: {symbol_counts.mean():.1f}")

    print("\n" + "=" * 80)
    print("✅ 진단 완료")
    print("=" * 80)

    return nan_stats_df


if __name__ == "__main__":
    try:
        result = diagnose_nan_patterns()

        print("\n💡 다음 단계:")
        print("1. NaN이 많은 컬럼의 원인 파악")
        print("2. VIEW 데이터 수집 과정 검토")
        print("3. dropna 적용 시 손실 허용 가능한지 판단")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
