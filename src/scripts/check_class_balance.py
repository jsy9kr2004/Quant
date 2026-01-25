"""
Class Balance Analysis Script

ML 학습 데이터의 클래스 분포를 분석합니다.
재학습 없이 저장된 parquet 파일을 읽어서 빠르게 분석합니다.

사용법:
    python src/scripts/check_class_balance.py

    또는 특정 연도 지정:
    python src/scripts/check_class_balance.py --year 2023

출력:
    - 각 분기별 클래스 분포
    - 전체 데이터 통계
    - 불균형 정도 분석
    - reports/class_balance_{timestamp}.csv
"""

import sys
import os
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

from src.infra.context_loader import load_config, MainContext
from src.infra.logger import get_logger


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Class Balance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='분석할 연도 (기본값: 모든 연도)'
    )

    parser.add_argument(
        '--quarter',
        type=str,
        default=None,
        choices=['Q1', 'Q2', 'Q3', 'Q4'],
        help='분석할 분기 (기본값: 모든 분기)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Binary classification threshold (기본값: 0.0, 즉 > 0이면 positive)'
    )

    return parser.parse_args()


def find_ml_data_files(data_dir, year=None, quarter=None):
    """ML 데이터 파일 목록 찾기"""
    logger = logging.getLogger('ClassBalance')

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return []

    # 패턴 매칭
    pattern = "rnorm_ml_"
    if year:
        pattern += f"{year}_"
    pattern += "*.parquet"

    files = sorted(data_path.glob(pattern))

    # 분기 필터링
    if quarter:
        files = [f for f in files if quarter in f.stem]

    logger.info(f"Found {len(files)} ML data files")
    return files


def analyze_file_balance(file_path, threshold=0.0):
    """개별 파일의 클래스 분포 분석"""
    logger = logging.getLogger('ClassBalance')

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"❌ Failed to read {file_path.name}: {e}")
        return None

    # 파일 정보
    file_info = {
        'file': file_path.name,
        'total_rows': len(df)
    }

    # Label 컬럼 찾기 (y, label, target 등 가능한 이름들)
    label_col = None
    for col in ['y', 'label', 'target', 'y_label']:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        logger.warning(f"⚠️  No label column found in {file_path.name}")
        logger.warning(f"   Available columns: {df.columns.tolist()[:10]}")
        return file_info

    # 클래스 분포 분석
    labels = df[label_col]

    # Continuous labels인 경우 threshold로 binary 변환
    binary_labels = (labels > threshold).astype(int)

    positive_count = binary_labels.sum()
    negative_count = len(binary_labels) - positive_count

    file_info.update({
        'label_column': label_col,
        'positive_count': int(positive_count),
        'negative_count': int(negative_count),
        'positive_ratio': positive_count / len(binary_labels) if len(binary_labels) > 0 else 0,
        'negative_ratio': negative_count / len(binary_labels) if len(binary_labels) > 0 else 0,
        'mean_label': float(labels.mean()),
        'std_label': float(labels.std()),
        'min_label': float(labels.min()),
        'max_label': float(labels.max()),
        'median_label': float(labels.median())
    })

    # 불균형 정도 계산 (Imbalance Ratio)
    if negative_count > 0:
        imbalance_ratio = positive_count / negative_count
    else:
        imbalance_ratio = float('inf')

    file_info['imbalance_ratio'] = imbalance_ratio

    return file_info


def print_summary(all_results):
    """전체 요약 출력"""
    logger = logging.getLogger('ClassBalance')

    logger.info("\n" + "="*80)
    logger.info("📊 Class Balance Summary")
    logger.info("="*80)

    total_rows = sum(r['total_rows'] for r in all_results)
    total_positive = sum(r.get('positive_count', 0) for r in all_results)
    total_negative = sum(r.get('negative_count', 0) for r in all_results)

    logger.info(f"\nTotal samples: {total_rows:,}")
    logger.info(f"Total positive: {total_positive:,} ({total_positive/total_rows*100:.2f}%)")
    logger.info(f"Total negative: {total_negative:,} ({total_negative/total_rows*100:.2f}%)")

    overall_imbalance = total_positive / total_negative if total_negative > 0 else float('inf')
    logger.info(f"Overall imbalance ratio: {overall_imbalance:.4f}")

    # 불균형 평가
    logger.info("\n" + "="*80)
    logger.info("⚖️  Imbalance Assessment")
    logger.info("="*80)

    if 0.9 <= overall_imbalance <= 1.1:
        logger.info("✅ Well balanced (ratio ≈ 1.0)")
    elif 0.5 <= overall_imbalance <= 2.0:
        logger.info("⚠️  Slightly imbalanced (0.5 < ratio < 2.0)")
    elif 0.2 <= overall_imbalance <= 5.0:
        logger.info("⚠️  Moderately imbalanced (0.2 < ratio < 5.0)")
    else:
        logger.info("❌ Highly imbalanced (ratio < 0.2 or > 5.0)")
        logger.info("   Consider using:")
        logger.info("   - Class weights in model training")
        logger.info("   - Oversampling (SMOTE)")
        logger.info("   - Undersampling majority class")
        logger.info("   - Different evaluation metrics (F1, AUC-ROC)")

    # 분기별 상세 정보
    logger.info("\n" + "="*80)
    logger.info("📋 Per-File Details")
    logger.info("="*80)

    for result in all_results:
        if 'positive_count' not in result:
            continue

        logger.info(f"\n{result['file']}:")
        logger.info(f"  Total: {result['total_rows']:,} samples")
        logger.info(f"  Positive: {result['positive_count']:,} ({result['positive_ratio']*100:.2f}%)")
        logger.info(f"  Negative: {result['negative_count']:,} ({result['negative_ratio']*100:.2f}%)")
        logger.info(f"  Imbalance ratio: {result['imbalance_ratio']:.4f}")
        logger.info(f"  Label stats: mean={result['mean_label']:.4f}, std={result['std_label']:.4f}")
        logger.info(f"                min={result['min_label']:.4f}, max={result['max_label']:.4f}, median={result['median_label']:.4f}")


def save_results(all_results):
    """결과 CSV 저장"""
    logger = logging.getLogger('ClassBalance')

    df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / f"class_balance_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"\n✅ Class balance report saved to: {csv_path}")


def main():
    """메인 실행 함수"""
    args = parse_args()

    # 로거 설정
    logger = get_logger('ClassBalance')
    logger.setLevel(logging.INFO)

    logger.info("="*80)
    logger.info("Class Balance Analysis")
    logger.info("="*80)
    if args.year:
        logger.info(f"Year: {args.year}")
    if args.quarter:
        logger.info(f"Quarter: {args.quarter}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("="*80)

    # 설정 로드
    try:
        config = load_config('config/conf.yaml')
        main_ctx = MainContext(config)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return 1

    # 데이터 디렉토리 설정
    data_dir = Path(main_ctx.root_path) / 'processed' / 'ml_data' / 'per_year'

    # ML 데이터 파일 찾기
    ml_files = find_ml_data_files(data_dir, args.year, args.quarter)

    if not ml_files:
        logger.error("❌ No ML data files found")
        logger.error(f"   Searched in: {data_dir}")
        logger.error("   Please run ML data generation first (python main.py)")
        return 1

    # 각 파일 분석
    all_results = []
    for file_path in ml_files:
        logger.info(f"\nAnalyzing: {file_path.name}")
        result = analyze_file_balance(file_path, args.threshold)
        if result:
            all_results.append(result)

    if not all_results:
        logger.error("❌ No valid results")
        return 1

    # 요약 출력
    print_summary(all_results)

    # 결과 저장
    save_results(all_results)

    logger.info("\n" + "="*80)
    logger.info("✅ Analysis completed successfully!")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
