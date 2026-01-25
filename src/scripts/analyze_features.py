"""
Feature Importance Analysis Script

기존 학습된 모델에서 특성 중요도를 분석합니다.
재학습 없이 저장된 모델 파일(*.sav)을 로드하여 빠르게 분석합니다.

사용법:
    python src/scripts/analyze_features.py

    또는 특정 모델 지정:
    python src/scripts/analyze_features.py --model clsmodel_0 --top-n 50

출력:
    - reports/feature_importance_{model}_{timestamp}.csv
    - 콘솔에 상위 N개 특성 출력
"""

import sys
import os
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from src.infra.context_loader import load_config, MainContext
from src.infra.logger import get_logger


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Feature Importance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default='clsmodel_0',
        choices=['clsmodel_0', 'clsmodel_1', 'clsmodel_2', 'clsmodel_3', 'model_0', 'model_1'],
        help='분석할 모델 선택 (기본값: clsmodel_0)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=50,
        help='출력할 상위 특성 개수 (기본값: 50)'
    )

    parser.add_argument(
        '--save-plot',
        action='store_true',
        help='특성 중요도 차트 저장'
    )

    return parser.parse_args()


def load_model_and_features(model_path, feature_columns_path):
    """모델과 특성 목록 로드"""
    logger = logging.getLogger('FeatureAnalysis')

    try:
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"❌ Model file not found: {model_path}")
        logger.error("Please run ML training first (python main.py with ML.RUN_REGRESSION=Y)")
        return None, None

    try:
        logger.info(f"Loading feature columns from: {feature_columns_path}")
        feature_columns = joblib.load(feature_columns_path)
        logger.info(f"✅ Loaded {len(feature_columns)} feature columns")
    except FileNotFoundError:
        logger.warning(f"⚠️  Feature columns file not found: {feature_columns_path}")
        logger.warning("Using numeric feature names instead")
        feature_columns = None

    return model, feature_columns


def analyze_feature_importance(model, feature_columns, model_name, top_n=50):
    """특성 중요도 분석"""
    logger = logging.getLogger('FeatureAnalysis')

    # 특성 중요도 추출
    try:
        importances = model.feature_importances_
    except AttributeError:
        logger.error("❌ This model does not have feature_importances_ attribute")
        return None

    # 특성 이름 매핑
    if feature_columns is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    else:
        if len(feature_columns) != len(importances):
            logger.warning(f"⚠️  Feature count mismatch: {len(feature_columns)} names vs {len(importances)} importances")
            feature_names = feature_columns[:len(importances)]
        else:
            feature_names = feature_columns

    # DataFrame 생성
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # 중요도 기준 정렬
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 통계 정보
    logger.info("="*80)
    logger.info(f"Feature Importance Statistics for {model_name}")
    logger.info("="*80)
    logger.info(f"Total features: {len(importance_df)}")
    logger.info(f"Mean importance: {importances.mean():.6f}")
    logger.info(f"Std importance: {importances.std():.6f}")
    logger.info(f"Max importance: {importances.max():.6f}")
    logger.info(f"Min importance: {importances.min():.6f}")

    # 중요도 분포
    zero_importance = (importances == 0).sum()
    logger.info(f"\nFeatures with zero importance: {zero_importance} ({zero_importance/len(importances)*100:.1f}%)")

    top_80_cumsum = importance_df['importance'].cumsum() / importance_df['importance'].sum()
    top_80_count = (top_80_cumsum <= 0.8).sum()
    logger.info(f"Features contributing to 80% importance: {top_80_count} ({top_80_count/len(importances)*100:.1f}%)")

    # 상위 N개 출력
    logger.info(f"\n📊 Top {top_n} Most Important Features:")
    logger.info("="*80)
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"{idx+1:3d}. {row['feature']:60s} {row['importance']:.6f}")

    return importance_df


def save_results(importance_df, model_name, save_plot=False):
    """결과 저장"""
    logger = logging.getLogger('FeatureAnalysis')

    # CSV 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / f"feature_importance_{model_name}_{timestamp}.csv"
    importance_df.to_csv(csv_path, index=False)
    logger.info(f"\n✅ Feature importance saved to: {csv_path}")

    # 플롯 저장 (옵션)
    if save_plot:
        try:
            plt.figure(figsize=(12, 8))
            top_30 = importance_df.head(30)
            plt.barh(range(len(top_30)), top_30['importance'])
            plt.yticks(range(len(top_30)), top_30['feature'], fontsize=8)
            plt.xlabel('Importance')
            plt.title(f'Top 30 Feature Importances - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            plot_path = output_dir / f"feature_importance_{model_name}_{timestamp}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info(f"✅ Plot saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not save plot: {e}")


def main():
    """메인 실행 함수"""
    args = parse_args()

    # 로거 설정
    logger = get_logger('FeatureAnalysis')
    logger.setLevel(logging.INFO)

    logger.info("="*80)
    logger.info("Feature Importance Analysis")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Top N features: {args.top_n}")
    logger.info(f"Save plot: {args.save_plot}")
    logger.info("="*80)

    # 설정 로드
    try:
        config = load_config('config/conf.yaml')
        main_ctx = MainContext(config)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return 1

    # 모델 경로 설정
    model_dir = Path(main_ctx.root_path) / 'MODELS'
    model_path = model_dir / f"{args.model}.sav"
    feature_columns_path = model_dir / "feature_columns.pkl"

    # 모델 및 특성 로드
    model, feature_columns = load_model_and_features(str(model_path), str(feature_columns_path))
    if model is None:
        return 1

    # 특성 중요도 분석
    importance_df = analyze_feature_importance(model, feature_columns, args.model, args.top_n)
    if importance_df is None:
        return 1

    # 결과 저장
    save_results(importance_df, args.model, args.save_plot)

    logger.info("\n" + "="*80)
    logger.info("✅ Analysis completed successfully!")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
