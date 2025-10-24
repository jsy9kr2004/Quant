"""
모델 성능 비교 실행 스크립트

여러 모델 버전을 비교하여 개선 여부를 판단합니다.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from optimization.model_comparator import ModelComparator
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/model_comparison.log'),
        logging.StreamHandler()
    ]
)


def load_train_test_data(data_path='./VIEW', train_ratio=0.8):
    """
    학습/테스트 데이터 로드

    Args:
        data_path: 데이터 디렉토리
        train_ratio: 학습 데이터 비율

    Returns:
        X_train, y_train, X_test, y_test
    """
    logging.info("데이터 로드 중...")

    try:
        # ===== 여기에 실제 데이터 로드 로직 구현 =====
        # 예시: CSV에서 로드
        # df = pd.read_csv(Path(data_path) / 'prepared_data.csv')

        # 임시: 샘플 데이터 생성
        logging.warning("실제 데이터가 없어 샘플 데이터를 생성합니다.")

        n_samples = 2000
        n_features = 20

        # 피처 생성
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # 타겟 생성 (이진 분류)
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.4, 0.6]))

        # Train/Test 분할
        split_idx = int(len(X) * train_ratio)

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        logging.info(f"학습 데이터: {len(X_train):,} samples")
        logging.info(f"테스트 데이터: {len(X_test):,} samples")
        logging.info(f"피처 수: {X_train.shape[1]}")

        return X_train, y_train, X_test, y_test

    except Exception as e:
        logging.error(f"데이터 로드 실패: {str(e)}")
        return None, None, None, None


def main():
    """메인 함수"""
    logging.info("="*60)
    logging.info("모델 성능 비교 시작")
    logging.info("="*60)

    # ===== 데이터 로드 =====
    X_train, y_train, X_test, y_test = load_train_test_data()

    if X_train is None:
        logging.error("데이터 로드 실패. 종료합니다.")
        return

    # ===== ModelComparator 초기화 =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparator = ModelComparator(experiment_name=f"model_comparison_{timestamp}")

    # ===== 비교할 모델들 추가 =====

    # 모델 1: XGBoost 기본 버전 (현재 사용 중인 모델)
    logging.info("\n모델 1: XGBoost v1 (기본 파라미터)")
    model_v1 = XGBoostModel(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1
    )
    model_v1.build_model({})
    comparator.add_model(
        model_name="XGBoost_v1_baseline",
        model_instance=model_v1,
        description="현재 사용 중인 기본 모델",
        hyperparameters={
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1
        }
    )

    # 모델 2: XGBoost 개선 버전
    logging.info("모델 2: XGBoost v2 (파라미터 튜닝)")
    model_v2 = XGBoostModel(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05
    )
    model_v2.build_model({})
    comparator.add_model(
        model_name="XGBoost_v2_tuned",
        model_instance=model_v2,
        description="파라미터 튜닝: 더 많은 트리, 더 깊은 depth, 낮은 learning rate",
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05
        }
    )

    # 모델 3: LightGBM
    logging.info("모델 3: LightGBM v1")
    model_v3 = LightGBMModel(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05
    )
    model_v3.build_model({})
    comparator.add_model(
        model_name="LightGBM_v1",
        model_instance=model_v3,
        description="LightGBM 알고리즘 (XGBoost 대체)",
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05
        }
    )

    # 모델 4: CatBoost
    logging.info("모델 4: CatBoost v1")
    model_v4 = CatBoostModel(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05
    )
    model_v4.build_model({})
    comparator.add_model(
        model_name="CatBoost_v1",
        model_instance=model_v4,
        description="CatBoost 알고리즘",
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05
        }
    )

    # ===== 모델 비교 실행 =====
    logging.info("\n" + "="*60)
    logging.info("모델 비교 실행 중...")
    logging.info("="*60)

    comparison_df = comparator.compare_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv_splits=3  # 3-fold cross-validation
    )

    # ===== 개선 여부 확인 =====
    logging.info("\n" + "="*60)
    logging.info("개선 여부 확인")
    logging.info("="*60)

    # Baseline(v1) 대비 각 모델의 개선 여부
    baseline_name = "XGBoost_v1_baseline"

    for model_name in ["XGBoost_v2_tuned", "LightGBM_v1", "CatBoost_v1"]:
        is_improved = comparator.is_improved(
            new_model_name=model_name,
            baseline_model_name=baseline_name,
            metric='accuracy',
            threshold=0.01  # 1% 이상 개선
        )

    # ===== 결과 저장 =====
    output_dir = Path('./results/model_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    comparator.save_results(str(output_dir))

    # 시각화 (선택 사항)
    try:
        comparator.plot_comparison(str(output_dir / f'comparison_chart_{timestamp}.png'))
    except Exception as e:
        logging.warning(f"차트 생성 실패: {str(e)}")

    # ===== 최고 성능 모델 =====
    best_model_info = comparator.get_best_model()

    if best_model_info:
        logging.info("\n" + "="*60)
        logging.info("🏆 최고 성능 모델")
        logging.info("="*60)
        logging.info(f"모델명: {best_model_info['model_name']}")
        logging.info(f"설명: {best_model_info['description']}")
        logging.info(f"Test Accuracy: {best_model_info['metrics'].get('accuracy', 0):.4f}")
        logging.info(f"F1 Score: {best_model_info['metrics'].get('f1', 0):.4f}")

        # 최고 모델 저장
        best_model_file = Path('./config/best_model.txt')
        best_model_file.parent.mkdir(parents=True, exist_ok=True)
        with open(best_model_file, 'w') as f:
            f.write(f"{best_model_info['model_name']}\n")
            f.write(f"accuracy: {best_model_info['metrics'].get('accuracy', 0):.4f}\n")
            f.write(f"f1: {best_model_info['metrics'].get('f1', 0):.4f}\n")

        logging.info(f"\n✅ 최고 모델 정보가 {best_model_file}에 저장되었습니다.")

    logging.info("\n" + "="*60)
    logging.info("완료")
    logging.info("="*60)
    logging.info(f"\n결과 확인:")
    logging.info(f"  - {output_dir}")


if __name__ == '__main__':
    # 로그 디렉토리 생성
    Path('./logs').mkdir(exist_ok=True)

    main()
