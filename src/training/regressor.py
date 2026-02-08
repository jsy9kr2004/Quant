"""2단계 분류 및 회귀를 사용한 주식 가격 변동 예측입니다.

이 모듈은 2단계 접근 방식을 사용하여 주식 가격 변동을 예측하는 정교한 머신러닝
파이프라인을 구현합니다:
    1. **분류 단계**: 여러 이진 분류기가 주식 가격의 상승 또는 하락 여부를 예측합니다.
    2. **회귀 단계**: 여러 회귀 모델이 가격 변동의 크기를 예측합니다.
    3. **앙상블 투표**: 회귀 예측을 적용하기 전에 여러 분류기의 예측을 결합하여
       주식을 필터링합니다.

2단계 전략은 회귀 모델의 예측을 신뢰하기 전에 여러 분류기의 합의를 요구하여
거짓 양성을 줄입니다.

주요 기능:
    - GPU 가속 XGBoost 및 LightGBM 모델
    - 다양한 하이퍼파라미터를 가진 여러 모델 앙상블
    - 효율성을 위한 parquet 형식의 분기별 데이터 처리
    - 누락된 데이터 및 특성 선택의 강력한 처리
    - 섹터 기반 예측 (선택 사항)
    - 상위 K개 주식 선택을 통한 종합적인 평가

사용 예시:
    from src.infra.config_loader import load_config
    conf = load_config('config/config.yaml')

    # 회귀 모델 초기화
    regressor = Regressor(conf)

    # 데이터 로드 및 준비
    regressor.dataload()

    # 모델 학습
    regressor.train()

    # 테스트 데이터로 평가
    regressor.evaluation()

    # 최신 데이터로 예측
    regressor.latest_prediction()

모델 조합:
    이 모듈은 여러 모델 변형을 학습하고 평가합니다:
    - 분류 모델 (4개 변형):
        * clsmodel_0, 1, 2: max_depth 8, 9, 10을 사용하는 XGBoost
        * clsmodel_3: max_depth 8을 사용하는 LightGBM
    - 회귀 모델 (2개 변형):
        * model_0: max_depth 8을 사용하는 XGBoost
        * model_1: max_depth 10을 사용하는 XGBoost
    - 앙상블 예측 (회귀 모델당):
        * prediction: 원시 회귀 출력
        * prediction_wbinary_0-3: 각 분류기로 필터링
        * prediction_wbinary_ensemble: 분류기 1 AND 3으로 필터링
        * prediction_wbinary_ensemble2: 분류기 1 AND 2로 필터링
        * prediction_wbinary_ensemble3: 다수결 투표 (3개 중 2개 이상)

TODO:
    - 섹터별 예측을 위한 PER_SECTOR=True 기능 구현
    - 이 파일에서 섹터 매핑 제거 (make_mldata.py에 있어야 함)
    - GridSearchCV 코드를 optimizer.py로 마이그레이션하거나 사용하지 않으면 제거
"""

import glob
import joblib
import logging
import torch
import os
import re
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any

from src.infra.g_variables import sector_map
from src.constants.data_schema import DataSchema
from src.training.data_processor import DataProcessor
import xgboost
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# ==============================================================================
# 현재 미사용 import (향후 모델 확장/실험용으로 보존)
# ==============================================================================
# --- 시각화: Feature importance 분석, 학습 곡선 시각화 등 ---
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# --- 대안 모델: 앙상블 다양성 확보, 벤치마크 비교용 ---
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPRegressor
#
# --- PyTorch 기반 모델: 딥러닝 회귀/분류 실험용 ---
# import torch.nn as nn
# import torch.nn.functional as nn_f
# import torch.optim as optim
# from torch.utils.data import DataLoader
#
# --- 하이퍼파라미터 탐색: GridSearchCV/RandomizedSearchCV 실험용 ---
# from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
# from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
#
# --- g_variables 확장 컬럼: 섹터별 feature 그룹 분석용 ---
# from src.infra.g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sparse_col_list
# ==============================================================================

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("⚠️  Optuna not installed. Hyperparameter tuning will be disabled.")

# 전역 설정
MODEL_SAVE_PATH = ""  # 학습된 모델 저장 경로 (메서드에서 설정됨)
THRESHOLD = 92  # 분류를 위한 백분위수 임계값 (92 = 상위 8%가 양성으로 예측됨)

# Preprocessing 설정은 config에서 읽음 (ml_backtest.py와 동일한 방식)
TARGET_FEATURES = 1000  # Feature selection 시 목표 feature 수

# ==============================================================================
# GPU/CPU Device Detection
# ==============================================================================

def detect_xgboost_device() -> str:
    """
    XGBoost에서 사용 가능한 최적의 디바이스를 자동으로 감지합니다.

    감지 순서:
    1. CuPy를 통한 CUDA GPU 사용 가능 여부 확인
    2. PyTorch를 통한 CUDA GPU 사용 가능 여부 확인
    3. GPU 있으면 'cuda:0' 반환
    4. GPU 없으면 'cpu' 반환

    Returns:
        str: 'cuda:0' (GPU 사용 가능) 또는 'cpu' (GPU 없음)

    Example:
        # 디바이스 자동 감지
        device = detect_xgboost_device()
        print(f"Using device: {device}")
        # 출력: Using device: cuda:0
    """
    # 먼저 CuPy를 통해 GPU 확인 (XGBoost는 CuPy를 사용할 수 있음)
    try:
        import cupy as cp
        _ = cp.cuda.Device(0)
        logging.info("🎮 GPU detected (via CuPy): Using CUDA acceleration")
        return 'cuda:0'
    except Exception as e:
        logging.debug(f"CuPy GPU detection failed: {e}")

    # CuPy가 실패하면 PyTorch로 시도
    try:
        if torch.cuda.is_available():
            logging.info("🎮 GPU detected (via PyTorch): Using CUDA acceleration")
            return 'cuda:0'
    except Exception as e:
        logging.debug(f"PyTorch GPU detection failed: {e}")

    logging.info("💻 No GPU detected: Using CPU")
    return 'cpu'


def check_cupy_available() -> bool:
    """
    CuPy가 설치되어 있고 사용 가능한지 확인합니다.

    CuPy는 GPU 예측 성능을 향상시키기 위해 데이터를 GPU로 전송하는 데 사용됩니다.
    CuPy가 없어도 모델은 정상 작동하지만, GPU 예측 시 device mismatch 워닝이 발생합니다.

    Returns:
        bool: CuPy 사용 가능 여부

    Example:
        # CuPy 사용 가능 여부 확인
        if check_cupy_available():
            import cupy as cp
            X_gpu = cp.asarray(X)
    """
    try:
        import cupy as cp
        # 실제 GPU 접근 테스트
        _ = cp.cuda.Device(0)
        return True
    except Exception:
        return False


def predict_with_gpu_support(model, X, use_gpu: bool):
    """
    GPU/CPU를 자동으로 처리하는 예측 함수입니다.

    GPU 사용 시 데이터를 GPU로 전송하여 device mismatch 워닝을 방지합니다.
    CuPy가 없거나 GPU 전송 실패 시 자동으로 CPU로 fallback합니다.

    Args:
        model: XGBoost 또는 LightGBM 모델
        X: 입력 데이터 (pandas DataFrame 또는 numpy array)
        use_gpu: GPU 예측 사용 여부

    Returns:
        numpy.ndarray: 예측 결과

    Example:
        # GPU 지원 예측 실행
        y_pred = predict_with_gpu_support(model, X_test, use_gpu=True)
    """
    if use_gpu:
        try:
            import cupy as cp
            # pandas DataFrame이면 values로 변환
            X_values = X.values if hasattr(X, 'values') else X
            X_gpu = cp.asarray(X_values)
            y_pred = model.predict(X_gpu)
            # 결과를 CPU로 다시 변환 (.get() 사용 - CuPy 권장 방식)
            return y_pred.get() if hasattr(y_pred, 'get') else cp.asnumpy(y_pred)
        except Exception as e:
            # GPU 예측 실패 시 CPU로 fallback
            logging.debug(f"GPU prediction failed, using CPU fallback: {e}")
            return model.predict(X)
    else:
        return model.predict(X)


def predict_proba_with_gpu_support(model, X, use_gpu: bool):
    """
    GPU/CPU를 자동으로 처리하는 확률 예측 함수입니다.

    Args:
        model: XGBoost 또는 LightGBM 분류 모델
        X: 입력 데이터 (pandas DataFrame 또는 numpy array)
        use_gpu: GPU 예측 사용 여부

    Returns:
        numpy.ndarray: 예측 확률 (shape: [n_samples, n_classes])

    Example:
        # GPU 지원 확률 예측
        y_proba = predict_proba_with_gpu_support(model, X_test, use_gpu=True)
        y_proba_class1 = y_proba[:, 1]  # 클래스 1의 확률

    Note:
        sklearn feature names warning may appear if LGBMClassifier was trained
        with DataFrame but prediction uses numpy array (required for GPU).
        This is expected behavior - prediction works correctly despite the warning.
    """
    if use_gpu:
        try:
            import cupy as cp

            # CUDA가 실제로 사용 가능한지 확인
            if not cp.cuda.is_available():
                logging.warning("GPU enabled but CUDA not available, using CPU fallback")
                return model.predict_proba(X)

            X_values = X.values if hasattr(X, 'values') else X
            X_gpu = cp.asarray(X_values)
            y_proba = model.predict_proba(X_gpu)
            # 결과를 CPU로 다시 변환 (.get() 사용 - CuPy 권장 방식)
            return y_proba.get() if hasattr(y_proba, 'get') else cp.asnumpy(y_proba)
        except Exception as e:
            logging.warning(f"GPU prediction failed, using CPU fallback: {e}")
            return model.predict_proba(X)
    else:
        return model.predict_proba(X)

# ==============================================================================
# Column Definitions (Unified with ml_backtest.py via DataSchema)
# ==============================================================================
# ✨ REFACTORED: Now using DataSchema for single source of truth
# This ensures regressor.py and ml_backtest.py use identical column definitions
# Prevents bugs from column name mismatches
y_col_list = DataSchema.get_excluded_cols()

# Backward compatibility: Keep old variable name for existing code
# TODO: Gradually replace all `y_col_list` references with `DataSchema.get_excluded_cols()`

# ==============================================================================
# Parallel Training Support (Ray)
# ==============================================================================
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("⚠️  Ray not installed. Parallel training will be disabled.")


@ray.remote
def train_single_period_remote(
    period_info: dict,
    root_path: str,
    use_classifier: bool,
    use_sector_model: bool,
    top_k: int,
    logger_level: int = logging.INFO
) -> Tuple[str, Optional[dict]]:
    """
    Ray remote function to train models for a single walk-forward period.

    This function is designed to run in parallel across multiple periods.
    Each worker independently:
    1. Loads training data until cutoff_date
    2. Trains models (classifiers + regressors)
    3. Loads prediction data at cutoff_date
    4. Generates predictions
    5. Selects top-K stocks

    Parameters:
    -----------
    period_info : dict
        Period configuration with keys:
        - cutoff_date: datetime
        - train_start_year: int
        - period_name: str
    root_path : str
        Root path for data files
    use_classifier : bool
        Whether to use classifiers
    use_sector_model : bool
        Whether to use sector-specific models
    top_k : int
        Number of top stocks to select
    logger_level : int
        Logging level for this worker

    Returns:
    --------
    Tuple[str, Optional[dict]]
        - cache_key: str (date string 'YYYY-MM-DD')
        - cache_entry: dict or None (predictions and metadata)
    """
    # Configure logging for this worker
    logging.basicConfig(
        level=logger_level,
        format='[Worker] %(levelname)s: %(message)s'
    )

    from src.training.data_processor import DataProcessor
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier

    cutoff_date = period_info['cutoff_date']
    train_start_year = period_info['train_start_year']
    period_name = period_info['period_name']
    cache_key = cutoff_date.strftime('%Y-%m-%d')

    try:
        logging.info(f"📅 Processing period: {cache_key} ({period_name})")

        # Step 1: Load training data
        logging.info(f"📂 Loading training data: {train_start_year} ~ {cache_key}")
        train_df = _load_data_until_cutoff_standalone(root_path, train_start_year, cutoff_date)

        if train_df.empty:
            logging.warning(f"⚠️  No training data for {cache_key}")
            return cache_key, None, "No training data"

        logging.info(f"✅ Loaded {len(train_df)} training samples")

        # Step 2: Train models
        logging.info("🔧 Training models...")
        models_info = _train_models_for_period_standalone(
            train_df, use_classifier, use_sector_model
        )

        # Step 3: Load prediction data
        logging.info(f"📂 Loading prediction data for {cache_key}")
        pred_df = _load_data_for_prediction_standalone(root_path, cutoff_date)

        if pred_df.empty:
            logging.warning(f"⚠️  No prediction data for {cache_key}")
            return cache_key, None, "No prediction data (rnorm_fs files missing or rebalance_date mismatch)"

        logging.info(f"✅ Loaded {len(pred_df)} stocks for prediction")

        # Step 4: Generate predictions
        logging.info("🔮 Generating predictions...")
        predictions_df = _generate_predictions_for_period_standalone(
            pred_df, models_info
        )

        # Step 5: Top-K selection
        logging.info(f"🎯 Selecting Top-{top_k} stocks...")
        predictions_df = _rank_and_select_top_k_standalone(predictions_df, top_k)

        # Build cache entry
        top_k_mask = predictions_df[DataSchema.SELECTED]
        cache_entry = {
            'predictions_df': predictions_df,
            'top_k_selected': predictions_df[top_k_mask]['symbol'].tolist(),
            'top_k_details': predictions_df[top_k_mask].copy(),
            'models_used': models_info,
            'train_samples': len(train_df),
            'predict_samples': len(pred_df)
        }

        logging.info(f"✅ Top-{top_k}: {cache_entry['top_k_selected']}")

        return cache_key, cache_entry, None

    except Exception as e:
        logging.error(f"❌ Error processing period {cache_key}: {e}")
        import traceback
        error_tb = traceback.format_exc()
        logging.error(error_tb)
        return cache_key, None, str(e)


# ==============================================================================
# Standalone Helper Functions for Ray Remote Workers
# ==============================================================================

def _load_data_until_cutoff_standalone(root_path: str, train_start_year: int, cutoff_date: datetime.datetime) -> pd.DataFrame:
    """Standalone version of _load_data_until_cutoff for Ray workers."""
    all_train_data = []
    current_year = train_start_year

    # ✅ FIX: Use correct path where make_mldata.py saves training data
    ml_data_dir = f"{root_path}/processed/ml_data/per_year"

    while current_year <= cutoff_date.year:
        # ✅ FIX: Use correct file pattern - rnorm_ml_YYYY_QN.parquet
        file_pattern = f"{ml_data_dir}/rnorm_ml_{current_year}_Q*.parquet"
        year_files = glob.glob(file_pattern)

        for file_path in sorted(year_files):
            try:
                df = pd.read_parquet(file_path)
                if 'rebalance_date' in df.columns:
                    df['rebalance_date'] = pd.to_datetime(df['rebalance_date'])
                    df = df[df['rebalance_date'] < cutoff_date]
                    if not df.empty:
                        all_train_data.append(df)
            except Exception as e:
                logging.warning(f"⚠️  Failed to load {file_path}: {e}")

        current_year += 1

    if not all_train_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_train_data, ignore_index=True)
    return combined_df


def _load_data_for_prediction_standalone(root_path: str, cutoff_date: datetime.datetime) -> pd.DataFrame:
    """Standalone version of _load_data_for_prediction for Ray workers."""
    year = cutoff_date.year

    ml_data_dir = f"{root_path}/processed/ml_data/per_year"
    # rnorm_fs_ files contain features only (no targets, for prediction)
    file_pattern = f"{ml_data_dir}/rnorm_fs_{year}_Q*.parquet"
    year_files = glob.glob(file_pattern)

    logging.info(f"   🔍 Prediction files pattern: {file_pattern}")
    logging.info(f"   🔍 Found {len(year_files)} files: {[os.path.basename(f) for f in year_files]}")

    pred_data = []
    for file_path in sorted(year_files):
        try:
            df = pd.read_parquet(file_path)
            if 'rebalance_date' in df.columns:
                df['rebalance_date'] = pd.to_datetime(df['rebalance_date'])
                unique_dates = sorted(df['rebalance_date'].unique())
                logging.info(f"   🔍 {os.path.basename(file_path)}: {len(df)} rows, rebalance_dates={[str(d)[:10] for d in unique_dates[:5]]}")
                df = df[df['rebalance_date'] == cutoff_date]
                if not df.empty:
                    pred_data.append(df)
                else:
                    logging.info(f"   🔍 No match for cutoff_date={cutoff_date} in {os.path.basename(file_path)}")
            else:
                logging.warning(f"   ⚠️  No 'rebalance_date' column in {os.path.basename(file_path)}, columns={list(df.columns)[:10]}")
        except Exception as e:
            logging.warning(f"⚠️  Failed to load {file_path}: {e}")

    if not pred_data:
        logging.warning(f"   ⚠️  No prediction data found for cutoff_date={cutoff_date}")
        return pd.DataFrame()

    return pd.concat(pred_data, ignore_index=True)


def _train_models_for_period_standalone(
    train_df: pd.DataFrame,
    use_classifier: bool,
    use_sector_model: bool
) -> dict:
    """Standalone version of _train_models_for_period for Ray workers."""
    from src.training.data_processor import DataProcessor
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier

    models_info = {
        'use_classifier': use_classifier,
        'use_sector': use_sector_model,
        'classifiers': {},
        'regressors': {},
        'sector_classifiers': {},
        'sector_regressors': {}
    }

    # Preprocess
    train_df = DataProcessor.preprocess_training_data(train_df, None)

    target_col = DataSchema.REGRESSION_TARGET
    if target_col not in train_df.columns:
        return models_info

    feature_cols = DataSchema.get_feature_cols(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    if use_sector_model:
        for sector in train_df['sector'].unique():
            sector_data = train_df[train_df['sector'] == sector]
            if len(sector_data) < 50:
                continue

            X_sector = sector_data[feature_cols]
            y_sector = sector_data[target_col]

            # Train sector classifiers
            if use_classifier:
                y_binary = (sector_data[target_col] > 0).astype(int)
                sector_clfs = {
                    0: XGBClassifier(max_depth=8, n_estimators=100, random_state=42),
                    1: XGBClassifier(max_depth=9, n_estimators=100, random_state=42),
                    2: XGBClassifier(max_depth=10, n_estimators=100, random_state=42),
                    3: LGBMClassifier(max_depth=8, n_estimators=100, random_state=42)
                }
                for clf in sector_clfs.values():
                    clf.fit(X_sector, y_binary)
                models_info['sector_classifiers'][sector] = sector_clfs

            # Train sector regressors
            sector_regs = {
                0: XGBRegressor(max_depth=8, n_estimators=100, random_state=42),
                1: XGBRegressor(max_depth=10, n_estimators=100, random_state=42)
            }
            for reg in sector_regs.values():
                reg.fit(X_sector, y_sector)
            models_info['sector_regressors'][sector] = sector_regs
    else:
        # Train global classifiers
        if use_classifier:
            y_binary = (train_df[target_col] > 0).astype(int)
            global_clfs = {
                0: XGBClassifier(max_depth=8, n_estimators=100, random_state=42),
                1: XGBClassifier(max_depth=9, n_estimators=100, random_state=42),
                2: XGBClassifier(max_depth=10, n_estimators=100, random_state=42),
                3: LGBMClassifier(max_depth=8, n_estimators=100, random_state=42)
            }
            for clf in global_clfs.values():
                clf.fit(X_train, y_binary)
            models_info['classifiers'] = global_clfs

        # Train global regressors
        global_regs = {
            0: XGBRegressor(max_depth=8, n_estimators=100, random_state=42),
            1: XGBRegressor(max_depth=10, n_estimators=100, random_state=42)
        }
        for reg in global_regs.values():
            reg.fit(X_train, y_train)
        models_info['regressors'] = global_regs

    return models_info


def _generate_predictions_for_period_standalone(
    pred_df: pd.DataFrame,
    models_info: dict
) -> pd.DataFrame:
    """Standalone version of _generate_predictions_for_period for Ray workers."""
    from src.training.data_processor import DataProcessor

    # Preprocess
    pred_df = DataProcessor.preprocess_training_data(pred_df, None)

    feature_cols = DataSchema.get_feature_cols(pred_df)
    X_pred = pred_df[feature_cols]

    pred_df[DataSchema.PRED_RETURN] = 0.0
    pred_df[DataSchema.PRED_PROBA] = 1.0

    if models_info['use_sector']:
        for sector in pred_df['sector'].unique():
            sector_mask = pred_df['sector'] == sector
            X_sector = pred_df.loc[sector_mask, feature_cols]

            if sector not in models_info['sector_regressors']:
                continue

            # Predict returns
            sector_regs = models_info['sector_regressors'][sector]
            y_pred_return = np.mean([reg.predict(X_sector) for reg in sector_regs.values()], axis=0)
            pred_df.loc[sector_mask, DataSchema.PRED_RETURN] = y_pred_return

            # Predict probabilities
            if models_info['use_classifier'] and sector in models_info['sector_classifiers']:
                sector_clfs = models_info['sector_classifiers'][sector]
                y_pred_proba = np.mean([clf.predict_proba(X_sector)[:, 1] for clf in sector_clfs.values()], axis=0)
                pred_df.loc[sector_mask, DataSchema.PRED_PROBA] = y_pred_proba
    else:
        # Global predictions
        global_regs = models_info['regressors']
        y_pred_return = np.mean([reg.predict(X_pred) for reg in global_regs.values()], axis=0)
        pred_df[DataSchema.PRED_RETURN] = y_pred_return

        if models_info['use_classifier']:
            global_clfs = models_info['classifiers']
            y_pred_proba = np.mean([clf.predict_proba(X_pred)[:, 1] for clf in global_clfs.values()], axis=0)
            pred_df[DataSchema.PRED_PROBA] = y_pred_proba

    # Calculate ML score
    pred_df[DataSchema.ML_SCORE] = pred_df[DataSchema.PRED_PROBA] * pred_df[DataSchema.PRED_RETURN]

    return pred_df


def _rank_and_select_top_k_standalone(predictions_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Standalone version of _rank_and_select_top_k for Ray workers."""
    # Sort by ML score descending
    predictions_df = predictions_df.sort_values(by=DataSchema.ML_SCORE, ascending=False)

    # Add rank
    predictions_df[DataSchema.RANK] = range(1, len(predictions_df) + 1)

    # Mark top-K as selected
    predictions_df[DataSchema.SELECTED] = False
    predictions_df.loc[predictions_df.index[:top_k], DataSchema.SELECTED] = True

    return predictions_df


class Regressor:
    """분류와 회귀를 사용하는 2단계 주식 가격 예측 모델입니다.

    이 클래스는 주식 가격 변동을 예측하는 종합적인 머신러닝 파이프라인을 구현합니다.
    분류 모델이 먼저 상승할 가능성이 높은 주식을 식별한 다음, 회귀 모델이 가격 변동의
    크기를 예측하는 2단계 접근 방식을 사용합니다. 여러 모델을 학습하고 앙상블 투표를
    사용하여 결합합니다.

    학습 파이프라인은 다음을 포함합니다:
        - parquet 파일에서 자동 데이터 로드 (분기별 데이터)
        - 누락 데이터 및 분산을 기반으로 한 특성 선택
        - 여러 XGBoost 및 LightGBM 모델 학습
        - 다양한 앙상블 전략을 사용한 평가
        - 상위 K개 주식 추천 생성

    Attributes:
        conf (Dict): YAML 파일의 설정 딕셔너리
        x_train (pd.DataFrame): 학습 특성
        y_train (pd.DataFrame): 학습 레이블 (price_dev_subavg - 가격 편차에서 평균을 뺀 값)
        y_train_cls (pd.DataFrame): 학습 분류 레이블 (이진: 상승/하락)
        x_test (pd.DataFrame): 테스트 특성
        y_test (pd.DataFrame): 테스트 레이블
        y_test_cls (pd.DataFrame): 테스트 분류 레이블
        train_df (pd.DataFrame): 전체 학습 데이터셋
        test_df (pd.DataFrame): 전체 테스트 데이터셋
        test_df_list (List[Tuple[str, pd.DataFrame]]): 각 테스트 기간에 대한 (파일경로, 데이터프레임) 리스트
        train_files (List[str]): 학습 데이터 파일 경로
        test_files (List[str]): 테스트 데이터 파일 경로
        root_path (str): 데이터 및 모델의 루트 디렉토리
        clsmodels (Dict[int, Any]): 학습된 분류 모델 딕셔너리
        models (Dict[int, Any]): 학습된 회귀 모델 딕셔너리
        drop_col_list (List[str]): 낮은 분산 또는 높은 누락률로 인해 삭제된 특성
        n_sector (int): 섹터 수 (PER_SECTOR 모드용)
        sector_list (List[str]): 섹터 이름 리스트 (PER_SECTOR 모드용)
        sector_train_dfs (Dict[str, pd.DataFrame]): 섹터별 학습 데이터
        sector_test_dfs (Dict[str, pd.DataFrame]): 섹터별 테스트 데이터
        sector_test_df_lists (List): 섹터별 테스트 데이터 리스트
        sector_models (Dict[Tuple[str, int], Any]): 섹터별 회귀 모델
        sector_cls_models (Dict): 섹터별 분류 모델
        sector_x_train (Dict[str, pd.DataFrame]): 섹터별 학습 특성
        sector_y_train (Dict[str, pd.DataFrame]): 섹터별 학습 레이블

    사용 예시:
        from src.infra.config_loader import load_config
        conf = load_config('config/config.yaml')

        # 회귀 모델 생성
        regressor = Regressor(conf)

        # 데이터 로드 (분기별 parquet 파일을 자동으로 로드)
        regressor.dataload()

        # 모든 모델 학습 (4개 분류기 + 2개 회귀 모델)
        regressor.train()

        # 테스트 기간에 대해 평가
        regressor.evaluation()

        # 최신 데이터로 예측
        regressor.latest_prediction()
    """

    def __init__(self, conf: Dict[str, Any]) -> None:
        """설정으로 Regressor를 초기화합니다.

        경로, 파일 리스트, 모델 및 데이터를 위한 빈 컨테이너를 설정합니다.
        설정된 연도 범위를 기반으로 분기별 데이터 파일을 자동으로 검색합니다.

        Args:
            conf: 다음 구조의 설정 딕셔너리:
                {
                    'DATA': {
                        'ROOT_PATH': '/path/to/data'
                    },
                    'ML': {
                        'TRAIN_START_YEAR': 2015,
                        'TRAIN_END_YEAR': 2021,
                        'TEST_START_YEAR': 2022,
                        'TEST_END_YEAR': 2023
                    }
                }

        Raises:
            ValueError: ROOT_PATH/processed/ml_data/per_year/에 학습 데이터 디렉토리가 없는 경우
        """
        self.conf = conf
        self.logger = logging.getLogger(self.__class__.__name__)
        self.x_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.x_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        logging.debug("Config: %s", self.conf)

        # 중첩된 구조에서 설정 값 추출
        data_config = conf.get('DATA', {})
        ml_config = conf.get('ML', {})
        features_config = conf.get('FEATURES', {})
        backtest_config = conf.get('BACKTEST', {})
        self.root_path: str = data_config.get('ROOT_PATH', '/home/user/Quant/data')

        # 섹터별 모델 사용 여부 (ml_backtest.py와 동일한 방식)
        self.use_sector_model = ml_config.get('USE_SECTOR_MODEL', 'N') == 'Y'
        self.sector_config = ml_config.get('SECTOR_CONFIG', {}) if self.use_sector_model else {}

        # Classifier 사용 여부 (ModelFactory와 동일한 방식)
        self.use_classifier = ml_config.get('USE_CLASSIFIER', 'Y') == 'Y'

        # Preprocessing 설정 (ml_backtest.py와 동일한 방식)
        self.use_winsorization = features_config.get('USE_WINSORIZATION', 'Y') == 'Y'
        self.use_feature_selection = features_config.get('USE_FEATURE_SELECTION', 'Y') == 'Y'

        # Top K 설정 (ml_backtest.py와 동일한 방식)
        self.top_k_num = int(backtest_config.get('TOP_K_NUM', 20))

        # ========================================================================
        # Walk-Forward Evaluation Settings (Phase 3: New Feature)
        # ========================================================================
        eval_config = conf.get('EVALUATION', {})
        self.use_walk_forward = eval_config.get('USE_WALK_FORWARD', 'N') == 'Y'
        self.walk_forward_periods = []  # Will be populated in dataload()
        self.predictions_cache = {}  # Store predictions for each cutoff_date

        if self.use_walk_forward:
            logging.info("🔄 Walk-Forward mode ENABLED")
            logging.info(f"   Window type: {eval_config.get('WINDOW_TYPE', 'expanding')}")
            logging.info(f"   Retrain frequency: {eval_config.get('RETRAIN_FREQUENCY', 'quarterly')}")
        else:
            logging.info("📊 Traditional train/test split mode (legacy)")

        aidata_dir = self.root_path + '/processed/ml_data/per_year/'
        logging.info("aidata path: %s", aidata_dir)
        if not os.path.exists(aidata_dir):
            logging.error("there is no ai data: %s", aidata_dir)
            return

        # 학습 파일 리스트 생성 (분기별 parquet 파일)
        self.train_files: List[str] = []
        train_start = int(ml_config.get('TRAIN_START_YEAR', 2015))
        train_end = int(ml_config.get('TRAIN_END_YEAR', 2021))
        for year in range(train_start, train_end + 1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # 5-10배 빠른 읽기를 위한 Parquet 형식
                path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.parquet"
                self.train_files.append(path)

        # 테스트 파일 리스트 생성 (분기별 parquet 파일)
        self.test_files: List[str] = []
        test_start = int(ml_config.get('TEST_START_YEAR', 2022))
        test_end = int(ml_config.get('TEST_END_YEAR', 2023))
        for year in range(test_start, test_end + 1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # 5-10배 빠른 읽기를 위한 Parquet 형식
                path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.parquet"
                self.test_files.append(path)

        logging.info("train file list: %s", self.train_files)
        logging.info("test file list: %s", self.test_files)

        # 데이터 컨테이너 초기화
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.test_df_list: List[Tuple[str, pd.DataFrame]] = []

        # 섹터 기반 예측 속성 (PER_SECTOR 모드용)
        self.n_sector: int = 0
        self.sector_list: List[str] = []
        self.sector_train_dfs: Dict[str, pd.DataFrame] = dict()
        self.sector_test_dfs: Dict[str, pd.DataFrame] = dict()
        self.sector_test_df_lists: List = []

        # 모델 컨테이너
        self.clsmodels: Dict[int, Any] = dict()  # Ensemble classifiers (global)
        self.models: Dict[int, Any] = dict()  # Ensemble regressors (global)
        self.sector_classifiers: Dict[Tuple[str, int], Any] = dict()  # Sector classifiers
        self.sector_models: Dict[Tuple[str, int], Any] = dict()  # Sector regressors (legacy name)

        # 섹터별 학습 데이터
        self.sector_x_train: Dict[str, pd.DataFrame] = dict()
        self.sector_y_train: Dict[str, pd.DataFrame] = dict()
        self.sector_y_train_cls: Dict[str, pd.DataFrame] = dict()  # Classification targets for sector models

        # 특성 선택 추적
        self.drop_col_list: List[str] = []

        # GPU/CPU 디바이스 설정
        self.device: str = detect_xgboost_device()
        self.use_gpu_prediction: bool = check_cupy_available() and self.device.startswith('cuda')

        logging.info("="*60)
        logging.info("XGBoost Device Configuration")
        logging.info("="*60)
        logging.info(f"Device: {self.device}")
        logging.info(f"CuPy available: {check_cupy_available()}")
        logging.info(f"GPU prediction enabled: {self.use_gpu_prediction}")
        logging.info("="*60)

        if self.device.startswith('cuda') and not self.use_gpu_prediction:
            logging.warning("⚠️  GPU training enabled but CuPy not available")
            logging.warning("   Predictions will work but with device mismatch warnings")
            logging.warning("   For better performance, install CuPy:")
            logging.warning("   pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12.x")

    def _export_nan_removal_details(self, x_data: pd.DataFrame, y_data: pd.DataFrame,
                                    y_cls_data: pd.DataFrame, stage: str = "train") -> None:
        """
        NaN 제거 시 실제로 제거되는 row들의 상세 정보를 CSV로 저장합니다.

        Args:
            x_data: 특성 데이터
            y_data: 회귀 타겟 데이터
            y_cls_data: 분류 타겟 데이터
            stage: 'train' 또는 'evaluation'
        """
        # 출력 디렉토리 생성
        removal_dir = os.path.join(self.root_path, "analysis/nan_removal", "training")
        os.makedirs(removal_dir, exist_ok=True)

        # NaN이 있는 행 찾기
        nan_mask_x = x_data.isna().any(axis=1)
        nan_mask_y = y_data.isna().any(axis=1)
        nan_mask_y_cls = y_cls_data.isna().any(axis=1)
        nan_mask_combined = nan_mask_x | nan_mask_y | nan_mask_y_cls

        total_nan_rows = nan_mask_combined.sum()

        if total_nan_rows == 0:
            logging.info(f"✅ [{stage.upper()}] No NaN values found - no removal needed")
            return

        logging.info(f"📝 [{stage.upper()}] Exporting NaN removal details...")

        # ===== 1. 제거될 row들 저장 =====
        nan_rows_x = x_data[nan_mask_combined].copy()
        nan_rows_y = y_data[nan_mask_combined].copy()
        nan_rows_y_cls = y_cls_data[nan_mask_combined].copy()

        # 합쳐서 저장 (x, y, y_cls 모두 포함)
        combined_nan_rows = pd.concat([nan_rows_x, nan_rows_y, nan_rows_y_cls], axis=1)

        # timestamp 추가
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rows_file = os.path.join(removal_dir, f"nan_removal_rows_{stage}_{timestamp}.csv")
        combined_nan_rows.to_csv(rows_file, index=True)
        logging.info(f"   Saved {len(combined_nan_rows)} rows to be removed → {os.path.basename(rows_file)}")

        # 컬럼별 NaN 통계
        nan_per_column = nan_rows_x.isna().sum()
        nan_columns_with_nans = nan_per_column[nan_per_column > 0].sort_values(ascending=False)

        if len(nan_columns_with_nans) > 0:
            logging.info(f"   NaN breakdown by column (top 10):")
            for col, count in list(nan_columns_with_nans.items())[:10]:
                logging.info(f"      {col}: {count} rows")

        # ===== 2. 요약 통계 저장 =====
        summary_data = []
        total_rows_before = len(x_data)
        total_clean_rows = total_rows_before - total_nan_rows
        nan_ratio = (total_nan_rows / total_rows_before * 100) if total_rows_before > 0 else 0

        summary_data.append({
            'metric': 'stage',
            'value': stage,
            'description': 'Training or Evaluation stage'
        })
        summary_data.append({
            'metric': 'total_rows_before_dropna',
            'value': total_rows_before,
            'description': '전체 row 개수 (제거 전)'
        })
        summary_data.append({
            'metric': 'nan_rows_removed',
            'value': total_nan_rows,
            'description': 'NaN으로 제거될 row 개수'
        })
        summary_data.append({
            'metric': 'nan_rows_from_x',
            'value': nan_mask_x.sum(),
            'description': 'x (특성)에서 NaN이 있는 row 개수'
        })
        summary_data.append({
            'metric': 'nan_rows_from_y',
            'value': nan_mask_y.sum(),
            'description': 'y (타겟)에서 NaN이 있는 row 개수'
        })
        summary_data.append({
            'metric': 'nan_rows_from_y_cls',
            'value': nan_mask_y_cls.sum(),
            'description': 'y_cls (분류 타겟)에서 NaN이 있는 row 개수'
        })
        summary_data.append({
            'metric': 'clean_rows_remaining',
            'value': total_clean_rows,
            'description': '제거 후 남은 row 개수'
        })
        summary_data.append({
            'metric': 'nan_removal_ratio_percent',
            'value': f"{nan_ratio:.2f}",
            'description': 'NaN 제거 비율 (%)'
        })
        summary_data.append({
            'metric': 'unique_columns_with_nan',
            'value': len(nan_columns_with_nans),
            'description': 'NaN이 있는 컬럼 개수'
        })

        # 요약을 CSV로 저장
        summary_file = os.path.join(removal_dir, f"nan_removal_summary_{stage}_{timestamp}.csv")
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"   Saved removal summary → {os.path.basename(summary_file)}")

        logging.info(f"✅ [{stage.upper()}] NaN removal details export complete")

    def clean_feature_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """LightGBM과 호환되도록 특성 이름을 정리합니다.

        LightGBM은 특성 이름에 특수 JSON 문자를 지원하지 않으며 고유한 특성 이름이
        필요합니다. 이 메서드는:
            1. 특수 문자를 제거합니다 (영숫자와 밑줄만 유지)
            2. 인덱스를 추가하여 중복 이름을 처리합니다

        Args:
            df: 문제가 있을 수 있는 특성 이름을 가진 DataFrame

        Returns:
            정리된 컬럼 이름을 가진 DataFrame

        사용 예시:
            df = pd.DataFrame({'price@2023': [1, 2], 'price@2024': [3, 4]})
            df = regressor.clean_feature_names(df)
            print(df.columns)
            Index(['price2023', 'price2024'], dtype='object')
        """
        # 컬럼 이름에서 특수 문자 제거
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
        new_n_list = list(new_names.values())

        # 인덱스를 추가하여 중복 이름 처리
        # [LightGBM] Feature appears more than one time.
        new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col
                     for i, (col, new_col) in enumerate(new_names.items())}
        df = df.rename(columns=new_names)
        return df

    @staticmethod
    def _extract_date_from_filepath(filepath: str) -> str:
        """파일 경로에서 날짜 정보를 추출합니다.

        이 메서드는 ML 데이터 파일명에서 연도와 분기 정보를 추출합니다.
        정규표현식을 우선 사용하고, 실패 시 파싱 방식으로 폴백합니다.

        Args:
            filepath: 파일 경로 (예: '/path/to/rnorm_ml_2023_Q1.parquet')

        Returns:
            날짜 문자열 (예: '2023_Q1'), 실패 시 'unknown_period'

        Examples:
            >>> Regressor._extract_date_from_filepath('/data/rnorm_ml_2023_Q1.parquet')
            '2023_Q1'
            >>> Regressor._extract_date_from_filepath('C:\\data\\rnorm_ml_2024_Q2.parquet')
            '2024_Q2'
        """
        filename = os.path.basename(filepath)

        # 정규표현식으로 안전하게 추출 (연도_분기 패턴)
        match = re.search(r'(\d{4})_(Q\d)', filename)
        if match:
            return f"{match.group(1)}_{match.group(2)}"

        # 폴백: 언더스코어로 파싱 (레거시 호환성)
        filename_without_ext = os.path.splitext(filename)[0]
        parts = filename_without_ext.split('_')
        if len(parts) >= 4:
            return f"{parts[2]}_{parts[3]}"

        logging.warning(f"⚠️  Cannot parse date from filename: {filename}")
        return "unknown_period"

    def _load_classifiers(self, model_save_path: str) -> None:
        """분류 모델들을 로드합니다.

        Args:
            model_save_path: 모델 저장 경로
        """
        for i in range(4):
            filename = f"{model_save_path}clsmodel_{i}.sav"
            self.clsmodels[i] = joblib.load(filename)
        logging.info("✅ Loaded 4 classification models")

    def _load_regressors(self, model_save_path: str) -> None:
        """회귀 모델들을 로드합니다.

        Args:
            model_save_path: 모델 저장 경로
        """
        for i in range(2):
            filename = f"{model_save_path}model_{i}.sav"
            self.models[i] = joblib.load(filename)
        logging.info("✅ Loaded 2 regression models")

    def _load_sector_models(self, model_save_path: str, sector_list: List[str]) -> None:
        """섹터별 회귀 모델들을 로드합니다.

        Args:
            model_save_path: 모델 저장 경로
            sector_list: 섹터 이름 리스트
        """
        for sec in sector_list:
            for i in range(2):
                k = (sec, i)
                filename = f"{model_save_path}{sec}_model_{i}.sav"
                self.sector_models[k] = joblib.load(filename)
        logging.info(f"✅ Loaded {len(self.sector_models)} sector regressor models for {len(sector_list)} sectors")

    def _load_sector_classifiers(self, model_save_path: str, sector_list: List[str]) -> None:
        """섹터별 분류 모델들을 로드합니다 (USE_CLASSIFIER=Y인 경우).

        Args:
            model_save_path: 모델 저장 경로
            sector_list: 섹터 이름 리스트
        """
        if not self.use_classifier:
            logging.info(" USE_CLASSIFIER=N: Skipping sector classifier loading")
            return

        for sec in sector_list:
            for i in range(4):  # 4 classifiers per sector (XGB depth 8/9/10 + CatBoost)
                k = (sec, i)
                filename = f"{model_save_path}{sec}_clsmodel_{i}.sav"
                self.sector_classifiers[k] = joblib.load(filename)
        logging.info(f"✅ Loaded {len(self.sector_classifiers)} sector classifier models for {len(sector_list)} sectors")

    @staticmethod
    def _build_prediction_column_names() -> List[str]:
        """모든 예측 컬럼 이름 리스트를 생성합니다.

        Returns:
            예측 컬럼 이름 리스트
        """
        pred_col_list = ['ai_pred_avg']
        for i in range(2):
            pred_col_list.extend([
                f'model_{i}_prediction',
                f'model_{i}_prediction_wbinary_0',
                f'model_{i}_prediction_wbinary_1',
                f'model_{i}_prediction_wbinary_2',
                f'model_{i}_prediction_wbinary_3',
                f'model_{i}_prediction_wbinary_ensemble',
                f'model_{i}_prediction_wbinary_ensemble2',
                f'model_{i}_prediction_wbinary_ensemble3'
            ])
        return pred_col_list

    @staticmethod
    def _get_classifier_column_names() -> dict:
        """분류기 예측 컬럼 이름들을 반환합니다.

        Returns:
            분류기 컬럼 이름 딕셔너리
        """
        return {
            0: 'clsmodel_0_prediction',
            1: 'clsmodel_1_prediction',
            2: 'clsmodel_2_prediction',
            3: 'clsmodel_3_prediction'
        }

    @staticmethod
    def _get_regression_column_names(model_idx: int) -> dict:
        """회귀 모델의 예측 컬럼 이름들을 반환합니다.

        Args:
            model_idx: 모델 인덱스

        Returns:
            회귀 컬럼 이름 딕셔너리
        """
        return {
            'prediction': f'model_{model_idx}_prediction',
            'wbinary_0': f'model_{model_idx}_prediction_wbinary_0',
            'wbinary_1': f'model_{model_idx}_prediction_wbinary_1',
            'wbinary_2': f'model_{model_idx}_prediction_wbinary_2',
            'wbinary_3': f'model_{model_idx}_prediction_wbinary_3',
            'wbinary_ensemble': f'model_{model_idx}_prediction_wbinary_ensemble',
            'wbinary_ensemble2': f'model_{model_idx}_prediction_wbinary_ensemble2',
            'wbinary_ensemble3': f'model_{model_idx}_prediction_wbinary_ensemble3',
            'loss': f'model_{model_idx}_loss',
            'loss_wbinary_0': f'model_{model_idx}_loss_wbinary_0',
            'loss_wbinary_1': f'model_{model_idx}_loss_wbinary_1',
            'loss_wbinary_2': f'model_{model_idx}_loss_wbinary_2',
            'loss_wbinary_3': f'model_{model_idx}_loss_wbinary_3'
        }

    @staticmethod
    def _apply_binary_filtering(
        df: pd.DataFrame,
        y_predict: np.ndarray,
        classifier_cols: dict,
        regression_col_names: dict,
        classifier_mode: str = 'positive_screen'
    ) -> pd.DataFrame:
        """분류기 결과를 사용하여 회귀 예측을 필터링합니다.

        모드에 따라 다른 조건 적용:
        - positive_screen: Class 0 (LOSS) → -1로 대체
        - negative_screen: Class 1 (BAD) → -1로 대체

        Args:
            df: 데이터프레임
            y_predict: 회귀 예측 값
            classifier_cols: 분류기 컬럼 이름 딕셔너리
            regression_col_names: 회귀 컬럼 이름 딕셔너리
            classifier_mode: 분류기 모드 ('positive_screen' or 'negative_screen')

        Returns:
            필터링된 예측이 추가된 데이터프레임
        """
        for i in range(4):
            col_name = regression_col_names[f'wbinary_{i}']
            clf_col = classifier_cols[i]

            if classifier_mode == "negative_screen":
                # negative_screen: Class 1 = BAD → 제외 (위험한 종목)
                df[col_name] = np.where(df[clf_col] == 1, -1, y_predict)
            else:  # positive_screen
                # positive_screen: Class 0 = LOSS → 제외 (비유망 종목)
                df[col_name] = np.where(df[clf_col] == 0, -1, y_predict)
        return df

    @staticmethod
    def _create_ensemble_predictions(
        df: pd.DataFrame,
        y_predict: np.ndarray,
        classifier_cols: dict,
        regression_col_names: dict,
        classifier_mode: str = 'positive_screen'
    ) -> pd.DataFrame:
        """앙상블 전략을 사용하여 예측을 생성합니다.

        3가지 앙상블 전략:
        - ensemble: 분류기 1 AND 3 모두 통과
        - ensemble2: 분류기 1 AND 2 모두 통과
        - ensemble3: 다수결 (3개 중 2개 이상 통과)

        모드에 따른 제외 조건:
        - positive_screen: Class 0 (LOSS) → 제외
        - negative_screen: Class 1 (BAD) → 제외

        Args:
            df: 데이터프레임
            y_predict: 회귀 예측 값
            classifier_cols: 분류기 컬럼 이름 딕셔너리
            regression_col_names: 회귀 컬럼 이름 딕셔너리
            classifier_mode: 분류기 모드 ('positive_screen' or 'negative_screen')

        Returns:
            앙상블 예측이 추가된 데이터프레임
        """
        # 제외할 클래스 결정
        exclude_class = 1 if classifier_mode == "negative_screen" else 0

        # 앙상블 1: 분류기 1 AND 3
        df[regression_col_names['wbinary_ensemble']] = np.where(
            ((df[classifier_cols[1]] == exclude_class) | (df[classifier_cols[3]] == exclude_class)),
            -1, y_predict)

        # 앙상블 2: 분류기 1 AND 2
        df[regression_col_names['wbinary_ensemble2']] = np.where(
            ((df[classifier_cols[1]] == exclude_class) | (df[classifier_cols[2]] == exclude_class)),
            -1, y_predict)

        # 앙상블 3: 다수결 (2개 이상이 제외 클래스 예측)
        condition = (
            (df[[classifier_cols[1], classifier_cols[2], classifier_cols[3]]] == exclude_class).sum(axis=1) >= 2
        )
        df[regression_col_names['wbinary_ensemble3']] = np.where(condition, -1, y_predict)

        return df

    @staticmethod
    def _calculate_prediction_losses(
        df: pd.DataFrame,
        y_predict: np.ndarray,
        regression_col_names: dict
    ) -> pd.DataFrame:
        """예측 오차(손실)를 계산합니다.

        Args:
            df: 데이터프레임 (label 컬럼 포함)
            y_predict: 회귀 예측 값
            regression_col_names: 회귀 컬럼 이름 딕셔너리

        Returns:
            손실이 추가된 데이터프레임
        """
        df[regression_col_names['loss']] = abs(df['label'] - y_predict)
        for i in range(4):
            loss_col = regression_col_names[f'loss_wbinary_{i}']
            pred_col = regression_col_names[f'wbinary_{i}']
            df[loss_col] = abs(df['label'] - df[pred_col])
        return df

    def dataload(self) -> None:
        """parquet 파일에서 학습 및 테스트 데이터를 로드하고 특성을 준비합니다.

        ✨ REFACTORED: Now uses DataProcessor.preprocess_training_data() for unified preprocessing!
        This ensures identical preprocessing logic with ml_backtest.py.

        이 메서드는 포괄적인 데이터 로드 및 전처리를 수행합니다:
            1. 모든 학습 파일을 로드하고 연결합니다 (parquet, with fillingDate filtering)
            2. DataProcessor.preprocess_training_data()를 사용한 통합 전처리:
               - Infinite value removal
               - Log transformation
               - Sparse column removal (>50% NaN)
               - Sparse row removal (>60% NaN)
               - (Optional) Winsorization
               - (Optional) Feature selection
            3. 섹터 기반 가격 편차를 계산합니다 (price_dev에서 섹터 평균을 뺀 값)
            4. 기간별 평가를 위해 테스트 파일을 개별적으로 로드합니다
            5. 학습 및 테스트를 위해 특성(X)과 레이블(y)을 분할합니다

        이 메서드는 경고를 로깅하고 계속 진행하여 누락된 파일을 우아하게 처리합니다.

        Raises:
            ValueError: 학습 데이터 파일을 찾을 수 없는 경우 (치명적 오류)

        부작용:
            - self.train_df, self.test_df, self.test_df_list 설정
            - self.x_train, self.y_train, self.y_train_cls 설정
            - self.x_test, self.y_test, self.y_test_cls 설정
            - 제거된 특성으로 self.drop_col_list 설정
            - 누락된 파일에 대한 경고 로깅
            - 데이터 형상 및 클래스 분포에 대한 정보 로깅

        참고:
            - 누락된 파일은 경고와 함께 건너뜁니다 (치명적이지 않음)
            - 빈 테스트 데이터는 허용됩니다 (평가가 건너뜀)
            - 섹터 매핑은 make_mldata.py에서 수행되어야 합니다 (코드의 TODO 참조)
        """
        logging.info("=" * 80)
        logging.info("📂 DATALOAD: Loading and preprocessing training/test data")
        logging.info("=" * 80)

        # ========================================================================
        # Walk-Forward Mode: Generate periods and skip traditional loading
        # ========================================================================
        if self.use_walk_forward:
            logging.info("🔄 Walk-Forward mode: Generating evaluation periods...")
            self._generate_walk_forward_periods()
            logging.info(f"✅ Generated {len(self.walk_forward_periods)} walk-forward periods")
            return  # Skip traditional data loading

        # ========================================================================
        # Traditional Mode: Load all training/test files
        # ========================================================================
        logging.info("📊 Traditional mode: Loading train/test files...")

        # STEP 1: Load all training files from parquet
        train_dfs = self._load_training_files()

        # Concatenate all training data
        self.train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
        logging.info(f"✅ Combined training data: {len(self.train_df)} rows from {len(train_dfs)} files")

        # STEP 2: Calculate sector-based features (BEFORE preprocessing)
        sector_list = self._calculate_sector_features()

        # STEP 3-4: Unified preprocessing and store training data
        self._preprocess_and_split_training_data()

        # STEP 5: Load and preprocess test files
        self._load_and_preprocess_test_files(sector_list)

        # ========================================================================
        # Final validation and logging
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("✅ DATALOAD COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Training data: {len(self.train_df)} rows, {len(self.x_train.columns)} features")
        logging.info(f"Test data: {len(self.test_df)} rows")
        logging.info(f"Dropped columns: {len(self.drop_col_list)}")

        # Class distribution
        positive_count = (self.y_train_cls.iloc[:, 0] > 0).sum()
        negative_count = (self.y_train_cls.iloc[:, 0] <= 0).sum()
        logging.info(f"Class distribution: positive={positive_count}, negative={negative_count} "
                    f"({positive_count/(positive_count+negative_count)*100:.1f}% positive)")

        if self.use_sector_model:
            logging.info(f"Sector models: {len(self.sector_list)} sectors")

        logging.info("=" * 80)

    def _load_training_files(self) -> list:
        """Load all training parquet files with fillingDate filtering.

        Returns:
            List of DataFrames loaded from parquet files.

        Raises:
            ValueError: If no training data files are found.
        """
        logging.info("STEP 1/5: Loading training parquet files...")
        train_dfs = []

        for fpath in self.train_files:
            logging.info(f"  Loading: {os.path.basename(fpath)}")

            # Skip missing files with warning
            if not os.path.exists(fpath):
                logging.warning(f"  ⚠️  Train file not found, skipping: {fpath}")
                continue

            # Read parquet (5-10x faster than CSV, 70-90% smaller)
            df = pd.read_parquet(fpath, engine='pyarrow')

            # ✅ UNIFIED: Apply fillingDate filtering (same as ml_backtest.py)
            # Only use data that has been publicly filed (prevents future leakage)
            if 'fillingDate' in df.columns:
                df['fillingDate'] = pd.to_datetime(df['fillingDate'], errors='coerce')
                before_filter = len(df)
                # For training, we don't have a specific cutoff, so just ensure fillingDate is valid
                df = df.dropna(subset=['fillingDate'])
                after_filter = len(df)
                if before_filter != after_filter:
                    logging.info(f"    Filtered by fillingDate: {before_filter} → {after_filter} rows")

            # Drop rows with missing target (price_diff)
            df = df.dropna(axis=0, subset=['price_diff'])

            if len(df) > 0:
                train_dfs.append(df)
                logging.info(f"    ✅ Loaded {len(df)} rows")
            else:
                logging.warning(f"    ⚠️  No valid rows after filtering")

        if not train_dfs:
            error_msg = "❌ FATAL ERROR: No training data files found!"
            logging.error(error_msg)
            raise ValueError("No training data files found. Please check your data directory and configuration.")

        return train_dfs

    def _calculate_sector_features(self) -> list:
        """Calculate sector-based price deviation features on training data.

        Computes sec_price_dev_subavg (sector-adjusted price deviation) for each sector.

        Returns:
            List of valid sector names found in the data.
        """
        # TODO: Move this to make_mldata.py (should be done during data generation)
        logging.info("\nSTEP 2/5: Calculating sector-based features...")

        if 'industry' not in self.train_df.columns:
            logging.warning("⚠️  'industry' column not found, skipping sector calculation")
            return []

        self.train_df["sector"] = self.train_df["industry"].map(sector_map)
        all_sectors = list(self.train_df['sector'].unique())
        sector_list = [
            x for x in all_sectors
            if pd.notna(x) and x is not None and isinstance(x, str) and x.strip()
        ]

        # Log invalid sectors and affected rows
        invalid_sectors = [
            x for x in all_sectors
            if not (pd.notna(x) and x is not None and isinstance(x, str) and x.strip())
        ]
        if invalid_sectors:
            logging.warning(f"  ⚠️  Found {len(invalid_sectors)} invalid sector value(s) in training data")
            total_invalid_rows = 0
            for inv_sec in invalid_sectors:
                if pd.isna(inv_sec):
                    count = self.train_df['sector'].isna().sum()
                    logging.warning(f"     - NaN/None: {count} rows excluded from sector calculations")
                    total_invalid_rows += count
                else:
                    count = (self.train_df['sector'] == inv_sec).sum()
                    logging.warning(f"     - '{inv_sec}': {count} rows excluded from sector calculations")
                    total_invalid_rows += count
            logging.warning(f"  Total rows excluded: {total_invalid_rows} / {len(self.train_df)}")

        logging.info(f"  Found {len(sector_list)} valid sectors: {sector_list}")

        # Calculate sector-adjusted price deviation
        for sec in sector_list:
            sec_mask = self.train_df['sector'] == sec
            sec_count = sec_mask.sum()
            sec_mean = self.train_df.loc[sec_mask, 'price_dev'].mean()
            self.train_df.loc[sec_mask, 'sec_price_dev_subavg'] = \
                self.train_df.loc[sec_mask, 'price_dev'] - sec_mean
            logging.debug(f"    {sec}: {sec_count} stocks, mean price_dev={sec_mean:.4f}")

        logging.info(f"✅ Sector features calculated")
        return sector_list

    def _preprocess_and_split_training_data(self) -> None:
        """Apply unified preprocessing, store training data, and split by sector.

        Performs DataProcessor.preprocess_training_data() and splits data into
        sector-specific subsets if USE_SECTOR_MODEL is enabled.

        Side effects:
            Sets self.x_train, self.y_train, self.y_train_cls, self.selected_features,
            self.drop_col_list, self.sector_list, self.sector_train_dfs,
            self.sector_x_train, self.sector_y_train, self.sector_y_train_cls.
        """
        # ========================================================================
        # STEP 3: Unified preprocessing using DataProcessor
        # ========================================================================
        logging.info("\nSTEP 3/5: Applying unified preprocessing (DataProcessor.preprocess_training_data)...")

        # Separate features and targets BEFORE preprocessing
        excluded_cols = DataSchema.get_excluded_cols()
        feature_cols = [col for col in self.train_df.columns if col not in excluded_cols]

        X_train = self.train_df[feature_cols].copy()
        y_train = self.train_df[[DataSchema.REGRESSION_TARGET]].copy()  # DataFrame with column name
        y_train_cls = self.train_df[[DataSchema.CLASSIFICATION_TARGET]].copy()  # DataFrame with column name

        logging.info(f"  Before preprocessing: {len(X_train)} rows, {len(feature_cols)} features")

        # 🎯 UNIFIED PREPROCESSING (Single Source of Truth)
        # This replaces ALL scattered preprocessing with ONE unified method
        X_train, y_train, y_train_cls, selected_features = DataProcessor.preprocess_training_data(
            X_train,
            y_train,
            y_cls=y_train_cls,
            config=self.conf,
            logger=logging.getLogger()
        )

        # Store selected features for later use
        if selected_features is not None:
            self.selected_features = selected_features
            logging.info(f"  Feature selection applied: {len(feature_cols)} → {len(selected_features)} features")
        else:
            self.selected_features = list(X_train.columns)

        # Track dropped columns for test data
        self.drop_col_list = [col for col in feature_cols if col not in X_train.columns]

        logging.info(f"✅ Preprocessing complete: {len(X_train)} rows, {len(X_train.columns)} features")
        logging.info(f"  Dropped {len(self.drop_col_list)} sparse columns")

        # ========================================================================
        # STEP 4: Store preprocessed training data
        # ========================================================================
        logging.info("\nSTEP 4/5: Storing preprocessed training data...")

        # Reconstruct train_df with preprocessed features + metadata
        metadata_cols = ['symbol', 'sector', 'industry'] if 'sector' in self.train_df.columns else ['symbol', 'industry']
        metadata_cols = [col for col in metadata_cols if col in self.train_df.columns]

        # Align metadata with preprocessed indices
        metadata_df = self.train_df.loc[X_train.index, metadata_cols].reset_index(drop=True)
        X_train_reset = X_train.reset_index(drop=True)
        y_train_reset = y_train.reset_index(drop=True)
        y_train_cls_reset = y_train_cls.reset_index(drop=True)

        # Combine into full train_df
        # Include sec_price_dev_subavg for sector models if available
        if self.use_sector_model and 'sec_price_dev_subavg' in self.train_df.columns:
            sec_target = self.train_df.loc[X_train.index, ['sec_price_dev_subavg']].reset_index(drop=True)
            self.train_df = pd.concat([metadata_df, X_train_reset, y_train_reset, y_train_cls_reset, sec_target], axis=1)
        else:
            self.train_df = pd.concat([metadata_df, X_train_reset, y_train_reset, y_train_cls_reset], axis=1)

        # Store X, y separately
        self.x_train = X_train
        self.y_train = y_train
        self.y_train_cls = y_train_cls

        logging.info(f"  Training data stored: {len(self.train_df)} rows")

        # PER_SECTOR mode: Split training data by sector
        if self.use_sector_model and 'sector' in self.train_df.columns:
            logging.info("  🔧 Sector model enabled: Splitting training data by sector...")

            # ✅ 섹터 카테고리화 적용 (config 기반)
            # CATEGORIZATION.ENABLED=Y일 때 원본 섹터를 카테고리로 통합
            self.train_df = DataProcessor.map_sectors_to_categories(
                self.train_df,
                self.conf,
                sector_column='sector',
                logger=logging.getLogger()
            )

            # 카테고리화된 섹터 사용
            all_sectors = list(self.train_df['sector_category'].unique())
            self.sector_list = [
                x for x in all_sectors
                if pd.notna(x) and x is not None and isinstance(x, str) and x.strip()
            ]

            # sector를 sector_category로 교체 (원본은 sector_original로 백업)
            self.train_df['sector_original'] = self.train_df['sector']
            self.train_df['sector'] = self.train_df['sector_category']

            # Log invalid sectors and affected rows
            invalid_sectors = [
                x for x in all_sectors
                if not (pd.notna(x) and x is not None and isinstance(x, str) and x.strip())
            ]
            if invalid_sectors:
                logging.warning(f"  ⚠️  Found {len(invalid_sectors)} invalid sector value(s) for sector model training")
                total_invalid_rows = 0
                for inv_sec in invalid_sectors:
                    if pd.isna(inv_sec):
                        count = self.train_df['sector'].isna().sum()
                        logging.warning(f"     - NaN/None: {count} rows excluded from sector models")
                        total_invalid_rows += count
                    else:
                        count = (self.train_df['sector'] == inv_sec).sum()
                        logging.warning(f"     - '{inv_sec}': {count} rows excluded from sector models")
                        total_invalid_rows += count
                logging.warning(f"  Total rows excluded from sector models: {total_invalid_rows} / {len(self.train_df)}")

            logging.info(f"  Found {len(self.sector_list)} valid sectors for training: {self.sector_list}")

            for sec in self.sector_list:
                self.sector_train_dfs[sec] = self.train_df[self.train_df['sector'] == sec].copy()
                sec_feature_cols = [col for col in self.x_train.columns if col in self.sector_train_dfs[sec].columns]
                self.sector_x_train[sec] = self.sector_train_dfs[sec][sec_feature_cols]
                self.sector_y_train[sec] = self.sector_train_dfs[sec][['sec_price_dev_subavg']]

                # Extract classification target for sector models (if USE_CLASSIFIER=Y)
                if self.use_classifier and DataSchema.CLASSIFICATION_TARGET in self.sector_train_dfs[sec].columns:
                    self.sector_y_train_cls[sec] = self.sector_train_dfs[sec][[DataSchema.CLASSIFICATION_TARGET]]

                logging.info(f"    {sec}: {len(self.sector_train_dfs[sec])} rows")

            logging.info(f"  ✅ Split into {len(self.sector_list)} sectors")

    def _load_and_preprocess_test_files(self, sector_list: list) -> None:
        """Load test files and apply same preprocessing as training data.

        Args:
            sector_list: List of valid sector names (from _calculate_sector_features,
                         used for test data sector feature calculation).

        Side effects:
            Sets self.test_df_list, self.test_df, self.x_test, self.y_test,
            self.y_test_cls, self.sector_test_df_lists.
        """
        logging.info("\nSTEP 5/5: Loading and preprocessing test files...")

        excluded_cols = DataSchema.get_excluded_cols()
        metadata_cols = ['symbol', 'sector', 'industry'] if 'sector' in self.train_df.columns else ['symbol', 'industry']
        metadata_cols = [col for col in metadata_cols if col in self.train_df.columns]

        self.test_df_list = []
        test_dfs = []

        for fpath in self.test_files:
            logging.info(f"  Loading: {os.path.basename(fpath)}")

            # Skip missing files with warning
            if not os.path.exists(fpath):
                logging.warning(f"  ⚠️  Test file not found, skipping: {fpath}")
                continue

            # Read parquet
            df = pd.read_parquet(fpath, engine='pyarrow')

            # Apply fillingDate filtering (same as train)
            if 'fillingDate' in df.columns:
                df['fillingDate'] = pd.to_datetime(df['fillingDate'], errors='coerce')
                df = df.dropna(subset=['fillingDate'])

            # Drop rows with missing target
            df = df.dropna(axis=0, subset=['price_diff'])

            if len(df) == 0:
                logging.warning(f"    ⚠️  No valid rows after filtering")
                continue

            # Calculate sector features (same as train)
            if 'industry' in df.columns:
                df["sector"] = df["industry"].map(sector_map)
                for sec in sector_list:
                    sec_mask = df['sector'] == sec
                    if sec_mask.sum() > 0:
                        sec_mean = df.loc[sec_mask, 'price_dev'].mean()
                        df.loc[sec_mask, 'sec_price_dev_subavg'] = \
                            df.loc[sec_mask, 'price_dev'] - sec_mean

            # Apply same column drops as training
            df = df.drop(columns=self.drop_col_list, errors='ignore')

            # Separate features and targets
            test_feature_cols = [col for col in df.columns if col not in excluded_cols]
            X_test = df[test_feature_cols].copy()
            y_test = df[[DataSchema.REGRESSION_TARGET]].copy()
            y_test_cls = df[[DataSchema.CLASSIFICATION_TARGET]].copy()

            # Apply SAME preprocessing as training (without fitting)
            # Note: We only apply log transform and NaN removal, not refitting scalers
            X_test = DataProcessor.log_transform_features(X_test)

            # Remove rows with NaN in labels
            nan_mask_y = y_test.isna().any(axis=1) | y_test_cls.isna().any(axis=1)
            if nan_mask_y.sum() > 0:
                X_test = X_test[~nan_mask_y]
                y_test = y_test[~nan_mask_y]
                y_test_cls = y_test_cls[~nan_mask_y]

            # Reconstruct test df with preprocessed data
            metadata_df_test = df.loc[X_test.index, metadata_cols].reset_index(drop=True) if 'sector' in df.columns else df.loc[X_test.index, ['symbol', 'industry']].reset_index(drop=True)
            X_test_reset = X_test.reset_index(drop=True)
            y_test_reset = y_test.reset_index(drop=True)
            y_test_cls_reset = y_test_cls.reset_index(drop=True)

            df_processed = pd.concat([metadata_df_test, X_test_reset, y_test_reset, y_test_cls_reset], axis=1)

            test_dfs.append(df_processed)
            self.test_df_list.append([fpath, df_processed])

            logging.info(f"    ✅ Loaded {len(df_processed)} rows")

            # PER_SECTOR mode: Split test data by sector
            if self.use_sector_model and 'sector' in df_processed.columns:
                for sec in self.sector_list:
                    sec_df = df_processed[df_processed['sector'] == sec].copy()
                    if len(sec_df) > 0:
                        self.sector_test_df_lists.append([fpath, sec_df, sec])

        # Combine all test data
        if test_dfs:
            self.test_df = pd.concat(test_dfs, axis=0, ignore_index=True)
            self.x_test = self.test_df[[col for col in self.x_train.columns if col in self.test_df.columns]]
            self.y_test = self.test_df[[DataSchema.REGRESSION_TARGET]]
            self.y_test_cls = self.test_df[[DataSchema.CLASSIFICATION_TARGET]]
            logging.info(f"✅ Combined test data: {len(self.test_df)} rows from {len(test_dfs)} files")
        else:
            logging.warning("⚠️  No test data available! Creating empty test datasets.")
            self.test_df = pd.DataFrame()
            self.x_test = pd.DataFrame(columns=self.x_train.columns)
            self.y_test = pd.DataFrame(columns=[DataSchema.REGRESSION_TARGET])
            self.y_test_cls = pd.DataFrame(columns=[DataSchema.CLASSIFICATION_TARGET])


    def def_model(
        self,
        optuna_params: Optional[Dict[str, Any]] = None,
        sector_optuna_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """분류 및 회귀 모델을 정의하고 초기화합니다.

        ✨ REFACTORED: Now uses ModelFactory for consistent model creation!

        앙상블 예측을 위해 다양한 하이퍼파라미터를 가진 여러 모델 변형을 생성합니다:

        분류 모델 (4개 변형):
            - clsmodels[0]: XGBClassifier, Optuna 최적화 or max_depth=8
            - clsmodels[1]: XGBClassifier, max_depth=9
            - clsmodels[2]: XGBClassifier, max_depth=10
            - clsmodels[3]: CatBoostClassifier, depth=7

        회귀 모델 (2개 변형):
            - models[0]: XGBRegressor, max_depth=8
            - models[1]: CatBoostRegressor, depth=7

        Args:
            optuna_params: Optuna로 찾은 최적 파라미터 (clsmodel_0에 적용)
            sector_optuna_params: 섹터별 Optuna 최적 파라미터 (sector models에 적용)
                                  Format: {'Technology': {...}, 'Financial': {...}, ...}

        부작용:
            - 4개의 분류 모델로 self.clsmodels 채우기
            - 2개의 회귀 모델로 self.models 채우기
            - PER_SECTOR=True인 경우 self.sector_models 채우기
        """
        from src.models.model_factory import create_models_for_regressor

        logging.info("🔧 Creating models using ModelFactory (ensures consistency with ml_backtest.py)")

        # Use ModelFactory to create all models
        classifiers, regressors, sector_classifiers, sector_regressors = create_models_for_regressor(
            config=self.conf,
            optuna_params=optuna_params,
            sector_list=self.sector_list if self.use_sector_model else None,
            use_sector_model=self.use_sector_model,
            sector_optuna_params=sector_optuna_params
        )

        # Assign to instance variables
        for i, clf in enumerate(classifiers):
            self.clsmodels[i] = clf

        for i, reg in enumerate(regressors):
            self.models[i] = reg

        if self.use_sector_model:
            self.sector_classifiers = sector_classifiers
            self.sector_models = sector_regressors

        # Logging
        msg = f"✅ Models created: {len(classifiers)} ensemble classifiers, {len(regressors)} ensemble regressors"
        if self.use_sector_model:
            msg += f", {len(sector_classifiers)} sector classifiers, {len(sector_regressors)} sector regressors"
        logging.info(msg)

    def _diagnose_extreme_values(self, X: np.ndarray, y: np.ndarray, name: str = "data") -> bool:
        """
        Phase 1: Optuna가 거부할 만한 극단값을 미리 진단합니다.

        Args:
            X: Feature 데이터 (numpy array)
            y: Label 데이터 (numpy array)
            name: 로그용 데이터 이름

        Returns:
            bool: QuantileDMatrix 생성 성공 여부
        """
        logging.info("=" * 80)
        logging.info(f"🔬 Phase 1: Diagnosing extreme values in {name}")
        logging.info("=" * 80)

        # 1. 기존 체크 (infinite)
        inf_count = np.isinf(X).sum()
        logging.info(f"Infinite values: {inf_count}")

        # 2. 극단값 체크 (여러 임계값)
        thresholds = [1e10, 1e8, 1e6, 1e4]
        max_abs = np.nanmax(np.abs(X[np.isfinite(X)])) if np.isfinite(X).any() else 0
        logging.info(f"Max absolute value: {max_abs:.2e}")

        for thresh in thresholds:
            count = (np.abs(X) > thresh).sum()
            if count > 0:
                pct = count / X.size * 100
                logging.warning(f"  Values > {thresh:.0e}: {count} ({pct:.3f}%)")

        # 3. QuantileDMatrix 시뮬레이션
        try:
            from xgboost import QuantileDMatrix
            test_dm = QuantileDMatrix(X, label=y)
            logging.info("✅ QuantileDMatrix creation successful")
            del test_dm
            return True
        except Exception as e:
            logging.error(f"❌ QuantileDMatrix creation failed: {e}")
            return False

    def _safe_clip_for_xgboost(self, X: pd.DataFrame, y: np.ndarray, max_abs_value: float = 1e7) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Phase 2: XGBoost를 위한 안전한 clipping (정보 손실 최소화)

        ✅ REFACTORED: Now uses DataProcessor.clip_extreme_values()

        철학:
        - Infinite: 제거 (계산 에러)
        - 극단값: Clipping (실제 데이터, 보존)
        - 임계값을 점진적으로 낮춰가며 최소한만 clip

        Args:
            X: Feature DataFrame
            y: Label array
            max_abs_value: Clipping 임계값 (기본: 1e7)

        Returns:
            (X_clipped, y): Clipping된 데이터
        """
        logging.info("=" * 80)
        logging.info(f"🔧 Phase 2: Safe clipping for XGBoost")
        logging.info("=" * 80)

        X_values = X.values
        original_max = np.nanmax(np.abs(X_values[np.isfinite(X_values)])) if np.isfinite(X_values).any() else 0

        # ✅ REFACTORED: Use DataProcessor for clipping
        X_clipped, _, n_extreme = DataProcessor.clip_extreme_values(
            X,
            y=None,
            threshold=max_abs_value,
            enabled=True
        )

        if n_extreme > 0:
            logging.warning(f"⚠️  Found extreme values (max: {original_max:.2e})")
            logging.warning(f"   These are REAL data, not errors")
            logging.warning(f"   Applying CLIPPING to [{-max_abs_value:.2e}, {max_abs_value:.2e}]")

            pct = n_extreme / X.size * 100
            logging.info(f"   Clipped {n_extreme} values ({pct:.3f}%)")
            logging.info(f"   New max: {np.nanmax(np.abs(X_clipped.values)):.2e}")

            return X_clipped, y
        else:
            logging.info(f"✅ No extreme values found (max: {original_max:.2e})")
            return X, y

    def _load_existing_optuna_params(
        self,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load existing Optuna optimization results from {ROOT_PATH}/models/optuna/.

        Parameters:
        ----------
        model_name : str
            Model identifier (e.g., 'clsmodel_0', 'sector_Technology')

        Returns:
        -------
        Optional[Dict[str, Any]]
            Best parameters if found and valid, None otherwise
        """
        import glob
        import json
        from pathlib import Path

        # Use ROOT_PATH/models/optuna/ for portability
        output_dir = os.path.join(self.root_path, 'models', 'optuna')
        output_path = Path(output_dir)

        if not output_path.exists():
            return None

        # Find latest optuna_best_params_{model_name}_*.json
        pattern = str(output_path / f'optuna_best_params_{model_name}_*.json')
        json_files = glob.glob(pattern)

        if not json_files:
            return None

        # Get the latest file
        latest_file = max(json_files, key=os.path.getmtime)

        try:
            with open(latest_file, 'r') as f:
                json_data = json.load(f)

            best_params = json_data.get('best_params', {})
            if not best_params:
                return None

            logging.info(f"   📂 Found existing Optuna result: {Path(latest_file).name}")
            logging.info(f"      Date: {json_data.get('optimization_date', 'unknown')}")
            logging.info(f"      Score: {json_data.get('best_score', 'unknown')}")

            return best_params

        except Exception as e:
            logging.warning(f"   ⚠️  Failed to load {latest_file}: {e}")
            return None

    def _is_optuna_result_fresh(
        self,
        model_name: str,
        max_age_days: int
    ) -> bool:
        """
        Check if Optuna result is fresh enough to reuse.

        Parameters:
        ----------
        model_name : str
            Model identifier
        max_age_days : int
            Maximum age in days (0 = always fresh)

        Returns:
        -------
        bool
            True if result is fresh, False otherwise
        """
        import glob
        from pathlib import Path
        from datetime import datetime, timedelta

        if max_age_days == 0:
            return True  # Always reuse if max_age is 0

        # Use ROOT_PATH/models/optuna/ for portability
        output_dir = os.path.join(self.root_path, 'models', 'optuna')
        output_path = Path(output_dir)

        if not output_path.exists():
            return False

        pattern = str(output_path / f'optuna_best_params_{model_name}_*.json')
        json_files = glob.glob(pattern)

        if not json_files:
            return False

        latest_file = max(json_files, key=os.path.getmtime)
        file_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
        age = datetime.now() - file_time

        is_fresh = age.days <= max_age_days
        logging.info(f"      Age: {age.days} days (max: {max_age_days} days) → {'✅ Fresh' if is_fresh else '❌ Stale'}")

        return is_fresh

    def _find_optimal_threshold(
        self,
        classifier: Any,
        x_train: pd.DataFrame,
        y_train_binary: pd.Series
    ) -> Dict[str, Any]:
        """
        분류기의 최적 제거 비율(remove percentage) 자동 탐색

        이 메서드는 분류기가 학습된 후, 몇 %를 제거할 때 가장 효과적인지 탐색하여
        최적의 필터링 비율을 찾습니다. 모드에 따라 작동 방식이 다릅니다.

        작동 원리:
            1. 학습된 분류기로 학습 데이터에 대한 예측 확률 계산
            2. 여러 제거 비율 (2~15%)에 대해 반복:
               - negative_screen: BAD 확률 상위 N% 제거
               - positive_screen: GOOD 확률 하위 N% 제거
               - 남은 종목의 precision, recall 계산
            3. 설정된 전략에 따라 최적 비율 선택:
               - "precision": Precision이 가장 높은 비율
               - "balance": Min precision 조건 만족하면서 데이터 최대화

        Args:
            classifier: 학습 완료된 분류 모델
            x_train: 학습 데이터 (features)
            y_train_binary: 실제 binary target
                           - negative_screen: 1=BAD, 0=OK
                           - positive_screen: 1=GOOD, 0=LOSS

        Returns:
            Dict containing:
                - remove_pct (int): 최적 제거 비율 (%)
                - percentile (int): 해당 percentile 값 (호환성)
                - threshold_value (float): 해당 비율의 확률값
                - precision (float): 선택된 종목들의 precision
                - recall (float): 선택된 종목들의 recall
                - n_selected (int): 선택된 종목 수
                - mode (str): classifier mode
                - all_results (dict): 모든 비율의 결과 (디버깅용)

        Example:
            >>> threshold_config = self._find_optimal_threshold(
            ...     classifier=self.clsmodels[0],
            ...     x_train=self.x_train,
            ...     y_train_binary=y_train_binary
            ... )
            >>> print(f"Optimal: Remove {threshold_config['remove_pct']}%")
            Optimal: Remove 10%

        Note:
            - Config에서 탐색 범위 및 기준 읽기:
              * CLASSIFIER_REMOVE_PCT_MIN/MAX
              * CLASSIFIER_THRESHOLD_MIN_PRECISION
              * CLASSIFIER_THRESHOLD_MIN_SAMPLES
              * CLASSIFIER_THRESHOLD_STRATEGY
              * CLASSIFIER_MODE
            - 결과는 threshold_config.pkl로 저장되어 평가/백테스트 시 재사용
        """
        # Config에서 설정 읽기
        ml_config = self.conf.get('ML', {})
        remove_pct_min = int(ml_config.get('CLASSIFIER_REMOVE_PCT_MIN', 2))
        remove_pct_max = int(ml_config.get('CLASSIFIER_REMOVE_PCT_MAX', 15))
        min_precision = float(ml_config.get('CLASSIFIER_THRESHOLD_MIN_PRECISION', 0.65))
        min_samples = int(ml_config.get('CLASSIFIER_THRESHOLD_MIN_SAMPLES', 30))
        strategy = ml_config.get('CLASSIFIER_THRESHOLD_STRATEGY', 'balance')
        classifier_mode = ml_config.get('CLASSIFIER_MODE', 'positive_screen')

        logging.info("="*80)
        logging.info("🔍 OPTIMAL FILTERING RATIO SEARCH")
        logging.info("="*80)
        logging.info(f"   Mode: {classifier_mode}")
        logging.info(f"   Remove range: {remove_pct_min}-{remove_pct_max}%")
        logging.info(f"   Min precision: {min_precision:.2f}")
        logging.info(f"   Min samples: {min_samples}")
        logging.info(f"   Strategy: {strategy}")
        logging.info("")

        # 예측 확률 계산
        y_probs = predict_proba_with_gpu_support(classifier, x_train, self.use_gpu_prediction)[:, 1]

        results = {}
        for remove_pct in range(remove_pct_min, remove_pct_max + 1):
            # 모드별 percentile 및 mask 계산
            if classifier_mode == "negative_screen":
                # Class 1 = BAD → 상위 N% 제거 (BAD 확률이 높은 것)
                percentile = 100 - remove_pct
                threshold = np.percentile(y_probs, percentile)
                mask = y_probs < threshold  # BAD 확률이 낮은 것만 선택
            else:  # positive_screen
                # Class 1 = GOOD → 하위 N% 제거 (GOOD 확률이 낮은 것)
                percentile = remove_pct
                threshold = np.percentile(y_probs, percentile)
                mask = y_probs > threshold  # GOOD 확률이 높은 것만 선택

            n_selected = mask.sum()
            if n_selected < min_samples:
                continue  # 너무 적으면 skip

            # 선택된 종목의 precision, recall 계산
            y_pred_binary = (y_probs[mask] > 0.5).astype(int)
            y_true_filtered = y_train_binary.values.ravel()[mask]

            # precision_score, recall_score는 zero_division 처리 필요
            try:
                precision = precision_score(y_true_filtered, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_filtered, y_pred_binary, zero_division=0)
            except Exception as e:
                logging.warning(f"  Remove {remove_pct}%: Error calculating metrics - {e}")
                continue

            results[remove_pct] = {
                'precision': precision,
                'recall': recall,
                'n_selected': n_selected,
                'threshold_value': threshold,
                'percentile': percentile
            }

            logging.info(
                f"  Remove {remove_pct:2d}%: threshold={threshold:.3f}, "
                f"selected={n_selected:5d}, precision={precision:.3f}, recall={recall:.3f}"
            )

        if not results:
            logging.warning("⚠️  No valid remove percentage found! Using default 8%")
            # Fallback to 8%
            remove_pct = 8
            if classifier_mode == "negative_screen":
                percentile = 100 - remove_pct  # 92
                threshold = np.percentile(y_probs, percentile)
                mask = y_probs < threshold
            else:
                percentile = remove_pct  # 8
                threshold = np.percentile(y_probs, percentile)
                mask = y_probs > threshold

            n_selected = mask.sum()
            y_pred_binary = (y_probs[mask] > 0.5).astype(int)
            y_true_filtered = y_train_binary.values.ravel()[mask]
            precision = precision_score(y_true_filtered, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_filtered, y_pred_binary, zero_division=0)

            return {
                'remove_pct': remove_pct,
                'percentile': percentile,
                'threshold_value': threshold,
                'precision': precision,
                'recall': recall,
                'n_selected': n_selected,
                'mode': classifier_mode,
                'all_results': {}
            }

        # 전략에 따라 최적 비율 선택
        if strategy == 'precision':
            # 전략 1: Precision 최대화
            optimal_remove_pct = max(results, key=lambda x: results[x]['precision'])
        else:  # strategy == 'balance'
            # 전략 2: Min precision 조건 만족하면서 최대 데이터
            optimal_remove_pct = None
            for remove_pct in sorted(results.keys()):
                if results[remove_pct]['precision'] >= min_precision:
                    if optimal_remove_pct is None or results[remove_pct]['n_selected'] > results[optimal_remove_pct]['n_selected']:
                        optimal_remove_pct = remove_pct

            # balance 전략에서 조건 만족 못하면 precision 최대값 선택
            if optimal_remove_pct is None:
                logging.warning(f"⚠️  No percentage meets min_precision={min_precision}, using max precision")
                optimal_remove_pct = max(results, key=lambda x: results[x]['precision'])

        logging.info("")
        logging.info("="*80)
        logging.info(f"🎯 OPTIMAL FILTERING: Remove {optimal_remove_pct}%")
        logging.info("="*80)
        logging.info(f"   Mode: {classifier_mode}")
        logging.info(f"   Threshold value: {results[optimal_remove_pct]['threshold_value']:.3f}")
        logging.info(f"   Precision: {results[optimal_remove_pct]['precision']:.3f}")
        logging.info(f"   Recall: {results[optimal_remove_pct]['recall']:.3f}")
        logging.info(f"   Selected stocks: {results[optimal_remove_pct]['n_selected']:,}")
        logging.info(f"   Percentage kept: {results[optimal_remove_pct]['n_selected'] / len(y_train_binary) * 100:.1f}%")
        logging.info("="*80)
        logging.info("")

        return {
            'remove_pct': optimal_remove_pct,
            'percentile': results[optimal_remove_pct]['percentile'],  # 호환성
            'threshold_value': results[optimal_remove_pct]['threshold_value'],
            'precision': results[optimal_remove_pct]['precision'],
            'recall': results[optimal_remove_pct]['recall'],
            'n_selected': results[optimal_remove_pct]['n_selected'],
            'mode': classifier_mode,
            'all_results': results  # 디버깅용
        }

    def _generate_walk_forward_periods(self) -> None:
        """
        Generate walk-forward evaluation periods from EVALUATION.PERIODS config.

        This method creates a list of cutoff dates for walk-forward training,
        similar to ml_backtest.py but for regressor evaluation.

        Populates:
        ----------
        self.walk_forward_periods : List[Dict]
            List of period information dicts with keys:
            - cutoff_date: datetime
            - train_start_year: int
            - period_name: str
        """
        from datetime import datetime
        from dateutil.relativedelta import relativedelta

        eval_config = self.conf.get('EVALUATION', {})
        periods = eval_config.get('PERIODS', [])
        train_start_year = eval_config.get('TRAIN_START_YEAR', 1996)
        rebalance_period = eval_config.get('REBALANCE_PERIOD', 3)

        if not periods:
            logging.warning("⚠️  No EVALUATION.PERIODS configured, walk-forward mode disabled")
            self.use_walk_forward = False
            return

        self.walk_forward_periods = []

        for period_config in periods:
            start_year = period_config['START_YEAR']
            end_year = period_config['END_YEAR']
            start_month = period_config.get('START_MONTH', 1)
            start_date = period_config.get('START_DATE', 1)

            start_dt = datetime(start_year, start_month, start_date)
            end_dt = datetime(end_year, 12, 31)

            # Generate rebalancing dates for this period
            current = start_dt
            while current <= end_dt:
                self.walk_forward_periods.append({
                    'cutoff_date': current,
                    'train_start_year': train_start_year,
                    'period_name': f"{start_year}-{end_year}"
                })
                current += relativedelta(months=rebalance_period)

        logging.info(f"  Generated {len(self.walk_forward_periods)} cutoff dates:")
        for i, period in enumerate(self.walk_forward_periods[:5]):  # Log first 5
            logging.info(f"    {i+1}. {period['cutoff_date'].date()}")
        if len(self.walk_forward_periods) > 5:
            logging.info(f"    ... and {len(self.walk_forward_periods) - 5} more")

    def train(self) -> None:
        """모든 분류 및 회귀 모델을 학습하고 디스크에 저장합니다.

        학습 파이프라인:
            1. def_model()로 모델 초기화
            2. LightGBM 호환성을 위해 특성 이름 정리
            3. 회귀 타겟을 이진 레이블로 변환 (price_dev > 0)
            4. 4개의 분류 모델 학습
            5. 2개의 회귀 모델 학습
            6. 모든 모델을 MODEL_SAVE_PATH에 저장
            7. PER_SECTOR=True인 경우 섹터별 모델 학습

        모든 모델은 나중에 로드하기 위해 joblib을 사용하여 .sav 파일로 저장됩니다.
        학습 점수(분류는 정확도, 회귀는 R²)가 로깅됩니다.

        부작용:
            - MODEL_SAVE_PATH 디렉토리가 없으면 생성
            - 모델을 디스크에 .sav 파일로 저장:
                * clsmodel_0.sav, clsmodel_1.sav, clsmodel_2.sav, clsmodel_3.sav
                * model_0.sav, model_1.sav
                * {sector}_model_0.sav, {sector}_model_1.sav (PER_SECTOR=True인 경우)
            - 모든 모델의 학습 점수 로깅

        참고:
            - 특성 중요도 분석 코드는 주석 처리됨 (필요시 주석 해제)
            - 하이퍼파라미터 튜닝을 위한 Grid search / random search 코드는 주석 처리됨
            - 모델은 속도를 위해 GPU에서 학습됩니다 (CUDA 지원 GPU 필요)
        """
        # 주석 처리됨: LightGBM 하이퍼파라미터 튜닝을 위한 Grid search
        # param_grid = {
        #     'n_estimators': [1000],
        #     'max_depth': [6, 8, 10, 12],
        #     'learning_rate': [0.01, 0.05, 0.1],
        #     'num_leaves': [31, 50, 70],
        #     'min_child_samples': [20, 30, 40]
        # }
        # lgbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',
        #                           device='gpu', boost_from_average=False)
        # grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid,
        #                            cv=5, scoring='accuracy', n_jobs=-1)
        # self.x_train = self.clean_feature_names(self.x_train)
        # y_train_binary = (self.y_train_cls > 0).astype(int)
        # grid_search.fit(self.x_train, y_train_binary)
        # print("Best parameters found: ", grid_search.best_params_)
        # print("Best accuracy: ", grid_search.best_score_)
        # exit()

        # 주석 처리됨: XGBoost 하이퍼파라미터 튜닝을 위한 Random search
        # params = {
        #     'learning_rate': np.arange(0.05, 0.3, 0.05),
        #     'max_depth': range(3, 10),
        #     'n_estimators': range(50, 500, 50),
        #     'colsample_bytree': np.arange(0.3, 1.0, 0.1),
        #     'subsample': np.arange(0.5, 1.0, 0.1),
        #     'gamma': [0, 1, 5]
        # }
        # xgb = xgboost.XGBRegressor()
        # cv = KFold(n_splits=5, shuffle=True)
        # search = RandomizedSearchCV(xgb, params, n_iter=100, cv=cv,
        #                             scoring='neg_mean_squared_error', random_state=42)
        # search.fit(self.x_train, self.y_train.values.ravel())
        # print(search.best_params_)
        # exit()

        # ========================================================================
        # Walk-Forward Mode: Train models for each period
        # ========================================================================
        if self.use_walk_forward:
            logging.info("="*80)
            logging.info("🔄 WALK-FORWARD TRAINING")
            logging.info("="*80)
            self._train_walk_forward()
            return  # Skip traditional training

        # ========================================================================
        # Traditional Mode: Train models once on all training data
        # ========================================================================
        logging.info("="*80)
        logging.info("📊 TRADITIONAL TRAINING (single train/test split)")
        logging.info("="*80)

        # 모델 저장 경로 설정
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'
        if not os.path.exists(MODEL_SAVE_PATH):
            logging.info("creating MODELS path: %s", MODEL_SAVE_PATH)
            os.makedirs(MODEL_SAVE_PATH)

        ml_config = self.conf.get('ML', {})
        use_optuna = ml_config.get('USE_OPTUNA', False)

        # Step 1: Optuna hyperparameter optimization (global)
        optuna_best_params = self._run_optuna_optimization(ml_config, use_optuna)

        # Step 2: Sector-specific Optuna optimization
        sector_optuna_params = self._run_sector_optuna_optimization(ml_config, use_optuna)

        # Step 3: Define models with Optuna results
        self.def_model(optuna_params=optuna_best_params, sector_optuna_params=sector_optuna_params)

        # LightGBM 호환성을 위해 특성 이름 정리 (Optuna를 사용하지 않은 경우에만)
        if not (use_optuna and OPTUNA_AVAILABLE):
            self.x_train = self.clean_feature_names(self.x_train)

        # Step 4: Train global models (classifiers + regressors)
        self._train_global_models(MODEL_SAVE_PATH, ml_config)

        # Step 5: Train sector models (if enabled)
        if self.use_sector_model:
            self._train_sector_models(MODEL_SAVE_PATH, ml_config)

    def _run_optuna_optimization(self, ml_config: dict, use_optuna) -> dict:
        """Run Optuna hyperparameter optimization for global classification model.

        Args:
            ml_config: ML configuration dictionary.
            use_optuna: Whether Optuna optimization is enabled.

        Returns:
            Best parameters dict from Optuna, or None if not available.
        """
        optuna_best_params = None

        # ========== Step 1: 항상 기존 Optuna 파라미터 로드 시도 ==========
        # USE_OPTUNA 설정과 무관하게, 기존에 최적화한 파라미터가 있으면 로드
        logging.info("")
        logging.info("="*80)
        logging.info("🔍 CHECKING FOR EXISTING OPTUNA PARAMETERS")
        logging.info("="*80)
        existing_params = self._load_existing_optuna_params('clsmodel_0')

        if existing_params:
            optuna_best_params = existing_params
            logging.info("✅ Found and loaded existing Optuna parameters!")
            logging.info(f"   Parameters: {existing_params}")
            logging.info("   These will be used regardless of USE_OPTUNA setting")
            logging.info("="*80)
        else:
            logging.info("📝 No existing Optuna parameters found")
            if not use_optuna:
                logging.info("   USE_OPTUNA=N, will use default parameters")
            else:
                logging.info("   USE_OPTUNA=Y, will run new optimization")
            logging.info("="*80)

        # ========== Step 2: USE_OPTUNA=Y이면 새로운 최적화 실행 ==========
        if use_optuna and OPTUNA_AVAILABLE:
            from src.training.optuna_utils import (
                optimize_xgboost_params,
                save_optuna_report,
                save_optuna_plots,
                PLOT_AVAILABLE
            )

            logging.info("="*80)
            logging.info("🔧 OPTUNA HYPERPARAMETER OPTIMIZATION")
            logging.info("="*80)

            # Optuna 설정
            n_trials = int(ml_config.get('OPTUNA_TRIALS', 50))
            cv_folds = int(ml_config.get('OPTUNA_CV_FOLDS', 5))
            timeout = ml_config.get('OPTUNA_TIMEOUT', None)
            save_report = ml_config.get('OPTUNA_SAVE_REPORT', True)
            save_plots = ml_config.get('OPTUNA_SAVE_PLOTS', True)

            # 탐색 공간 (config에서 읽기 or 기본값)
            # Option A: 메모리 안전 + 성능 보장 (과적합 방지)
            search_space_config = ml_config.get('OPTUNA_SEARCH_SPACE', {})
            search_space = {
                'n_estimators': search_space_config.get('n_estimators', [100, 500]),  # 2000→500: 메모리 안전
                'learning_rate': search_space_config.get('learning_rate', [0.01, 0.3]),  # 0.001→0.01: 너무 낮은 lr 방지
                'max_depth': search_space_config.get('max_depth', [3, 8]),  # 15→8: 과적합 방지, 메모리 안전
                'subsample': search_space_config.get('subsample', [0.5, 1.0]),
                'colsample_bytree': search_space_config.get('colsample_bytree', [0.5, 1.0]),
                'gamma': search_space_config.get('gamma', [0, 10])
            }

            # LightGBM 호환성을 위해 특성 이름 정리 (Optuna 전에 수행)
            self.x_train = self.clean_feature_names(self.x_train)

            # Baseline 파라미터 (기존 설정)
            baseline_params = {
                'n_estimators': 500,
                'learning_rate': 0.1,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0
            }

            # Classification 모델 최적화 (clsmodel_0 기준)
            logging.info(f"Optimizing XGBoost Classifier (clsmodel_0)")
            logging.info(f"  Trials: {n_trials}, CV folds: {cv_folds}")
            logging.info(f"  Search space: {search_space}")

            # ✅ REFACTORED: Binary label generation using DataProcessor (for Optuna)
            y_train_binary_optuna = DataProcessor.create_binary_target(self.y_train_cls, config=self.conf, logger=logging.getLogger())

            # ========== Phase 1 & 2: 극단값 진단 및 클리핑 ==========
            # Optuna CV 실행 전에 데이터 품질 확인 및 처리
            logging.info("")
            logging.info("="*80)
            logging.info("🔬 PRE-OPTUNA DATA QUALITY CHECK")
            logging.info("="*80)

            # Phase 1: 진단
            X_values = self.x_train.values
            y_values = y_train_binary_optuna.values if hasattr(y_train_binary_optuna, 'values') else y_train_binary_optuna
            diagnosis_ok = self._diagnose_extreme_values(X_values, y_values, "x_train")

            # Phase 2: 극단값이 발견되면 클리핑 적용
            if not diagnosis_ok:
                logging.warning("⚠️  QuantileDMatrix test failed - applying clipping")
                self.x_train, _ = self._safe_clip_for_xgboost(self.x_train, y_values, max_abs_value=1e7)

                # 재진단
                logging.info("")
                logging.info("🔄 Re-diagnosing after clipping...")
                X_values_clipped = self.x_train.values
                diagnosis_ok_after = self._diagnose_extreme_values(X_values_clipped, y_values, "x_train_clipped")

                if not diagnosis_ok_after:
                    logging.error("❌ Still failing after clipping - trying lower threshold (1e6)")
                    self.x_train, _ = self._safe_clip_for_xgboost(self.x_train, y_values, max_abs_value=1e6)
            else:
                logging.info("✅ Data quality check passed - no clipping needed")

            logging.info("="*80)
            logging.info("")

            # Baseline 성능 측정
            from sklearn.model_selection import cross_val_score
            baseline_model = xgboost.XGBClassifier(
                **baseline_params,
                tree_method='hist',
                device='cpu',
                objective='binary:logistic',
                eval_metric='logloss',
                missing=np.nan
            )
            baseline_scores = cross_val_score(
                baseline_model, self.x_train, y_train_binary_optuna,
                cv=cv_folds, scoring='accuracy', n_jobs=-1
            )
            baseline_score = baseline_scores.mean()
            logging.info(f"📊 Baseline accuracy: {baseline_score:.4f} (±{baseline_scores.std():.4f})")

            # ========== Optuna Reuse Logic (Classification) ==========
            reuse_existing = ml_config.get('OPTUNA_REUSE_EXISTING', 'N') == 'Y'
            max_age_days = int(ml_config.get('OPTUNA_REUSE_MAX_AGE_DAYS', 7))

            study = None
            best_params = None

            if reuse_existing:
                logging.info("")
                logging.info("🔍 Checking for existing Optuna results...")
                existing_params = self._load_existing_optuna_params('clsmodel_0')

                if existing_params and self._is_optuna_result_fresh('clsmodel_0', max_age_days):
                    logging.info("✅ Reusing existing Optuna results (saved time!)")
                    optuna_best_params = existing_params

                    # Skip optimization, but still show results
                    logging.info("="*80)
                    logging.info("✅ USING CACHED OPTUNA RESULTS")
                    logging.info("="*80)
                    logging.info(f"Best params: {existing_params}")
                    logging.info("="*80)
                else:
                    if not existing_params:
                        logging.info("⏩ No existing results found, running optimization...")
                    else:
                        logging.info("⏩ Existing results are stale, running optimization...")

                    # Run optimization
                    study, best_params = optimize_xgboost_params(
                        self.x_train,
                        y_train_binary_optuna,
                        search_space,
                        n_trials=n_trials,
                        cv_folds=cv_folds,
                        timeout=timeout,
                        task='classification'
                    )
            else:
                logging.info("")
                logging.info("⏩ OPTUNA_REUSE_EXISTING=N, running optimization...")

                # Run optimization
                study, best_params = optimize_xgboost_params(
                    self.x_train,
                    y_train_binary_optuna,
                    search_space,
                    n_trials=n_trials,
                    cv_folds=cv_folds,
                    timeout=timeout,
                    task='classification'
                )

            # Process optimization results (only if we ran optimization)
            if study and best_params:
                optuna_best_params = best_params
                improvement = study.best_value - baseline_score
                logging.info("="*80)
                logging.info("✅ OPTIMIZATION RESULTS")
                logging.info("="*80)
                logging.info(f"Best accuracy: {study.best_value:.4f} ({improvement:+.4f}, {improvement/baseline_score*100:+.2f}%)")
                logging.info(f"Best params: {best_params}")

                # 리포트 저장 (ROOT_PATH/models/optuna/)
                if save_report:
                    save_optuna_report(
                        study, baseline_params, baseline_score,
                        'clsmodel_0', 'reports', root_path=self.root_path
                    )

                # 차트 저장 (ROOT_PATH/models/optuna/)
                if save_plots and PLOT_AVAILABLE:
                    save_optuna_plots(study, 'clsmodel_0', 'reports', root_path=self.root_path)

                logging.info("="*80)
            elif not reuse_existing or (reuse_existing and not existing_params):
                # Only show warning if we tried to optimize and failed
                logging.warning("⚠️  Optuna optimization failed, using baseline params")

        elif use_optuna and not OPTUNA_AVAILABLE:
            logging.warning("⚠️  USE_OPTUNA=Y but Optuna not installed. Using default params.")
            logging.warning("   Install with: pip install optuna plotly kaleido")

        return optuna_best_params

    def _run_sector_optuna_optimization(self, ml_config: dict, use_optuna) -> dict:
        """Run Optuna hyperparameter optimization for sector-specific models.

        Args:
            ml_config: ML configuration dictionary.
            use_optuna: Whether Optuna optimization is enabled.

        Returns:
            Dictionary mapping sector names to their best Optuna parameters.
        """
        sector_optuna_params = {}
        optuna_optimize_sectors = ml_config.get('OPTUNA_OPTIMIZE_SECTORS', 'N') == 'Y'

        # ========== Step 1: 항상 기존 섹터별 Optuna 파라미터 로드 시도 ==========
        # USE_OPTUNA 설정과 무관하게, 기존에 최적화한 파라미터가 있으면 로드
        if self.use_sector_model and len(self.sector_list) > 0:
            logging.info("")
            logging.info("="*80)
            logging.info("🔍 CHECKING FOR EXISTING SECTOR-SPECIFIC OPTUNA PARAMETERS")
            logging.info("="*80)

            loaded_count = 0
            for sec in self.sector_list:
                existing_sector_params = self._load_existing_optuna_params(f'sector_{sec}')
                if existing_sector_params:
                    sector_optuna_params[sec] = existing_sector_params
                    loaded_count += 1
                    logging.info(f"   ✅ {sec}: Loaded existing parameters")

            if loaded_count > 0:
                logging.info(f"✅ Loaded {loaded_count}/{len(self.sector_list)} sector-specific Optuna parameters")
                logging.info("   These will be used regardless of OPTUNA_OPTIMIZE_SECTORS setting")
            else:
                logging.info("📝 No existing sector-specific Optuna parameters found")
                if not (optuna_optimize_sectors and use_optuna):
                    logging.info("   Will use SECTOR_CONFIG or default parameters")
                else:
                    logging.info("   Will run new sector-specific optimization")
            logging.info("="*80)

        # ========== Step 2: OPTUNA_OPTIMIZE_SECTORS=Y이면 새로운 최적화 실행 ==========
        if self.use_sector_model and optuna_optimize_sectors and use_optuna and OPTUNA_AVAILABLE:
            from src.training.optuna_utils import (
                optimize_xgboost_params,
                save_optuna_report,
                save_optuna_plots,
                PLOT_AVAILABLE
            )

            logging.info("="*80)
            logging.info("🔧 SECTOR-SPECIFIC OPTUNA OPTIMIZATION")
            logging.info("="*80)

            # ✅ UNIFIED: Final check before sector Optuna using DataProcessor
            DataProcessor.check_duplicate_index(
                self.x_train,
                "Before sector Optuna (CRITICAL - where error occurs)",
                logging.getLogger()
            )

            # Sector Optuna 설정
            sector_trials = int(ml_config.get('OPTUNA_SECTOR_TRIALS', 30))
            sector_cv_folds = int(ml_config.get('OPTUNA_SECTOR_CV_FOLDS', 2))
            sector_timeout = ml_config.get('OPTUNA_SECTOR_TIMEOUT', 180)

            # 탐색 공간 (classifier와 동일)
            search_space_config = ml_config.get('OPTUNA_SEARCH_SPACE', {})
            search_space = {
                'n_estimators': search_space_config.get('n_estimators', [100, 500]),
                'learning_rate': search_space_config.get('learning_rate', [0.01, 0.3]),
                'max_depth': search_space_config.get('max_depth', [3, 8]),
                'subsample': search_space_config.get('subsample', [0.5, 1.0]),
                'colsample_bytree': search_space_config.get('colsample_bytree', [0.5, 1.0]),
                'gamma': search_space_config.get('gamma', [0, 10])
            }

            save_report = ml_config.get('OPTUNA_SAVE_REPORT', True)
            save_plots = ml_config.get('OPTUNA_SAVE_PLOTS', True)

            # self.x_train에 sector 정보 임시 추가 (인덱스 기반으로)
            x_train_with_sector = self.x_train.copy()
            x_train_with_sector['sector'] = self.train_df.loc[self.x_train.index, 'sector']

            logging.info(f"Optimizing {len(self.sector_list)} sectors individually...")
            logging.info(f"  Trials per sector: {sector_trials}, CV folds: {sector_cv_folds}")

            for sec_idx, sec in enumerate(self.sector_list):
                logging.info("")
                logging.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logging.info(f"🔧 Sector {sec_idx+1}/{len(self.sector_list)}: {sec}")
                logging.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                # ✅ 이미 Step 1에서 로드된 파라미터가 있으면 스킵
                if sec in sector_optuna_params:
                    logging.info(f"  ⏭️  Skipping optimization - parameters already loaded from existing file")
                    logging.info(f"     Parameters: {sector_optuna_params[sec]}")
                    continue

                # 섹터별 데이터 필터링
                sector_mask = x_train_with_sector['sector'] == sec
                X_sector = x_train_with_sector[sector_mask].drop('sector', axis=1)
                y_sector_reg = self.y_train[sector_mask].iloc[:, 0]  # 회귀 타겟

                logging.info(f"  Sector data: {len(X_sector)} samples, {len(X_sector.columns)} features")

                # Baseline 파라미터
                baseline_params = {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 7,
                    'subsample': 0.8,
                    'colsample_bytree': 0.7,
                    'gamma': 0.01
                }

                # Baseline 성능 측정
                from sklearn.model_selection import cross_val_score
                baseline_model = xgboost.XGBRegressor(
                    **baseline_params,
                    tree_method='hist',
                    device='cpu',
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    missing=np.nan
                )
                baseline_scores = cross_val_score(
                    baseline_model, X_sector, y_sector_reg,
                    cv=sector_cv_folds, scoring='neg_mean_squared_error', n_jobs=-1
                )
                baseline_score = -baseline_scores.mean()  # MSE (양수)
                logging.info(f"  📊 Baseline MSE: {baseline_score:.6f} (±{baseline_scores.std():.6f})")

                # ========== Optuna Reuse Logic (Sector) ==========
                reuse_existing = ml_config.get('OPTUNA_REUSE_EXISTING', 'N') == 'Y'
                max_age_days = int(ml_config.get('OPTUNA_REUSE_MAX_AGE_DAYS', 7))

                study = None
                best_params = None

                if reuse_existing:
                    logging.info(f"  🔍 Checking for existing Optuna results for sector '{sec}'...")
                    existing_params = self._load_existing_optuna_params(f'sector_{sec}')

                    if existing_params and self._is_optuna_result_fresh(f'sector_{sec}', max_age_days):
                        logging.info(f"  ✅ Reusing existing Optuna results for '{sec}' (saved time!)")
                        sector_optuna_params[sec] = existing_params
                        logging.info(f"  Best params: {existing_params}")
                        continue  # Skip to next sector
                    else:
                        if not existing_params:
                            logging.info(f"  ⏩ No existing results found for '{sec}', running optimization...")
                        else:
                            logging.info(f"  ⏩ Existing results are stale for '{sec}', running optimization...")
                else:
                    logging.info(f"  ⏩ OPTUNA_REUSE_EXISTING=N, running optimization for '{sec}'...")

                # Optuna 최적화 실행 (reuse가 안 되는 경우에만)
                try:
                    study, best_params = optimize_xgboost_params(
                        X_sector,
                        y_sector_reg,
                        search_space,
                        n_trials=sector_trials,
                        cv_folds=sector_cv_folds,
                        timeout=sector_timeout,
                        task='regression'
                    )

                    if study and best_params:
                        sector_optuna_params[sec] = best_params
                        improvement = baseline_score - study.best_value  # MSE는 낮을수록 좋음
                        improvement_pct = improvement / baseline_score * 100
                        logging.info(f"  ✅ Best MSE: {study.best_value:.6f} ({-improvement:+.6f}, {improvement_pct:+.2f}% improvement)")
                        logging.info(f"  Best params: {best_params}")

                        # 리포트 저장 (ROOT_PATH/models/optuna/)
                        if save_report:
                            save_optuna_report(
                                study, baseline_params, baseline_score,
                                f'sector_{sec}', 'reports', root_path=self.root_path
                            )

                        # 차트 저장 (ROOT_PATH/models/optuna/)
                        if save_plots and PLOT_AVAILABLE:
                            save_optuna_plots(study, f'sector_{sec}', 'reports', root_path=self.root_path)
                    else:
                        logging.warning(f"  ⚠️  Optimization failed for sector {sec}, using baseline params")
                        sector_optuna_params[sec] = baseline_params

                except Exception as e:
                    logging.error(f"  ❌ Error optimizing sector {sec}: {e}")
                    logging.warning(f"  Using baseline params for {sec}")
                    sector_optuna_params[sec] = baseline_params

            logging.info("="*80)
            logging.info(f"✅ Sector optimization complete: {len(sector_optuna_params)}/{len(self.sector_list)} sectors optimized")
            logging.info("="*80)
            logging.info("")

        elif self.use_sector_model and optuna_optimize_sectors and not (use_optuna and OPTUNA_AVAILABLE):
            logging.warning("⚠️  OPTUNA_OPTIMIZE_SECTORS=Y but Optuna not available. Using SECTOR_CONFIG only.")

        return sector_optuna_params

    def _train_global_models(self, model_save_path: str, ml_config: dict) -> None:
        """Train global classifiers and regressors using 2-stage pipeline.

        Stage 1: Train classifiers on all data, find optimal threshold.
        Stage 2: Filter data by threshold, train regressors on filtered data.

        Args:
            model_save_path: Directory path where models will be saved.
            ml_config: ML configuration dictionary.
        """
        # ========================================
        # 🎯 UNIFIED PREPROCESSING (Single Source of Truth)
        # ========================================
        # Replaces ALL scattered preprocessing with ONE unified method
        # Used by BOTH regressor.py and ml_backtest.py for IDENTICAL preprocessing
        #
        # Steps performed (in order):
        # 1. Remove infinite values from X and y
        # 2. Replace remaining infinite with NaN
        # 3. Remove rows with infinite in y labels (CRITICAL)
        # 4. Log transformation (extreme value compression)
        # 5. Remove columns with >50% NaN
        # 6. Remove rows with NaN in y labels
        # 7. (Optional) Winsorization
        # 8. (Optional) Feature selection

        self.x_train, self.y_train, self.y_train_cls, self.selected_features = \
            DataProcessor.preprocess_training_data(
                self.x_train,
                self.y_train,
                self.y_train_cls,
                self.conf,
                logging.getLogger()
            )

        # ✅ Create binary target for classification after preprocessing
        # (preprocessing may change indices, so recreate after)
        # analyze=True: Show threshold analysis on first training
        y_train_binary = DataProcessor.create_binary_target(
            self.y_train_cls,
            config=self.conf,
            logger=logging.getLogger(),
            analyze=True  # Analyze threshold candidates
        )

        # ===== Stage 1: Train Classifiers (모든 데이터) =====
        logging.info("="*80)
        logging.info("🎯 STAGE 1: CLASSIFIER TRAINING (All data)")
        logging.info("="*80)

        for i, model in self.clsmodels.items():
            logging.info(f"Training classifier {i}...")
            model.fit(self.x_train, y_train_binary)
            filename = model_save_path + 'clsmodel_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            score = model.score(self.x_train, y_train_binary)
            logging.info(f"  Classifier {i} accuracy: {score:.4f}")

        # ===== Optimal Threshold Search (자동 탐색 or 고정값) =====
        auto_search = ml_config.get('CLASSIFIER_THRESHOLD_AUTO_SEARCH', 'Y') == 'Y'

        if auto_search:
            # 자동 탐색: 대표 분류기 (clsmodel_0) 사용
            threshold_config = self._find_optimal_threshold(
                classifier=self.clsmodels[0],
                x_train=self.x_train,
                y_train_binary=y_train_binary
            )
        else:
            # 고정값 사용
            fixed_remove_pct = int(ml_config.get('CLASSIFIER_REMOVE_PCT', 8))
            classifier_mode = ml_config.get('CLASSIFIER_MODE', 'positive_screen')
            y_probs = predict_proba_with_gpu_support(self.clsmodels[0], self.x_train, self.use_gpu_prediction)[:, 1]

            # 모드별 percentile 및 mask 계산
            if classifier_mode == "negative_screen":
                percentile = 100 - fixed_remove_pct
                threshold = np.percentile(y_probs, percentile)
                mask = y_probs < threshold  # BAD 확률이 낮은 것만 선택
            else:  # positive_screen
                percentile = fixed_remove_pct
                threshold = np.percentile(y_probs, percentile)
                mask = y_probs > threshold  # GOOD 확률이 높은 것만 선택

            n_selected = mask.sum()

            logging.info("="*80)
            logging.info(f"📌 USING FIXED FILTERING: Remove {fixed_remove_pct}%")
            logging.info("="*80)
            logging.info(f"   Mode: {classifier_mode}")
            logging.info(f"   Threshold value: {threshold:.3f}")
            logging.info(f"   Selected stocks: {n_selected:,} ({n_selected / len(y_train_binary) * 100:.1f}%)")
            logging.info("="*80)

            threshold_config = {
                'remove_pct': fixed_remove_pct,
                'percentile': percentile,
                'threshold_value': threshold,
                'n_selected': n_selected,
                'mode': classifier_mode,
                'auto_search': False
            }

        # Threshold config 저장
        threshold_config_file = model_save_path + 'threshold_config.pkl'
        joblib.dump(threshold_config, threshold_config_file)
        logging.info(f"✅ Saved threshold config to {threshold_config_file}")
        logging.info("")

        # ===== Stage 2: Filter Data by Threshold =====
        logging.info("="*80)
        logging.info("🎯 STAGE 2: DATA FILTERING")
        logging.info("="*80)

        y_probs = predict_proba_with_gpu_support(self.clsmodels[0], self.x_train, self.use_gpu_prediction)[:, 1]
        threshold = threshold_config['threshold_value']
        classifier_mode = threshold_config.get('mode', 'positive_screen')

        # 모드별 mask 계산
        if classifier_mode == "negative_screen":
            safe_mask = y_probs < threshold  # BAD 확률이 낮은 것 선택
        else:  # positive_screen
            safe_mask = y_probs > threshold  # GOOD 확률이 높은 것 선택

        x_train_filtered = self.x_train[safe_mask]
        y_train_filtered = self.y_train[safe_mask]

        logging.info(f"   Original training data: {len(self.x_train):,} stocks")
        logging.info(f"   After filtering: {len(x_train_filtered):,} stocks")
        logging.info(f"   Removed: {len(self.x_train) - len(x_train_filtered):,} stocks ({(len(self.x_train) - len(x_train_filtered)) / len(self.x_train) * 100:.1f}%)")
        logging.info("="*80)
        logging.info("")

        # ===== Stage 3: Train Regressors (필터링된 데이터) =====
        logging.info("="*80)
        logging.info("🎯 STAGE 3: REGRESSOR TRAINING (Filtered data)")
        logging.info("="*80)

        for i, model in self.models.items():
            logging.info(f"Training regressor {i}...")
            model.fit(x_train_filtered, y_train_filtered.values.ravel())
            filename = model_save_path + 'model_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            score = model.score(x_train_filtered, y_train_filtered.values.ravel())
            logging.info(f"  Regressor {i} R² score: {score:.4f}")

        logging.info("="*80)
        logging.info("✅ 2-STAGE TRAINING COMPLETE")
        logging.info("="*80)
        logging.info("")

        # ===== Feature columns 저장 (예측 시 피처 정렬용) =====
        # 필터링 전 전체 feature 저장 (예측 시에는 전체 데이터 사용)
        feature_columns = self.x_train.columns.tolist()
        feature_columns_file = model_save_path + 'feature_columns.pkl'
        joblib.dump(feature_columns, feature_columns_file)
        logging.info(f"✅ Saved {len(feature_columns)} feature columns to {feature_columns_file}")

        # 주석 처리됨: 특성 중요도 분석
        # logging.info("end fitting RandomForestRegressor")
        # ftr_importances_values = model.feature_importances_
        # ftr_importances = pd.Series(ftr_importances_values, index=self.x_train.columns)
        # ftr_importances.to_csv(model_save_path+'model_importances.csv')
        # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
        # logging.info(ftr_top20)

    def _train_sector_models(self, model_save_path: str, ml_config: dict) -> None:
        """Train sector-specific classifiers and regressors using 2-stage pipeline.

        Applies unified preprocessing to each sector, trains sector classifiers
        (if USE_CLASSIFIER=Y), finds sector thresholds, filters data, and trains
        sector regressors on filtered data.

        Args:
            model_save_path: Directory path where models will be saved.
            ml_config: ML configuration dictionary.
        """
        logging.info("="*80)
        logging.info("🎯 SECTOR MODELS: Applying unified preprocessing to each sector")
        logging.info("="*80)

        for sec in self.sector_list:
            logging.info(f"\n📊 Processing sector: {sec}")
            logging.info("-" * 60)

            # ✅ UNIFIED: Use SAME preprocessing as unified model (SINGLE SOURCE OF TRUTH)
            # Pass y_cls if USE_CLASSIFIER=Y, otherwise None
            y_cls_input = self.sector_y_train_cls.get(sec) if self.use_classifier else None
            x_sector_clean, y_sector_clean, y_cls_clean, _ = DataProcessor.preprocess_training_data(
                self.sector_x_train[sec],
                self.sector_y_train[sec],
                y_cls=y_cls_input,
                config=self.conf,
                logger=logging.getLogger()
            )

            # Update sector training data with preprocessed results
            self.sector_x_train[sec] = x_sector_clean
            self.sector_y_train[sec] = y_sector_clean
            if self.use_classifier and y_cls_clean is not None:
                self.sector_y_train_cls[sec] = y_cls_clean

        logging.info("="*80)
        logging.info("✅ All sector preprocessing complete")
        logging.info("="*80)
        logging.info("")

        # ===== Train sector classifiers (if USE_CLASSIFIER=Y) =====
        if self.use_classifier:
            logging.info("="*80)
            logging.info("🎯 SECTOR CLASSIFIERS: Training sector-specific classifiers")
            logging.info("="*80)

            for sec_idx, sec in enumerate(self.sector_list):
                logging.info(f"\n📊 Training classifiers for sector: {sec}")

                # Get preprocessed classification target and create binary target
                y_cls_sector = self.sector_y_train_cls[sec]
                y_train_binary = DataProcessor.create_binary_target(y_cls_sector, config=self.conf, logger=logging.getLogger())

                # Train all 4 classifiers for this sector
                for i in range(4):
                    k = (sec, i)
                    model = self.sector_classifiers[k]
                    logging.info(f"  Training sector classifier {i} for {sec}...")

                    # Use numpy arrays for consistent sklearn API behavior across model families
                    # (CatBoost/XGBoost both support DataFrame too, but numpy avoids edge cases)
                    X_train = self.sector_x_train[sec].values

                    model.fit(X_train, y_train_binary)
                    filename = model_save_path + '{}_clsmodel_{}.sav'.format(sec, str(i))
                    joblib.dump(model, filename)

                    # Score calculation (use values for consistency)
                    score = model.score(X_train, y_train_binary)
                    logging.info(f"  Sector classifier {i} score: {score:.4f}")

                logging.info(f"✅ Trained and saved 4 classifiers for {sec}")

            logging.info("="*80)
            logging.info(f"✅ All sector classifiers complete ({len(self.sector_list)} sectors x 4 classifiers)")
            logging.info("="*80)
            logging.info("")

        # ===== Train sector regressors (2-stage with filtering) =====
        logging.info("="*80)
        logging.info("🎯 SECTOR REGRESSORS: Training with 2-stage filtering")
        logging.info("="*80)

        # Store sector threshold configs
        sector_threshold_configs = {}
        auto_search = ml_config.get('CLASSIFIER_THRESHOLD_AUTO_SEARCH', 'Y') == 'Y'

        for sec_idx, sec in enumerate(self.sector_list):
            logging.info(f"\n📊 Training regressors for sector: {sec}")
            logging.info("-" * 60)

            if self.use_classifier:
                # ===== Sector-specific threshold search =====
                y_cls_sector = self.sector_y_train_cls[sec]
                y_train_binary_sector = DataProcessor.create_binary_target(y_cls_sector, config=self.conf, logger=logging.getLogger())

                if auto_search:
                    # Use sector-specific classifier (0) for threshold search
                    sector_threshold_config = self._find_optimal_threshold(
                        classifier=self.sector_classifiers[(sec, 0)],
                        x_train=self.sector_x_train[sec],
                        y_train_binary=y_train_binary_sector
                    )
                else:
                    # Use same fixed remove percentage for all sectors
                    fixed_remove_pct = int(ml_config.get('CLASSIFIER_REMOVE_PCT', 8))
                    classifier_mode = ml_config.get('CLASSIFIER_MODE', 'positive_screen')
                    y_probs_sector = predict_proba_with_gpu_support(
                        self.sector_classifiers[(sec, 0)],
                        self.sector_x_train[sec],
                        self.use_gpu_prediction
                    )[:, 1]

                    # 모드별 percentile 및 mask 계산
                    if classifier_mode == "negative_screen":
                        percentile = 100 - fixed_remove_pct
                        threshold_sector = np.percentile(y_probs_sector, percentile)
                        mask_sector = y_probs_sector < threshold_sector
                    else:  # positive_screen
                        percentile = fixed_remove_pct
                        threshold_sector = np.percentile(y_probs_sector, percentile)
                        mask_sector = y_probs_sector > threshold_sector

                    n_selected_sector = mask_sector.sum()

                    logging.info(f"  Using fixed filtering: Remove {fixed_remove_pct}%")
                    logging.info(f"  Selected: {n_selected_sector}/{len(y_train_binary_sector)}")

                    sector_threshold_config = {
                        'remove_pct': fixed_remove_pct,
                        'percentile': percentile,
                        'threshold_value': threshold_sector,
                        'n_selected': n_selected_sector,
                        'mode': classifier_mode,
                        'auto_search': False
                    }

                # Save sector threshold config
                sector_threshold_configs[sec] = sector_threshold_config

                # Filter data by threshold
                y_probs_sector = predict_proba_with_gpu_support(
                    self.sector_classifiers[(sec, 0)],
                    self.sector_x_train[sec],
                    self.use_gpu_prediction
                )[:, 1]
                threshold_sector = sector_threshold_config['threshold_value']
                classifier_mode_sector = sector_threshold_config.get('mode', 'positive_screen')

                # 모드별 mask 계산
                if classifier_mode_sector == "negative_screen":
                    safe_mask_sector = y_probs_sector < threshold_sector
                else:  # positive_screen
                    safe_mask_sector = y_probs_sector > threshold_sector

                x_train_sector_filtered = self.sector_x_train[sec][safe_mask_sector]
                y_train_sector_filtered = self.sector_y_train[sec][safe_mask_sector]

                logging.info(f"  Filtered: {len(x_train_sector_filtered)}/{len(self.sector_x_train[sec])} stocks")
            else:
                # No classifier: use all data
                x_train_sector_filtered = self.sector_x_train[sec]
                y_train_sector_filtered = self.sector_y_train[sec]
                logging.info(f"  No classifier: using all {len(x_train_sector_filtered)} stocks")

            # Train 2 regressors on filtered data
            for i in range(2):
                k = (sec, i)
                model = self.sector_models[k]
                model.fit(x_train_sector_filtered, y_train_sector_filtered.values.ravel())
                filename = model_save_path + '{}_model_{}.sav'.format(sec, str(i))
                joblib.dump(model, filename)

                score = model.score(x_train_sector_filtered, y_train_sector_filtered.values.ravel())
                logging.info(f"  Regressor {i} R² score: {score:.4f}")

            logging.info(f"✅ Sector {sec} complete")

        # Save all sector threshold configs
        if self.use_classifier:
            sector_threshold_file = model_save_path + 'sector_threshold_configs.pkl'
            joblib.dump(sector_threshold_configs, sector_threshold_file)
            logging.info(f"\n✅ Saved sector threshold configs to {sector_threshold_file}")

        logging.info("="*80)
        logging.info(f"✅ All sector regressors complete ({len(self.sector_list)} sectors x 2 regressors)")
        logging.info("="*80)
        logging.info("")


    def _train_walk_forward(self) -> None:
        """
        Walk-Forward training: Train and predict for each cutoff period.

        This method implements walk-forward analysis with optional parallel execution:
        1. Determine optimal number of workers (using MemoryProfiler)
        2. For each cutoff_date (in parallel or sequential):
           a. Load data up to cutoff_date (expanding/rolling window)
           b. Train models
           c. Load next period data
           d. Make predictions
           e. Store predictions in self.predictions_cache

        The predictions_cache will be saved to pkl for use by:
        - evaluation() method (accuracy metrics)
        - ml_backtest.py (if USE_CACHED_PREDICTIONS=Y)

        Parallel execution:
        - Uses Ray to train multiple periods in parallel
        - MemoryProfiler auto-learns optimal worker count
        - Handles OOM by invalidating profile and retrying conservatively
        """
        from src.training.data_processor import DataProcessor
        from src.scripts.memory_profiler import MemoryProfiler

        MODEL_SAVE_PATH = self.root_path + '/MODELS/'
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)

        eval_config = self.conf.get('EVALUATION', {})
        top_k = eval_config.get('TOP_K_NUM', 10)

        # Parallel training configuration
        parallel_enabled = eval_config.get('PARALLEL_TRAINING', 'N') == 'Y'
        num_workers_config = eval_config.get('NUM_WORKERS', 'auto')
        profile_file = self.root_path + '/config/' + eval_config.get('MEMORY_PROFILE_FILE', 'memory_profile.json')

        logging.info(f"\n🔄 Walk-Forward Training: {len(self.walk_forward_periods)} periods")
        logging.info(f"🎯 Top-K Selection: {top_k}")
        logging.info(f"⚡ Parallel Training: {'Enabled' if parallel_enabled else 'Disabled'}")

        # Determine execution mode
        use_parallel = parallel_enabled and RAY_AVAILABLE and len(self.walk_forward_periods) > 1

        if parallel_enabled and not RAY_AVAILABLE:
            logging.warning("⚠️  Parallel training requested but Ray not available. Using sequential mode.")

        if use_parallel:
            self._train_walk_forward_parallel(MODEL_SAVE_PATH, top_k, num_workers_config, profile_file)
        else:
            self._train_walk_forward_sequential(MODEL_SAVE_PATH, top_k)

    def _save_predictions_to_csv(self, model_save_path: str) -> None:
        """
        Save predictions cache to CSV files for user-friendly inspection.

        Creates 3 types of CSV files:
        1. regressor_predictions_all.csv: All predictions combined (full dataset)
        2. regressor_predictions_YYYYMMDD.csv: Per-date predictions (detailed view)
        3. regressor_predictions_top_k.csv: Selected stocks only (portfolio view)

        Parameters:
        ----------
        model_save_path : str
            Directory to save CSV files
        """
        import os
        from pathlib import Path

        if not self.predictions_cache:
            logging.warning("⚠️  No predictions cache to export!")
            return

        # Create CSV directory
        csv_dir = Path(model_save_path) / 'predictions_csv'
        csv_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n📁 Saving predictions to CSV: {csv_dir}")

        # Prepare data for combined CSV
        all_predictions = []
        top_k_predictions = []

        for date_str, cache_entry in self.predictions_cache.items():
            predictions_df = cache_entry.get('predictions_df')

            if predictions_df is None or predictions_df.empty:
                logging.warning(f"   ⚠️  {date_str}: Empty predictions, skipping")
                continue

            # Add rebalance_date column for traceability
            predictions_with_date = predictions_df.copy()
            predictions_with_date.insert(0, 'rebalance_date', date_str)

            # All predictions
            all_predictions.append(predictions_with_date)

            # Save per-date CSV
            date_csv_file = csv_dir / f'predictions_{date_str.replace("-", "")}.csv'
            predictions_with_date.to_csv(date_csv_file, index=False, encoding='utf-8-sig')
            logging.info(f"   ✅ {date_str}: {len(predictions_with_date)} predictions → {date_csv_file.name}")

            # Top-K predictions
            if DataSchema.SELECTED in predictions_df.columns:
                top_k_df = predictions_with_date[predictions_with_date[DataSchema.SELECTED] == True].copy()
                if not top_k_df.empty:
                    top_k_predictions.append(top_k_df)

        # Save combined CSV (all predictions)
        if all_predictions:
            all_combined = pd.concat(all_predictions, ignore_index=True)
            all_csv_file = csv_dir / 'regressor_predictions_all.csv'
            all_combined.to_csv(all_csv_file, index=False, encoding='utf-8-sig')
            logging.info(f"\n   📊 Combined: {len(all_combined)} predictions → {all_csv_file.name}")
        else:
            logging.warning("   ⚠️  No predictions to combine!")

        # Save top-K CSV (selected stocks only)
        if top_k_predictions:
            top_k_combined = pd.concat(top_k_predictions, ignore_index=True)
            top_k_csv_file = csv_dir / 'regressor_predictions_top_k.csv'
            top_k_combined.to_csv(top_k_csv_file, index=False, encoding='utf-8-sig')
            logging.info(f"   🎯 Top-K: {len(top_k_combined)} selected stocks → {top_k_csv_file.name}")
        else:
            logging.warning("   ⚠️  No Top-K selections to export!")

        logging.info(f"\n✅ CSV export completed! Files saved to: {csv_dir}")

    def _train_walk_forward_sequential(self, model_save_path: str, top_k: int) -> None:
        """Sequential walk-forward training (original implementation)."""
        from src.training.data_processor import DataProcessor

        logging.info("="*80)
        logging.info("🔁 Sequential Mode")
        logging.info("="*80)

        # Initialize predictions cache
        self.predictions_cache = {}

        for idx, period_info in enumerate(self.walk_forward_periods, 1):
            cutoff_date = period_info['cutoff_date']
            train_start_year = period_info['train_start_year']
            period_name = period_info['period_name']

            logging.info(f"\n{'='*80}")
            logging.info(f"📅 Period {idx}/{len(self.walk_forward_periods)}: {cutoff_date.strftime('%Y-%m-%d')} ({period_name})")
            logging.info(f"{'='*80}")

            # Step 1: Load training data until cutoff_date
            logging.info(f"📂 Loading training data: {train_start_year} ~ {cutoff_date.strftime('%Y-%m-%d')}")
            train_df = self._load_data_until_cutoff(train_start_year, cutoff_date)

            if train_df.empty:
                logging.warning(f"⚠️  No training data available for {cutoff_date}, skipping...")
                continue

            logging.info(f"✅ Loaded {len(train_df)} training samples")

            # Step 2: Train models on this data
            logging.info("🔧 Training models...")
            models_info = self._train_models_for_period(train_df)

            # Step 3: Load prediction data (stocks at cutoff_date)
            logging.info(f"📂 Loading prediction data for {cutoff_date.strftime('%Y-%m-%d')}")
            pred_df = self._load_data_for_prediction(cutoff_date)

            if pred_df.empty:
                logging.warning(f"⚠️  No prediction data available for {cutoff_date}, skipping...")
                continue

            logging.info(f"✅ Loaded {len(pred_df)} stocks for prediction")

            # Step 4: Generate predictions
            logging.info("🔮 Generating predictions...")
            predictions_df = self._generate_predictions_for_period(pred_df, models_info)

            # Step 5: Top-K selection and ranking
            logging.info(f"🎯 Selecting Top-{top_k} stocks...")
            predictions_df = self._rank_and_select_top_k(predictions_df, top_k)

            # Store in cache
            cache_key = cutoff_date.strftime('%Y-%m-%d')
            top_k_mask = predictions_df[DataSchema.SELECTED]
            self.predictions_cache[cache_key] = {
                'predictions_df': predictions_df,
                'top_k_selected': predictions_df[top_k_mask]['symbol'].tolist(),
                'top_k_details': predictions_df[top_k_mask].copy(),
                'models_used': models_info,
                'train_samples': len(train_df),
                'predict_samples': len(pred_df)
            }

            logging.info(f"✅ Top-{top_k}: {self.predictions_cache[cache_key]['top_k_selected']}")

        # Save predictions cache to pkl
        eval_config = self.conf.get('EVALUATION', {})
        cache_file = model_save_path + eval_config.get('PREDICTIONS_CACHE_FILE', 'regressor_predictions.pkl')
        logging.info(f"\n💾 Saving predictions cache to {cache_file}")
        joblib.dump(self.predictions_cache, cache_file)
        logging.info(f"✅ Walk-forward training completed! {len(self.predictions_cache)} periods saved.")

        # Save predictions cache to CSV (user-friendly format)
        self._save_predictions_to_csv(model_save_path)
        logging.info(f"✅ CSV exports completed!")

    def _train_walk_forward_parallel(self, model_save_path: str, top_k: int, num_workers_config: str, profile_file: str) -> None:
        """Parallel walk-forward training using Ray and MemoryProfiler."""
        from src.scripts.memory_profiler import MemoryProfiler

        logging.info("="*80)
        logging.info("⚡ Parallel Mode (Ray)")
        logging.info("="*80)

        # Initialize MemoryProfiler
        profiler = MemoryProfiler(profile_path=profile_file)

        # Build config for profiler
        profiler_config = {
            'train_start_year': self.conf.get('EVALUATION', {}).get('TRAIN_START_YEAR'),
            'num_periods': len(self.walk_forward_periods),
            'use_sector': self.use_sector_model,
            'use_classifier': self.use_classifier,
            'top_k': top_k
        }

        # Determine number of workers
        if num_workers_config == 'auto':
            worker_info = profiler.get_optimal_workers(profiler_config)
            num_workers = worker_info['recommended_workers']
            confidence = worker_info['confidence']
            reason = worker_info['reason']

            logging.info(f"🤖 Auto-determined workers: {num_workers}")
            logging.info(f"   Confidence: {confidence}")
            logging.info(f"   Reason: {reason}")
        else:
            try:
                num_workers = int(num_workers_config)
                logging.info(f"👤 Manual workers: {num_workers}")
            except ValueError:
                logging.warning(f"⚠️  Invalid NUM_WORKERS value '{num_workers_config}', defaulting to 1")
                num_workers = 1

        # Cap workers at number of periods
        num_workers = min(num_workers, len(self.walk_forward_periods))

        logging.info(f"🚀 Starting parallel training with {num_workers} workers...")

        # Start profiling
        profiler.start_run(profiler_config, num_workers)

        try:
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=True)
                logging.info("✅ Ray initialized")

            # Submit all periods as Ray tasks
            logging.info(f"📤 Submitting {len(self.walk_forward_periods)} tasks to Ray...")
            futures = []
            for idx, period_info in enumerate(self.walk_forward_periods):
                future = train_single_period_remote.remote(
                    period_info=period_info,
                    root_path=self.root_path,
                    use_classifier=self.use_classifier,
                    use_sector_model=self.use_sector_model,
                    top_k=top_k,
                    logger_level=logging.INFO
                )
                futures.append(future)

                # Sample memory for first few periods
                if idx < 3:
                    import time
                    time.sleep(2)  # Give worker time to start
                    train_samples = len(self._load_data_until_cutoff(
                        period_info['train_start_year'],
                        period_info['cutoff_date']
                    ))
                    profiler.record_period_memory(idx, train_samples)

            # Wait for all results
            logging.info(f"⏳ Waiting for {len(futures)} tasks to complete...")
            results = ray.get(futures)

            # Build predictions cache from results
            self.predictions_cache = {}
            for result in results:
                if len(result) == 3:
                    cache_key, cache_entry, error_msg = result
                else:
                    cache_key, cache_entry = result[0], result[1]
                    error_msg = None
                if cache_entry is not None:
                    self.predictions_cache[cache_key] = cache_entry
                    logging.info(f"✅ Period {cache_key}: {len(cache_entry['top_k_selected'])} stocks selected")
                else:
                    if error_msg:
                        logging.warning(f"⚠️  Period {cache_key}: Failed - {error_msg}")
                    else:
                        logging.warning(f"⚠️  Period {cache_key}: No results (empty data or error)")

            # Save predictions cache
            eval_config = self.conf.get('EVALUATION', {})
            cache_file = model_save_path + eval_config.get('PREDICTIONS_CACHE_FILE', 'regressor_predictions.pkl')
            logging.info(f"\n💾 Saving predictions cache to {cache_file}")
            joblib.dump(self.predictions_cache, cache_file)
            logging.info(f"✅ Parallel walk-forward training completed! {len(self.predictions_cache)} periods saved.")

            # Save predictions cache to CSV (user-friendly format)
            self._save_predictions_to_csv(model_save_path)
            logging.info(f"✅ CSV exports completed!")

            # End profiling (success)
            profiler.end_run(success=True)

        except ray.exceptions.OutOfMemoryError as e:
            logging.error(f"❌ Out of Memory Error: {e}")
            logging.error("🔧 Profile will be invalidated. Next run will use conservative settings.")

            # End profiling (OOM failure)
            profiler.end_run(success=False, error="OutOfMemoryError")

            # Fall back to sequential mode
            logging.info("⚠️  Falling back to sequential mode...")
            self._train_walk_forward_sequential(model_save_path, top_k)

        except Exception as e:
            logging.error(f"❌ Parallel training failed: {e}")
            import traceback
            traceback.print_exc()

            # End profiling (failure)
            profiler.end_run(success=False, error=str(e))

            # Fall back to sequential mode
            logging.info("⚠️  Falling back to sequential mode...")
            self._train_walk_forward_sequential(model_save_path, top_k)

        finally:
            # Cleanup Ray resources
            if ray.is_initialized():
                ray.shutdown()
                logging.info("🧹 Ray shutdown")

    def _load_data_until_cutoff(self, train_start_year: int, cutoff_date: datetime) -> pd.DataFrame:
        """
        Load training data from train_start_year until cutoff_date.

        This implements the expanding window approach where each iteration
        uses all data from the start until the current cutoff date.

        Parameters:
        -----------
        train_start_year : int
            Starting year for training data
        cutoff_date : datetime
            Cutoff date (exclusive) - data before this date is used for training

        Returns:
        --------
        pd.DataFrame
            Training data with all features and targets
        """
        from src.training.data_processor import DataProcessor

        # Load all quarterly data files from train_start_year onwards
        all_train_data = []
        current_year = train_start_year

        # ✅ FIX: Use correct path where make_mldata.py saves training data
        ml_data_dir = f"{self.root_path}/processed/ml_data/per_year"

        while current_year <= cutoff_date.year:
            # ✅ FIX: Use correct file pattern - rnorm_ml_YYYY_QN.parquet
            # rnorm_ml_ files contain features + targets (for training)
            file_pattern = f"{ml_data_dir}/rnorm_ml_{current_year}_Q*.parquet"
            year_files = glob.glob(file_pattern)

            for file_path in sorted(year_files):
                try:
                    df = pd.read_parquet(file_path)
                    if 'rebalance_date' in df.columns:
                        df['rebalance_date'] = pd.to_datetime(df['rebalance_date'])
                        # Only include data before cutoff_date
                        df = df[df['rebalance_date'] < cutoff_date]
                        if not df.empty:
                            all_train_data.append(df)
                except Exception as e:
                    logging.warning(f"⚠️  Failed to load {file_path}: {e}")

            current_year += 1

        if not all_train_data:
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_train_data, ignore_index=True)
        logging.info(f"   Loaded {len(combined_df)} samples from {len(all_train_data)} files")

        return combined_df

    def _load_data_for_prediction(self, cutoff_date: datetime) -> pd.DataFrame:
        """
        Load data for prediction at the specific cutoff_date.

        This loads the stocks available at cutoff_date for making predictions.

        Parameters:
        -----------
        cutoff_date : datetime
            The rebalancing date for which to make predictions

        Returns:
        --------
        pd.DataFrame
            Prediction data with all features
        """
        # Load data files that might contain the cutoff_date
        year = cutoff_date.year

        # ✅ FIX: Use correct path and file pattern
        ml_data_dir = f"{self.root_path}/processed/ml_data/per_year"
        # rnorm_fs_ files contain features only (no targets, for prediction)
        file_pattern = f"{ml_data_dir}/rnorm_fs_{year}_Q*.parquet"
        year_files = glob.glob(file_pattern)

        pred_data = []
        for file_path in sorted(year_files):
            try:
                df = pd.read_parquet(file_path)
                if 'rebalance_date' in df.columns:
                    df['rebalance_date'] = pd.to_datetime(df['rebalance_date'])
                    # Only select stocks at exactly this cutoff_date
                    df = df[df['rebalance_date'] == cutoff_date]
                    if not df.empty:
                        pred_data.append(df)
            except Exception as e:
                logging.warning(f"⚠️  Failed to load {file_path}: {e}")

        if not pred_data:
            return pd.DataFrame()

        # Combine all prediction data
        combined_df = pd.concat(pred_data, ignore_index=True)
        logging.info(f"   Found {len(combined_df)} stocks at {cutoff_date.strftime('%Y-%m-%d')}")

        return combined_df

    def _train_models_for_period(self, train_df: pd.DataFrame) -> dict:
        """
        Train all models (classifiers + regressors) for a specific period.

        This method trains models in-memory without saving to disk.
        The trained models are returned in a dict for immediate use in prediction.

        Parameters:
        -----------
        train_df : pd.DataFrame
            Training data for this period

        Returns:
        --------
        dict
            Dictionary containing trained models:
            {
                'use_classifier': bool,
                'use_sector': bool,
                'classifiers': {...},  # Global classifiers (if use_classifier=Y)
                'regressors': {...},   # Global regressors
                'sector_classifiers': {...},  # Sector classifiers (if use_sector=Y)
                'sector_regressors': {...}    # Sector regressors (if use_sector=Y)
            }
        """
        from src.training.data_processor import DataProcessor

        models_info = {
            'use_classifier': self.use_classifier,
            'use_sector': self.use_sector_model,
            'classifiers': {},
            'regressors': {},
            'sector_classifiers': {},
            'sector_regressors': {}
        }

        # Preprocess training data
        # ⚠️ TODO: This call is incorrect - preprocess_training_data expects (X, y, y_cls, config, logger)
        # For now, commenting out to avoid errors. Proper preprocessing happens later.
        # train_df = DataProcessor.preprocess_training_data(train_df, None)

        # Get target column
        target_col = DataSchema.REGRESSION_TARGET
        if target_col not in train_df.columns:
            logging.error(f"❌ Target column '{target_col}' not found in training data")
            return models_info

        # Get feature columns
        feature_cols = DataSchema.get_feature_cols(train_df)
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        # Train global models
        if self.use_sector_model:
            logging.info("   Training sector-specific models...")
            # Train sector models
            for sector in train_df['sector'].unique():
                sector_data = train_df[train_df['sector'] == sector]
                if len(sector_data) < 50:  # Skip sectors with too few samples
                    continue

                X_sector = sector_data[feature_cols]
                y_sector = sector_data[target_col]

                # ✅ CRITICAL FIX: Preprocess sector data BEFORE training
                # This removes inf/NaN and applies winsorization
                # Without this, XGBoost fails with "Input data contains inf or value too large"
                try:
                    X_sector_clean, y_sector_clean, _, _ = DataProcessor.preprocess_training_data(
                        X_sector.copy(),
                        y_sector.to_frame(),  # Convert Series to DataFrame
                        y_cls=None,
                        config=self.conf,  # ✅ Use self.conf (not self.config)
                        logger=logging
                    )

                    # Convert y back to Series (preprocess_training_data returns DataFrame)
                    y_sector_clean = y_sector_clean.iloc[:, 0]

                except Exception as e:
                    logging.error(f"   ❌ {sector}: Preprocessing failed - {str(e)}")
                    continue

                # Train sector classifiers
                if self.use_classifier:
                    sector_clfs = self._train_sector_classifiers_inline(X_sector_clean, y_sector_clean)
                    models_info['sector_classifiers'][sector] = sector_clfs

                # Train sector regressors
                sector_regs = self._train_sector_regressors_inline(X_sector_clean, y_sector_clean)
                models_info['sector_regressors'][sector] = sector_regs

            logging.info(f"   ✅ Trained models for {len(models_info['sector_regressors'])} sectors")
        else:
            logging.info("   Training global models...")

            # ✅ CRITICAL FIX: Preprocess global data BEFORE training
            # This removes inf/NaN and applies winsorization
            try:
                X_train_clean, y_train_clean, _, _ = DataProcessor.preprocess_training_data(
                    X_train.copy(),
                    y_train.to_frame(),  # Convert Series to DataFrame
                    y_cls=None,
                    config=self.conf,  # ✅ Use self.conf (not self.config)
                    logger=logging
                )

                # Convert y back to Series
                y_train_clean = y_train_clean.iloc[:, 0]

            except Exception as e:
                logging.error(f"   ❌ Global model preprocessing failed - {str(e)}")
                return models_info

            # Train global classifiers
            if self.use_classifier:
                global_clfs = self._train_global_classifiers_inline(X_train_clean, y_train_clean)
                models_info['classifiers'] = global_clfs

            # Train global regressors
            global_regs = self._train_global_regressors_inline(X_train_clean, y_train_clean)
            models_info['regressors'] = global_regs

            logging.info(f"   ✅ Trained {len(models_info['classifiers'])} classifiers + {len(models_info['regressors'])} regressors")

        return models_info

    def _generate_predictions_for_period(self, pred_df: pd.DataFrame, models_info: dict) -> pd.DataFrame:
        """
        Generate predictions for all stocks in pred_df using trained models.

        Parameters:
        -----------
        pred_df : pd.DataFrame
            DataFrame containing stocks to predict (with features)
        models_info : dict
            Dictionary containing trained models from _train_models_for_period()

        Returns:
        --------
        pd.DataFrame
            Predictions with columns: symbol, sector, pred_return, pred_proba, ml_score
        """
        from src.training.data_processor import DataProcessor

        # Preprocess prediction data
        # ⚠️ TODO: This call is incorrect - preprocess_training_data expects (X, y, y_cls, config, logger)
        # For now, commenting out to avoid errors. Proper preprocessing happens later.
        # pred_df = DataProcessor.preprocess_training_data(pred_df, None)

        # Get feature columns
        feature_cols = DataSchema.get_feature_cols(pred_df)
        X_pred = pred_df[feature_cols]

        # Initialize prediction columns
        pred_df[DataSchema.PRED_RETURN] = 0.0
        pred_df[DataSchema.PRED_PROBA] = 1.0  # Default to 1.0 if no classifier

        # Generate predictions based on model type
        if models_info['use_sector']:
            # Sector-specific predictions
            for sector in pred_df['sector'].unique():
                sector_mask = pred_df['sector'] == sector
                X_sector = pred_df.loc[sector_mask, feature_cols]

                if sector not in models_info['sector_regressors']:
                    continue  # Skip if no model for this sector

                # Predict returns
                sector_regs = models_info['sector_regressors'][sector]
                y_pred_return = np.mean([reg.predict(X_sector) for reg in sector_regs.values()], axis=0)
                pred_df.loc[sector_mask, DataSchema.PRED_RETURN] = y_pred_return

                # Predict probabilities (if classifier enabled)
                if models_info['use_classifier'] and sector in models_info['sector_classifiers']:
                    sector_clfs = models_info['sector_classifiers'][sector]
                    y_pred_proba = np.mean([clf.predict_proba(X_sector)[:, 1] for clf in sector_clfs.values()], axis=0)
                    pred_df.loc[sector_mask, DataSchema.PRED_PROBA] = y_pred_proba
        else:
            # Global predictions
            # Predict returns
            global_regs = models_info['regressors']
            y_pred_return = np.mean([reg.predict(X_pred) for reg in global_regs.values()], axis=0)
            pred_df[DataSchema.PRED_RETURN] = y_pred_return

            # Predict probabilities (if classifier enabled)
            if models_info['use_classifier']:
                global_clfs = models_info['classifiers']
                y_pred_proba = np.mean([clf.predict_proba(X_pred)[:, 1] for clf in global_clfs.values()], axis=0)
                pred_df[DataSchema.PRED_PROBA] = y_pred_proba

        # Calculate ML score
        pred_df[DataSchema.ML_SCORE] = pred_df[DataSchema.PRED_PROBA] * pred_df[DataSchema.PRED_RETURN]

        # Select relevant columns for output
        output_cols = ['symbol', 'sector', DataSchema.PRED_RETURN, DataSchema.PRED_PROBA, DataSchema.ML_SCORE]
        # Add company_name if available
        if DataSchema.COMPANY_NAME in pred_df.columns:
            output_cols.insert(2, DataSchema.COMPANY_NAME)

        predictions_df = pred_df[output_cols].copy()

        return predictions_df

    def _rank_and_select_top_k(self, predictions_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        Rank stocks by ml_score and select top K.

        Parameters:
        -----------
        predictions_df : pd.DataFrame
            DataFrame with predictions (must contain ml_score column)
        top_k : int
            Number of top stocks to select

        Returns:
        --------
        pd.DataFrame
            Predictions with added columns: rank, selected
        """
        # Sort by ml_score descending
        predictions_df = predictions_df.sort_values(DataSchema.ML_SCORE, ascending=False).reset_index(drop=True)

        # Add rank (1-indexed)
        predictions_df[DataSchema.RANK] = range(1, len(predictions_df) + 1)

        # Select top K
        predictions_df[DataSchema.SELECTED] = predictions_df[DataSchema.RANK] <= top_k

        logging.info(f"   Ranked {len(predictions_df)} stocks, selected top {top_k}")

        return predictions_df

    def _temporal_train_val_split(self, X: pd.DataFrame, y: pd.Series, val_ratio: float = 0.2) -> tuple:
        """
        Split data into train/validation sets based on temporal order.

        For time series data, we use the last val_ratio of data as validation
        to simulate forward testing (training on past, validating on future).

        Args:
            X: Feature DataFrame
            y: Target Series
            val_ratio: Proportion of data to use for validation (default: 0.2)

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # ✅ FIX: Reset indices to ensure .iloc works correctly
        # When X and y come from filtered DataFrames (e.g., sector filtering),
        # their indices might be non-contiguous (e.g., [100, 200, 300, ...])
        # .iloc uses positional indexing, so we need 0-based contiguous indices
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # ✅ FIX: Remove NaN values in target before splitting
        # NaN in target means no future price data (delisting, bankruptcy, etc.)
        # XGBoost/LightGBM cannot handle NaN in labels
        nan_mask = y.isna()
        if nan_mask.any():
            n_nan = nan_mask.sum()
            n_total = len(y)
            pct_nan = (n_nan / n_total) * 100
            logging.warning(f"   ⚠️  Found {n_nan} NaN in target ({pct_nan:.2f}%) - removing these rows")

            # Remove rows with NaN in target
            valid_mask = ~nan_mask
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)

            logging.info(f"   After NaN removal: {len(y)} samples ({n_total - n_nan} valid)")

        split_idx = int(len(X) * (1 - val_ratio))

        X_train_split = X.iloc[:split_idx]
        X_val_split = X.iloc[split_idx:]
        y_train_split = y.iloc[:split_idx]
        y_val_split = y.iloc[split_idx:]

        logging.info(f"   Temporal split: Train={len(X_train_split)}, Val={len(X_val_split)}")

        return X_train_split, X_val_split, y_train_split, y_val_split

    def _train_global_classifiers_inline(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train global classifiers with early stopping validation.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Preprocessed feature data (inf/NaN already removed by DataProcessor)
        y_train : pd.Series
            Preprocessed target data (inf/NaN already removed by DataProcessor)
        """
        classifiers = {}

        # ✅ Data is already preprocessed by DataProcessor.preprocess_training_data()
        # - inf removed
        # - NaN removed
        # - Winsorization applied
        # Just need to convert to binary (0/1)
        y_regression = y_train

        # Convert clean data to binary (0/1)
        # No need to check NaN again - already removed in preprocessing
        y_binary = (y_regression > 0).astype(int)

        # ✅ ADD: Temporal train/val split for early stopping
        X_tr, X_val, y_tr, y_val = self._temporal_train_val_split(X_train, y_binary, val_ratio=0.2)

        # Train 4 classifiers with early stopping
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        # ✅ CHANGE: Increase n_estimators, add early_stopping_rounds
        classifiers[0] = XGBClassifier(max_depth=8, n_estimators=500, random_state=42, early_stopping_rounds=50)
        classifiers[1] = XGBClassifier(max_depth=9, n_estimators=500, random_state=42, early_stopping_rounds=50)
        classifiers[2] = XGBClassifier(max_depth=10, n_estimators=500, random_state=42, early_stopping_rounds=50)
        classifiers[3] = LGBMClassifier(max_depth=8, n_estimators=500, random_state=42, early_stopping_rounds=50)

        for idx, clf in classifiers.items():
            # ✅ ADD: Fit with validation set for early stopping
            if isinstance(clf, LGBMClassifier):
                clf.fit(X_tr, y_tr,
                       eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
                best_iter = clf.best_iteration_ if hasattr(clf, 'best_iteration_') else len(clf.evals_result_['valid_0']['binary_logloss'])
                val_score = clf.evals_result_['valid_0']['binary_logloss'][best_iter - 1]
                logging.info(f"   Classifier {idx} (LGBM): best_iter={best_iter}, val_logloss={val_score:.4f}")
            else:  # XGBoost
                clf.fit(X_tr, y_tr,
                       eval_set=[(X_val, y_val)],
                       verbose=False)
                best_iter = clf.best_iteration
                val_score = clf.evals_result()['validation_0']['logloss'][best_iter]
                logging.info(f"   Classifier {idx} (XGB): best_iter={best_iter}, val_logloss={val_score:.4f}")

        return classifiers

    def _train_global_regressors_inline(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train global regressors with early stopping validation."""
        regressors = {}

        # ✅ ADD: Temporal train/val split for early stopping
        X_tr, X_val, y_tr, y_val = self._temporal_train_val_split(X_train, y_train, val_ratio=0.2)

        from xgboost import XGBRegressor

        # ✅ CHANGE: Increase n_estimators, add early_stopping_rounds
        regressors[0] = XGBRegressor(max_depth=8, n_estimators=500, random_state=42, early_stopping_rounds=50)
        regressors[1] = XGBRegressor(max_depth=10, n_estimators=500, random_state=42, early_stopping_rounds=50)

        for idx, reg in regressors.items():
            # ✅ ADD: Fit with validation set for early stopping
            reg.fit(X_tr, y_tr,
                   eval_set=[(X_val, y_val)],
                   verbose=False)
            best_iter = reg.best_iteration
            val_score = reg.evals_result()['validation_0']['rmse'][best_iter]
            logging.info(f"   Regressor {idx} (XGB): best_iter={best_iter}, val_rmse={val_score:.4f}")

        return regressors

    def _train_sector_classifiers_inline(self, X_sector: pd.DataFrame, y_sector: pd.Series) -> dict:
        """Train sector classifiers with early stopping validation.

        Parameters:
        -----------
        X_sector : pd.DataFrame
            Preprocessed feature data (inf/NaN already removed by DataProcessor)
        y_sector : pd.Series
            Preprocessed target data (inf/NaN already removed by DataProcessor)
        """
        classifiers = {}

        # ✅ Data is already preprocessed by DataProcessor.preprocess_training_data()
        # - inf removed
        # - NaN removed
        # - Winsorization applied
        # Just need to convert to binary (0/1)
        y_regression = y_sector

        # Convert clean data to binary (0/1)
        # No need to check NaN again - already removed in preprocessing
        y_binary = (y_regression > 0).astype(int)

        # ✅ ADD: Temporal train/val split for early stopping
        X_tr, X_val, y_tr, y_val = self._temporal_train_val_split(X_sector, y_binary, val_ratio=0.2)

        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        # ✅ CHANGE: Increase n_estimators, add early_stopping_rounds
        classifiers[0] = XGBClassifier(max_depth=8, n_estimators=500, random_state=42, early_stopping_rounds=50)
        classifiers[1] = XGBClassifier(max_depth=9, n_estimators=500, random_state=42, early_stopping_rounds=50)
        classifiers[2] = XGBClassifier(max_depth=10, n_estimators=500, random_state=42, early_stopping_rounds=50)
        classifiers[3] = LGBMClassifier(max_depth=8, n_estimators=500, random_state=42, early_stopping_rounds=50)

        for idx, clf in classifiers.items():
            # ✅ ADD: Fit with validation set for early stopping
            if isinstance(clf, LGBMClassifier):
                clf.fit(X_tr, y_tr,
                       eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
                best_iter = clf.best_iteration_ if hasattr(clf, 'best_iteration_') else len(clf.evals_result_['valid_0']['binary_logloss'])
                val_score = clf.evals_result_['valid_0']['binary_logloss'][best_iter - 1]
                logging.info(f"   Sector Classifier {idx} (LGBM): best_iter={best_iter}, val_logloss={val_score:.4f}")
            else:  # XGBoost
                clf.fit(X_tr, y_tr,
                       eval_set=[(X_val, y_val)],
                       verbose=False)
                best_iter = clf.best_iteration
                val_score = clf.evals_result()['validation_0']['logloss'][best_iter]
                logging.info(f"   Sector Classifier {idx} (XGB): best_iter={best_iter}, val_logloss={val_score:.4f}")

        return classifiers

    def _train_sector_regressors_inline(self, X_sector: pd.DataFrame, y_sector: pd.Series) -> dict:
        """Train sector regressors with early stopping validation."""
        regressors = {}

        # ✅ ADD: Temporal train/val split for early stopping
        X_tr, X_val, y_tr, y_val = self._temporal_train_val_split(X_sector, y_sector, val_ratio=0.2)

        from xgboost import XGBRegressor

        # ✅ CHANGE: Increase n_estimators, add early_stopping_rounds
        regressors[0] = XGBRegressor(max_depth=8, n_estimators=500, random_state=42, early_stopping_rounds=50)
        regressors[1] = XGBRegressor(max_depth=10, n_estimators=500, random_state=42, early_stopping_rounds=50)

        for idx, reg in regressors.items():
            # ✅ ADD: Fit with validation set for early stopping
            reg.fit(X_tr, y_tr,
                   eval_set=[(X_val, y_val)],
                   verbose=False)
            best_iter = reg.best_iteration
            val_score = reg.evals_result()['validation_0']['rmse'][best_iter]
            logging.info(f"   Sector Regressor {idx} (XGB): best_iter={best_iter}, val_rmse={val_score:.4f}")

        return regressors

    def evaluation(self) -> None:
        """학습된 모델을 테스트 데이터로 평가하고 종합 보고서를 생성합니다.

        이 메서드는 2단계 예측 및 앙상블 투표 전략을 구현합니다:

        평가 파이프라인:
            1. 디스크에서 학습된 모델 로드
            2. 각 테스트 기간(분기별):
                a. 4개의 모든 분류기를 실행하여 이진 예측 얻기
                b. 2개의 모든 회귀 모델을 실행하여 가격 변동 크기 얻기
                c. 분류기를 결합하여 앙상블 예측 생성:
                   - prediction_wbinary_0-3: 각 분류기로 개별적으로 필터링
                   - prediction_wbinary_ensemble: 분류기 1 AND 3으로 필터링
                   - prediction_wbinary_ensemble2: 분류기 1 AND 2로 필터링
                   - prediction_wbinary_ensemble3: 다수결 투표 (3개 중 2개 이상 동의)
                d. 모든 예측 변형에 대한 손실 계산
                e. 각 예측 방법에 대해 상위 K개 주식 선택 (K=3, 7, 15)
                f. 상위 K개 선택에 대한 주식당 평균 수익 계산
            3. 결과를 CSV 파일로 저장
            4. 종합 평가 보고서 생성

        모델 조합:
            2개의 회귀 모델 각각에 대해 8개의 예측 변형을 생성합니다:
                1. model_i_prediction: 원시 회귀 출력
                2-5. model_i_prediction_wbinary_0-3: 각 분류기로 필터링
                   (분류기가 하락을 예측하면 예측값을 -1로 설정)
                6. model_i_prediction_wbinary_ensemble: cls1 AND cls3으로 필터링
                7. model_i_prediction_wbinary_ensemble2: cls1 AND cls2로 필터링
                8. model_i_prediction_wbinary_ensemble3: 다수결 투표 필터

            총: 2개 회귀 모델 × 8개 변형 = 16개 예측 방법

        출력 파일 (MODEL_SAVE_PATH에 저장):
            - prediction_ai_{date}.csv: 각 테스트 기간의 예측
            - prediction_ai.csv: 모든 예측 연결
            - pred_df_topk.csv: 모든 모델의 상위 K개 평가 메트릭
            - prediction_{date}_{model}_{col}_top{s}-{e}.csv: 모델당 상위 K개 주식

        참고:
            - 분류기 확률을 이진으로 변환하기 위해 THRESHOLD (기본값 92) 사용
            - 상위 8%의 주식 (100-92=8%)이 양성으로 예측됨
            - PER_SECTOR=True인 경우 섹터별 모델도 평가합니다
        """
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # 학습된 분류 모델 로드
        self.models = dict()
        self.clsmodels = dict()
        self._load_classifiers(MODEL_SAVE_PATH)
        self._load_regressors(MODEL_SAVE_PATH)

        # Load threshold configs
        threshold_config, THRESHOLD_PERCENTILE = self._load_threshold_config(MODEL_SAVE_PATH)
        sector_threshold_configs = self._load_sector_threshold_configs(MODEL_SAVE_PATH)

        # 통합 메서드로 예측 컬럼 이름 생성
        pred_col_list = self._build_prediction_column_names()

        # Evaluate global models across all test periods
        full_df, model_eval_hist = self._evaluate_global_models(
            MODEL_SAVE_PATH, threshold_config, THRESHOLD_PERCENTILE, pred_col_list
        )

        # Generate global evaluation report
        col_name = ['start_date', 'model', 'krange', 'avg_earning_per_stock', 'cur_model_pred',
                   'loss_y_and_pred', 'cur_model_pred_ispositive', 'avg_pred', 'model0_pred',
                   'model1_pred', 'model0_pred_wbinary_0', 'model1_pred_wbinary_0',
                   'model0_pred_wbinary_1', 'model1_pred_wbinary_1', 'model0_pred_wbinary_2',
                   'model1_pred_wbinary_2', 'model0_pred_wbinary_3', 'model1_pred_wbinary_3',
                   'model0_pred_wbinary_ensemble', 'model1_pred_wbinary_ensemble',
                   'model0_pred_wbinary_ensemble2', 'model1_pred_wbinary_ensemble2',
                   'model0_pred_wbinary_ensemble3', 'model1_pred_wbinary_ensemble3']

        pred_df = pd.DataFrame(model_eval_hist, columns=col_name)
        logging.info(pred_df)
        pred_df.to_csv(MODEL_SAVE_PATH+'pred_df_topk.csv', index=False)
        full_df.to_csv(MODEL_SAVE_PATH+'prediction_ai.csv', index=False)

        # Evaluate sector models (if enabled)
        if self.use_sector_model:
            self._evaluate_sector_models(
                MODEL_SAVE_PATH, threshold_config, THRESHOLD_PERCENTILE, sector_threshold_configs
            )

    def _load_threshold_config(self, model_save_path: str) -> tuple:
        """Load threshold configuration from saved file.

        Args:
            model_save_path: Directory path where model files are saved.

        Returns:
            Tuple of (threshold_config dict, THRESHOLD_PERCENTILE int).
        """
        threshold_config_file = model_save_path + 'threshold_config.pkl'
        try:
            threshold_config = joblib.load(threshold_config_file)
            THRESHOLD_PERCENTILE = threshold_config['percentile']
            logging.info("="*80)
            logging.info("📂 LOADED THRESHOLD CONFIG")
            logging.info("="*80)
            logging.info(f"   Percentile: {THRESHOLD_PERCENTILE}")
            logging.info(f"   Threshold value: {threshold_config.get('threshold_value', 'N/A'):.3f}" if 'threshold_value' in threshold_config else "")
            if 'precision' in threshold_config:
                logging.info(f"   Precision: {threshold_config['precision']:.3f}")
                logging.info(f"   Selected stocks: {threshold_config['n_selected']:,}")
            logging.info("="*80)
            logging.info("")
        except FileNotFoundError:
            # Fallback to config or hardcoded value
            ml_config = self.conf.get('ML', {})
            THRESHOLD_PERCENTILE = int(ml_config.get('CLASSIFIER_THRESHOLD_PERCENTILE', 92))
            logging.warning(f"⚠️  Threshold config not found, using default: {THRESHOLD_PERCENTILE}")
            threshold_config = {'percentile': THRESHOLD_PERCENTILE}

        return threshold_config, THRESHOLD_PERCENTILE

    def _load_sector_threshold_configs(self, model_save_path: str) -> dict:
        """Load sector-specific threshold configurations.

        Args:
            model_save_path: Directory path where model files are saved.

        Returns:
            Dictionary mapping sector names to their threshold configs.
        """
        sector_threshold_configs = {}
        if self.use_sector_model and self.use_classifier:
            sector_threshold_file = model_save_path + 'sector_threshold_configs.pkl'
            try:
                sector_threshold_configs = joblib.load(sector_threshold_file)
                logging.info(f"✅ Loaded sector threshold configs for {len(sector_threshold_configs)} sectors")
            except FileNotFoundError:
                logging.warning("⚠️  Sector threshold configs not found, will use global threshold")

        return sector_threshold_configs

    def _evaluate_global_models(self, model_save_path: str, threshold_config: dict,
                                threshold_percentile: int, pred_col_list: list) -> tuple:
        """Evaluate global models across all test periods.

        Runs classifiers and regressors on each test period, generates ensemble
        predictions, and evaluates top-K stock selection.

        Args:
            model_save_path: Directory path for saving evaluation results.
            threshold_config: Threshold configuration dict.
            threshold_percentile: Percentile for classifier threshold.
            pred_col_list: List of prediction column names.

        Returns:
            Tuple of (full_df DataFrame with all predictions, model_eval_hist list).
        """
        model_eval_hist = []  # 모든 기간의 평가 결과 저장
        full_df = pd.DataFrame()  # 예측이 포함된 모든 테스트 데이터 누적

        # 각 테스트 기간을 개별적으로 평가
        for test_idx, (testdate, df) in enumerate(self.test_df_list):

            logging.info("evaluation date : ")
            # 파일 경로에서 날짜 추출 (통합 유틸리티 메서드 사용)
            tdate = self._extract_date_from_filepath(testdate)
            filename = os.path.basename(testdate)

            self.logger.debug(f"in test loop filename : {filename}")
            self.logger.debug(f"in test loop tdate : {tdate}")

            # 이 테스트 기간의 특성과 레이블 준비
            x_test = df[df.columns.difference(y_col_list)]
            y_test = df[['price_dev_subavg']]
            y_test_cls = df[['price_dev']]
            # ✅ REFACTORED: Use DataProcessor for binary target
            y_test_binary = DataProcessor.create_binary_target(y_test_cls, config=self.conf, logger=logging.getLogger())

            df['label'] = y_test  # 실제 가격 변동
            df['label_binary'] = y_test_binary  # 실제 이진 레이블

            # LightGBM을 위해 특성 이름 정리
            x_test = self.clean_feature_names(x_test)

            # ===== Feature Alignment (피처 정렬) =====
            # 학습 시 사용한 피처 리스트 로드
            feature_columns_file = model_save_path + 'feature_columns.pkl'
            try:
                train_feature_columns = joblib.load(feature_columns_file)
                logging.info(f"✅ Loaded {len(train_feature_columns)} train feature columns")
                logging.info(f"   Test data has {len(x_test.columns)} features")

                # 누락된 피처는 NaN으로 채우기 (XGBoost가 처리)
                missing_features = set(train_feature_columns) - set(x_test.columns)
                if missing_features:
                    logging.warning(f"   ⚠️  {len(missing_features)} features missing in test data, filling with NaN")
                    for col in missing_features:
                        x_test.loc[:, col] = np.nan

                # 추가 피처는 제거
                extra_features = set(x_test.columns) - set(train_feature_columns)
                if extra_features:
                    logging.info(f"   Removing {len(extra_features)} extra features from test data")
                    x_test = x_test.drop(columns=list(extra_features))

                # 피처 순서 맞추기 (중요!)
                x_test = x_test[train_feature_columns]
                logging.info(f"   ✅ Feature alignment complete: {len(x_test.columns)} features")

            except FileNotFoundError:
                logging.warning(f"⚠️  Feature columns file not found: {feature_columns_file}")
                logging.warning("   Proceeding without feature alignment (may cause errors)")

            # ✅ Winsorization: Apply if enabled during training
            # Must match training preprocessing for consistency
            if self.use_winsorization:
                x_test = DataProcessor.winsorize_features(
                    x_test,
                    lower_percentile=0.01,
                    upper_percentile=0.99,
                    enabled=True
                )
                logging.info(f"   ✅ Winsorization applied to test data")

            # ===== NaN 값 처리 (Evaluation 단계 - 개선된 전략) =====
            logging.info(f"🔬 Checking for NaN values in test data ({tdate})...")

            original_test_rows = len(x_test)

            # 1단계: y에 NaN이 있는 행만 제거 (레이블이 없으면 평가 불가)
            nan_mask_y_test = y_test.isna().any(axis=1)
            nan_mask_y_test_cls = y_test_cls.isna().any(axis=1)
            nan_mask_labels_test = nan_mask_y_test | nan_mask_y_test_cls

            if nan_mask_labels_test.sum() > 0:
                logging.warning(f"⚠️  Removing {nan_mask_labels_test.sum()} rows with NaN labels")
                x_test = x_test[~nan_mask_labels_test]
                y_test = y_test[~nan_mask_labels_test]
                y_test_cls = y_test_cls[~nan_mask_labels_test]
                y_test_binary = y_test_binary[~nan_mask_labels_test]
                df = df[~nan_mask_labels_test]

            # 2단계: 나머지 NaN은 그대로 유지 (XGBoost의 missing=np.nan이 처리)
            remaining_nan_count = x_test.isna().sum().sum()
            if remaining_nan_count > 0:
                logging.info(f"   Keeping {remaining_nan_count} NaN values for XGBoost to handle")

            logging.info(f"✅ NaN handling complete: {original_test_rows} → {len(x_test)} rows ({len(x_test)/original_test_rows*100:.1f}% retained)")
            if remaining_nan_count > 0:
                logging.info(f"   Remaining NaN values: {remaining_nan_count} (will be handled by XGBoost)")

            # NaN 제거 완료 후 preds 배열 초기화 (회귀 예측 저장용)
            preds = np.empty((0, x_test.shape[0]))

            # === 분류 단계 ===
            # 4개의 모든 분류기를 실행하고 성능 평가
            for i, model in self.clsmodels.items():
                logging.info(f"classification model # {i}")
                pred_col_name = 'clsmodel_' + str(i) + '_prediction'
                correct_col_name = 'clsmodel_' + str(i) + '_correct'

                # 예측 확률 가져오기 (클래스 1의 확률 = 가격 상승)
                # GPU 지원으로 device mismatch 워닝 방지
                y_probs = predict_proba_with_gpu_support(model, x_test, self.use_gpu_prediction)[:, 1]

                # 백분위수 임계값을 사용하여 확률을 이진 예측으로 변환
                # 학습 시 자동 탐색된 threshold 사용
                threshold = np.percentile(y_probs, threshold_percentile)
                classifier_mode = threshold_config.get('mode', 'positive_screen')

                # 모드별 binary 예측
                if classifier_mode == "negative_screen":
                    y_predict_binary = (y_probs < threshold).astype(int)
                else:  # positive_screen
                    y_predict_binary = (y_probs > threshold).astype(int)

                logging.info(f"20% positive threshold == {threshold}")
                logging.info(classification_report(y_test_binary, y_predict_binary))

                # 분류기 예측 저장
                df[pred_col_name] = y_predict_binary
                df[correct_col_name] = (y_test_binary.values.ravel() == y_predict_binary).astype(int)

                acc = accuracy_score(df['label_binary'], df[pred_col_name])
                logging.info(f"Accuracy for {pred_col_name}: {acc:.4f}")


            # === 회귀 단계 ===
            # 2개의 모든 회귀 모델을 실행하고 앙상블 예측 생성
            # 통합 유틸리티 메서드로 컬럼 이름 가져오기
            classifier_cols = self._get_classifier_column_names()

            # CLASSIFIER_MODE 가져오기 (threshold_config 또는 config에서)
            classifier_mode = threshold_config.get('mode', self.conf.get('ML', {}).get('CLASSIFIER_MODE', 'positive_screen'))

            for i, model in self.models.items():
                # 통합 유틸리티 메서드로 컬럼 이름 가져오기
                reg_cols = self._get_regression_column_names(i)

                # 원시 회귀 예측 가져오기
                # GPU 지원으로 device mismatch 워닝 방지
                y_predict = predict_with_gpu_support(model, x_test, self.use_gpu_prediction)

                # 원시 회귀 예측 저장
                df[reg_cols['prediction']] = y_predict

                # 통합 메서드로 바이너리 필터링 적용 (모드 전달)
                df = self._apply_binary_filtering(df, y_predict, classifier_cols, reg_cols, classifier_mode)

                # 통합 메서드로 앙상블 예측 생성 (모드 전달)
                df = self._create_ensemble_predictions(df, y_predict, classifier_cols, reg_cols, classifier_mode)

                # 평균화를 위한 원시 예측 저장
                preds = np.vstack((preds, y_predict[None,:]))

                # 통합 메서드로 예측 손실 계산
                df = self._calculate_prediction_losses(df, y_predict, reg_cols)

                # 이 기간의 평가 메트릭 로깅
                logging.info(f"eval : model i : {i} loss : {df[reg_cols['loss']].mean()} "
                           f"loss_wbin_0 {df[reg_cols['loss_wbinary_0']].mean()} "
                           f"loss_wbin_1 {df[reg_cols['loss_wbinary_1']].mean()} "
                           f"loss_wbin_2 {df[reg_cols['loss_wbinary_2']].mean()} "
                           f"loss_wbin_3 {df[reg_cols['loss_wbinary_3']].mean()}")

                # 누적 메트릭 로깅 (지금까지의 모든 기간)
                if test_idx != 0:
                    logging.info(f"accumulated eval : model i : {i} "
                               f"loss : {full_df[reg_cols['loss']].mean()} "
                               f"loss_wbin_0 {full_df[reg_cols['loss_wbinary_0']].mean()} "
                               f"loss_wbin_1 {full_df[reg_cols['loss_wbinary_1']].mean()} "
                               f"loss_wbin_2 {full_df[reg_cols['loss_wbinary_2']].mean()} "
                               f"loss_wbin_3 {full_df[reg_cols['loss_wbinary_3']].mean()}")

            # 모든 회귀 모델의 평균 예측 계산
            df['ai_pred_avg'] = np.average(preds, axis=0)
            df['ai_pred_avg_loss'] = abs(df['label']-df['ai_pred_avg'])

            # 결과 누적
            full_df = pd.concat([full_df, df], ignore_index=True)
            df.to_csv(model_save_path + "prediction_ai_{}.csv".format(tdate))

            # === 상위 K개 주식 선택 ===
            # 각 예측 방법에 대해 상위 K개 주식을 선택하고 평균 수익 계산
            topk_period_earning_sums = []
            topk_list = [(0,3), (0,7), (0,15)]  # 상위 3, 7, 15개 주식

            for s, e in topk_list:
                logging.info("top" + str(s) + " ~ "  + str(e) )
                k = str(s) + '~' + str(e)

                # 각 예측 방법 평가
                for col in pred_col_list:
                    # 예측을 기반으로 상위 K개 주식 선택
                    top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]

                    logging.info("")
                    logging.info(col)
                    logging.info(("label"))
                    logging.info((top_k_df['price_dev'].sum()/(e-s+1)))
                    logging.info(("pred"))
                    logging.info((top_k_df[col].sum()/(e-s+1)))
                    topk_period_earning_sums.append(top_k_df['price_dev'].sum())

                    # 상위 K개 주식을 CSV로 저장
                    top_k_df.to_csv(model_save_path+'prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))

                    # 이 모델 및 상위 K개 범위에 대한 평가 메트릭 기록
                    model_eval_hist.append([
                        tdate, col, k,
                        top_k_df['price_dev'].sum()/(e-s+1),  # 주식당 평균 실제 수익
                        top_k_df[col].sum()/(e-s+1),  # 주식당 평균 예측 수익
                        abs(top_k_df[col].sum()/(e-s+1) - top_k_df['price_dev'].sum()/(e-s+1)),  # 손실
                        int(top_k_df[col].sum()/(e-s+1) > 0),  # 예측이 양수인가?
                        top_k_df['ai_pred_avg'].sum()/(e-s+1),
                        top_k_df['model_0_prediction'].sum()/(e-s+1),
                        top_k_df['model_1_prediction'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_0'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_0'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_1'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_1'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_2'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_2'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_3'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_3'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_ensemble'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_ensemble'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_ensemble2'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_ensemble2'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_ensemble3'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_ensemble3'].sum()/(e-s+1)
                    ])

        return full_df, model_eval_hist

    def _evaluate_sector_models(self, model_save_path: str, threshold_config: dict,
                                threshold_percentile: int, sector_threshold_configs: dict) -> None:
        """Evaluate sector-specific models and generate sector evaluation report.

        Args:
            model_save_path: Directory path for saving evaluation results.
            threshold_config: Global threshold configuration dict.
            threshold_percentile: Global percentile for classifier threshold.
            sector_threshold_configs: Sector-specific threshold configs.
        """
        testdates = set()
        allsector_topk_df = pd.DataFrame()
        self.sector_models = dict()
        self.sector_classifiers = dict()

        # 통합 섹터 모델 로딩 메서드 사용
        self._load_sector_models(model_save_path, self.sector_list)
        self._load_sector_classifiers(model_save_path, self.sector_list)

        sector_model_eval_hist = []

        # 각 섹터 및 테스트 기간 평가
        for test_idx, (testdate, df, sec) in enumerate(self.sector_test_df_lists):
            self.logger.debug("sec evaluation date : ")
            # 파일 경로에서 날짜 추출 (통합 유틸리티 메서드 사용)
            tdate = self._extract_date_from_filepath(testdate)
            if tdate == "unknown_period":
                logging.warning(f"⚠️  Skipping sector evaluation due to unknown period: {testdate}")
                continue
            self.logger.debug(f"tdate: {tdate}, sector: {sec}")
            testdates.add(tdate)

            x_test_full = df[df.columns.difference(y_col_list)]
            y_test = df[['price_dev_subavg']]
            y_test_2 = df[['price_dev_subavg']]

            if len(x_test_full) == 0:
                continue

            # ===== Feature Alignment (Unified via DataProcessor) =====
            # Uses DataProcessor.align_features_to_model() to ensure consistency
            # with ml_backtest.py and eliminate code duplication

            # Align for global classifier
            logging.info(f"   Global classifier: Aligning features...")
            x_test_full = DataProcessor.align_features_to_model(
                x_test_full,
                self.clsmodels[2],
                logging.getLogger()
            )

            # Align for sector models
            logging.info(f"   Sector {sec}: Aligning features...")
            x_test_sector = DataProcessor.align_features_to_model(
                x_test_full.copy(),
                self.sector_models[(sec, 0)],
                logging.getLogger()
            )

            sector_preds = np.empty((0, x_test_sector.shape[0]))
            df['label'] = y_test

            # 섹터 기반 필터링을 위해 분류기 2 사용 (GLOBAL classifier - use aligned full features!)
            # GPU 지원으로 device mismatch 워닝 방지
            y_probs = predict_proba_with_gpu_support(self.clsmodels[2], x_test_full, self.use_gpu_prediction)[:, 1]

            # Use sector-specific threshold if available, otherwise use global threshold
            if sec in sector_threshold_configs:
                sector_config = sector_threshold_configs[sec]
                sector_threshold_pct = sector_config['percentile']
                threshold = np.percentile(y_probs, sector_threshold_pct)
                sector_mode = sector_config.get('mode', 'positive_screen')
            else:
                threshold = np.percentile(y_probs, threshold_percentile)
                sector_mode = threshold_config.get('mode', 'positive_screen')

            # 모드별 binary 예측
            if sector_mode == "negative_screen":
                y_predict_binary = (y_probs < threshold).astype(int)
            else:  # positive_screen
                y_predict_binary = (y_probs > threshold).astype(int)

            # 섹터별 모델 실행 (use sector-specific features!)
            for i in range(2):
                k = (sec, i)
                model = self.sector_models[k]
                pred_col_name = 'model_' + str(i) + '_prediction'
                pred_col_name_wbin = 'model_' + str(i) + '_prediction_wbinary_2'
                # GPU 지원으로 device mismatch 워닝 방지
                y_predict = predict_with_gpu_support(model, x_test_sector, self.use_gpu_prediction)
                df[pred_col_name] = y_predict

                df[pred_col_name_wbin] = np.where(y_predict_binary == 0, -1, y_predict)
                self.logger.debug(f"i{i} sec {sec} x_test:{x_test_sector.shape} preds:{sector_preds.shape} y_pred:{y_predict[None,:].shape}")
                sector_preds = np.vstack((sector_preds, y_predict[None,:]))

            df['ai_pred_avg'] = np.average(sector_preds, axis=0)
            df.to_csv(model_save_path+ "sec_{}_prediction_ai_{}.csv".format(sec, tdate))

            # 섹터별 예측의 상위 K개 평가
            # Sector models only have basic predictions (no ensemble variants)
            sector_pred_col_list = [
                'ai_pred_avg',
                'model_0_prediction',
                'model_0_prediction_wbinary_2',
                'model_1_prediction',
                'model_1_prediction_wbinary_2'
            ]
            topk_period_earning_sums = []
            topk_list = [(0,3), (0,7)]
            for s, e in topk_list:
                logging.info("top" + str(s) + " ~ "  + str(e) )
                k = str(s) + '~' + str(e)
                for col in sector_pred_col_list:
                    top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                    logging.info(col)
                    logging.info(("label"))
                    logging.info((top_k_df['price_dev'].sum()/(e-s+1)))
                    logging.info(("pred"))
                    logging.info((top_k_df[col].sum()/(e-s+1)))
                    topk_period_earning_sums.append(top_k_df['price_dev'].sum())
                    top_k_df.to_csv(model_save_path+'prediction_{}_{}_{}_top{}-{}.csv'.format(tdate, sec, col, s, e))
                    top_k_df['start_date'] = tdate
                    top_k_df['col'] = col
                    allsector_topk_df = pd.concat([allsector_topk_df, top_k_df])
                    sector_model_eval_hist.append([
                        tdate, sec, col, k,
                        top_k_df['price_dev'].sum()/(e-s+1),
                        top_k_df[col].sum()/(e-s+1),
                        abs(top_k_df[col].sum()/(e-s+1) - top_k_df['price_dev'].sum()/(e-s+1)),
                        int(top_k_df[col].sum()/(e-s+1) > 0),
                        top_k_df['ai_pred_avg'].sum()/(e-s+1),
                        top_k_df['model_0_prediction'].sum()/(e-s+1),
                        top_k_df['model_1_prediction'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_2'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_2'].sum()/(e-s+1)
                    ])

        col_name = ['start_date', 'sector', 'model', 'krange', 'avg_earning_per_stock',
                   'cur_model_pred', 'loss_y_and_pred', 'cur_model_pred_ispositive',
                   'avg_pred', 'model0_pred', 'model1_pred',
                   'model0_pred_wbinary_2', 'model1_pred_wbinary_2']
        pred_df = pd.DataFrame(sector_model_eval_hist, columns=col_name)
        self.logger.debug(f"pred_df:\n{pred_df}")
        pred_df.to_csv(model_save_path+'allsector_pred_df.csv', index=False)



    def latest_prediction(self) -> None:
        """주식 선택을 위해 가장 최근 데이터로 예측합니다.

        이 메서드는 최신 분기별 데이터를 로드하고 현재 주식 선택을 위한 예측을
        생성합니다. evaluation()과 동일한 2단계 예측 및 앙상블 투표 전략을 따르지만
        과거 테스트 데이터가 아닌 가장 최근 데이터로만 작업합니다.

        파이프라인:
            1. 모든 학습된 분류 및 회귀 모델 로드
            2. 최신 연도 데이터(모든 분기)를 읽고 심볼당 가장 최근 것 유지
            3. 충분한 데이터가 있는 주식으로 필터링 (>60% non-NaN)
            4. 4개의 분류 모델을 실행하여 이진 예측 얻기
            5. 2개의 회귀 모델을 실행하여 가격 변동 크기 얻기
            6. 다양한 투표 전략을 사용하여 앙상블 예측 생성
            7. 상위 K개 주식 추천 생성
            8. 예측을 CSV 파일로 저장

        출력 파일 (MODEL_SAVE_PATH에 저장):
            - latest_prediction.csv: 최신 데이터의 모든 예측
            - latest_prediction_{model}_{col}_top{s}-{e}.csv: 모델당 상위 K개 주식
            - sec_{sector}_latest_prediction.csv: 섹터별 예측 (PER_SECTOR=True인 경우)
            - allsector_latest_pred_df.csv: 섹터 기반 상위 K개 요약 (PER_SECTOR=True인 경우)
        """
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # 통합 모델 로딩 메서드 사용
        self.clsmodels = dict()
        self.models = dict()
        self._load_classifiers(MODEL_SAVE_PATH)
        self._load_regressors(MODEL_SAVE_PATH)

        # Load threshold config
        threshold_config, THRESHOLD_PERCENTILE = self._load_threshold_config(MODEL_SAVE_PATH)

        # Load and prepare latest data
        ldf, ldf_with_sector = self._load_latest_data(MODEL_SAVE_PATH)
        if ldf is None:
            return

        # Run global predictions
        pred_col_list = self._build_prediction_column_names()
        self._predict_latest_global(ldf, MODEL_SAVE_PATH, threshold_config, THRESHOLD_PERCENTILE, pred_col_list)

        # Run sector predictions (if enabled)
        if self.use_sector_model and ldf_with_sector is not None:
            self._predict_latest_sectors(ldf_with_sector, ldf, MODEL_SAVE_PATH)

    def _load_latest_data(self, model_save_path: str) -> tuple:
        """Load and prepare latest quarterly data for prediction.

        Loads the most recent year's data, deduplicates by symbol, applies
        sector categorization, and filters rows with excessive NaN values.

        Args:
            model_save_path: Directory path where models are saved.

        Returns:
            Tuple of (ldf DataFrame without sector, ldf_with_sector DataFrame or None).
            Returns (None, None) if data loading fails.
        """
        aidata_dir = self.root_path + '/processed/ml_data/per_year/'

        # 최신 연도 데이터(모든 분기)를 로드하고 심볼당 가장 최근 것 유지
        # 자동으로 가장 최근 연도 감지 및 Parquet 형식 로드
        import glob

        # 모든 rnorm_fs 파일 찾기
        fs_files = sorted(glob.glob(aidata_dir + 'rnorm_fs_*.parquet'))

        if not fs_files:
            logging.error(f"No rnorm_fs files found in {aidata_dir}")
            logging.error("Cannot generate latest prediction without feature data")
            return None, None

        # 파일명에서 연도 추출하여 가장 최근 연도 찾기
        try:
            years = [int(os.path.basename(f).split('_')[2]) for f in fs_files]
            latest_year = max(years)
            logging.info(f"Latest year detected: {latest_year}")
        except (IndexError, ValueError) as e:
            logging.error(f"Failed to parse year from filenames: {e}")
            return None, None

        # ✅ Use unified data loading (DataProcessor.load_quarterly_data)
        # This ensures consistency with ml_backtest.py and prevents duplicate indices
        try:
            ldf = DataProcessor.load_quarterly_data(
                data_dir=aidata_dir,
                year=latest_year,
                file_prefix='rnorm_fs',
                logger=logging.getLogger()
            )
        except ValueError as e:
            logging.error(f"Failed to load latest data: {e}")
            return None, None

        # year_period를 기준으로 내림차순 정렬하고 심볼당 첫 번째(가장 최근) 유지
        ldf = ldf.sort_values(by='year_period', ascending=False)
        ldf = ldf.drop_duplicates(subset='symbol', keep='first')
        logging.info(f"  After deduplication: {len(ldf)} unique symbols")

        ldf = ldf.drop(columns=self.drop_col_list, errors='ignore')

        # ✅ 섹터 카테고리화 적용 (학습 시와 동일하게)
        # CATEGORIZATION.ENABLED=Y일 때 원본 섹터를 카테고리로 통합
        if self.use_sector_model and 'sector' in ldf.columns:
            ldf = DataProcessor.map_sectors_to_categories(
                ldf,
                self.conf,
                sector_column='sector',
                logger=logging.getLogger()
            )

            # sector를 sector_category로 교체 (원본은 sector_original로 백업)
            ldf['sector_original'] = ldf['sector']
            ldf['sector'] = ldf['sector_category']

        # 섹터 리스트 추출 (NaN, None, 빈 문자열 필터링)
        # 카테고리화가 활성화되었다면 카테고리 이름 사용
        sector_col = 'sector_category' if self.use_sector_model and 'sector_category' in ldf.columns else 'sector'
        all_sectors = list(ldf[sector_col].unique())
        self.sector_list = [
            x for x in all_sectors
            if pd.notna(x) and x is not None and isinstance(x, str) and x.strip()
        ]

        # Log invalid sectors and affected stocks
        invalid_sectors = [
            x for x in all_sectors
            if not (pd.notna(x) and x is not None and isinstance(x, str) and x.strip())
        ]
        if invalid_sectors:
            logging.warning(f"  ⚠️  Found {len(invalid_sectors)} invalid sector value(s) in latest data")
            total_invalid_rows = 0
            for inv_sec in invalid_sectors:
                if pd.isna(inv_sec):
                    count = ldf['sector'].isna().sum()
                    logging.warning(f"     - NaN/None: {count} stocks excluded from sector predictions")
                    total_invalid_rows += count
                else:
                    count = (ldf['sector'] == inv_sec).sum()
                    logging.warning(f"     - '{inv_sec}': {count} stocks excluded from sector predictions")
                    total_invalid_rows += count
            logging.warning(f"  Total stocks excluded from sector predictions: {total_invalid_rows} / {len(ldf)}")

        if len(self.sector_list) == 0:
            logging.warning("⚠️  No valid sectors found in data")
        else:
            logging.info(f"  Found {len(self.sector_list)} valid sectors: {self.sector_list}")

        # 과도한 누락 데이터가 있는 행 필터링 (>60% NaN)
        # ⚠️ IMPORTANT: Apply BEFORE dropping sector column
        self.logger.info(f"before dtable len : {len(ldf)}")
        ldf = DataProcessor.drop_many_nan_row(ldf, threshold=0.6)
        self.logger.info(f"after dtable len : {len(ldf)}")

        # 섹터별 예측을 위해 sector 컬럼이 있는 복사본 저장
        # ⚠️ CRITICAL: Save AFTER NaN row removal to ensure same row count
        ldf_with_sector = ldf.copy() if self.use_sector_model else None

        # sector 컬럼 제거 (global 모델은 sector를 사용하지 않음)
        ldf = ldf.drop('sector', axis=1)

        return ldf, ldf_with_sector

    def _predict_latest_global(self, ldf: pd.DataFrame, model_save_path: str,
                               threshold_config: dict, threshold_percentile: int,
                               pred_col_list: list) -> None:
        """Run global classification and regression predictions on latest data.

        Performs feature alignment, runs all classifiers and regressors,
        generates ensemble predictions, and saves top-K recommendations.

        Args:
            ldf: DataFrame with latest data (sector column removed).
            model_save_path: Directory path for saving predictions.
            threshold_config: Threshold configuration dict.
            threshold_percentile: Percentile for classifier threshold.
            pred_col_list: List of prediction column names.
        """
        # 입력 특성 준비
        input = ldf[ldf.columns.difference(y_col_list)]
        input = self.clean_feature_names(input)

        # ===== Feature Alignment (피처 정렬) =====
        # 학습 시 사용한 피처 리스트 로드 (evaluation()과 동일한 방식)
        feature_columns_file = model_save_path + 'feature_columns.pkl'
        try:
            train_feature_columns = joblib.load(feature_columns_file)
            logging.info(f"✅ Loaded {len(train_feature_columns)} train feature columns")
            logging.info(f"   Latest prediction data has {len(input.columns)} features")

            # 누락된 피처는 NaN으로 채우기 (XGBoost가 처리)
            missing_features = set(train_feature_columns) - set(input.columns)
            if missing_features:
                logging.warning(f"   ⚠️  {len(missing_features)} features missing in prediction data, filling with NaN")
                if len(missing_features) <= 10:
                    for col in missing_features:
                        logging.warning(f"      - {col}")
                else:
                    for col in list(missing_features)[:10]:
                        logging.warning(f"      - {col}")
                    logging.warning(f"      ... and {len(missing_features)-10} more")
                for col in missing_features:
                    input.loc[:, col] = np.nan

            # 추가 피처는 제거
            extra_features = set(input.columns) - set(train_feature_columns)
            if extra_features:
                logging.info(f"   Removing {len(extra_features)} extra features from prediction data")
                if len(extra_features) <= 10:
                    for col in extra_features:
                        logging.info(f"      - {col}")
                else:
                    for col in list(extra_features)[:10]:
                        logging.info(f"      - {col}")
                    logging.info(f"      ... and {len(extra_features)-10} more")
                input = input.drop(columns=list(extra_features))

            # 피처 순서 맞추기 (중요!)
            input = input[train_feature_columns]
            logging.info(f"   ✅ Feature alignment complete: {len(input.columns)} features")

        except FileNotFoundError:
            logging.error(f"❌ Feature columns file not found: {feature_columns_file}")
            logging.error("   Cannot proceed with prediction - feature alignment required")
            logging.error("   Please run training first to generate feature_columns.pkl")
            return

        # ✅ Winsorization: Apply if enabled during training
        # Must match training preprocessing for consistency
        if self.use_winsorization:
            input = DataProcessor.winsorize_features(
                input,
                lower_percentile=0.01,
                upper_percentile=0.99,
                enabled=True
            )
            logging.info(f"   ✅ Winsorization applied to latest prediction data")

        preds = np.empty((0, input.shape[0]))

        # === 분류 단계 ===
        # 모든 분류기 실행
        for i, model in self.clsmodels.items():
            logging.info(f"classification model # {i}")
            pred_col_name = 'clsmodel_' + str(i) + '_prediction'
            # GPU 지원으로 device mismatch 워닝 방지
            y_probs = predict_proba_with_gpu_support(model, input, self.use_gpu_prediction)[:, 1]
            # 백분위수 임계값을 사용하여 이진으로 변환 (학습 시 자동 탐색된 threshold 사용)
            threshold = np.percentile(y_probs, threshold_percentile)
            y_predict_binary = (y_probs > threshold).astype(int)
            logging.info(f"20% positive threshold == {threshold}")
            ldf[pred_col_name] = y_predict_binary

        # === 회귀 단계 ===
        # 모든 회귀 모델을 실행하고 앙상블 예측 생성
        # 통합 유틸리티 메서드로 컬럼 이름 가져오기
        classifier_cols = self._get_classifier_column_names()

        # CLASSIFIER_MODE 가져오기 (threshold_config 또는 config에서)
        classifier_mode = threshold_config.get('mode', self.conf.get('ML', {}).get('CLASSIFIER_MODE', 'positive_screen'))

        for i, model in self.models.items():
            # 통합 유틸리티 메서드로 컬럼 이름 가져오기
            reg_cols = self._get_regression_column_names(i)

            # 원시 회귀 예측 가져오기
            # GPU 지원으로 device mismatch 워닝 방지
            y_predict = predict_with_gpu_support(model, input, self.use_gpu_prediction)

            # 원시 예측 저장
            ldf[reg_cols['prediction']] = y_predict

            # 통합 메서드로 바이너리 필터링 적용 (모드 전달)
            ldf = self._apply_binary_filtering(ldf, y_predict, classifier_cols, reg_cols, classifier_mode)

            # 통합 메서드로 앙상블 예측 생성 (모드 전달)
            ldf = self._create_ensemble_predictions(ldf, y_predict, classifier_cols, reg_cols, classifier_mode)

            preds = np.vstack((preds, y_predict[None,:]))

        # 평균 예측 계산
        ldf['ai_pred_avg'] = np.average(preds, axis=0)
        ldf.to_csv(model_save_path+"latest_prediction.csv")

        # 상위 K개 주식 추천 생성 (config의 TOP_K_NUM 사용, ml_backtest.py와 동일)
        topk_list = [(0, self.top_k_num - 1)]
        for s, e in topk_list:
            logging.info("top" + str(s) + " ~ " + str(e))
            for col in pred_col_list:
                top_k_df = ldf.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                top_k_df.to_csv(model_save_path+'latest_prediction_{}_top{}-{}.csv'.format(col, s, e))

    def _predict_latest_sectors(self, ldf_with_sector: pd.DataFrame,
                                ldf: pd.DataFrame, model_save_path: str) -> None:
        """Run sector-specific predictions on latest data.

        Loads sector models, runs predictions per sector, generates top-K
        recommendations, and restores original sector names for reports.

        Args:
            ldf_with_sector: DataFrame with sector column preserved.
            ldf: DataFrame used for sector name restoration.
            model_save_path: Directory path for saving predictions.
        """
        self.sector_models = dict()
        self.sector_classifiers = dict()
        # Use the saved copy with sector information
        ldf = ldf_with_sector.copy()

        # 섹터별 모델 로드
        # 통합 섹터 모델 로딩 메서드 사용
        self._load_sector_models(model_save_path, self.sector_list)
        self._load_sector_classifiers(model_save_path, self.sector_list)

        all_preds = []

        # 섹터별로 예측 수행
        for sec in self.sector_list:
            sec_df = ldf[ldf['sector']==sec].copy()

            # ✅ Use unified preprocessing (DataProcessor.prepare_sector_data)
            # Ensures consistency with ml_backtest.py
            first_sector_model = self.sector_models[(sec, 0)]
            sec_input = DataProcessor.prepare_sector_data(
                sector_df=sec_df,
                sector_model=first_sector_model,
                y_col_list=y_col_list,
                use_winsorization=self.use_winsorization,
                logger=logging.getLogger()
            )

            # 전처리된 데이터를 indata로 사용
            indata = sec_input
            preds = np.empty((0, indata.shape[0]))

            # 섹터별 모델 실행
            for i in range(2):
                k = (sec, i)
                model = self.sector_models[k]
                pred_col_name = 'model_' + str(i) + '_prediction'
                # GPU 지원으로 device mismatch 워닝 방지
                y_predict3 = predict_with_gpu_support(model, indata, self.use_gpu_prediction)
                # 원본 sec_df에 예측 결과 추가 (인덱스 정렬)
                sec_df.loc[sec_input.index, pred_col_name] = y_predict3
                preds = np.vstack((preds, y_predict3[None,:]))

            sec_df.loc[sec_input.index, 'ai_pred_avg'] = np.average(preds, axis=0)
            sec_df.to_csv(model_save_path+"sec_{}_latest_prediction.csv".format(sec))

            # 섹터별 상위 K개 (config의 TOP_K_NUM 사용, ml_backtest.py와 동일)
            # ⚠️ 섹터 예측은 wbinary 컬럼을 생성하지 않으므로, 실제 생성된 컬럼만 사용
            sector_pred_col_list = ['ai_pred_avg', 'model_0_prediction', 'model_1_prediction']
            topk_list = [(0, self.top_k_num - 1)]
            for s, e in topk_list:
                logging.info("top" + str(s) + " ~ " + str(e))
                for col in sector_pred_col_list:
                    top_k_df = sec_df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                    top_k_df.to_csv(model_save_path+'latest_prediction_{}_{}_top{}-{}.csv'.format(col, sec, s, e))
                    symbols = top_k_df['symbol'].to_list()
                    preds = top_k_df[col].to_list()
                    for i, sym in enumerate(symbols):
                        all_preds.append([(e-s), sec, col, i, sym, preds[i]])

        # ✅ 원본 섹터 이름 복원 (레포트용)
        # 카테고리 이름으로 학습/예측했지만, 사용자에게는 원본 섹터 이름 표시
        if 'sector_original' in ldf.columns:
            # all_preds의 섹터 이름을 원본으로 매핑
            category_to_original = {}
            for cat in ldf['sector_category'].unique():
                originals = ldf[ldf['sector_category'] == cat]['sector_original'].unique()
                if len(originals) > 0:
                    # 카테고리에 포함된 원본 섹터들을 쉼표로 구분하여 표시
                    # None 값 필터링 (섹터 정보가 없는 종목 제외)
                    valid_originals = [o for o in originals if o is not None]
                    if valid_originals:
                        category_to_original[cat] = ', '.join(sorted(valid_originals))
                    else:
                        category_to_original[cat] = 'Unknown'

            # all_preds에서 섹터 이름 변환
            for pred_row in all_preds:
                cat_name = pred_row[1]  # sector는 index 1
                if cat_name in category_to_original:
                    pred_row[1] = category_to_original[cat_name]

        # 섹터 기반 요약 저장
        col_name = ['k', 'sector', 'model', 'i', 'symbol', 'pred']
        pred_df = pd.DataFrame(all_preds, columns=col_name)
        pred_df.to_csv(model_save_path+'allsector_latest_pred_df.csv', index=False)


    def predict_for_date(self, target_date: str = "latest", top_k: int = 10) -> pd.DataFrame:
        """
        특정 날짜 기준으로 주식 추천을 생성합니다 (학습 없이 예측만).

        이미 학습된 모델을 로드하여 target_date 기준으로 사용 가능한 데이터로
        예측을 수행합니다. 학습, 평가, 백테스트를 건너뛰고 빠르게 추천을 받을 수 있습니다.

        Parameters:
        ----------
        target_date : str
            예측 기준 날짜:
            - "latest": 가장 최근 데이터 사용 (기존 latest_prediction()과 동일)
            - "2025-01-11": 해당 날짜 기준으로 filingDate <= target_date인 최신 데이터 사용
        top_k : int
            추천할 종목 수 (기본값: 10)

        Returns:
        -------
        pd.DataFrame
            추천 종목 DataFrame (symbol, company, sector, ml_score, pred_return 등)
        """
        from datetime import datetime as dt

        MODEL_SAVE_PATH = self.root_path + '/MODELS/'
        aidata_dir = self.root_path + '/processed/ml_data/per_year/'

        # ===== 1. 모델 로드 =====
        logging.info("="*80)
        logging.info("🎯 PREDICTION-ONLY MODE")
        logging.info(f"   Target Date: {target_date}")
        logging.info(f"   Top-K: {top_k}")
        logging.info("="*80)

        self.clsmodels = dict()
        self.models = dict()

        try:
            self._load_classifiers(MODEL_SAVE_PATH)
            self._load_regressors(MODEL_SAVE_PATH)
            logging.info("✅ Models loaded successfully")
        except FileNotFoundError as e:
            logging.error(f"❌ Model files not found: {e}")
            logging.error("   Please run training first (ML.RUN_REGRESSION=Y)")
            return pd.DataFrame()

        # Threshold config 로드
        threshold_config, THRESHOLD_PERCENTILE = self._load_threshold_config(MODEL_SAVE_PATH)
        classifier_mode = threshold_config.get('mode', self.conf.get('ML', {}).get('CLASSIFIER_MODE', 'positive_screen'))

        # Feature columns 로드
        feature_columns_file = MODEL_SAVE_PATH + 'feature_columns.pkl'
        try:
            train_feature_columns = joblib.load(feature_columns_file)
            logging.info(f"✅ Feature columns loaded: {len(train_feature_columns)} features")
        except FileNotFoundError:
            logging.error(f"❌ Feature columns file not found: {feature_columns_file}")
            logging.error("   Please run training first")
            return pd.DataFrame()

        # ===== 2. 데이터 로드 및 필터링 =====
        ldf, date_str = self._load_data_for_date_prediction(aidata_dir, target_date)
        if ldf is None:
            return pd.DataFrame()

        # ===== 3. 전처리 =====
        input_data, meta_df = self._prepare_prediction_features(ldf, train_feature_columns)

        # ===== 4. 예측 =====
        logging.info("")
        logging.info("🔮 Running predictions...")

        result_df = meta_df.copy()

        # 분류기 실행
        clf_probs_list = []
        for i, model in self.clsmodels.items():
            y_probs = predict_proba_with_gpu_support(model, input_data, self.use_gpu_prediction)[:, 1]
            clf_probs_list.append(y_probs)
            result_df[f'clf_{i}_proba'] = y_probs

        # 앙상블 평균 확률
        avg_proba = np.mean(clf_probs_list, axis=0)
        result_df['pred_up_proba'] = avg_proba

        # 회귀 예측
        reg_preds_list = []
        for i, model in self.models.items():
            y_pred = predict_with_gpu_support(model, input_data, self.use_gpu_prediction)
            reg_preds_list.append(y_pred)
            result_df[f'reg_{i}_pred'] = y_pred

        # 앙상블 평균 수익률 예측
        avg_return = np.mean(reg_preds_list, axis=0)
        result_df['pred_return'] = avg_return

        # ===== 5. ML Score 계산 및 결과 저장 =====
        return self._compute_scores_and_save(
            result_df, avg_proba, avg_return, THRESHOLD_PERCENTILE,
            classifier_mode, top_k, date_str, target_date, MODEL_SAVE_PATH
        )

    def _load_data_for_date_prediction(self, aidata_dir: str, target_date: str) -> tuple:
        """Load and filter data for prediction based on target date.

        Args:
            aidata_dir: Directory path containing ML data files.
            target_date: Target date string ("latest" or "YYYY-MM-DD").

        Returns:
            Tuple of (ldf DataFrame, date_str string) or (None, None) on failure.
        """
        logging.info("")
        logging.info("📊 Loading data...")

        # 모든 parquet 파일 찾기
        fs_files = sorted(glob.glob(aidata_dir + 'rnorm_fs_*.parquet'))
        if not fs_files:
            logging.error(f"❌ No data files found in {aidata_dir}")
            return None, None

        # 모든 연도 데이터 로드
        all_data = []
        for f in fs_files:
            try:
                df = pd.read_parquet(f)
                all_data.append(df)
            except Exception as e:
                logging.warning(f"   Failed to load {f}: {e}")

        if not all_data:
            logging.error("❌ No data loaded")
            return None, None

        ldf = pd.concat(all_data, ignore_index=True)
        logging.info(f"   Loaded {len(ldf)} rows from {len(all_data)} files")

        # 날짜 필터링
        if target_date.lower() != "latest":
            # 특정 날짜 기준 필터링
            try:
                target_dt = pd.to_datetime(target_date)
                logging.info(f"   Filtering data for target_date: {target_date}")

                # filingDate 기준 필터링 (공시일 <= target_date)
                if 'filingDate' in ldf.columns:
                    ldf['filingDate'] = pd.to_datetime(ldf['filingDate'], errors='coerce')
                    before_filter = len(ldf)
                    ldf = ldf[ldf['filingDate'] <= target_dt]
                    logging.info(f"   After filingDate filter: {len(ldf)} rows (removed {before_filter - len(ldf)})")
                else:
                    logging.warning("   ⚠️ 'filingDate' column not found, using all data")

                date_str = target_date.replace("-", "")
            except Exception as e:
                logging.error(f"❌ Invalid date format: {target_date}")
                logging.error(f"   Expected format: YYYY-MM-DD (e.g., 2025-01-11)")
                return None, None
        else:
            date_str = "latest"
            logging.info("   Using latest available data")

        # 심볼당 가장 최근 데이터만 유지
        if 'year_period' in ldf.columns:
            ldf = ldf.sort_values(by='year_period', ascending=False)
            ldf = ldf.drop_duplicates(subset='symbol', keep='first')
            logging.info(f"   After deduplication: {len(ldf)} unique symbols")

        if len(ldf) == 0:
            logging.error("❌ No data available for the specified date")
            return None, None

        return ldf, date_str

    def _prepare_prediction_features(self, ldf: pd.DataFrame,
                                     train_feature_columns: list) -> tuple:
        """Prepare input features for prediction with alignment and preprocessing.

        Args:
            ldf: Raw data DataFrame.
            train_feature_columns: List of feature column names from training.

        Returns:
            Tuple of (input_data DataFrame, meta_df DataFrame with metadata).
        """
        # 과도한 NaN 행 제거
        ldf = DataProcessor.drop_many_nan_row(ldf, threshold=0.6)
        logging.info(f"   After NaN filter: {len(ldf)} rows")

        # 메타데이터 컬럼 보존
        meta_cols = ['symbol', 'company', 'sector', 'filingDate', 'year_period']
        meta_df = ldf[[c for c in meta_cols if c in ldf.columns]].copy()

        # 입력 특성 준비
        ldf_features = ldf.drop(columns=self.drop_col_list, errors='ignore')
        ldf_features = ldf_features.drop(columns=['sector'], errors='ignore')

        input_data = ldf_features[ldf_features.columns.difference(y_col_list)]
        input_data = self.clean_feature_names(input_data)

        # Feature alignment
        missing_features = set(train_feature_columns) - set(input_data.columns)
        if missing_features:
            logging.info(f"   Adding {len(missing_features)} missing features with NaN")
            for col in missing_features:
                input_data[col] = np.nan

        extra_features = set(input_data.columns) - set(train_feature_columns)
        if extra_features:
            logging.info(f"   Removing {len(extra_features)} extra features")
            input_data = input_data.drop(columns=list(extra_features))

        input_data = input_data[train_feature_columns]
        logging.info(f"   Feature alignment complete: {len(input_data.columns)} features")

        # Winsorization
        if self.use_winsorization:
            input_data = DataProcessor.winsorize_features(
                input_data,
                lower_percentile=0.01,
                upper_percentile=0.99,
                enabled=True
            )

        return input_data, meta_df

    def _compute_scores_and_save(self, result_df: pd.DataFrame,
                                 avg_proba: np.ndarray, avg_return: np.ndarray,
                                 threshold_percentile: int, classifier_mode: str,
                                 top_k: int, date_str: str, target_date: str,
                                 model_save_path: str) -> pd.DataFrame:
        """Compute ML scores, apply filtering, save results and display recommendations.

        Args:
            result_df: DataFrame with prediction results.
            avg_proba: Average classifier probabilities.
            avg_return: Average regressor predictions.
            threshold_percentile: Percentile for classifier threshold.
            classifier_mode: Classifier mode ("positive_screen" or "negative_screen").
            top_k: Number of top stocks to recommend.
            date_str: Date string for file naming.
            target_date: Original target date string for display.
            model_save_path: Directory path for saving results.

        Returns:
            DataFrame with top-K recommended stocks.
        """
        # ML Score 계산 (Hard Filtering)
        threshold = np.percentile(avg_proba, threshold_percentile)

        if classifier_mode == "negative_screen":
            # BAD 확률이 threshold 이상이면 제외
            pass_mask = avg_proba < threshold
        else:  # positive_screen
            # GOOD 확률이 threshold 이하면 제외
            pass_mask = avg_proba > threshold

        result_df['ml_score'] = np.where(pass_mask, avg_return, -np.inf)
        result_df['passed_filter'] = pass_mask

        n_passed = pass_mask.sum()
        logging.info(f"   Classifier filter ({classifier_mode}): {n_passed}/{len(pass_mask)} passed")

        # 결과 정렬 및 저장
        result_df = result_df.sort_values('ml_score', ascending=False)

        # 전체 예측 저장
        output_file = f"{model_save_path}prediction_{date_str}.csv"
        result_df.to_csv(output_file, index=False)
        logging.info(f"✅ Full predictions saved: {output_file}")

        # Top-K 추천 저장
        top_k_df = result_df[result_df['passed_filter']].head(top_k)
        top_k_file = f"{model_save_path}prediction_{date_str}_top{top_k}.csv"
        top_k_df.to_csv(top_k_file, index=False)
        logging.info(f"✅ Top-{top_k} recommendations saved: {top_k_file}")

        # 결과 출력
        logging.info("")
        logging.info("="*80)
        logging.info(f"🏆 TOP {top_k} RECOMMENDATIONS (as of {target_date})")
        logging.info("="*80)

        display_cols = ['symbol', 'company', 'sector', 'ml_score', 'pred_return', 'pred_up_proba']
        display_cols = [c for c in display_cols if c in top_k_df.columns]

        for idx, row in top_k_df.iterrows():
            symbol = row.get('symbol') or 'N/A'
            company_raw = row.get('company')
            company = str(company_raw)[:30] if pd.notna(company_raw) else 'N/A'
            sector = row.get('sector') or 'N/A'
            ml_score = row.get('ml_score')
            ml_score = float(ml_score) if pd.notna(ml_score) else 0.0
            pred_return = row.get('pred_return')
            pred_return = float(pred_return) if pd.notna(pred_return) else 0.0

            logging.info(f"   {symbol:6s} | {company:30s} | {sector:20s} | Score: {ml_score:+.4f} | Return: {pred_return:+.2%}")

        logging.info("="*80)

        return top_k_df

