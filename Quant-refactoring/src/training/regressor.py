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
    from config.config_loader import load_config
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
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as nn_f
import torch.optim as optim
from dateutil.relativedelta import relativedelta
import datetime
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any

# datasets 라이브러리는 사용되지 않으므로 import 제거됨
# from datasets import Dataset
from config.g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map, sparse_col_list
from src.constants.data_schema import DataSchema  # ✨ Unified column definitions
from src.training.data_processor import DataProcessor  # ✨ Unified preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from torch.utils.data import DataLoader
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

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
            # 결과를 CPU로 다시 변환
            return cp.asnumpy(y_pred)
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
            return cp.asnumpy(y_proba)
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
        from config.config_loader import load_config
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
        self.x_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.x_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        print(self.conf)

        # 중첩된 구조에서 설정 값 추출
        data_config = conf.get('DATA', {})
        ml_config = conf.get('ML', {})
        features_config = conf.get('FEATURES', {})
        backtest_config = conf.get('BACKTEST', {})
        self.root_path: str = data_config.get('ROOT_PATH', '/home/user/Quant/data')

        # 섹터별 모델 사용 여부 (ml_backtest.py와 동일한 방식)
        self.use_sector_model = ml_config.get('USE_SECTOR_MODEL', 'N') == 'Y'
        self.sector_config = ml_config.get('SECTOR_CONFIG', {}) if self.use_sector_model else {}

        # Preprocessing 설정 (ml_backtest.py와 동일한 방식)
        self.use_winsorization = features_config.get('USE_WINSORIZATION', 'Y') == 'Y'
        self.use_feature_selection = features_config.get('USE_FEATURE_SELECTION', 'Y') == 'Y'

        # Top K 설정 (ml_backtest.py와 동일한 방식)
        self.top_k_num = int(backtest_config.get('TOP_K_NUM', 20))

        aidata_dir = self.root_path + '/processed/ml_data/per_year/'
        print("aidata path : " + aidata_dir)
        if not os.path.exists(aidata_dir):
            print("there is no ai data : " + aidata_dir)
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

        print("train file list : ", self.train_files)
        print("test file list : ", self.test_files)

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
        self.clsmodels: Dict[int, Any] = dict()  # 분류 모델
        self.models: Dict[int, Any] = dict()  # 회귀 모델
        self.sector_models: Dict[Tuple[str, int], Any] = dict()  # 섹터별 모델
        self.sector_cls_models: Dict = dict()

        # 섹터별 학습 데이터
        self.sector_x_train: Dict[str, pd.DataFrame] = dict()
        self.sector_y_train: Dict[str, pd.DataFrame] = dict()

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
        """섹터별 모델들을 로드합니다.

        Args:
            model_save_path: 모델 저장 경로
            sector_list: 섹터 이름 리스트
        """
        for sec in sector_list:
            for i in range(2):
                k = (sec, i)
                filename = f"{model_save_path}{sec}_model_{i}.sav"
                self.sector_models[k] = joblib.load(filename)
        logging.info(f"✅ Loaded sector models for {len(sector_list)} sectors")

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
        regression_col_names: dict
    ) -> pd.DataFrame:
        """분류기 결과를 사용하여 회귀 예측을 필터링합니다.

        분류기가 하락(0)을 예측하면 회귀 출력을 -1로 대체합니다.

        Args:
            df: 데이터프레임
            y_predict: 회귀 예측 값
            classifier_cols: 분류기 컬럼 이름 딕셔너리
            regression_col_names: 회귀 컬럼 이름 딕셔너리

        Returns:
            필터링된 예측이 추가된 데이터프레임
        """
        for i in range(4):
            col_name = regression_col_names[f'wbinary_{i}']
            clf_col = classifier_cols[i]
            df[col_name] = np.where(df[clf_col] == 0, -1, y_predict)
        return df

    @staticmethod
    def _create_ensemble_predictions(
        df: pd.DataFrame,
        y_predict: np.ndarray,
        classifier_cols: dict,
        regression_col_names: dict
    ) -> pd.DataFrame:
        """앙상블 전략을 사용하여 예측을 생성합니다.

        3가지 앙상블 전략:
        - ensemble: 분류기 1 AND 3 모두 상승
        - ensemble2: 분류기 1 AND 2 모두 상승
        - ensemble3: 다수결 (3개 중 2개 이상)

        Args:
            df: 데이터프레임
            y_predict: 회귀 예측 값
            classifier_cols: 분류기 컬럼 이름 딕셔너리
            regression_col_names: 회귀 컬럼 이름 딕셔너리

        Returns:
            앙상블 예측이 추가된 데이터프레임
        """
        # 앙상블 1: 분류기 1 AND 3
        df[regression_col_names['wbinary_ensemble']] = np.where(
            ((df[classifier_cols[1]] == 0) | (df[classifier_cols[3]] == 0)),
            -1, y_predict)

        # 앙상블 2: 분류기 1 AND 2
        df[regression_col_names['wbinary_ensemble2']] = np.where(
            ((df[classifier_cols[1]] == 0) | (df[classifier_cols[2]] == 0)),
            -1, y_predict)

        # 앙상블 3: 다수결 (2개 이상이 하락 예측)
        condition = (
            (df[[classifier_cols[1], classifier_cols[2], classifier_cols[3]]] == 0).sum(axis=1) >= 2
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
        # STEP 1: Load all training files from parquet
        # ========================================================================
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

        # Concatenate all training data
        self.train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
        logging.info(f"✅ Combined training data: {len(self.train_df)} rows from {len(train_dfs)} files")

        # ========================================================================
        # STEP 2: Calculate sector-based features (BEFORE preprocessing)
        # ========================================================================
        # TODO: Move this to make_mldata.py (should be done during data generation)
        logging.info("\nSTEP 2/5: Calculating sector-based features...")

        if 'industry' in self.train_df.columns:
            self.train_df["sector"] = self.train_df["industry"].map(sector_map)
            sector_list = list(self.train_df['sector'].unique())
            sector_list = [x for x in sector_list if str(x) != 'nan']

            logging.info(f"  Found {len(sector_list)} sectors: {sector_list}")

            # Calculate sector-adjusted price deviation
            for sec in sector_list:
                sec_mask = self.train_df['sector'] == sec
                sec_count = sec_mask.sum()
                sec_mean = self.train_df.loc[sec_mask, 'price_dev'].mean()
                self.train_df.loc[sec_mask, 'sec_price_dev_subavg'] = \
                    self.train_df.loc[sec_mask, 'price_dev'] - sec_mean
                logging.debug(f"    {sec}: {sec_count} stocks, mean price_dev={sec_mean:.4f}")

            logging.info(f"✅ Sector features calculated")
        else:
            logging.warning("⚠️  'industry' column not found, skipping sector calculation")
            sector_list = []

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
        self.train_df = pd.concat([metadata_df, X_train_reset, y_train_reset, y_train_cls_reset], axis=1)

        # Store X, y separately
        self.x_train = X_train
        self.y_train = y_train
        self.y_train_cls = y_train_cls

        logging.info(f"  Training data stored: {len(self.train_df)} rows")

        # PER_SECTOR mode: Split training data by sector
        if self.use_sector_model and 'sector' in self.train_df.columns:
            logging.info("  🔧 Sector model enabled: Splitting training data by sector...")
            self.sector_list = list(self.train_df['sector'].unique())
            self.sector_list = [x for x in self.sector_list if str(x) != 'nan']

            for sec in self.sector_list:
                self.sector_train_dfs[sec] = self.train_df[self.train_df['sector'] == sec].copy()
                sec_feature_cols = [col for col in self.x_train.columns if col in self.sector_train_dfs[sec].columns]
                self.sector_x_train[sec] = self.sector_train_dfs[sec][sec_feature_cols]
                self.sector_y_train[sec] = self.sector_train_dfs[sec][['sec_price_dev_subavg']]
                logging.info(f"    {sec}: {len(self.sector_train_dfs[sec])} rows")

            logging.info(f"  ✅ Split into {len(self.sector_list)} sectors")

        # ========================================================================
        # STEP 5: Load and preprocess test files
        # ========================================================================
        logging.info("\nSTEP 5/5: Loading and preprocessing test files...")

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
            - clsmodels[3]: LGBMClassifier, max_depth=8

        회귀 모델 (2개 변형):
            - models[0]: XGBRegressor, max_depth=8
            - models[1]: XGBRegressor, max_depth=10

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
        classifiers, regressors, sector_models = create_models_for_regressor(
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
            self.sector_models = sector_models

        logging.info(f"✅ Models created: {len(classifiers)} classifiers, {len(regressors)} regressors" +
                    (f", {len(sector_models)} sector models" if self.use_sector_model else ""))

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

        # 모델 저장 경로 설정
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # 필요시 저장 디렉토리 생성
        if not os.path.exists(MODEL_SAVE_PATH):
            print("creating MODELS path : " + MODEL_SAVE_PATH)
            os.makedirs(MODEL_SAVE_PATH)

        # ========== Optuna Hyperparameter Optimization ==========
        ml_config = self.conf.get('ML', {})
        use_optuna = ml_config.get('USE_OPTUNA', False)
        optuna_best_params = None

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
            y_train_binary_optuna = DataProcessor.create_binary_target(self.y_train_cls)

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

        # ========== Sector-specific Optuna Optimization ==========
        sector_optuna_params = {}
        optuna_optimize_sectors = ml_config.get('OPTUNA_OPTIMIZE_SECTORS', 'N') == 'Y'

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

        # ========== 모델 정의 (Optuna 결과 반영) ==========
        self.def_model(optuna_params=optuna_best_params, sector_optuna_params=sector_optuna_params)

        # LightGBM 호환성을 위해 특성 이름 정리 (Optuna를 사용하지 않은 경우에만)
        if not (use_optuna and OPTUNA_AVAILABLE):
            self.x_train = self.clean_feature_names(self.x_train)

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
        y_train_binary = DataProcessor.create_binary_target(self.y_train_cls)

        # 모든 분류 모델 학습
        for i, model in self.clsmodels.items():
            logging.info("start fitting classifier")
            model.fit(self.x_train, y_train_binary)
            filename = MODEL_SAVE_PATH + 'clsmodel_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, y_train_binary))

        # 모든 회귀 모델 학습
        for i, model in self.models.items():
            logging.info("start fitting XGBRegressor")
            model.fit(self.x_train, self.y_train.values.ravel())
            filename = MODEL_SAVE_PATH + 'model_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, self.y_train))

        # ===== Feature columns 저장 (예측 시 피처 정렬용) =====
        feature_columns = self.x_train.columns.tolist()
        feature_columns_file = MODEL_SAVE_PATH + 'feature_columns.pkl'
        joblib.dump(feature_columns, feature_columns_file)
        logging.info(f"✅ Saved {len(feature_columns)} feature columns to {feature_columns_file}")

            # 주석 처리됨: 특성 중요도 분석
            # logging.info("end fitting RandomForestRegressor")
            # ftr_importances_values = model.feature_importances_
            # ftr_importances = pd.Series(ftr_importances_values, index=self.x_train.columns)
            # ftr_importances.to_csv(MODEL_SAVE_PATH+'model_importances.csv')
            # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            # logging.info(ftr_top20)

        # 섹터별 모델 학습 (PER_SECTOR=True인 경우)
        if self.use_sector_model:
            logging.info("="*80)
            logging.info("🎯 SECTOR MODELS: Applying unified preprocessing to each sector")
            logging.info("="*80)

            for sec in self.sector_list:
                logging.info(f"\n📊 Processing sector: {sec}")
                logging.info("-" * 60)

                # ✅ UNIFIED: Use SAME preprocessing as unified model (SINGLE SOURCE OF TRUTH)
                # Sector models have NO y_cls (classification target), so pass None
                x_sector_clean, y_sector_clean, _, _ = DataProcessor.preprocess_training_data(
                    self.sector_x_train[sec],
                    self.sector_y_train[sec],
                    y_cls=None,  # No classification for sector models
                    config=self.conf,
                    logger=logging.getLogger()
                )

                # Update sector training data with preprocessed results
                self.sector_x_train[sec] = x_sector_clean
                self.sector_y_train[sec] = y_sector_clean

            logging.info("="*80)
            logging.info("✅ All sector preprocessing complete")
            logging.info("="*80)
            logging.info("")

            # Train sector models
            for sec_idx, sec in enumerate(self.sector_list):
                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    model.fit(self.sector_x_train[sec], self.sector_y_train[sec].values.ravel())
                    filename = MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))

                    joblib.dump(model, filename)
                    logging.info("model {} score : ".format(str(i)))
                    logging.info(model.score(self.sector_x_train[sec], self.sector_y_train[sec]))
                    logging.info("end fitting per sector XGBRegressor")

                    # 주석 처리됨: 섹터별 특성 중요도 분석
                    # ftr_importances_values = model.feature_importances_
                    # ftr_importances = pd.Series(ftr_importances_values,
                    #                            index=self.sector_x_train[sec].columns)
                    # ftr_importances.to_csv(MODEL_SAVE_PATH + sec + '_model_importances.csv')
                    # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
                    # logging.info(ftr_top20)


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

        앙상블 투표 로직:
            - prediction_wbinary_0: 분류기 0의 예측 사용
              cls0이 하락(0)을 예측하면 회귀 출력을 -1로 설정
            - prediction_wbinary_1: 분류기 1의 예측 사용
            - prediction_wbinary_2: 분류기 2의 예측 사용
            - prediction_wbinary_3: 분류기 3의 예측 사용
            - prediction_wbinary_ensemble: cls1 AND cls3 모두 상승을 예측해야 함
              둘 중 하나라도 하락을 예측하면 회귀 출력을 -1로 설정
            - prediction_wbinary_ensemble2: cls1 AND cls2 모두 상승을 예측해야 함
            - prediction_wbinary_ensemble3: 다수결 투표 - 3개 중 최소 2개가 상승을 예측해야 함
              cls1, cls2, cls3를 투표에 사용

        출력 파일 (MODEL_SAVE_PATH에 저장):
            - prediction_ai_{date}.csv: 각 테스트 기간의 예측
            - prediction_ai.csv: 모든 예측 연결
            - pred_df_topk.csv: 모든 모델의 상위 K개 평가 메트릭
            - prediction_{date}_{model}_{col}_top{s}-{e}.csv: 모델당 상위 K개 주식

        부작용:
            - MODEL_SAVE_PATH/*.sav에서 모델 로드
            - MODEL_SAVE_PATH에 평가 CSV 파일 생성
            - 분류 보고서 및 메트릭 로깅
            - 각 예측 방법의 상위 K개 수익 로깅

        참고:
            - 분류기 확률을 이진으로 변환하기 위해 THRESHOLD (기본값 92) 사용
            - 상위 8%의 주식 (100-92=8%)이 양성으로 예측됨
            - 평가에는 기간별 및 누적 메트릭이 모두 포함됨
            - PER_SECTOR=True인 경우 섹터별 모델도 평가합니다
        """
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # 학습된 분류 모델 로드
        self.models = dict()
        self.clsmodels = dict()

        # 통합 모델 로딩 메서드 사용
        self._load_classifiers(MODEL_SAVE_PATH)
        self._load_regressors(MODEL_SAVE_PATH)

        # 통합 메서드로 예측 컬럼 이름 생성
        pred_col_list = self._build_prediction_column_names()

        model_eval_hist = []  # 모든 기간의 평가 결과 저장
        full_df = pd.DataFrame()  # 예측이 포함된 모든 테스트 데이터 누적

        # 각 테스트 기간을 개별적으로 평가
        for test_idx, (testdate, df) in enumerate(self.test_df_list):

            logging.info("evaluation date : ")
            # 파일 경로에서 날짜 추출 (통합 유틸리티 메서드 사용)
            tdate = self._extract_date_from_filepath(testdate)
            filename = os.path.basename(testdate)

            print(f"in test loop filename : {filename}")
            print(f"in test loop tdate : {tdate}")

            # 이 테스트 기간의 특성과 레이블 준비
            x_test = df[df.columns.difference(y_col_list)]
            y_test = df[['price_dev_subavg']]
            y_test_cls = df[['price_dev']]
            # ✅ REFACTORED: Use DataProcessor for binary target
            y_test_binary = DataProcessor.create_binary_target(y_test_cls)

            df['label'] = y_test  # 실제 가격 변동
            df['label_binary'] = y_test_binary  # 실제 이진 레이블

            # LightGBM을 위해 특성 이름 정리
            x_test = self.clean_feature_names(x_test)

            # ===== Feature Alignment (피처 정렬) =====
            # 학습 시 사용한 피처 리스트 로드
            feature_columns_file = MODEL_SAVE_PATH + 'feature_columns.pkl'
            try:
                train_feature_columns = joblib.load(feature_columns_file)
                logging.info(f"✅ Loaded {len(train_feature_columns)} train feature columns")
                logging.info(f"   Test data has {len(x_test.columns)} features")

                # 누락된 피처는 NaN으로 채우기 (XGBoost가 처리)
                missing_features = set(train_feature_columns) - set(x_test.columns)
                if missing_features:
                    logging.warning(f"   ⚠️  {len(missing_features)} features missing in test data, filling with NaN")
                    for col in missing_features:
                        x_test[col] = np.nan

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
                # THRESHOLD=92는 상위 8%가 양성으로 예측됨을 의미
                threshold = np.percentile(y_probs, THRESHOLD)
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

            for i, model in self.models.items():
                # 통합 유틸리티 메서드로 컬럼 이름 가져오기
                reg_cols = self._get_regression_column_names(i)

                # 원시 회귀 예측 가져오기
                # GPU 지원으로 device mismatch 워닝 방지
                y_predict = predict_with_gpu_support(model, x_test, self.use_gpu_prediction)

                # 원시 회귀 예측 저장
                df[reg_cols['prediction']] = y_predict

                # 통합 메서드로 바이너리 필터링 적용
                df = self._apply_binary_filtering(df, y_predict, classifier_cols, reg_cols)

                # 통합 메서드로 앙상블 예측 생성
                df = self._create_ensemble_predictions(df, y_predict, classifier_cols, reg_cols)

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
                               f"loss : {full_df[loss_col_name].mean()} "
                               f"loss_wbin_0 {full_df[loss_bin_col_name_0].mean()} "
                               f"loss_wbin_1 {full_df[loss_bin_col_name_1].mean()} "
                               f"loss_wbin_2 {full_df[loss_bin_col_name_2].mean()} "
                               f"loss_wbin_3 {full_df[loss_bin_col_name_3].mean()}")

            # 모든 회귀 모델의 평균 예측 계산
            df['ai_pred_avg'] = np.average(preds, axis=0)
            df['ai_pred_avg_loss'] = abs(df['label']-df['ai_pred_avg'])

            # 결과 누적
            full_df = pd.concat([full_df, df], ignore_index=True)
            df.to_csv(MODEL_SAVE_PATH + "prediction_ai_{}.csv".format(tdate))

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
                    top_k_df.to_csv(MODEL_SAVE_PATH+'prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))

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

        # 종합 평가 보고서 생성
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

        # === 섹터 기반 평가 (PER_SECTOR=True인 경우) ===
        if self.use_sector_model:
            testdates = set()
            allsector_topk_df = pd.DataFrame()
            self.sector_models = dict()

            # 통합 섹터 모델 로딩 메서드 사용
            self._load_sector_models(MODEL_SAVE_PATH, self.sector_list)

            sector_model_eval_hist = []

            # 각 섹터 및 테스트 기간 평가
            for test_idx, (testdate, df, sec) in enumerate(self.sector_test_df_lists):
                print("sec evaluation date : ")
                # 파일 경로에서 날짜 추출 (통합 유틸리티 메서드 사용)
                tdate = self._extract_date_from_filepath(testdate)
                if tdate == "unknown_period":
                    logging.warning(f"⚠️  Skipping sector evaluation due to unknown period: {testdate}")
                    continue
                print(tdate)
                print(sec)
                testdates.add(tdate)

                x_test = df[df.columns.difference(y_col_list)]
                y_test = df[['price_dev_subavg']]
                y_test_2 = df[['price_dev_subavg']]

                if len(x_test) == 0:
                    continue

                sector_preds = np.empty((0, x_test.shape[0]))
                df['label'] = y_test

                # 섹터 기반 필터링을 위해 분류기 2 사용
                # GPU 지원으로 device mismatch 워닝 방지
                y_probs = predict_proba_with_gpu_support(self.clsmodels[2], x_test, self.use_gpu_prediction)[:, 1]
                threshold = np.percentile(y_probs, THRESHOLD)
                y_predict_binary = (y_probs > threshold).astype(int)

                # 섹터별 모델 실행
                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    pred_col_name_wbin = 'model_' + str(i) + '_prediction_wbinary_2'
                    # GPU 지원으로 device mismatch 워닝 방지
                    y_predict = predict_with_gpu_support(model, x_test, self.use_gpu_prediction)
                    df[pred_col_name] = y_predict

                    df[pred_col_name_wbin] = np.where(y_predict_binary == 0, -1, y_predict)
                    print(f"i{i} sec {sec}")
                    print(x_test.shape)
                    print(sector_preds.shape)
                    print(y_predict[None,:].shape)
                    sector_preds = np.vstack((sector_preds, y_predict[None,:]))

                df['ai_pred_avg'] = np.average(sector_preds, axis=0)
                df.to_csv(MODEL_SAVE_PATH+ "sec_{}_prediction_ai_{}.csv".format(sec, tdate))

                # 섹터별 예측의 상위 K개 평가
                topk_period_earning_sums = []
                topk_list = [(0,3), (0,7)]
                for s, e in topk_list:
                    logging.info("top" + str(s) + " ~ "  + str(e) )
                    k = str(s) + '~' + str(e)
                    for col in pred_col_list:
                        top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                        logging.info(col)
                        logging.info(("label"))
                        logging.info((top_k_df['price_dev'].sum()/(e-s+1)))
                        logging.info(("pred"))
                        logging.info((top_k_df[col].sum()/(e-s+1)))
                        topk_period_earning_sums.append(top_k_df['price_dev'].sum())
                        top_k_df.to_csv(MODEL_SAVE_PATH+'prediction_{}_{}_{}_top{}-{}.csv'.format(tdate, sec, col, s, e))
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
            print(pred_df)
            pred_df.to_csv(MODEL_SAVE_PATH+'allsector_pred_df.csv'.format(sec), index=False)


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
            7. 상위 K개 주식 추천 생성 (K=3, 7, 15)
            8. 예측을 CSV 파일로 저장

        출력 파일 (MODEL_SAVE_PATH에 저장):
            - latest_prediction.csv: 최신 데이터의 모든 예측
            - latest_prediction_{model}_{col}_top{s}-{e}.csv: 모델당 상위 K개 주식
            - sec_{sector}_latest_prediction.csv: 섹터별 예측 (PER_SECTOR=True인 경우)
            - allsector_latest_pred_df.csv: 섹터 기반 상위 K개 요약 (PER_SECTOR=True인 경우)

        부작용:
            - MODEL_SAVE_PATH/*.sav에서 모델 로드
            - MODEL_SAVE_PATH에 예측 CSV 파일 생성
            - 예측 임계값 및 상위 K개 범위 로깅

        참고:
            - year_period를 사용하여 심볼당 가장 최근 데이터만 유지
            - 분류에는 evaluation()과 동일한 THRESHOLD (92) 적용
            - PER_SECTOR=True인 경우 섹터별 예측 사용 가능
            - FIXME: 2024 데이터를 읽도록 하드코딩됨 (설정 가능해야 함)
        """
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # 통합 모델 로딩 메서드 사용
        self.clsmodels = dict()
        self.models = dict()
        self._load_classifiers(MODEL_SAVE_PATH)
        self._load_regressors(MODEL_SAVE_PATH)

        aidata_dir = self.root_path + '/processed/ml_data/per_year/'

        # 통합 메서드로 예측 컬럼 이름 생성
        pred_col_list = self._build_prediction_column_names()

        # 최신 연도 데이터(모든 분기)를 로드하고 심볼당 가장 최근 것 유지
        # 자동으로 가장 최근 연도 감지 및 Parquet 형식 로드
        import glob

        # 모든 rnorm_fs 파일 찾기
        fs_files = sorted(glob.glob(aidata_dir + 'rnorm_fs_*.parquet'))

        if not fs_files:
            logging.error(f"No rnorm_fs files found in {aidata_dir}")
            logging.error("Cannot generate latest prediction without feature data")
            return

        # 파일명에서 연도 추출하여 가장 최근 연도 찾기
        try:
            years = [int(os.path.basename(f).split('_')[2]) for f in fs_files]
            latest_year = max(years)
            logging.info(f"Latest year detected: {latest_year}")
        except (IndexError, ValueError) as e:
            logging.error(f"Failed to parse year from filenames: {e}")
            return

        # 최신 연도의 모든 분기 로드
        ldf = pd.DataFrame()
        loaded_quarters = []

        for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
            latest_data_path = aidata_dir + f'rnorm_fs_{latest_year}_{Q}.parquet'

            if os.path.exists(latest_data_path):
                df = pd.read_parquet(latest_data_path)
                ldf = pd.concat([ldf, df], axis=0)
                loaded_quarters.append(Q)
                logging.info(f"Loaded {latest_year}_{Q}: {len(df)} rows")
            else:
                logging.warning(f"Latest data file not found: {os.path.basename(latest_data_path)}")

        if ldf.empty:
            logging.error(f"No data loaded for latest year {latest_year}")
            logging.error(f"Checked quarters: Q1, Q2, Q3, Q4")
            return

        logging.info(f"Total loaded for {latest_year}: {len(ldf)} rows from {loaded_quarters}")

        # year_period를 기준으로 내림차순 정렬하고 심볼당 첫 번째(가장 최근) 유지
        ldf = ldf.sort_values(by='year_period', ascending=False)
        ldf = ldf.drop_duplicates(subset='symbol', keep='first')
        ldf = ldf.drop(columns=self.drop_col_list, errors='ignore')

        # Parquet 파일은 인덱스 컬럼 없음 (CSV와 달리)

        # 섹터 리스트 추출
        self.sector_list = list(ldf['sector'].unique())
        self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
        ldf = ldf.drop('sector', axis=1)

        # 과도한 누락 데이터가 있는 행 필터링 (>60% NaN)
        # ✅ REFACTORED: Use DataProcessor for excessive NaN row removal
        print("before dtable len : ", len(ldf))
        ldf = DataProcessor.drop_many_nan_row(ldf, threshold=0.6)
        print("after dtable len : ", len(ldf))

        # 입력 특성 준비
        input = ldf[ldf.columns.difference(y_col_list)]
        input = self.clean_feature_names(input)

        # ===== Feature Alignment (피처 정렬) =====
        # 학습 시 사용한 피처 리스트 로드 (evaluation()과 동일한 방식)
        feature_columns_file = MODEL_SAVE_PATH + 'feature_columns.pkl'
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
                    input[col] = np.nan

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
            # 백분위수 임계값을 사용하여 이진으로 변환
            threshold = np.percentile(y_probs, THRESHOLD)
            y_predict_binary = (y_probs > threshold).astype(int)
            logging.info(f"20% positive threshold == {threshold}")
            ldf[pred_col_name] = y_predict_binary

        # === 회귀 단계 ===
        # 모든 회귀 모델을 실행하고 앙상블 예측 생성
        # 통합 유틸리티 메서드로 컬럼 이름 가져오기
        classifier_cols = self._get_classifier_column_names()

        for i, model in self.models.items():
            # 통합 유틸리티 메서드로 컬럼 이름 가져오기
            reg_cols = self._get_regression_column_names(i)

            # 원시 회귀 예측 가져오기
            # GPU 지원으로 device mismatch 워닝 방지
            y_predict = predict_with_gpu_support(model, input, self.use_gpu_prediction)

            # 원시 예측 저장
            ldf[reg_cols['prediction']] = y_predict

            # 통합 메서드로 바이너리 필터링 적용
            ldf = self._apply_binary_filtering(ldf, y_predict, classifier_cols, reg_cols)

            # 통합 메서드로 앙상블 예측 생성
            ldf = self._create_ensemble_predictions(ldf, y_predict, classifier_cols, reg_cols)

            preds = np.vstack((preds, y_predict[None,:]))

        # 평균 예측 계산
        ldf['ai_pred_avg'] = np.average(preds, axis=0)
        ldf.to_csv(MODEL_SAVE_PATH+"latest_prediction.csv")

        # 상위 K개 주식 추천 생성 (config의 TOP_K_NUM 사용, ml_backtest.py와 동일)
        topk_list = [(0, self.top_k_num - 1)]
        for s, e in topk_list:
            logging.info("top" + str(s) + " ~ " + str(e))
            for col in pred_col_list:
                top_k_df = ldf.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                top_k_df.to_csv(MODEL_SAVE_PATH+'latest_prediction_{}_top{}-{}.csv'.format(col, s, e))

        # === 섹터별 예측 (PER_SECTOR=True인 경우) ===
        if self.use_sector_model:
            self.sector_models = dict()
            ldf = pd.read_csv(latest_data_path)

            # 섹터별 모델 로드
            # 통합 섹터 모델 로딩 메서드 사용
            self._load_sector_models(MODEL_SAVE_PATH, self.sector_list)

            all_preds = []

            # 섹터별로 예측 수행
            for sec in self.sector_list:
                sec_df = ldf[ldf['sector']==sec]
                sec_df = sec_df.drop('sector', axis=1)
                indata = sec_df[sec_df.columns.difference(['symbol'])]
                print(indata)
                preds = np.empty((0, indata.shape[0]))

                # 섹터별 모델 실행
                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    # GPU 지원으로 device mismatch 워닝 방지
                    y_predict3 = predict_with_gpu_support(model, indata, self.use_gpu_prediction)
                    sec_df[pred_col_name] = y_predict3
                    preds = np.vstack((preds, y_predict3[None,:]))

                sec_df['ai_pred_avg'] = np.average(preds, axis=0)
                sec_df.to_csv(MODEL_SAVE_PATH+"sec_{}_latest_prediction.csv".format(sec))

                # 섹터별 상위 K개 (config의 TOP_K_NUM 사용, ml_backtest.py와 동일)
                topk_list = [(0, self.top_k_num - 1)]
                for s, e in topk_list:
                    logging.info("top" + str(s) + " ~ " + str(e))
                    for col in pred_col_list:
                        top_k_df = sec_df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                        top_k_df.to_csv(MODEL_SAVE_PATH+'latest_prediction_{}_{}_top{}-{}.csv'.format(col, sec, s, e))
                        symbols = top_k_df['symbol'].to_list()
                        preds = top_k_df[col].to_list()
                        for i, sym in enumerate(symbols):
                            all_preds.append([(e-s), sec, col, i, sym, preds[i]])

            # 섹터 기반 요약 저장
            col_name = ['k', 'sector', 'model', 'i', 'symbol', 'pred']
            pred_df = pd.DataFrame(all_preds, columns=col_name)
            pred_df.to_csv(MODEL_SAVE_PATH+'allsector_latest_pred_df.csv', index=False)
