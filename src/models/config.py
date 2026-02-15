"""머신러닝 알고리즘의 모델 설정 모듈.

이 모듈은 XGBoost, LightGBM, CatBoost 모델의 사전 설정된 하이퍼파라미터를
포함하며, 퀀트 트레이딩 애플리케이션에 최적화되어 있습니다.

설정 구성:
1. 모델 유형 (XGBoost, LightGBM, CatBoost)
2. 작업 유형 (분류, 회귀)
3. 설정 변형 (default, deep 등)

모든 설정은 빠른 학습을 위해 GPU 가속을 사용합니다. 파라미터는
금융 시계열 데이터에 대한 광범위한 실험을 통해 튜닝되었습니다.

설정 구조:
    - XGBOOST_CLASSIFIER_CONFIGS: XGBoost 분류 설정
    - XGBOOST_REGRESSOR_CONFIGS: XGBoost 회귀 설정
    - LIGHTGBM_CLASSIFIER_CONFIGS: LightGBM 분류 설정
    - LIGHTGBM_REGRESSOR_CONFIGS: LightGBM 회귀 설정
    - CATBOOST_CLASSIFIER_CONFIGS: CatBoost 분류 설정
    - CATBOOST_REGRESSOR_CONFIGS: CatBoost 회귀 설정
    - OPTUNA_SEARCH_SPACE: 최적화를 위한 하이퍼파라미터 탐색 범위
    - ENSEMBLE_CONFIG: 앙상블 기법 설정
    - THRESHOLD_PERCENTILE: 종목 선택 임계값

사용 예시:
    from src.models.config import XGBOOST_CLASSIFIER_CONFIGS

    # Get default XGBoost classifier configuration
    default_config = XGBOOST_CLASSIFIER_CONFIGS['default']

    # Get deeper XGBoost classifier configuration
    deep_config = XGBOOST_CLASSIFIER_CONFIGS['depth_10']

    # Create custom configuration by modifying defaults
    custom_config = XGBOOST_CLASSIFIER_CONFIGS['default'].copy()
    custom_config['learning_rate'] = 0.05
    custom_config['max_depth'] = 10

Note:
    - 모든 GPU 기반 설정은 CUDA 호환 GPU가 필요합니다
    - XGBoost: GPU 가속을 위해 tree_method='hist' + device='cuda:0' 사용
    - LightGBM: GPU 가속을 위해 device='gpu' 사용
    - CatBoost: GPU 가속을 위해 task_type='GPU' 사용
"""

from typing import Dict, Any, Tuple

# =============================================================================
# XGBoost 설정
# =============================================================================

XGBOOST_CLASSIFIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'tree_method': 'hist',           # 히스토그램 알고리즘 (device 파라미터로 GPU 사용)
        'device': 'cuda:0',              # GPU 디바이스 (cuda:0: 첫 번째 GPU, cpu: CPU)
        'n_estimators': 500,             # 부스팅 라운드 수
        'learning_rate': 0.1,            # 스텝 크기 축소 (eta)
        'gamma': 0,                      # 분할을 위한 최소 손실 감소
        'subsample': 0.8,                # 트리당 샘플 비율 (0.8 = 80%)
        'colsample_bytree': 0.8,         # 트리당 feature 비율 (0.8 = 80%)
        'max_depth': 8,                  # 최대 트리 깊이
        'objective': 'binary:logistic',  # 이진 분류 목적 함수
        'eval_metric': 'logloss'         # 평가 지표 (로그 손실)
    },
    'depth_9': {
        'tree_method': 'hist',
        'device': 'cuda:0',
        'n_estimators': 500,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 9,                  # 복잡한 패턴을 위한 더 깊은 트리
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    },
    'depth_10': {
        'tree_method': 'hist',
        'device': 'cuda:0',
        'n_estimators': 500,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 10,                 # 매우 복잡한 패턴을 위한 더욱 깊은 트리
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
}

XGBOOST_REGRESSOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'tree_method': 'hist',
        'device': 'cuda:0',
        'n_estimators': 1000,            # 회귀를 위한 더 많은 반복 (1000)
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 8,
        'objective': 'reg:squarederror',  # 제곱 오차 회귀 목적 함수
        'eval_metric': 'rmse'             # 평균 제곱근 오차
    },
    'depth_10': {
        'tree_method': 'hist',
        'device': 'cuda:0',
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 10,                  # 회귀를 위한 더 깊은 트리 (분류기와 일관성 유지)
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    },
    'deep': {
        'tree_method': 'hist',
        'device': 'cuda:0',
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 10,                  # 회귀를 위한 더 깊은 트리 (depth_10의 별칭)
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
}

# =============================================================================
# LightGBM 설정
# =============================================================================

LIGHTGBM_CLASSIFIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'boosting_type': 'gbdt',          # Gradient Boosting Decision Tree
        'objective': 'binary',            # 이진 분류
        'n_estimators': 1000,             # 부스팅 반복 횟수
        'max_depth': 8,                   # 최대 트리 깊이
        'learning_rate': 0.1,             # 학습률 (eta)
        'device': 'gpu',                  # 학습에 GPU 사용
        'boost_from_average': False       # 평균에서 부스팅 시작 안 함 (불균형 데이터에 유리)
    }
}

LIGHTGBM_REGRESSOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'boosting_type': 'gbdt',          # Gradient Boosting Decision Tree
        'objective': 'regression',        # 회귀 목적 함수
        'n_estimators': 1000,             # 부스팅 반복 횟수
        'max_depth': 8,                   # 최대 트리 깊이
        'learning_rate': 0.1,             # 학습률
        'subsample': 0.8,                 # 트리당 샘플 비율
        'colsample_bytree': 0.8,          # 트리당 feature 비율
        'device': 'gpu'                   # 학습에 GPU 사용
    }
}

# =============================================================================
# CatBoost 설정
# =============================================================================

CATBOOST_CLASSIFIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'iterations': 1000,               # 부스팅 반복 횟수
        'learning_rate': 0.1,             # 학습률
        'depth': 8,                       # 트리 깊이 (CatBoost는 'max_depth'가 아닌 'depth' 사용)
        'task_type': 'GPU',               # 학습에 GPU 사용
        'loss_function': 'Logloss',       # 이진 분류용 로그 손실
        'eval_metric': 'AUC',             # 평가용 ROC 곡선 아래 면적
        'bootstrap_type': 'Bernoulli',    # 베르누이 부트스트랩 (랜덤 샘플링)
        'subsample': 0.8,                 # 베르누이 부트스트랩 샘플링 비율
        'verbose': 100                    # 100 반복마다 지표 출력
    },
    'deep': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 10,                      # 복잡한 패턴을 위한 더 깊은 트리
        'task_type': 'GPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    }
}

CATBOOST_REGRESSOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    'default': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 8,
        'task_type': 'GPU',
        'loss_function': 'RMSE',          # 평균 제곱근 오차
        'eval_metric': 'RMSE',            # 평가용 RMSE
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    },
    'deep': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 10,                      # 회귀를 위한 더 깊은 트리
        'task_type': 'GPU',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    }
}

# =============================================================================
# Optuna 하이퍼파라미터 탐색 공간
# =============================================================================

# Optuna를 사용한 하이퍼파라미터 최적화를 위한 탐색 범위 정의
# 각 튜플은 파라미터의 (최솟값, 최댓값)을 나타냅니다
OPTUNA_SEARCH_SPACE: Dict[str, Dict[str, Tuple[int, int] | Tuple[float, float]]] = {
    'xgboost': {
        'max_depth': (6, 12),              # 트리 깊이 범위
        'learning_rate': (0.01, 0.3),      # 학습률 범위
        'n_estimators': (500, 2000),       # 트리 수 범위
        'subsample': (0.6, 1.0),           # 샘플링 비율 범위
        'colsample_bytree': (0.6, 1.0),    # feature 비율 범위
        'gamma': (0, 5),                   # 최소 손실 감소 범위
        'min_child_weight': (1, 10)        # 최소 인스턴스 가중치 합 범위
    },
    'lightgbm': {
        'max_depth': (6, 12),              # 트리 깊이 범위
        'learning_rate': (0.01, 0.3),      # 학습률 범위
        'n_estimators': (500, 2000),       # 트리 수 범위
        'num_leaves': (20, 100),           # 트리당 최대 리프 수 (LightGBM 전용)
        'min_child_samples': (10, 50),     # 리프당 최소 샘플 수
        'subsample': (0.6, 1.0),           # 샘플링 비율 범위
        'colsample_bytree': (0.6, 1.0)     # feature 비율 범위
    },
    'catboost': {
        'depth': (6, 12),                  # 트리 깊이 범위
        'learning_rate': (0.01, 0.3),      # 학습률 범위
        'iterations': (500, 2000),         # 반복 횟수 범위
        'l2_leaf_reg': (1, 10),            # L2 정규화 계수
        'subsample': (0.6, 1.0)            # 샘플링 비율 범위
    }
}

# =============================================================================
# 앙상블 설정
# =============================================================================

ENSEMBLE_CONFIG: Dict[str, Dict[str, Any]] = {
    'stacking': {
        'cv_folds': 5,                     # 메타러너 학습을 위한 교차 검증 폴드 수
        'meta_learner': 'ridge',           # 메타러너: 'ridge', 'lasso', 또는 'elasticnet'
        'meta_learner_params': {
            'alpha': 1.0                   # Ridge 회귀의 정규화 강도
        }
    },
    'voting': {
        'voting': 'soft',                  # 투표 방식: 'hard' (다수결) 또는 'soft' (확률 평균)
        'weights': None                    # 모델 가중치: None은 동일 가중치, 또는 float 리스트
    }
}

# =============================================================================
# 모델 선택 설정
# =============================================================================

# 종목 선택을 위한 임계 백분위수
# 92번째 백분위수는 모델 예측 기반 상위 8% 종목을 선택하는 것을 의미합니다
# 높은 백분위수 = 더 선별적 (더 적은 종목, 잠재적으로 더 높은 품질)
# 낮은 백분위수 = 덜 선별적 (더 많은 종목, 더 나은 분산투자)
THRESHOLD_PERCENTILE: int = 92  # 92번째 백분위수 이상의 예측값을 가진 종목 선택 (상위 8%)
