"""
Model configuration for different ML algorithms
"""

# XGBoost 설정
XGBOOST_CLASSIFIER_CONFIGS = {
    'default': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 500,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    },
    'depth_9': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 500,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 9,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    },
    'depth_10': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 500,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 10,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
}

XGBOOST_REGRESSOR_CONFIGS = {
    'default': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 8,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    },
    'deep': {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 10,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
}

# LightGBM 설정
LIGHTGBM_CLASSIFIER_CONFIGS = {
    'default': {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.1,
        'device': 'gpu',
        'boost_from_average': False
    }
}

LIGHTGBM_REGRESSOR_CONFIGS = {
    'default': {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'device': 'gpu'
    }
}

# CatBoost 설정 (신규)
CATBOOST_CLASSIFIER_CONFIGS = {
    'default': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 8,
        'task_type': 'GPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    },
    'deep': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 10,
        'task_type': 'GPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    }
}

CATBOOST_REGRESSOR_CONFIGS = {
    'default': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 8,
        'task_type': 'GPU',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    },
    'deep': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 10,
        'task_type': 'GPU',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'verbose': 100
    }
}

# Optuna 하이퍼파라미터 탐색 범위
OPTUNA_SEARCH_SPACE = {
    'xgboost': {
        'max_depth': (6, 12),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (500, 2000),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'gamma': (0, 5),
        'min_child_weight': (1, 10)
    },
    'lightgbm': {
        'max_depth': (6, 12),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (500, 2000),
        'num_leaves': (20, 100),
        'min_child_samples': (10, 50),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    },
    'catboost': {
        'depth': (6, 12),
        'learning_rate': (0.01, 0.3),
        'iterations': (500, 2000),
        'l2_leaf_reg': (1, 10),
        'subsample': (0.6, 1.0)
    }
}

# 앙상블 설정
ENSEMBLE_CONFIG = {
    'stacking': {
        'cv_folds': 5,
        'meta_learner': 'ridge',  # 'ridge', 'lasso', 'elasticnet'
        'meta_learner_params': {
            'alpha': 1.0
        }
    },
    'voting': {
        'voting': 'soft',  # 'hard' or 'soft'
        'weights': None  # None for equal weights
    }
}

# 모델 선택 임계값 설정
THRESHOLD_PERCENTILE = 92  # 상위 8% 선택
