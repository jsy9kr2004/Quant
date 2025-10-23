"""
XGBoost model wrapper
"""

import xgboost as xgb
from .base_model import BaseModel
from .config import XGBOOST_CLASSIFIER_CONFIGS, XGBOOST_REGRESSOR_CONFIGS
from typing import Dict, Any


class XGBoostModel(BaseModel):
    """XGBoost 모델 래퍼"""

    def __init__(self, task: str = 'classification', config_name: str = 'default'):
        """
        Initialize XGBoost model

        Args:
            task: 'classification' or 'regression'
            config_name: 설정 이름 ('default', 'depth_9', 'depth_10', 'deep')
        """
        super().__init__(model_type='xgboost', task=task)
        self.config_name = config_name

        # 기본 설정 로드
        if task == 'classification':
            self.default_params = XGBOOST_CLASSIFIER_CONFIGS.get(config_name,
                                                                 XGBOOST_CLASSIFIER_CONFIGS['default'])
        else:
            self.default_params = XGBOOST_REGRESSOR_CONFIGS.get(config_name,
                                                                XGBOOST_REGRESSOR_CONFIGS['default'])

    def build_model(self, params: Dict[str, Any] = None):
        """
        XGBoost 모델 생성

        Args:
            params: 커스텀 파라미터 (없으면 기본값 사용)
        """
        if params is None:
            params = self.default_params
        else:
            # 기본값과 병합
            merged_params = self.default_params.copy()
            merged_params.update(params)
            params = merged_params

        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)

        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            early_stopping_rounds: int = 50, verbose: bool = True):
        """
        XGBoost 학습 (early stopping 지원)

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터
            y_val: 검증 레이블
            early_stopping_rounds: Early stopping 라운드
            verbose: 로그 출력 여부
        """
        kwargs = {
            'verbose': verbose
        }

        if X_val is not None and y_val is not None:
            kwargs['early_stopping_rounds'] = early_stopping_rounds
            kwargs['eval_metric'] = self.model.eval_metric

        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
