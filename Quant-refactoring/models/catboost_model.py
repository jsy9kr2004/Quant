"""
CatBoost model wrapper (NEW)
"""

from catboost import CatBoostClassifier, CatBoostRegressor
from .base_model import BaseModel
from .config import CATBOOST_CLASSIFIER_CONFIGS, CATBOOST_REGRESSOR_CONFIGS
from typing import Dict, Any


class CatBoostModel(BaseModel):
    """
    CatBoost 모델 래퍼 (신규)

    Advantages:
    - Robust to overfitting (중요!)
    - Handles categorical features automatically
    - Ordered boosting reduces prediction bias
    - Fast GPU training
    """

    def __init__(self, task: str = 'classification', config_name: str = 'default'):
        """
        Initialize CatBoost model

        Args:
            task: 'classification' or 'regression'
            config_name: 설정 이름 ('default', 'deep')
        """
        super().__init__(model_type='catboost', task=task)
        self.config_name = config_name

        # 기본 설정 로드
        if task == 'classification':
            self.default_params = CATBOOST_CLASSIFIER_CONFIGS.get(config_name,
                                                                  CATBOOST_CLASSIFIER_CONFIGS['default'])
        else:
            self.default_params = CATBOOST_REGRESSOR_CONFIGS.get(config_name,
                                                                 CATBOOST_REGRESSOR_CONFIGS['default'])

    def build_model(self, params: Dict[str, Any] = None):
        """
        CatBoost 모델 생성

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
            self.model = CatBoostClassifier(**params)
        else:
            self.model = CatBoostRegressor(**params)

        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            early_stopping_rounds: int = 50, verbose: bool = True):
        """
        CatBoost 학습

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
            eval_set = (X_val, y_val)
            kwargs['eval_set'] = eval_set
            kwargs['early_stopping_rounds'] = early_stopping_rounds

        return super().fit(X_train, y_train, X_val, y_val, **kwargs)

    def get_feature_importance(self, top_n: int = 20):
        """
        CatBoost 특징 중요도 (여러 타입 지원)

        CatBoost는 여러 중요도 타입 제공:
        - PredictionValuesChange
        - LossFunctionChange
        - FeatureImportance
        """
        import pandas as pd

        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        # 기본: PredictionValuesChange
        importances = self.model.get_feature_importance()

        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df
