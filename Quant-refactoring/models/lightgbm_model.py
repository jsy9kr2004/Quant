"""
LightGBM model wrapper
"""

import lightgbm as lgb
import re
from .base_model import BaseModel
from .config import LIGHTGBM_CLASSIFIER_CONFIGS, LIGHTGBM_REGRESSOR_CONFIGS
from typing import Dict, Any


class LightGBMModel(BaseModel):
    """LightGBM 모델 래퍼"""

    def __init__(self, task: str = 'classification', config_name: str = 'default'):
        """
        Initialize LightGBM model

        Args:
            task: 'classification' or 'regression'
            config_name: 설정 이름 ('default')
        """
        super().__init__(model_type='lightgbm', task=task)
        self.config_name = config_name

        # 기본 설정 로드
        if task == 'classification':
            self.default_params = LIGHTGBM_CLASSIFIER_CONFIGS.get(config_name,
                                                                  LIGHTGBM_CLASSIFIER_CONFIGS['default'])
        else:
            self.default_params = LIGHTGBM_REGRESSOR_CONFIGS.get(config_name,
                                                                 LIGHTGBM_REGRESSOR_CONFIGS['default'])

    def build_model(self, params: Dict[str, Any] = None):
        """
        LightGBM 모델 생성

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
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

        return self

    @staticmethod
    def clean_feature_names(df):
        """
        LightGBM용 특징 이름 정리
        (JSON 특수문자 제거)

        Args:
            df: DataFrame

        Returns:
            정리된 DataFrame
        """
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
        new_n_list = list(new_names.values())

        # 중복 이름 처리
        new_names = {
            col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }

        df = df.rename(columns=new_names)
        return df

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            early_stopping_rounds: int = 50, verbose: int = 100):
        """
        LightGBM 학습

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터
            y_val: 검증 레이블
            early_stopping_rounds: Early stopping 라운드
            verbose: 로그 출력 빈도
        """
        # 특징 이름 정리 (LightGBM 요구사항)
        if hasattr(X_train, 'columns'):
            X_train = self.clean_feature_names(X_train)
            if X_val is not None:
                X_val = self.clean_feature_names(X_val)

        kwargs = {}

        if X_val is not None and y_val is not None:
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=verbose)
            ]
            kwargs['callbacks'] = callbacks

        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
