"""그래디언트 부스팅을 위한 XGBoost 모델 래퍼입니다.

이 모듈은 XGBoost(Extreme Gradient Boosting) 모델을 위한 래퍼 클래스를 제공하며,
GPU 가속을 지원하는 분류 및 회귀 작업을 모두 지원합니다.

XGBoost는 의사결정 트리를 기본 학습기로 사용하는 강력한 그래디언트 부스팅
프레임워크입니다. 다음과 같은 특징이 있습니다:
- 높은 성능과 속도
- GPU 가속 지원
- 과적합 방지를 위한 내장 정규화
- 특성 중요도 점수
- 조기 종료 기능

XGBoostModel 클래스는 BaseModel을 확장하여 다음을 제공합니다:
- 일반적인 사용 사례를 위한 사전 구성 설정
- 커스텀 하이퍼파라미터 지원
- GPU 가속 학습
- 조기 종료 지원
- 학습 파이프라인과의 쉬운 통합

사용 예제:
    기본 분류:
        from models.xgboost_model import XGBoostModel

        # 기본 설정으로 모델 생성 및 빌드
        model = XGBoostModel(task='classification', config_name='default')
        model.build_model()

        # 조기 종료와 함께 학습
        model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)

        # 예측
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # 평가
        metrics = model.evaluate(X_test, y_test)
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    커스텀 설정:
        # 커스텀 하이퍼파라미터 사용
        custom_params = {
            'max_depth': 10,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.9
        }
        model = XGBoostModel(task='classification')
        model.build_model(custom_params)

    다른 깊이 설정:
        # 사전 구성된 깊은 모델 사용
        model = XGBoostModel(task='classification', config_name='depth_10')
        model.build_model()

    회귀:
        model = XGBoostModel(task='regression', config_name='deep')
        model.build_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

Attributes:
    model_type (str): 항상 'xgboost'
    task (str): 작업 유형 ('classification' 또는 'regression')
    config_name (str): 설정 프리셋 이름
    default_params (Dict): 모델의 기본 하이퍼파라미터
    model: XGBoost 분류기 또는 회귀기 인스턴스
"""

import xgboost as xgb
from .base_model import BaseModel
from .config import XGBOOST_CLASSIFIER_CONFIGS, XGBOOST_REGRESSOR_CONFIGS
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np


class XGBoostModel(BaseModel):
    """GPU 가속을 지원하는 XGBoost 모델 래퍼입니다.

    이 클래스는 XGBoost 분류기와 회귀기를 래핑하여 BaseModel API와
    호환되는 일관된 인터페이스를 제공합니다. 다양한 시나리오에 최적화된
    사전 구성 설정을 포함하며 GPU 가속을 지원합니다.

    XGBoost 기능:
    - 트리 기반 그래디언트 부스팅
    - tree_method='gpu_hist'를 통한 GPU 가속
    - 정규화 (L1 및 L2)
    - 컬럼 및 행 서브샘플링
    - 조기 종료 지원

    Attributes:
        model_type (str): 항상 'xgboost'인 타입 식별자.
        task (str): 'classification' 또는 'regression' 작업 유형.
        config_name (str): 사용 중인 설정 프리셋 이름.
        default_params (Dict[str, Any]): 기본 하이퍼파라미터 딕셔너리.
        model (xgb.XGBClassifier or xgb.XGBRegressor): 기반 XGBoost 모델.
        feature_names (Optional[List[str]]): 특성 이름 리스트.
        is_trained (bool): 모델이 학습되었는지 나타내는 플래그.

    Example:
        >>> # 기본 설정으로 분류
        >>> model = XGBoostModel(task='classification', config_name='default')
        >>> model.build_model()
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>>
        >>> # 특성 중요도 가져오기
        >>> importance_df = model.get_feature_importance(top_n=10)
        >>> print(importance_df)
        >>>
        >>> # 저장 및 로드
        >>> model.save('model.pkl')
        >>> loaded_model = XGBoostModel()
        >>> loaded_model.load('model.pkl')
    """

    def __init__(
        self,
        task: str = 'classification',
        config_name: str = 'default'
    ) -> None:
        """지정된 설정으로 XGBoost 모델을 초기화합니다.

        Args:
            task (str, optional): 'classification' 또는 'regression' 작업 유형.
                기본값은 'classification'.
            config_name (str, optional): 사용할 설정 프리셋 이름.
                분류용: 'default', 'depth_9', 'depth_10'
                회귀용: 'default', 'deep'
                기본값은 'default'.

        Example:
            >>> # 기본 분류 모델
            >>> model = XGBoostModel(task='classification')
            >>>
            >>> # 더 깊은 분류 모델
            >>> model = XGBoostModel(task='classification', config_name='depth_10')
            >>>
            >>> # 회귀 모델
            >>> model = XGBoostModel(task='regression', config_name='deep')
        """
        super().__init__(model_type='xgboost', task=task)
        self.config_name = config_name

        # Load default configuration based on task type
        if task == 'classification':
            self.default_params = XGBOOST_CLASSIFIER_CONFIGS.get(
                config_name,
                XGBOOST_CLASSIFIER_CONFIGS['default']
            )
        else:
            self.default_params = XGBOOST_REGRESSOR_CONFIGS.get(
                config_name,
                XGBOOST_REGRESSOR_CONFIGS['default']
            )

    def build_model(self, params: Optional[Dict[str, Any]] = None) -> 'XGBoostModel':
        """지정되거나 기본 파라미터로 XGBoost 모델을 빌드합니다.

        작업 유형에 따라 XGBoost 분류기 또는 회귀기를 생성합니다.
        커스텀 파라미터가 제공되면 기본 파라미터와 병합되며,
        커스텀 파라미터가 우선합니다.

        Args:
            params (Optional[Dict[str, Any]], optional): 커스텀 하이퍼파라미터.
                None인 경우 설정의 기본 파라미터 사용.
                제공된 경우 기본값과 병합 (커스텀 파라미터가 기본값 덮어씀).
                일반적인 파라미터:
                - max_depth (int): 최대 트리 깊이
                - learning_rate (float): 스텝 크기 축소
                - n_estimators (int): 부스팅 라운드 수
                - subsample (float): 트리당 사용되는 샘플 비율
                - colsample_bytree (float): 트리당 사용되는 특성 비율
                - gamma (float): 분할을 위한 최소 손실 감소
                기본값은 None.

        Returns:
            XGBoostModel: 메서드 체이닝을 위한 self.

        Example:
            >>> # 기본 파라미터 사용
            >>> model.build_model()
            >>>
            >>> # 커스텀 파라미터 사용
            >>> custom_params = {
            ...     'max_depth': 12,
            ...     'learning_rate': 0.05,
            ...     'n_estimators': 2000,
            ...     'gamma': 1.0
            ... }
            >>> model.build_model(custom_params)
            >>>
            >>> # 부분 덮어쓰기 (다른 파라미터는 기본값 사용)
            >>> model.build_model({'max_depth': 10})
        """
        if params is None:
            params = self.default_params
        else:
            # Merge custom parameters with defaults
            merged_params = self.default_params.copy()
            merged_params.update(params)
            params = merged_params

        # Create appropriate model type based on task
        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)

        return self

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> 'XGBoostModel':
        """선택적 조기 종료와 함께 XGBoost 모델을 학습시킵니다.

        제공된 데이터로 XGBoost 모델을 학습시킵니다. 검증 데이터가
        제공되면 검증 메트릭을 모니터링하여 과적합을 방지하는
        조기 종료를 활성화합니다.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): 학습 특성.
            y_train (Union[pd.Series, np.ndarray]): 학습 레이블.
            X_val (Optional[Union[pd.DataFrame, np.ndarray]], optional):
                조기 종료를 위한 검증 특성. 기본값은 None.
            y_val (Optional[Union[pd.Series, np.ndarray]], optional):
                조기 종료를 위한 검증 레이블. 기본값은 None.
            early_stopping_rounds (int, optional): 개선이 없는 라운드 수,
                이후 학습이 중지됨. 검증 데이터가 제공된 경우에만 사용.
                기본값은 50.
            verbose (bool, optional): 학습 진행 상황 출력 여부.
                기본값은 True.

        Returns:
            XGBoostModel: 메서드 체이닝을 위한 self.

        Raises:
            ValueError: 모델이 빌드되지 않은 경우.

        Example:
            >>> # 조기 종료 없이 학습
            >>> model.fit(X_train, y_train, verbose=True)
            >>>
            >>> # 조기 종료와 함께 학습
            >>> model.fit(
            ...     X_train, y_train,
            ...     X_val, y_val,
            ...     early_stopping_rounds=100,
            ...     verbose=True
            ... )
            >>>
            >>> # 조용한 학습
            >>> model.fit(X_train, y_train, verbose=False)

        Note:
            조기 종료는 모델 설정에서 지정된 평가 메트릭을 모니터링합니다
            (예: 분류의 경우 'logloss', 회귀의 경우 'rmse').
        """
        kwargs = {
            'verbose': verbose
        }

        # Add early stopping parameters if validation data is provided
        if X_val is not None and y_val is not None:
            kwargs['early_stopping_rounds'] = early_stopping_rounds
            kwargs['eval_metric'] = self.model.eval_metric

        # Call parent fit method with prepared kwargs
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
