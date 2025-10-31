"""Microsoft LightGBM을 사용한 그래디언트 부스팅 모델 래퍼입니다.

이 모듈은 GPU 가속을 지원하는 LightGBM (Light Gradient Boosting Machine) 래퍼 클래스를
제공하며, 분류 및 회귀 작업을 모두 지원합니다.

LightGBM은 Microsoft가 개발한 트리 기반 학습 알고리즘을 사용하는 그래디언트 부스팅
프레임워크입니다. 주요 장점:
- 빠른 학습 속도와 높은 효율성
- 낮은 메모리 사용량
- 다른 프레임워크보다 높은 정확도
- GPU 가속 지원
- 대규모 데이터 처리에 탁월
- 범주형 특성 지원

LightGBMModel 클래스는 BaseModel을 확장하여 다음을 제공합니다:
- 트레이딩에 최적화된 사전 구성 설정
- LightGBM 호환성을 위한 자동 특성 이름 정리
- 커스텀 하이퍼파라미터 지원
- GPU 가속 학습
- 콜백을 통한 조기 종료
- 학습 파이프라인과의 쉬운 통합

사용 예제:
    기본 분류:
        from models.lightgbm_model import LightGBMModel

        # 모델 생성 및 빌드
        model = LightGBMModel(task='classification', config_name='default')
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
            'n_estimators': 1500,
            'num_leaves': 64
        }
        model = LightGBMModel(task='classification')
        model.build_model(custom_params)

    특성 이름 정리:
        # LightGBM은 유효한 특성 이름 필요 (특수 문자 없음)
        # 모델이 자동으로 특성 이름 정리
        df_cleaned = LightGBMModel.clean_feature_names(df)

    회귀:
        model = LightGBMModel(task='regression')
        model.build_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

Attributes:
    model_type (str): 항상 'lightgbm'
    task (str): 작업 유형 ('classification' 또는 'regression')
    config_name (str): 설정 프리셋 이름
    default_params (Dict): 모델의 기본 하이퍼파라미터
    model: LightGBM 분류기 또는 회귀기 인스턴스

Note:
    LightGBM은 특성 이름에 대한 엄격한 요구사항이 있습니다. [, ], <, >와 같은
    특수 문자를 포함할 수 없습니다. clean_feature_names() 메서드가 유효하지 않은
    문자를 제거하여 자동으로 처리합니다.
"""

import lightgbm as lgb
import re
from .base_model import BaseModel
from .config import LIGHTGBM_CLASSIFIER_CONFIGS, LIGHTGBM_REGRESSOR_CONFIGS
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np


class LightGBMModel(BaseModel):
    """GPU 가속 및 자동 특성 정리를 지원하는 LightGBM 모델 래퍼입니다.

    이 클래스는 LightGBM 분류기 및 회귀기를 래핑하여 BaseModel API와 호환되는
    일관된 인터페이스를 제공합니다. LightGBM의 명명 요구사항을 준수하기 위한
    자동 특성 이름 정리 기능을 포함합니다.

    LightGBM 특징:
    - Leaf-wise 트리 성장 (XGBoost의 level-wise와 대조적)
    - 더 빠른 학습을 위한 히스토그램 기반 알고리즘
    - device='gpu'를 통한 GPU 가속
    - 네이티브 범주형 특성 지원
    - XGBoost보다 낮은 메모리 사용량

    Attributes:
        model_type (str): 유형 식별자, 항상 'lightgbm'.
        task (str): 작업 유형, 'classification' 또는 'regression'.
        config_name (str): 사용 중인 설정 프리셋 이름.
        default_params (Dict[str, Any]): 기본 하이퍼파라미터 딕셔너리.
        model (lgb.LGBMClassifier or lgb.LGBMRegressor): 기반 LightGBM 모델.
        feature_names (Optional[List[str]]): 특성 이름 리스트 (정리됨).
        is_trained (bool): 모델이 학습되었는지 나타내는 플래그.

    사용 예시:
        # 기본 설정으로 분류
        model = LightGBMModel(task='classification')
        model.build_model()
        model.fit(X_train, y_train, X_val, y_val)

        # 학습 전 특성 이름 정리
        X_train_clean = LightGBMModel.clean_feature_names(X_train)
        X_test_clean = LightGBMModel.clean_feature_names(X_test)

        # 특성 중요도 가져오기
        importance_df = model.get_feature_importance(top_n=10)
    """

    def __init__(
        self,
        task: str = 'classification',
        config_name: str = 'default'
    ) -> None:
        """지정된 설정으로 LightGBM 모델을 초기화합니다.

        Args:
            task (str, optional): 작업 유형, 'classification' 또는 'regression'.
                기본값은 'classification'.
            config_name (str, optional): 사용할 설정 프리셋 이름.
                현재는 두 작업 모두 'default'만 사용 가능합니다.
                기본값은 'default'.

        사용 예시:
            # 기본 분류 모델
            model = LightGBMModel(task='classification')

            # 회귀 모델
            model = LightGBMModel(task='regression')
        """
        super().__init__(model_type='lightgbm', task=task)
        self.config_name = config_name

        # 작업 유형에 따라 기본 설정 로드
        if task == 'classification':
            self.default_params = LIGHTGBM_CLASSIFIER_CONFIGS.get(
                config_name,
                LIGHTGBM_CLASSIFIER_CONFIGS['default']
            )
        else:
            self.default_params = LIGHTGBM_REGRESSOR_CONFIGS.get(
                config_name,
                LIGHTGBM_REGRESSOR_CONFIGS['default']
            )

    def build_model(self, params: Optional[Dict[str, Any]] = None) -> 'LightGBMModel':
        """지정된 또는 기본 파라미터로 LightGBM 모델을 빌드합니다.

        작업 유형에 따라 LightGBM 분류기 또는 회귀기를 생성합니다.
        커스텀 파라미터가 제공되면 기본 파라미터와 병합하며,
        커스텀 파라미터가 우선순위를 가집니다.

        Args:
            params (Optional[Dict[str, Any]], optional): 커스텀 하이퍼파라미터.
                None인 경우 설정의 기본 파라미터 사용.
                제공된 경우 기본값과 병합 (커스텀 파라미터가 기본값 재정의).
                일반적인 파라미터:
                - max_depth (int): 최대 트리 깊이
                - learning_rate (float): 부스팅 학습률
                - n_estimators (int): 부스팅 반복 횟수
                - num_leaves (int): 트리당 최대 리프 개수
                - subsample (float): 학습용 샘플 비율
                - colsample_bytree (float): 학습용 특성 비율
                - min_child_samples (int): 리프당 최소 샘플 수
                기본값은 None.

        Returns:
            LightGBMModel: 메서드 체이닝을 위한 self.

        사용 예시:
            # 기본 파라미터 사용
            model.build_model()

            # 커스텀 파라미터 사용
            custom_params = {
                'max_depth': 12,
                'learning_rate': 0.05,
                'n_estimators': 2000,
                'num_leaves': 64
            }
            model.build_model(custom_params)

            # 부분 재정의
            model.build_model({'num_leaves': 128})
        """
        if params is None:
            params = self.default_params
        else:
            # 커스텀 파라미터를 기본값과 병합
            merged_params = self.default_params.copy()
            merged_params.update(params)
            params = merged_params

        # 작업에 따라 적절한 모델 유형 생성
        if self.task == 'classification':
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

        return self

    @staticmethod
    def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
        """LightGBM 요구사항을 준수하도록 특성 이름을 정리합니다.

        LightGBM은 특성 이름이 영숫자와 언더스코어만 포함하도록 요구합니다.
        이 메서드는 유효하지 않은 문자(괄호, 특수 기호 등)를 제거하고
        인덱스를 추가하여 중복 이름을 처리합니다.

        Args:
            df (pd.DataFrame): 유효하지 않을 수 있는 특성 이름을 가진 DataFrame.

        Returns:
            pd.DataFrame: 정리된 특성 이름을 가진 DataFrame.

        사용 예시:
            # 컬럼 이름에 특수 문자가 있는 DataFrame
            df = pd.DataFrame({
                'feature[1]': [1, 2, 3],
                'feature<2>': [4, 5, 6],
                'normal_feature': [7, 8, 9]
            })
            df_clean = LightGBMModel.clean_feature_names(df)
            print(df_clean.columns)
            Index(['feature1', 'feature2', 'normal_feature'], dtype='object')

            # 정리 후 중복 이름 처리
            df = pd.DataFrame({
                'feature[1]': [1, 2, 3],
                'feature(1)': [4, 5, 6]
            })
            df_clean = LightGBMModel.clean_feature_names(df)
            print(df_clean.columns)
            Index(['feature1', 'feature1_1'], dtype='object')

        Note:
            - A-Z, a-z, 0-9, 언더스코어를 제외한 모든 문자 제거
            - 컬럼 순서 보존
            - 정리 후 중복 이름 자동 처리
        """
        # 영숫자가 아닌 모든 문자 제거 (언더스코어 제외)
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
        new_n_list = list(new_names.values())

        # 인덱스를 추가하여 중복 이름 처리
        # 정리된 이름이 여러 번 나타나면 _1, _2 등 추가
        new_names = {
            col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }

        df = df.rename(columns=new_names)
        return df

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 100
    ) -> 'LightGBMModel':
        """자동 특성 정리 및 조기 종료와 함께 LightGBM 모델을 학습합니다.

        제공된 데이터로 LightGBM 모델을 학습합니다. LightGBM 요구사항을 준수하도록
        특성 이름을 자동으로 정리합니다. 검증 데이터가 제공되면 LightGBM 콜백을
        사용하여 조기 종료를 활성화합니다.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): 훈련 특성.
                DataFrame인 경우 특성 이름이 자동으로 정리됩니다.
            y_train (Union[pd.Series, np.ndarray]): 훈련 레이블.
            X_val (Optional[Union[pd.DataFrame, np.ndarray]], optional): 조기 종료를
                위한 검증 특성. DataFrame인 경우 정리됩니다.
                기본값은 None.
            y_val (Optional[Union[pd.Series, np.ndarray]], optional): 조기 종료를
                위한 검증 레이블. 기본값은 None.
            early_stopping_rounds (int, optional): 개선이 없을 경우 학습을 중지할
                라운드 수. 검증 데이터가 제공된 경우에만 사용됩니다.
                기본값은 50.
            verbose (int, optional): 로깅 빈도. 'verbose' 반복마다 메트릭을 출력합니다.
                0으로 설정하면 무음 학습. 기본값은 100.

        Returns:
            LightGBMModel: 메서드 체이닝을 위한 self.

        Raises:
            ValueError: 모델이 빌드되지 않은 경우.

        사용 예시:
            # 조기 종료 없이 학습
            model.fit(X_train, y_train, verbose=100)

            # 조기 종료와 함께 학습
            model.fit(
                X_train, y_train,
                X_val, y_val,
                early_stopping_rounds=100,
                verbose=50
            )

            # 무음 학습
            model.fit(X_train, y_train, verbose=0)

        Note:
            - 입력이 DataFrame인 경우 특성 이름이 자동으로 정리됨
            - 조기 종료는 lgb.early_stopping 및 lgb.log_evaluation 콜백 사용
            - 검증 메트릭은 작업에 따라 다름 (분류는 binary log loss,
              회귀는 RMSE)
        """
        # 입력이 DataFrame인 경우 특성 이름 정리 (LightGBM 요구사항)
        if hasattr(X_train, 'columns'):
            X_train = self.clean_feature_names(X_train)
            if X_val is not None:
                X_val = self.clean_feature_names(X_val)

        kwargs = {}

        # 검증 데이터가 제공된 경우 조기 종료 및 로깅을 위한 콜백 추가
        if X_val is not None and y_val is not None:
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=verbose)
            ]
            kwargs['callbacks'] = callbacks

        # 준비된 kwargs로 부모 fit 메서드 호출
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
