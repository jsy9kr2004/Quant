"""Yandex CatBoost를 사용한 그래디언트 부스팅 모델 래퍼입니다.

이 모듈은 GPU 가속을 지원하는 CatBoost (Categorical Boosting) 래퍼 클래스를 제공하며,
분류 및 회귀 작업을 모두 지원합니다.

CatBoost는 Yandex가 개발한 그래디언트 부스팅 프레임워크로, 범주형 특성 처리에
탁월하며 즉시 사용 가능한 강력한 성능을 제공합니다. 주요 장점:
- 과적합에 강함 (트레이딩 모델에 중요)
- 전처리 없이 범주형 특성 자동 처리
- Ordered boosting으로 예측 편향 감소
- 최소 메모리 사용량으로 빠른 GPU 학습
- 결측치 내장 처리
- 낮은 하이퍼파라미터 튜닝 요구사항
- 빠른 예측을 위한 대칭 트리 구조

CatBoostModel 클래스는 BaseModel을 확장하여 다음을 제공합니다:
- 트레이딩에 최적화된 사전 구성 설정 (default 및 deep)
- 네이티브 범주형 특성 지원
- 커스텀 하이퍼파라미터 지원
- GPU 가속 학습
- 조기 종료 지원
- 다양한 특성 중요도 유형
- 학습 파이프라인과의 쉬운 통합

사용 예시:
    기본 분류:
        from models.catboost_model import CatBoostModel

        # 모델 생성 및 빌드
        model = CatBoostModel(task='classification', config_name='default')
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
            'depth': 10,
            'learning_rate': 0.05,
            'iterations': 2000,
            'l2_leaf_reg': 5
        }
        model = CatBoostModel(task='classification')
        model.build_model(custom_params)

    복잡한 패턴을 위한 깊은 모델:
        # 사전 구성된 더 깊은 모델 사용
        model = CatBoostModel(task='classification', config_name='deep')
        model.build_model()

    회귀:
        model = CatBoostModel(task='regression')
        model.build_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    특성 중요도:
        # CatBoost는 여러 중요도 유형 지원
        importance_df = model.get_feature_importance(top_n=10)
        # 기본적으로 PredictionValuesChange 사용

Attributes:
    model_type (str): 항상 'catboost'
    task (str): 작업 유형 ('classification' 또는 'regression')
    config_name (str): 구성 프리셋 이름
    default_params (Dict): 모델의 기본 하이퍼파라미터
    model: CatBoost classifier 또는 regressor 인스턴스

Note:
    CatBoost는 특히 다음의 경우에 권장됩니다:
    - 과적합이 우려되는 경우 (대칭 트리 및 ordered boosting 사용)
    - 범주형 특성을 다룰 때
    - 최소한의 튜닝으로 강력한 성능이 필요할 때
    - GPU 메모리가 제한적일 때
"""

from catboost import CatBoostClassifier, CatBoostRegressor
from .base_model import BaseModel
from .config import CATBOOST_CLASSIFIER_CONFIGS, CATBOOST_REGRESSOR_CONFIGS
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np


class CatBoostModel(BaseModel):
    """GPU 가속 및 강력한 과적합 방지 기능을 갖춘 CatBoost 모델 래퍼입니다.

    이 클래스는 CatBoost classifier와 regressor를 래핑하여 BaseModel API와
    호환되는 일관된 인터페이스를 제공합니다. CatBoost는 과적합에 강하여
    트레이딩 애플리케이션에 특히 적합합니다.

    CatBoost 특징:
    - Ordered boosting: 예측 편향과 과적합 감소
    - 대칭 트리: 빠른 예측, 균형 잡힌 트리 구조
    - 네이티브 범주형 특성 지원 (인코딩 불필요)
    - task_type='GPU'를 통한 GPU 가속
    - 내장 정규화 (L2 leaf regularization)
    - 결측치 자동 처리
    - 다양한 특성 중요도 유형

    트레이딩에서의 장점:
    - 과적합에 강함 (금융 데이터에 중요)
    - 시장 체제 변화에 더 잘 대응
    - 하이퍼파라미터 선택에 덜 민감
    - 즉시 사용 가능한 뛰어난 성능

    Attributes:
        model_type (str): 유형 식별자, 항상 'catboost'.
        task (str): 작업 유형, 'classification' 또는 'regression'.
        config_name (str): 사용 중인 구성 프리셋 이름.
        default_params (Dict[str, Any]): 기본 하이퍼파라미터 딕셔너리.
        model (CatBoostClassifier or CatBoostRegressor): 기반 CatBoost 모델.
        feature_names (Optional[List[str]]): 특성 이름 리스트.
        is_trained (bool): 모델이 학습되었는지 나타내는 플래그.

    사용 예시:
        # 기본 설정으로 분류
        model = CatBoostModel(task='classification')
        model.build_model()
        model.fit(X_train, y_train, X_val, y_val)

        # 복잡한 패턴을 위한 깊은 모델
        model = CatBoostModel(task='classification', config_name='deep')
        model.build_model()

        # 특성 중요도 가져오기
        importance_df = model.get_feature_importance(top_n=10)
    """

    def __init__(
        self,
        task: str = 'classification',
        config_name: str = 'default'
    ) -> None:
        """지정된 구성으로 CatBoost 모델을 초기화합니다.

        Args:
            task (str, optional): 작업 유형, 'classification' 또는 'regression'.
                기본값은 'classification'.
            config_name (str, optional): 사용할 구성 프리셋 이름.
                두 작업 모두: 'default', 'deep'
                - 'default': 깊이 8, 대부분의 경우에 적합
                - 'deep': 깊이 10, 복잡한 패턴용
                기본값은 'default'.

        사용 예시:
            # 기본 분류 모델
            model = CatBoostModel(task='classification')

            # 복잡한 패턴을 위한 더 깊은 분류 모델
            model = CatBoostModel(task='classification', config_name='deep')

            # 회귀 모델
            model = CatBoostModel(task='regression')
        """
        super().__init__(model_type='catboost', task=task)
        self.config_name = config_name

        # 작업 유형에 따라 기본 구성 로드
        if task == 'classification':
            self.default_params = CATBOOST_CLASSIFIER_CONFIGS.get(
                config_name,
                CATBOOST_CLASSIFIER_CONFIGS['default']
            )
        else:
            self.default_params = CATBOOST_REGRESSOR_CONFIGS.get(
                config_name,
                CATBOOST_REGRESSOR_CONFIGS['default']
            )

    def build_model(self, params: Optional[Dict[str, Any]] = None) -> 'CatBoostModel':
        """지정된 파라미터 또는 기본 파라미터로 CatBoost 모델을 빌드합니다.

        작업 유형에 따라 CatBoost classifier 또는 regressor를 생성합니다.
        커스텀 파라미터가 제공되면 기본 파라미터와 병합되며,
        커스텀 파라미터가 우선합니다.

        Args:
            params (Optional[Dict[str, Any]], optional): 커스텀 하이퍼파라미터.
                None인 경우 구성의 기본 파라미터 사용.
                제공된 경우 기본값과 병합 (커스텀 파라미터가 기본값 재정의).
                일반 파라미터:
                - depth (int): 트리 깊이 (CatBoost는 'max_depth'가 아닌 'depth' 사용)
                - learning_rate (float): 학습률
                - iterations (int): 부스팅 반복 횟수
                - l2_leaf_reg (float): L2 정규화 계수
                - subsample (float): 배깅을 위한 샘플 비율
                - border_count (int): 수치형 특성의 분할 수
                - bootstrap_type (str): Bootstrap 유형 ('Bayesian', 'Bernoulli', 'MVS')
                기본값은 None.

        Returns:
            CatBoostModel: 메서드 체이닝을 위한 self.

        사용 예시:
            # 기본 파라미터 사용
            model.build_model()

            # 커스텀 파라미터 사용
            custom_params = {
                'depth': 12,
                'learning_rate': 0.03,
                'iterations': 2000,
                'l2_leaf_reg': 10
            }
            model.build_model(custom_params)

            # 부분 재정의 (다른 파라미터는 기본값 사용)
            model.build_model({'depth': 10, 'iterations': 1500})
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
            self.model = CatBoostClassifier(**params)
        else:
            self.model = CatBoostRegressor(**params)

        return self

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> 'CatBoostModel':
        """선택적 조기 종료와 함께 CatBoost 모델을 학습합니다.

        제공된 데이터로 CatBoost 모델을 학습합니다. 검증 데이터가 제공되면
        검증 메트릭을 모니터링하여 조기 종료를 활성화하여 과적합을 방지합니다.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): 훈련 특성.
                범주형 특성을 포함할 수 있음 (자동 처리됨).
            y_train (Union[pd.Series, np.ndarray]): 훈련 레이블.
            X_val (Optional[Union[pd.DataFrame, np.ndarray]], optional): 조기 종료를
                위한 검증 특성. 기본값은 None.
            y_val (Optional[Union[pd.Series, np.ndarray]], optional): 조기 종료를
                위한 검증 레이블. 기본값은 None.
            early_stopping_rounds (int, optional): 개선이 없는 라운드 수로,
                이 수가 지나면 학습이 중지됩니다. 검증 데이터가 제공된 경우에만 사용.
                기본값은 50.
            verbose (bool, optional): 학습 진행 상황 출력 여부.
                True인 경우 'verbose' 반복마다 메트릭 출력 (config에서).
                기본값은 True.

        Returns:
            CatBoostModel: 메서드 체이닝을 위한 self.

        Raises:
            ValueError: 모델이 빌드되지 않은 경우.

        사용 예시:
            # 조기 종료 없이 학습
            model.fit(X_train, y_train, verbose=True)

            # 조기 종료와 함께 학습
            model.fit(
                X_train, y_train,
                X_val, y_val,
                early_stopping_rounds=100,
                verbose=True
            )

            # 조용한 학습
            model.fit(X_train, y_train, verbose=False)

        Note:
            - CatBoost는 범주형 특성을 자동으로 감지하고 처리
            - 조기 종료는 config에 지정된 평가 메트릭을 모니터링
              (예: 분류의 경우 'AUC', 회귀의 경우 'RMSE')
            - 더 나은 일반화를 위해 기본적으로 ordered boosting 사용
        """
        kwargs = {
            'verbose': verbose
        }

        # 검증 데이터가 제공된 경우 조기 종료 파라미터 추가
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            kwargs['eval_set'] = eval_set
            kwargs['early_stopping_rounds'] = early_stopping_rounds

        # 준비된 kwargs로 부모 fit 메서드 호출
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)

    def get_feature_importance(self, top_n: Optional[int] = 20) -> pd.DataFrame:
        """학습된 CatBoost 모델에서 특성 중요도 점수를 가져옵니다.

        CatBoost는 여러 유형의 특성 중요도를 제공합니다. 이 메서드는 기본적으로
        'PredictionValuesChange'를 사용하며, 이는 특성 값이 변경될 때 예측의
        평균 변화를 측정합니다.

        CatBoost에서 사용 가능한 중요도 유형:
        - PredictionValuesChange: 평균 예측 변화 (기본값)
        - LossFunctionChange: 손실 함수에 미치는 영향
        - FeatureImportance: 분할 기반 중요도

        Args:
            top_n (Optional[int], optional): 반환할 상위 특성 개수.
                None인 경우 모든 특성을 반환합니다. 기본값은 20.

        Returns:
            pd.DataFrame: 'feature'와 'importance' 열이 있는 DataFrame,
                중요도 내림차순으로 정렬됨.

        Raises:
            ValueError: 모델이 학습되지 않은 경우.

        사용 예시:
            # 상위 10개의 가장 중요한 특성 가져오기
            importance_df = model.get_feature_importance(top_n=10)
            print(importance_df)

            # 특성 중요도 플롯
            importance_df.plot(x='feature', y='importance', kind='barh')

            # 모든 특성 가져오기
            importance_df = model.get_feature_importance(top_n=None)

        Note:
            CatBoost의 특성 중요도는 학습 중 사용된 ordered boosting 방식을
            고려하므로 트리 기반 메서드보다 더 강력합니다.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        # PredictionValuesChange를 사용하여 특성 중요도 가져오기 (기본값)
        importances = self.model.get_feature_importance()

        # 저장된 특성 이름 사용 또는 일반 이름 생성
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names

        # 특성 이름과 중요도 점수로 DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # 중요도 내림차순으로 정렬
        importance_df = importance_df.sort_values('importance', ascending=False)

        # 지정된 경우 상위 N개 특성 반환
        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df
