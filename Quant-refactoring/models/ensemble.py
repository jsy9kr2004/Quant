"""여러 기본 학습기를 결합하기 위한 앙상블 모델입니다.

이 모듈은 여러 기본 모델의 예측을 결합하기 위한 두 가지 앙상블 전략을 제공합니다:
Stacking과 Voting. 앙상블 방법은 다양한 알고리즘의 장점을 활용하여 개별 모델보다
우수한 성능을 발휘하는 경우가 많습니다.

앙상블 전략:
1. Stacking Ensemble:
   - 메타 학습을 사용하여 기본 모델 예측 결합
   - 기본 모델 출력에 대해 메타 학습기(예: Ridge, Lasso) 학습
   - 일반적으로 단순 평균보다 나은 성능 제공
   - 교차 검증을 사용하여 메타 특성 생성

2. Voting Ensemble:
   - 간단하지만 효과적인 결합 방법
   - Hard voting: 다수결 투표 (분류)
   - Soft voting: 평균 확률 (분류)
   - 평균 예측 (회귀)
   - 다양한 모델 기여도에 대한 가중 투표 옵션

사용 예제:
    Stacking Ensemble:
        from models.ensemble import StackingEnsemble
        from models.xgboost_model import XGBoostModel
        from models.lightgbm_model import LightGBMModel
        from models.catboost_model import CatBoostModel

        # 기본 모델 생성
        xgb = XGBoostModel(task='classification').build_model()
        lgb = LightGBMModel(task='classification').build_model()
        cat = CatBoostModel(task='classification').build_model()

        # 기본 모델 학습
        xgb.fit(X_train, y_train, X_val, y_val)
        lgb.fit(X_train, y_train, X_val, y_val)
        cat.fit(X_train, y_train, X_val, y_val)

        # 스태킹 앙상블 생성
        base_models = [('xgb', xgb.model), ('lgb', lgb.model), ('cat', cat.model)]
        stacking = StackingEnsemble(
            base_models=base_models,
            task='classification',
            meta_learner='ridge',
            cv=5
        )
        stacking.build_ensemble()
        stacking.fit(X_train, y_train)

        # 예측
        predictions = stacking.predict(X_test)
        probabilities = stacking.predict_proba(X_test)

    Voting Ensemble:
        from models.ensemble import VotingEnsemble

        # soft voting으로 투표 앙상블 생성
        base_models = [('xgb', xgb.model), ('lgb', lgb.model), ('cat', cat.model)]
        voting = VotingEnsemble(
            base_models=base_models,
            task='classification',
            voting='soft',
            weights=[1.0, 1.2, 1.1]  # 선택적 가중치
        )
        voting.build_ensemble()
        voting.fit(X_train, y_train)

        # 예측 생성
        predictions = voting.predict(X_test)

Attributes:
    StackingEnsemble: 메타 학습 기반 앙상블
    VotingEnsemble: 투표 기반 앙상블

Note:
    - Stacking은 일반적으로 더 나은 성능을 보이지만 더 복잡함
    - Voting은 더 간단하고 학습 속도가 빠름
    - 두 방법 모두 다양한 기본 모델에서 이점을 얻음
    - 앙상블을 생성하기 전에 기본 모델이 이미 학습되어 있어야 함
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from typing import List, Tuple, Dict, Any, Optional, Union
import joblib
from pathlib import Path


class StackingEnsemble:
    """메타 학습을 사용하여 기본 모델을 결합하는 스태킹 앙상블입니다.

    Stacking (Stacked Generalization)은 메타 학습기를 사용하여 여러 기본 모델을
    결합하는 앙상블 방법입니다. 기본 모델이 예측을 수행하고, 메타 학습기는 이러한
    예측을 사용하여 학습되어 최종 출력을 생성합니다.

    프로세스:
    1. 기본 모델이 훈련 데이터에서 학습됨
    2. 교차 검증을 사용하여 기본 모델 예측 생성
    3. 기본 모델 예측을 사용하여 메타 학습기 학습
    4. 최종 예측은 기본 모델 + 메타 학습기 사용

    이 접근 방식은 메타 학습기가 기본 모델 예측을 결합하는 최적의 방법을 학습할 수
    있기 때문에 일반적으로 단순 평균보다 우수한 성능을 발휘합니다.

    Attributes:
        base_models (List[Tuple[str, Any]]): 기본 학습기를 위한 (이름, 모델) 튜플 리스트.
        task (str): 'classification' 또는 'regression' 작업 유형.
        meta_learner_name (str): 메타 학습기 알고리즘 이름.
        meta_learner_params (Dict[str, Any]): 메타 학습기의 하이퍼파라미터.
        cv (int): 메타 특성 생성을 위한 교차 검증 폴드 수.
        ensemble (StackingClassifier or StackingRegressor): scikit-learn 스태킹 모델.
        is_trained (bool): 앙상블이 학습되었는지 나타내는 플래그.

    사용 예시:
        # 기본 모델 생성 (이미 학습됨)
        base_models = [
            ('xgboost', xgb_model.model),
            ('lightgbm', lgb_model.model),
            ('catboost', cat_model.model)
        ]

        # 스태킹 앙상블 생성
        stacking = StackingEnsemble(
            base_models=base_models,
            task='classification',
            meta_learner='ridge',
            cv=5
        )
        stacking.build_ensemble()
        stacking.fit(X_train, y_train)

        # 각 기본 모델에서 예측 가져오기
        base_preds = stacking.get_base_predictions(X_test)
    """

    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        task: str = 'classification',
        meta_learner: str = 'ridge',
        meta_learner_params: Optional[Dict[str, Any]] = None,
        cv: int = 5
    ) -> None:
        """스태킹 앙상블을 초기화합니다.

        Args:
            base_models (List[Tuple[str, Any]]): (이름, 모델) 튜플 리스트.
                각 튜플은 문자열 이름과 학습된 모델 인스턴스를 포함합니다.
                예시: [('xgb', xgb_model.model), ('lgb', lgb_model.model)]
            task (str, optional): 'classification' 또는 'regression' 작업 유형.
                기본값은 'classification'.
            meta_learner (str, optional): 메타 학습기 알고리즘 이름.
                분류의 경우: 'logistic'
                회귀의 경우: 'ridge', 'lasso', 'elasticnet'
                기본값은 'ridge'.
            meta_learner_params (Optional[Dict[str, Any]], optional): 메타 학습기의
                하이퍼파라미터. None인 경우 기본값 사용. 기본값은 None.
            cv (int, optional): 메타 학습기 학습을 위한 메타 특성 생성에 사용되는
                교차 검증 폴드 수. 기본값은 5.

        사용 예시:
            stacking = StackingEnsemble(
                base_models=[('xgb', xgb.model), ('lgb', lgb.model)],
                task='classification',
                meta_learner='logistic',
                meta_learner_params={'C': 1.0},
                cv=5
            )
        """
        self.base_models = base_models
        self.task = task
        self.meta_learner_name = meta_learner
        self.meta_learner_params = meta_learner_params or {}
        self.cv = cv
        self.ensemble = None
        self.is_trained = False

    def _create_meta_learner(self) -> Union[LogisticRegression, Ridge, Lasso, ElasticNet]:
        """메타 학습기 모델을 생성합니다.

        Returns:
            Union[LogisticRegression, Ridge, Lasso, ElasticNet]: 설정된 메타 학습기.

        Note:
            분류의 경우, meta_learner_name에 관계없이 LogisticRegression이 사용됩니다.
            회귀의 경우, 지정된 알고리즘(ridge/lasso/elasticnet)이 사용됩니다.
        """
        if self.task == 'classification':
            # 분류의 경우, 로지스틱 회귀를 메타 학습기로 사용
            if self.meta_learner_name == 'logistic':
                return LogisticRegression(**self.meta_learner_params)
            else:
                # 분류의 기본값은 로지스틱 회귀
                return LogisticRegression(**self.meta_learner_params)
        else:  # regression
            # 회귀의 경우, 여러 메타 학습기 옵션 지원
            if self.meta_learner_name == 'ridge':
                return Ridge(**self.meta_learner_params)
            elif self.meta_learner_name == 'lasso':
                return Lasso(**self.meta_learner_params)
            elif self.meta_learner_name == 'elasticnet':
                return ElasticNet(**self.meta_learner_params)
            else:
                # 회귀의 기본값은 Ridge
                return Ridge(**self.meta_learner_params)

    def build_ensemble(self) -> 'StackingEnsemble':
        """스태킹 앙상블 모델을 빌드합니다.

        지정된 기본 모델과 메타 학습기로 scikit-learn StackingClassifier 또는
        StackingRegressor를 생성합니다. 교차 검증을 사용하여 메타 학습기 학습을 위한
        예측을 생성합니다.

        Returns:
            StackingEnsemble: 메서드 체이닝을 위한 self.

        사용 예시:
            stacking.build_ensemble()
            print("앙상블이 성공적으로 빌드되었습니다")
        """
        meta_learner = self._create_meta_learner()

        # 작업 유형에 따라 스태킹 모델 생성
        if self.task == 'classification':
            self.ensemble = StackingClassifier(
                estimators=self.base_models,
                final_estimator=meta_learner,
                cv=self.cv,
                n_jobs=-1
            )
        else:
            self.ensemble = StackingRegressor(
                estimators=self.base_models,
                final_estimator=meta_learner,
                cv=self.cv,
                n_jobs=-1
            )

        logging.info(f"Stacking ensemble created with {len(self.base_models)} base models")
        logging.info(f"   Meta-learner: {self.meta_learner_name}")
        logging.info(f"   CV folds: {self.cv}")

        return self

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> 'StackingEnsemble':
        """스태킹 앙상블을 학습합니다.

        제공된 데이터를 사용하여 스태킹 앙상블을 학습합니다. 기본 모델은 이미 학습된
        상태여야 합니다. 앙상블은 내부 교차 검증을 사용하여 메타 학습기 학습을 위한
        메타 특성을 생성합니다.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): 훈련 특성.
            y_train (Union[pd.Series, np.ndarray]): 훈련 레이블.

        Returns:
            StackingEnsemble: 메서드 체이닝을 위한 self.

        Raises:
            ValueError: 앙상블이 빌드되지 않은 경우.

        사용 예시:
            stacking.fit(X_train, y_train)
            print("앙상블 학습이 완료되었습니다")
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build_ensemble() first.")

        logging.info("Training stacking ensemble...")
        logging.info(f"  Training samples: {len(X_train):,}")

        # 참고: 기본 모델은 이미 학습된 상태여야 함
        # 스태킹은 내부적으로 CV를 사용하여 메타 특성 생성
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True

        logging.info("Stacking ensemble training completed")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """스태킹 앙상블을 사용하여 예측을 생성합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 예측을 위한 입력 특성.

        Returns:
            np.ndarray: 예측 값. 분류의 경우 클래스 레이블 반환.
                회귀의 경우 연속 값 반환.

        Raises:
            ValueError: 앙상블이 학습되지 않은 경우.

        사용 예시:
            predictions = stacking.predict(X_test)
            print(f"예측: {predictions[:5]}")
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """스태킹 앙상블을 사용하여 클래스 확률을 예측합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 예측을 위한 입력 특성.

        Returns:
            np.ndarray: 확률 추정치. 형상은 (n_samples, n_classes).

        Raises:
            ValueError: 작업이 분류가 아니거나 앙상블이 학습되지 않은 경우.

        사용 예시:
            proba = stacking.predict_proba(X_test)
            positive_proba = proba[:, 1]  # 양성 클래스의 확률
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict_proba(X)

    def get_base_predictions(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """각 기본 모델에서 예측을 가져옵니다.

        개별 모델 기여도를 분석하는 데 유용하도록, 모든 기본 모델의 예측을 DataFrame으로
        반환합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 예측을 위한 입력 특성.

        Returns:
            pd.DataFrame: 각 기본 모델에 대한 열을 포함하는 DataFrame으로,
                각 모델의 예측이 담겨 있습니다. 분류의 경우 양성 클래스의
                확률을 반환합니다.

        Raises:
            ValueError: 앙상블이 학습되지 않은 경우.

        사용 예시:
            base_preds = stacking.get_base_predictions(X_test)
            print(base_preds.head())
            # 기본 모델 예측 간의 상관관계 분석
            print(base_preds.corr())
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        predictions = {}

        # 각 기본 모델에서 예측 가져오기
        for name, model in self.base_models:
            if self.task == 'classification':
                # 분류의 경우, 양성 클래스의 확률 가져오기
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                # 회귀의 경우, 직접 예측 가져오기
                predictions[name] = model.predict(X)

        return pd.DataFrame(predictions)

    def save(self, path: str) -> None:
        """학습된 앙상블을 디스크에 저장합니다.

        Args:
            path (str): 앙상블을 저장할 파일 경로.

        Raises:
            ValueError: 앙상블이 학습되지 않은 경우.

        사용 예시:
            stacking.save('/path/to/stacking_ensemble.pkl')
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Cannot save untrained ensemble.")

        # 상위 디렉토리가 없으면 생성
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # 메타데이터와 함께 앙상블 패키징
        ensemble_data = {
            'ensemble': self.ensemble,
            'task': self.task,
            'meta_learner': self.meta_learner_name,
            'cv': self.cv
        }

        joblib.dump(ensemble_data, path)
        logging.info(f"Ensemble saved to: {path}")

    def load(self, path: str) -> None:
        """디스크에서 학습된 앙상블을 로드합니다.

        Args:
            path (str): 저장된 앙상블 파일 경로.

        Raises:
            FileNotFoundError: 앙상블 파일이 존재하지 않는 경우.

        사용 예시:
            stacking = StackingEnsemble(base_models=[], task='classification')
            stacking.load('/path/to/stacking_ensemble.pkl')
            predictions = stacking.predict(X_test)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        # 파일에서 앙상블 데이터 로드
        ensemble_data = joblib.load(path)

        # 앙상블 상태 복원
        self.ensemble = ensemble_data['ensemble']
        self.task = ensemble_data['task']
        self.meta_learner_name = ensemble_data['meta_learner']
        self.cv = ensemble_data['cv']
        self.is_trained = True

        logging.info(f"Ensemble loaded from: {path}")


class VotingEnsemble:
    """투표를 통해 기본 모델을 결합하는 투표 앙상블입니다.

    Voting은 투표(분류) 또는 평균(회귀)을 통해 여러 기본 모델의 예측을 결합하는
    간단하지만 효과적인 앙상블 방법입니다.

    투표 유형:
    - Hard voting (분류): 각 모델이 클래스에 투표하고, 다수결로 결정
    - Soft voting (분류): 예측 확률의 평균, 더 강건함
    - Averaging (회귀): 예측의 단순 평균

    투표 앙상블의 특징:
    - Stacking보다 간단함 (메타 학습 없음)
    - 학습 속도가 빠름 (교차 검증 불필요)
    - 종종 놀랍도록 효과적임
    - 해석 및 디버깅이 용이함

    Attributes:
        base_models (List[Tuple[str, Any]]): 기본 학습기를 위한 (이름, 모델) 튜플 리스트.
        task (str): 'classification' 또는 'regression' 작업 유형.
        voting (str): 투표 유형 (분류의 경우 'hard' 또는 'soft').
        weights (Optional[List[float]]): 각 모델에 대한 선택적 가중치.
        ensemble (VotingClassifier or VotingRegressor): scikit-learn 투표 모델.
        is_trained (bool): 앙상블이 학습되었는지 나타내는 플래그.

    사용 예시:
        # 동일 가중치로 투표 앙상블 생성
        base_models = [
            ('xgboost', xgb_model.model),
            ('lightgbm', lgb_model.model),
            ('catboost', cat_model.model)
        ]
        voting = VotingEnsemble(
            base_models=base_models,
            task='classification',
            voting='soft'
        )
        voting.build_ensemble()
        voting.fit(X_train, y_train)

        # 가중 투표 (특정 모델에 더 많은 중요도 부여)
        voting = VotingEnsemble(
            base_models=base_models,
            voting='soft',
            weights=[1.0, 1.2, 0.8]  # lightgbm에 더 많은 가중치
        )
    """

    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        task: str = 'classification',
        voting: str = 'soft',
        weights: Optional[List[float]] = None
    ) -> None:
        """투표 앙상블을 초기화합니다.

        Args:
            base_models (List[Tuple[str, Any]]): (이름, 모델) 튜플 리스트.
                각 튜플은 문자열 이름과 학습된 모델 인스턴스를 포함합니다.
            task (str, optional): 'classification' 또는 'regression' 작업 유형.
                기본값은 'classification'.
            voting (str, optional): 분류를 위한 투표 유형.
                'hard': 예측된 클래스에 대한 다수결 투표
                'soft': 평균 확률에 대한 투표 (더 강건함)
                회귀의 경우 무시됨 (항상 평균 사용).
                기본값은 'soft'.
            weights (Optional[List[float]], optional): 각 모델의 가중치.
                None인 경우 모든 모델이 동일한 가중치를 가집니다. 제공되는 경우
                base_models와 동일한 길이여야 합니다. 기본값은 None.

        사용 예시:
            # 동일 가중치 투표
            voting = VotingEnsemble(
                base_models=[('xgb', xgb.model), ('lgb', lgb.model)],
                task='classification',
                voting='soft'
            )

            # 가중 투표
            voting = VotingEnsemble(
                base_models=[('xgb', xgb.model), ('lgb', lgb.model)],
                voting='soft',
                weights=[1.5, 1.0]  # xgboost에 1.5배 가중치
            )
        """
        self.base_models = base_models
        self.task = task
        self.voting = voting
        self.weights = weights
        self.ensemble = None
        self.is_trained = False

    def build_ensemble(self) -> 'VotingEnsemble':
        """투표 앙상블 모델을 빌드합니다.

        지정된 기본 모델과 투표 전략으로 scikit-learn VotingClassifier 또는
        VotingRegressor를 생성합니다.

        Returns:
            VotingEnsemble: 메서드 체이닝을 위한 self.

        사용 예시:
            voting.build_ensemble()
            print("투표 앙상블이 성공적으로 빌드되었습니다")
        """
        # 작업 유형에 따라 투표 모델 생성
        if self.task == 'classification':
            self.ensemble = VotingClassifier(
                estimators=self.base_models,
                voting=self.voting,
                weights=self.weights,
                n_jobs=-1
            )
        else:
            self.ensemble = VotingRegressor(
                estimators=self.base_models,
                weights=self.weights,
                n_jobs=-1
            )

        logging.info(f"Voting ensemble created with {len(self.base_models)} base models")
        if self.task == 'classification':
            logging.info(f"   Voting: {self.voting}")
        if self.weights:
            logging.info(f"   Weights: {self.weights}")

        return self

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> 'VotingEnsemble':
        """투표 앙상블을 학습합니다.

        제공된 데이터로 투표 앙상블을 학습합니다. 기본 모델은 이미 학습된 상태여야
        합니다.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): 훈련 특성.
            y_train (Union[pd.Series, np.ndarray]): 훈련 레이블.

        Returns:
            VotingEnsemble: 메서드 체이닝을 위한 self.

        Raises:
            ValueError: 앙상블이 빌드되지 않은 경우.

        사용 예시:
            voting.fit(X_train, y_train)
            print("앙상블 학습이 완료되었습니다")
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build_ensemble() first.")

        logging.info("Training voting ensemble...")
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True
        logging.info("Voting ensemble training completed")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """투표 앙상블을 사용하여 예측을 생성합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 예측을 위한 입력 특성.

        Returns:
            np.ndarray: 예측 값. 분류의 경우 클래스 레이블 반환.
                회귀의 경우 연속 값 반환.

        Raises:
            ValueError: 앙상블이 학습되지 않은 경우.

        사용 예시:
            predictions = voting.predict(X_test)
            print(f"예측: {predictions[:5]}")
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """투표 앙상블을 사용하여 클래스 확률을 예측합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 예측을 위한 입력 특성.

        Returns:
            np.ndarray: 확률 추정치. 형상은 (n_samples, n_classes).

        Raises:
            ValueError: 작업이 분류가 아니거나 앙상블이 학습되지 않은 경우.

        사용 예시:
            proba = voting.predict_proba(X_test)
            positive_proba = proba[:, 1]  # 양성 클래스의 확률
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict_proba(X)

    def save(self, path: str) -> None:
        """학습된 앙상블을 디스크에 저장합니다.

        Args:
            path (str): 앙상블을 저장할 파일 경로.

        Raises:
            ValueError: 앙상블이 학습되지 않은 경우.

        사용 예시:
            voting.save('/path/to/voting_ensemble.pkl')
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained.")

        # 상위 디렉토리가 없으면 생성
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.ensemble, path)
        logging.info(f"Voting ensemble saved to: {path}")

    def load(self, path: str) -> None:
        """디스크에서 학습된 앙상블을 로드합니다.

        Args:
            path (str): 저장된 앙상블 파일 경로.

        Raises:
            FileNotFoundError: 앙상블 파일이 존재하지 않는 경우.

        사용 예시:
            voting = VotingEnsemble(base_models=[], task='classification')
            voting.load('/path/to/voting_ensemble.pkl')
            predictions = voting.predict(X_test)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        # 파일에서 앙상블 로드
        self.ensemble = joblib.load(path)
        self.is_trained = True
        logging.info(f"Voting ensemble loaded from: {path}")
