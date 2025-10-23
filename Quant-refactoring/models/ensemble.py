"""
Ensemble models: Stacking and Voting
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from typing import List, Tuple, Dict, Any
import joblib
from pathlib import Path


class StackingEnsemble:
    """
    Stacking 앙상블 모델

    Features:
    - Multiple base models
    - Configurable meta-learner
    - Better than simple averaging
    - Cross-validation for meta-learner training
    """

    def __init__(self,
                 base_models: List[Tuple[str, Any]],
                 task: str = 'classification',
                 meta_learner: str = 'ridge',
                 meta_learner_params: Dict[str, Any] = None,
                 cv: int = 5):
        """
        Initialize Stacking Ensemble

        Args:
            base_models: [(name, model), ...] 리스트
            task: 'classification' or 'regression'
            meta_learner: 'ridge', 'lasso', 'elasticnet', 'logistic'
            meta_learner_params: 메타 학습기 파라미터
            cv: Cross-validation folds
        """
        self.base_models = base_models
        self.task = task
        self.meta_learner_name = meta_learner
        self.meta_learner_params = meta_learner_params or {}
        self.cv = cv
        self.ensemble = None
        self.is_trained = False

    def _create_meta_learner(self):
        """메타 학습기 생성"""
        if self.task == 'classification':
            if self.meta_learner_name == 'logistic':
                return LogisticRegression(**self.meta_learner_params)
            else:
                # 분류에는 로지스틱 회귀 기본
                return LogisticRegression(**self.meta_learner_params)
        else:  # regression
            if self.meta_learner_name == 'ridge':
                return Ridge(**self.meta_learner_params)
            elif self.meta_learner_name == 'lasso':
                return Lasso(**self.meta_learner_params)
            elif self.meta_learner_name == 'elasticnet':
                return ElasticNet(**self.meta_learner_params)
            else:
                return Ridge(**self.meta_learner_params)

    def build_ensemble(self):
        """앙상블 모델 생성"""
        meta_learner = self._create_meta_learner()

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

        logging.info(f"✅ Stacking ensemble created with {len(self.base_models)} base models")
        logging.info(f"   Meta-learner: {self.meta_learner_name}")
        logging.info(f"   CV folds: {self.cv}")

        return self

    def fit(self, X_train, y_train):
        """
        앙상블 학습

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build_ensemble() first.")

        logging.info("Training stacking ensemble...")
        logging.info(f"  Training samples: {len(X_train):,}")

        # 각 base model이 이미 학습되어 있다고 가정
        # Stacking은 내부적으로 CV를 통해 메타 학습

        self.ensemble.fit(X_train, y_train)
        self.is_trained = True

        logging.info("✅ Stacking ensemble training completed")

        return self

    def predict(self, X):
        """예측"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X):
        """확률 예측 (분류만)"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict_proba(X)

    def get_base_predictions(self, X) -> pd.DataFrame:
        """
        각 base model의 예측값 반환

        Args:
            X: 예측 데이터

        Returns:
            각 모델의 예측값 DataFrame
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        predictions = {}

        for name, model in self.base_models:
            if self.task == 'classification':
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                predictions[name] = model.predict(X)

        return pd.DataFrame(predictions)

    def save(self, path: str):
        """앙상블 저장"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Cannot save untrained ensemble.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        ensemble_data = {
            'ensemble': self.ensemble,
            'task': self.task,
            'meta_learner': self.meta_learner_name,
            'cv': self.cv
        }

        joblib.dump(ensemble_data, path)
        logging.info(f"💾 Ensemble saved to: {path}")

    def load(self, path: str):
        """앙상블 로드"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        ensemble_data = joblib.load(path)

        self.ensemble = ensemble_data['ensemble']
        self.task = ensemble_data['task']
        self.meta_learner_name = ensemble_data['meta_learner']
        self.cv = ensemble_data['cv']
        self.is_trained = True

        logging.info(f"📂 Ensemble loaded from: {path}")


class VotingEnsemble:
    """
    Voting 앙상블 (단순하지만 효과적)

    Hard voting: 다수결
    Soft voting: 확률 평균
    """

    def __init__(self,
                 base_models: List[Tuple[str, Any]],
                 task: str = 'classification',
                 voting: str = 'soft',
                 weights: List[float] = None):
        """
        Initialize Voting Ensemble

        Args:
            base_models: [(name, model), ...] 리스트
            task: 'classification' or 'regression'
            voting: 'hard' or 'soft' (classification)
            weights: 모델별 가중치 (None이면 균등)
        """
        self.base_models = base_models
        self.task = task
        self.voting = voting
        self.weights = weights
        self.ensemble = None
        self.is_trained = False

    def build_ensemble(self):
        """앙상블 모델 생성"""
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

        logging.info(f"✅ Voting ensemble created with {len(self.base_models)} base models")
        if self.task == 'classification':
            logging.info(f"   Voting: {self.voting}")
        if self.weights:
            logging.info(f"   Weights: {self.weights}")

        return self

    def fit(self, X_train, y_train):
        """앙상블 학습"""
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build_ensemble() first.")

        logging.info("Training voting ensemble...")
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True
        logging.info("✅ Voting ensemble training completed")

        return self

    def predict(self, X):
        """예측"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X):
        """확률 예측"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict_proba(X)

    def save(self, path: str):
        """앙상블 저장"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.ensemble, path)
        logging.info(f"💾 Voting ensemble saved to: {path}")

    def load(self, path: str):
        """앙상블 로드"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        self.ensemble = joblib.load(path)
        self.is_trained = True
        logging.info(f"📂 Voting ensemble loaded from: {path}")
