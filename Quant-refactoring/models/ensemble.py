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
            weights=[1.0, 1.2, 1.1]  # Optional weights
        )
        voting.build_ensemble()
        voting.fit(X_train, y_train)

        # Make predictions
        predictions = voting.predict(X_test)

Attributes:
    StackingEnsemble: Meta-learning based ensemble
    VotingEnsemble: Voting based ensemble

Note:
    - Stacking usually performs better but is more complex
    - Voting is simpler and faster to train
    - Both methods benefit from diverse base models
    - Ensure base models are already trained before creating ensemble
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
    """Stacking ensemble that uses meta-learning to combine base models.

    Stacking (Stacked Generalization) is an ensemble method that combines multiple
    base models using a meta-learner. The base models make predictions, and the
    meta-learner is trained on these predictions to generate final outputs.

    The process:
    1. Base models are trained on the training data
    2. Base model predictions are generated using cross-validation
    3. Meta-learner is trained on base model predictions
    4. Final predictions use base models + meta-learner

    This approach typically outperforms simple averaging because the meta-learner
    can learn the optimal way to combine base model predictions.

    Attributes:
        base_models (List[Tuple[str, Any]]): List of (name, model) tuples for base learners.
        task (str): Task type, either 'classification' or 'regression'.
        meta_learner_name (str): Name of the meta-learner algorithm.
        meta_learner_params (Dict[str, Any]): Hyperparameters for meta-learner.
        cv (int): Number of cross-validation folds for meta-feature generation.
        ensemble (StackingClassifier or StackingRegressor): The scikit-learn stacking model.
        is_trained (bool): Flag indicating if ensemble has been trained.

    Example:
        >>> # Create base models (already trained)
        >>> base_models = [
        ...     ('xgboost', xgb_model.model),
        ...     ('lightgbm', lgb_model.model),
        ...     ('catboost', cat_model.model)
        ... ]
        >>>
        >>> # Create stacking ensemble
        >>> stacking = StackingEnsemble(
        ...     base_models=base_models,
        ...     task='classification',
        ...     meta_learner='ridge',
        ...     cv=5
        ... )
        >>> stacking.build_ensemble()
        >>> stacking.fit(X_train, y_train)
        >>>
        >>> # Get predictions from each base model
        >>> base_preds = stacking.get_base_predictions(X_test)
    """

    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        task: str = 'classification',
        meta_learner: str = 'ridge',
        meta_learner_params: Optional[Dict[str, Any]] = None,
        cv: int = 5
    ) -> None:
        """Initialize Stacking Ensemble.

        Args:
            base_models (List[Tuple[str, Any]]): List of (name, model) tuples.
                Each tuple contains a string name and a trained model instance.
                Example: [('xgb', xgb_model.model), ('lgb', lgb_model.model)]
            task (str, optional): Task type, either 'classification' or 'regression'.
                Defaults to 'classification'.
            meta_learner (str, optional): Meta-learner algorithm name.
                For classification: 'logistic'
                For regression: 'ridge', 'lasso', 'elasticnet'
                Defaults to 'ridge'.
            meta_learner_params (Optional[Dict[str, Any]], optional): Hyperparameters
                for the meta-learner. If None, uses defaults. Defaults to None.
            cv (int, optional): Number of cross-validation folds used to generate
                meta-features for training the meta-learner. Defaults to 5.

        Example:
            >>> stacking = StackingEnsemble(
            ...     base_models=[('xgb', xgb.model), ('lgb', lgb.model)],
            ...     task='classification',
            ...     meta_learner='logistic',
            ...     meta_learner_params={'C': 1.0},
            ...     cv=5
            ... )
        """
        self.base_models = base_models
        self.task = task
        self.meta_learner_name = meta_learner
        self.meta_learner_params = meta_learner_params or {}
        self.cv = cv
        self.ensemble = None
        self.is_trained = False

    def _create_meta_learner(self) -> Union[LogisticRegression, Ridge, Lasso, ElasticNet]:
        """Create the meta-learner model.

        Returns:
            Union[LogisticRegression, Ridge, Lasso, ElasticNet]: Configured meta-learner.

        Note:
            For classification, LogisticRegression is used regardless of meta_learner_name.
            For regression, the specified algorithm (ridge/lasso/elasticnet) is used.
        """
        if self.task == 'classification':
            # For classification, use logistic regression as meta-learner
            if self.meta_learner_name == 'logistic':
                return LogisticRegression(**self.meta_learner_params)
            else:
                # Default to logistic regression for classification
                return LogisticRegression(**self.meta_learner_params)
        else:  # regression
            # For regression, support multiple meta-learner options
            if self.meta_learner_name == 'ridge':
                return Ridge(**self.meta_learner_params)
            elif self.meta_learner_name == 'lasso':
                return Lasso(**self.meta_learner_params)
            elif self.meta_learner_name == 'elasticnet':
                return ElasticNet(**self.meta_learner_params)
            else:
                # Default to Ridge for regression
                return Ridge(**self.meta_learner_params)

    def build_ensemble(self) -> 'StackingEnsemble':
        """Build the stacking ensemble model.

        Creates a scikit-learn StackingClassifier or StackingRegressor with
        the specified base models and meta-learner. Uses cross-validation to
        generate predictions for meta-learner training.

        Returns:
            StackingEnsemble: Self for method chaining.

        Example:
            >>> stacking.build_ensemble()
            >>> print("Ensemble built successfully")
        """
        meta_learner = self._create_meta_learner()

        # Create stacking model based on task type
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
        """Train the stacking ensemble.

        Trains the stacking ensemble using the provided data. The base models
        should already be trained. The ensemble uses internal cross-validation
        to generate meta-features for training the meta-learner.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Training features.
            y_train (Union[pd.Series, np.ndarray]): Training labels.

        Returns:
            StackingEnsemble: Self for method chaining.

        Raises:
            ValueError: If ensemble has not been built.

        Example:
            >>> stacking.fit(X_train, y_train)
            >>> print("Ensemble training completed")
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build_ensemble() first.")

        logging.info("Training stacking ensemble...")
        logging.info(f"  Training samples: {len(X_train):,}")

        # Note: Base models should already be trained
        # Stacking internally uses CV to generate meta-features
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True

        logging.info("Stacking ensemble training completed")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate predictions using the stacking ensemble.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features for prediction.

        Returns:
            np.ndarray: Predicted values. For classification, returns class labels.
                For regression, returns continuous values.

        Raises:
            ValueError: If ensemble has not been trained.

        Example:
            >>> predictions = stacking.predict(X_test)
            >>> print(f"Predictions: {predictions[:5]}")
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities using the stacking ensemble.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features for prediction.

        Returns:
            np.ndarray: Probability estimates. Shape is (n_samples, n_classes).

        Raises:
            ValueError: If task is not classification or if ensemble is not trained.

        Example:
            >>> proba = stacking.predict_proba(X_test)
            >>> positive_proba = proba[:, 1]  # Probability of positive class
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict_proba(X)

    def get_base_predictions(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Get predictions from each base model.

        Returns predictions from all base models as a DataFrame, useful for
        analyzing individual model contributions.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features for prediction.

        Returns:
            pd.DataFrame: DataFrame with columns for each base model containing
                their predictions. For classification, returns probabilities of
                the positive class.

        Raises:
            ValueError: If ensemble has not been trained.

        Example:
            >>> base_preds = stacking.get_base_predictions(X_test)
            >>> print(base_preds.head())
            >>> # Analyze correlation between base model predictions
            >>> print(base_preds.corr())
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        predictions = {}

        # Get predictions from each base model
        for name, model in self.base_models:
            if self.task == 'classification':
                # For classification, get probability of positive class
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                # For regression, get direct predictions
                predictions[name] = model.predict(X)

        return pd.DataFrame(predictions)

    def save(self, path: str) -> None:
        """Save the trained ensemble to disk.

        Args:
            path (str): File path where the ensemble should be saved.

        Raises:
            ValueError: If ensemble has not been trained.

        Example:
            >>> stacking.save('/path/to/stacking_ensemble.pkl')
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Cannot save untrained ensemble.")

        # Create parent directories if they don't exist
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Package ensemble with metadata
        ensemble_data = {
            'ensemble': self.ensemble,
            'task': self.task,
            'meta_learner': self.meta_learner_name,
            'cv': self.cv
        }

        joblib.dump(ensemble_data, path)
        logging.info(f"Ensemble saved to: {path}")

    def load(self, path: str) -> None:
        """Load a trained ensemble from disk.

        Args:
            path (str): File path to the saved ensemble.

        Raises:
            FileNotFoundError: If the ensemble file does not exist.

        Example:
            >>> stacking = StackingEnsemble(base_models=[], task='classification')
            >>> stacking.load('/path/to/stacking_ensemble.pkl')
            >>> predictions = stacking.predict(X_test)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        # Load ensemble data from file
        ensemble_data = joblib.load(path)

        # Restore ensemble state
        self.ensemble = ensemble_data['ensemble']
        self.task = ensemble_data['task']
        self.meta_learner_name = ensemble_data['meta_learner']
        self.cv = ensemble_data['cv']
        self.is_trained = True

        logging.info(f"Ensemble loaded from: {path}")


class VotingEnsemble:
    """Voting ensemble that combines base models through voting.

    Voting is a simple but effective ensemble method that combines predictions
    from multiple base models through voting (classification) or averaging (regression).

    Voting types:
    - Hard voting (classification): Each model votes for a class, majority wins
    - Soft voting (classification): Average predicted probabilities, more robust
    - Averaging (regression): Simple average of predictions

    Voting ensembles are:
    - Simpler than stacking (no meta-learning)
    - Faster to train (no cross-validation needed)
    - Often surprisingly effective
    - Easy to interpret and debug

    Attributes:
        base_models (List[Tuple[str, Any]]): List of (name, model) tuples for base learners.
        task (str): Task type, either 'classification' or 'regression'.
        voting (str): Voting type ('hard' or 'soft' for classification).
        weights (Optional[List[float]]): Optional weights for each model.
        ensemble (VotingClassifier or VotingRegressor): The scikit-learn voting model.
        is_trained (bool): Flag indicating if ensemble has been trained.

    Example:
        >>> # Create voting ensemble with equal weights
        >>> base_models = [
        ...     ('xgboost', xgb_model.model),
        ...     ('lightgbm', lgb_model.model),
        ...     ('catboost', cat_model.model)
        ... ]
        >>> voting = VotingEnsemble(
        ...     base_models=base_models,
        ...     task='classification',
        ...     voting='soft'
        ... )
        >>> voting.build_ensemble()
        >>> voting.fit(X_train, y_train)
        >>>
        >>> # Weighted voting (give more importance to certain models)
        >>> voting = VotingEnsemble(
        ...     base_models=base_models,
        ...     voting='soft',
        ...     weights=[1.0, 1.2, 0.8]  # More weight on lightgbm
        ... )
    """

    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        task: str = 'classification',
        voting: str = 'soft',
        weights: Optional[List[float]] = None
    ) -> None:
        """Initialize Voting Ensemble.

        Args:
            base_models (List[Tuple[str, Any]]): List of (name, model) tuples.
                Each tuple contains a string name and a trained model instance.
            task (str, optional): Task type, either 'classification' or 'regression'.
                Defaults to 'classification'.
            voting (str, optional): Voting type for classification.
                'hard': Majority voting on predicted classes
                'soft': Voting on averaged probabilities (more robust)
                Ignored for regression (always uses averaging).
                Defaults to 'soft'.
            weights (Optional[List[float]], optional): Weights for each model.
                If None, all models have equal weight. Must have same length as
                base_models if provided. Defaults to None.

        Example:
            >>> # Equal weight voting
            >>> voting = VotingEnsemble(
            ...     base_models=[('xgb', xgb.model), ('lgb', lgb.model)],
            ...     task='classification',
            ...     voting='soft'
            ... )
            >>>
            >>> # Weighted voting
            >>> voting = VotingEnsemble(
            ...     base_models=[('xgb', xgb.model), ('lgb', lgb.model)],
            ...     voting='soft',
            ...     weights=[1.5, 1.0]  # 1.5x weight on xgboost
            ... )
        """
        self.base_models = base_models
        self.task = task
        self.voting = voting
        self.weights = weights
        self.ensemble = None
        self.is_trained = False

    def build_ensemble(self) -> 'VotingEnsemble':
        """Build the voting ensemble model.

        Creates a scikit-learn VotingClassifier or VotingRegressor with the
        specified base models and voting strategy.

        Returns:
            VotingEnsemble: Self for method chaining.

        Example:
            >>> voting.build_ensemble()
            >>> print("Voting ensemble built successfully")
        """
        # Create voting model based on task type
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
        """Train the voting ensemble.

        Trains the voting ensemble on the provided data. The base models should
        already be trained.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Training features.
            y_train (Union[pd.Series, np.ndarray]): Training labels.

        Returns:
            VotingEnsemble: Self for method chaining.

        Raises:
            ValueError: If ensemble has not been built.

        Example:
            >>> voting.fit(X_train, y_train)
            >>> print("Ensemble training completed")
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build_ensemble() first.")

        logging.info("Training voting ensemble...")
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True
        logging.info("Voting ensemble training completed")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate predictions using the voting ensemble.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features for prediction.

        Returns:
            np.ndarray: Predicted values. For classification, returns class labels.
                For regression, returns continuous values.

        Raises:
            ValueError: If ensemble has not been trained.

        Example:
            >>> predictions = voting.predict(X_test)
            >>> print(f"Predictions: {predictions[:5]}")
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities using the voting ensemble.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features for prediction.

        Returns:
            np.ndarray: Probability estimates. Shape is (n_samples, n_classes).

        Raises:
            ValueError: If task is not classification or if ensemble is not trained.

        Example:
            >>> proba = voting.predict_proba(X_test)
            >>> positive_proba = proba[:, 1]  # Probability of positive class
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")

        return self.ensemble.predict_proba(X)

    def save(self, path: str) -> None:
        """Save the trained ensemble to disk.

        Args:
            path (str): File path where the ensemble should be saved.

        Raises:
            ValueError: If ensemble has not been trained.

        Example:
            >>> voting.save('/path/to/voting_ensemble.pkl')
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained.")

        # Create parent directories if they don't exist
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.ensemble, path)
        logging.info(f"Voting ensemble saved to: {path}")

    def load(self, path: str) -> None:
        """Load a trained ensemble from disk.

        Args:
            path (str): File path to the saved ensemble.

        Raises:
            FileNotFoundError: If the ensemble file does not exist.

        Example:
            >>> voting = VotingEnsemble(base_models=[], task='classification')
            >>> voting.load('/path/to/voting_ensemble.pkl')
            >>> predictions = voting.predict(X_test)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        # Load ensemble from file
        self.ensemble = joblib.load(path)
        self.is_trained = True
        logging.info(f"Voting ensemble loaded from: {path}")
