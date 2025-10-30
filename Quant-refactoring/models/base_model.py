"""모든 머신러닝 모델을 위한 기본 모델 클래스입니다.

이 모듈은 퀀트 트레이딩 시스템에서 사용되는 모든 ML 모델을 위한 추상 기본 클래스를
제공합니다. 모델 학습, 예측, 평가, 저장을 위한 일관된 인터페이스를 정의합니다.

BaseModel 클래스는 다음과 같은 공통 기능을 구현합니다:
- 검증 지원이 포함된 모델 학습
- 예측 및 확률 예측
- 다양한 메트릭을 사용한 모델 평가
- 특성 중요도 추출
- 모델 저장/로드
- 교차 검증 지원

사용 예제:
    BaseModel을 상속하여 커스텀 모델 생성:

        from models.base_model import BaseModel
        from typing import Dict, Any

        class CustomModel(BaseModel):
            def __init__(self, task: str = 'classification'):
                super().__init__(model_type='custom', task=task)

            def build_model(self, params: Dict[str, Any]):
                # 모델 생성 로직 구현
                self.model = YourModelClass(**params)
                return self

        # 모델 사용
        model = CustomModel(task='classification')
        model.build_model({'param1': value1})
        model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)

Attributes:
    model_type (str): 모델 유형 (예: 'xgboost', 'lightgbm', 'catboost')
    task (str): 작업 유형 ('classification' 또는 'regression')
    model: 기반 ML 모델 인스턴스
    feature_names (list): 특성 이름 리스트
    is_trained (bool): 모델이 학습되었는지 나타내는 플래그
"""

from abc import ABC, abstractmethod
import joblib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report


class BaseModel(ABC):
    """모든 머신러닝 모델을 위한 추상 기본 클래스입니다.

    이 클래스는 머신러닝 모델의 학습, 예측, 평가, 저장을 위한 일관된 인터페이스를
    제공합니다. 모든 구체적인 모델 구현은 이 클래스를 상속하고 추상 메서드를
    구현해야 합니다.

    이 클래스는 분류 및 회귀 작업을 모두 지원하며, 모델 평가, 특성 중요도 분석,
    교차 검증을 위한 유틸리티를 제공합니다.

    Attributes:
        model_type (str): 모델의 유형 (예: 'xgboost', 'lightgbm', 'catboost').
        task (str): 'classification' 또는 'regression' 작업 유형.
        model (Any): 기반 ML 모델 인스턴스 (빌드되기 전까지 None).
        feature_names (Optional[List[str]]): 학습에 사용된 특성 이름 리스트.
        is_trained (bool): 모델이 학습되었는지 나타내는 플래그.

    Example:
        >>> from models.xgboost_model import XGBoostModel
        >>> model = XGBoostModel(task='classification')
        >>> model.build_model({'max_depth': 8, 'n_estimators': 100})
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """

    def __init__(self, model_type: str, task: str = 'classification') -> None:
        """기본 모델을 초기화합니다.

        Args:
            model_type (str): 모델 유형 (예: 'xgboost', 'lightgbm', 'catboost').
            task (str, optional): 'classification' 또는 'regression' 작업 유형.
                기본값은 'classification'.

        Example:
            >>> model = BaseModel(model_type='custom', task='regression')
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False

    @abstractmethod
    def build_model(self, params: Dict[str, Any]) -> 'BaseModel':
        """Build the model with specified parameters.

        This is an abstract method that must be implemented by subclasses.
        It should create and configure the underlying ML model instance.

        Args:
            params (Dict[str, Any]): Dictionary of model hyperparameters.

        Returns:
            BaseModel: Self for method chaining.

        Raises:
            NotImplementedError: If not implemented by subclass.

        Example:
            >>> model.build_model({'max_depth': 8, 'learning_rate': 0.1})
        """
        pass

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any
    ) -> 'BaseModel':
        """Train the model on the provided data.

        Trains the model using the training data and optionally validates on
        validation data. The method extracts feature names from pandas DataFrames
        and logs training progress.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Training features.
            y_train (Union[pd.Series, np.ndarray]): Training labels.
            X_val (Optional[Union[pd.DataFrame, np.ndarray]], optional): Validation
                features. Defaults to None.
            y_val (Optional[Union[pd.Series, np.ndarray]], optional): Validation
                labels. Defaults to None.
            **kwargs: Additional arguments passed to the underlying model's fit method.

        Returns:
            BaseModel: Self for method chaining.

        Raises:
            ValueError: If model has not been built (model is None).

        Example:
            >>> model.fit(X_train, y_train, X_val, y_val, verbose=True)
            >>> # With early stopping
            >>> model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Extract feature names from DataFrame columns if available
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None

        logging.info(f"Training {self.model_type} {self.task} model...")
        logging.info(f"  Training samples: {len(X_train):,}")
        if X_val is not None:
            logging.info(f"  Validation samples: {len(X_val):,}")

        # Train the model with or without validation set
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, **kwargs)
        else:
            self.model.fit(X_train, y_train, **kwargs)

        self.is_trained = True
        logging.info(f"Training completed successfully")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate predictions for the input data.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features for prediction.

        Returns:
            np.ndarray: Predicted values. For classification, returns class labels.
                For regression, returns continuous values.

        Raises:
            ValueError: If model has not been trained.

        Example:
            >>> predictions = model.predict(X_test)
            >>> print(f"First 5 predictions: {predictions[:5]}")
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities for the input data.

        This method is only available for classification tasks. Returns the
        probability estimates for each class.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features for prediction.

        Returns:
            np.ndarray: Probability estimates. Shape is (n_samples, n_classes).
                For binary classification, column 0 is probability of class 0,
                and column 1 is probability of class 1.

        Raises:
            ValueError: If task is not classification or if model is not trained.

        Example:
            >>> proba = model.predict_proba(X_test)
            >>> # Get probability of positive class
            >>> positive_proba = proba[:, 1]
            >>> print(f"Probability of positive class: {positive_proba[:5]}")
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Evaluate the model on the provided data.

        Computes and returns evaluation metrics appropriate for the task type.
        For classification, returns accuracy, precision, recall, and F1 score.
        For regression, returns RMSE and MAE.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features for evaluation.
            y (Union[pd.Series, np.ndarray]): True labels for evaluation.
            threshold (Optional[float], optional): Classification threshold for
                converting probabilities to class labels. Only used for classification.
                If None, uses the model's default threshold (0.5). Defaults to None.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
                For classification: {'accuracy', 'precision', 'recall', 'f1'}
                For regression: {'rmse', 'mae'}

        Raises:
            ValueError: If model has not been trained.

        Example:
            >>> # Classification
            >>> metrics = model.evaluate(X_test, y_test)
            >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
            >>> print(f"F1 Score: {metrics['f1']:.4f}")
            >>>
            >>> # With custom threshold
            >>> metrics = model.evaluate(X_test, y_test, threshold=0.6)
            >>>
            >>> # Regression
            >>> metrics = model.evaluate(X_test, y_test)
            >>> print(f"RMSE: {metrics['rmse']:.4f}")
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        metrics = {}

        if self.task == 'classification':
            # Get probability predictions for positive class
            y_pred_proba = self.predict_proba(X)[:, 1]

            # Apply threshold if specified, otherwise use default predictions
            if threshold is not None:
                y_pred = (y_pred_proba >= threshold).astype(int)
            else:
                y_pred = self.predict(X)

            # Compute classification metrics
            metrics['accuracy'] = accuracy_score(y, y_pred)

            # Get precision, recall, and F1 from classification report
            report = classification_report(y, y_pred, output_dict=True)
            metrics['precision'] = report['1']['precision']
            metrics['recall'] = report['1']['recall']
            metrics['f1'] = report['1']['f1-score']

            logging.info(f"Evaluation - Accuracy: {metrics['accuracy']:.4f}, "
                        f"Precision: {metrics['precision']:.4f}, "
                        f"Recall: {metrics['recall']:.4f}, "
                        f"F1: {metrics['f1']:.4f}")

        else:  # regression
            y_pred = self.predict(X)
            # Compute regression metrics
            metrics['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics['mae'] = np.mean(np.abs(y - y_pred))

            logging.info(f"Evaluation - RMSE: {metrics['rmse']:.4f}, "
                        f"MAE: {metrics['mae']:.4f}")

        return metrics

    def get_feature_importance(self, top_n: Optional[int] = 20) -> pd.DataFrame:
        """Get feature importance scores from the trained model.

        Extracts and returns feature importance scores if the model supports them.
        Results are sorted by importance in descending order.

        Args:
            top_n (Optional[int], optional): Number of top features to return.
                If None, returns all features. Defaults to 20.

        Returns:
            pd.DataFrame: DataFrame with columns 'feature' and 'importance',
                sorted by importance in descending order. Returns empty DataFrame
                if model doesn't support feature importance.

        Raises:
            ValueError: If model has not been trained.

        Example:
            >>> importance_df = model.get_feature_importance(top_n=10)
            >>> print(importance_df)
            >>> # Plot feature importance
            >>> importance_df.plot(x='feature', y='importance', kind='barh')
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        # Check if model supports feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            logging.warning("Model does not support feature importance")
            return pd.DataFrame()

        # Use stored feature names or generate generic names
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names

        # Create DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance in descending order
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Return top N features if specified
        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Serializes the model and its metadata to a file using joblib.
        Creates parent directories if they don't exist.

        Args:
            path (str): File path where the model should be saved.

        Raises:
            ValueError: If model has not been trained.

        Example:
            >>> model.save('/path/to/model.pkl')
            >>> # Or with automatic directory creation
            >>> model.save('/new/directory/model.pkl')
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")

        # Create parent directories if they don't exist
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Package model with metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task': self.task,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, path)
        logging.info(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        """Load a trained model from disk.

        Deserializes a previously saved model and restores its state.

        Args:
            path (str): File path to the saved model.

        Raises:
            FileNotFoundError: If the model file does not exist.

        Example:
            >>> model = XGBoostModel()
            >>> model.load('/path/to/saved_model.pkl')
            >>> predictions = model.predict(X_test)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load model data from file
        model_data = joblib.load(path)

        # Restore model state
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        self.feature_names = model_data.get('feature_names')
        self.is_trained = True

        logging.info(f"Model loaded from: {path}")

    def get_params(self) -> Dict[str, Any]:
        """Get the model's hyperparameters.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameter names and values.
                Returns empty dict if model has not been built.

        Example:
            >>> params = model.get_params()
            >>> print(f"Learning rate: {params.get('learning_rate')}")
        """
        if self.model is None:
            return {}
        return self.model.get_params()

    def set_params(self, **params: Any) -> None:
        """Set the model's hyperparameters.

        Args:
            **params: Keyword arguments representing hyperparameter names and values.

        Raises:
            ValueError: If model has not been built.

        Example:
            >>> model.set_params(learning_rate=0.01, max_depth=10)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.set_params(**params)

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        dates: Optional[pd.Series] = None,
        cv_splits: int = 5,
        verbose: bool = True
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Perform cross-validation on the model.

        Uses time-series aware cross-validation if dates are provided, otherwise
        uses standard k-fold cross-validation. This helps assess model performance
        and generalization ability.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature data.
            y (Union[pd.Series, np.ndarray]): Target data.
            dates (Optional[pd.Series], optional): Date information for time-series
                cross-validation. If None, uses standard k-fold CV. Defaults to None.
            cv_splits (int, optional): Number of cross-validation folds. Defaults to 5.
            verbose (bool, optional): Whether to print detailed logs. Defaults to True.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: A tuple containing:
                - Average scores across all folds (dict of metric name to value)
                - List of scores for each fold

        Example:
            >>> # Standard cross-validation
            >>> avg_scores, all_scores = model.cross_validate(X, y, cv_splits=5)
            >>> print(f"Average accuracy: {avg_scores['accuracy']:.4f}")
            >>>
            >>> # Time-series cross-validation
            >>> avg_scores, all_scores = model.cross_validate(
            ...     X, y, dates=date_series, cv_splits=5
            ... )
        """
        from validation.time_series_cv import TimeSeriesCV

        cv = TimeSeriesCV(n_splits=cv_splits)
        avg_scores, all_scores = cv.cross_validate_model(
            self, X, y, dates=dates, verbose=verbose
        )

        return avg_scores, all_scores

    def fit_with_cv(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        dates: Optional[pd.Series] = None,
        cv_splits: int = 5
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Perform cross-validation and then train on full dataset.

        This method first evaluates the model using cross-validation to estimate
        its performance, then trains a final model on all available data.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature data.
            y (Union[pd.Series, np.ndarray]): Target data.
            dates (Optional[pd.Series], optional): Date information for time-series
                cross-validation. Defaults to None.
            cv_splits (int, optional): Number of cross-validation folds. Defaults to 5.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: A tuple containing:
                - Average scores across all folds
                - List of scores for each fold

        Example:
            >>> # Cross-validate and train final model
            >>> avg_scores, all_scores = model.fit_with_cv(X, y, cv_splits=5)
            >>> print(f"CV Accuracy: {avg_scores['accuracy']:.4f}")
            >>> # Model is now trained on full dataset
            >>> predictions = model.predict(X_test)
        """
        # Perform cross-validation
        avg_scores, all_scores = self.cross_validate(X, y, dates, cv_splits)

        # Train final model on full dataset
        logging.info(f"\n{'='*60}")
        logging.info("Training final model on full dataset...")
        logging.info(f"{'='*60}")
        self.fit(X, y)

        return avg_scores, all_scores

    def __repr__(self) -> str:
        """Return string representation of the model.

        Returns:
            str: String describing the model type, task, and training status.

        Example:
            >>> print(model)
            XGBOOST classification model (trained)
        """
        status = "trained" if self.is_trained else "not trained"
        return f"{self.model_type.upper()} {self.task} model ({status})"
