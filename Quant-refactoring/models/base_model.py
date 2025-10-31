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

    사용 예시:
        from models.xgboost_model import XGBoostModel
        model = XGBoostModel(task='classification')
        model.build_model({'max_depth': 8, 'n_estimators': 100})
        model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    """

    def __init__(self, model_type: str, task: str = 'classification') -> None:
        """기본 모델을 초기화합니다.

        Args:
            model_type (str): 모델 유형 (예: 'xgboost', 'lightgbm', 'catboost').
            task (str, optional): 'classification' 또는 'regression' 작업 유형.
                기본값은 'classification'.

        사용 예시:
            model = BaseModel(model_type='custom', task='regression')
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False

    @abstractmethod
    def build_model(self, params: Dict[str, Any]) -> 'BaseModel':
        """지정된 파라미터로 모델을 빌드합니다.

        이 추상 메서드는 서브클래스에서 반드시 구현되어야 합니다.
        기반 ML 모델 인스턴스를 생성하고 설정합니다.

        Args:
            params (Dict[str, Any]): 모델 하이퍼파라미터 딕셔너리.

        Returns:
            BaseModel: 메서드 체이닝을 위한 self.

        Raises:
            NotImplementedError: 서브클래스에서 구현되지 않은 경우.

        사용 예시:
            model.build_model({'max_depth': 8, 'learning_rate': 0.1})
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
        """제공된 데이터로 모델을 학습합니다.

        훈련 데이터를 사용하여 모델을 학습하고, 선택적으로 검증 데이터에 대해
        검증을 수행합니다. 이 메서드는 pandas DataFrame에서 특성 이름을 추출하고
        학습 진행 상황을 로깅합니다.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): 훈련 특성.
            y_train (Union[pd.Series, np.ndarray]): 훈련 레이블.
            X_val (Optional[Union[pd.DataFrame, np.ndarray]], optional): 검증 특성.
                기본값은 None.
            y_val (Optional[Union[pd.Series, np.ndarray]], optional): 검증 레이블.
                기본값은 None.
            **kwargs: 기반 모델의 fit 메서드로 전달될 추가 인자.

        Returns:
            BaseModel: 메서드 체이닝을 위한 self.

        Raises:
            ValueError: 모델이 빌드되지 않은 경우 (model이 None).

        사용 예시:
            model.fit(X_train, y_train, X_val, y_val, verbose=True)
            # early stopping과 함께 사용
            model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
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
        """입력 데이터에 대한 예측을 생성합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 예측을 위한 입력 특성.

        Returns:
            np.ndarray: 예측 값. 분류의 경우 클래스 레이블 반환.
                회귀의 경우 연속 값 반환.

        Raises:
            ValueError: 모델이 학습되지 않은 경우.

        사용 예시:
            predictions = model.predict(X_test)
            print(f"처음 5개 예측: {predictions[:5]}")
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """입력 데이터에 대한 클래스 확률을 예측합니다.

        이 메서드는 분류 작업에만 사용 가능합니다. 각 클래스에 대한 확률 추정치를
        반환합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 예측을 위한 입력 특성.

        Returns:
            np.ndarray: 확률 추정치. 형상은 (n_samples, n_classes).
                이진 분류의 경우, 열 0은 클래스 0의 확률,
                열 1은 클래스 1의 확률.

        Raises:
            ValueError: 작업이 분류가 아니거나 모델이 학습되지 않은 경우.

        사용 예시:
            proba = model.predict_proba(X_test)
            # 양성 클래스의 확률 가져오기
            positive_proba = proba[:, 1]
            print(f"양성 클래스 확률: {positive_proba[:5]}")
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
        """제공된 데이터에서 모델을 평가합니다.

        작업 유형에 적합한 평가 메트릭을 계산하여 반환합니다.
        분류의 경우, 정확도, 정밀도, 재현율, F1 점수를 반환합니다.
        회귀의 경우, RMSE와 MAE를 반환합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 평가를 위한 특성.
            y (Union[pd.Series, np.ndarray]): 평가를 위한 실제 레이블.
            threshold (Optional[float], optional): 확률을 클래스 레이블로 변환하기 위한
                분류 임계값. 분류에만 사용됩니다.
                None인 경우 모델의 기본 임계값(0.5) 사용. 기본값은 None.

        Returns:
            Dict[str, float]: 평가 메트릭을 담은 딕셔너리.
                분류: {'accuracy', 'precision', 'recall', 'f1'}
                회귀: {'rmse', 'mae'}

        Raises:
            ValueError: 모델이 학습되지 않은 경우.

        사용 예시:
            # 분류
            metrics = model.evaluate(X_test, y_test)
            print(f"정확도: {metrics['accuracy']:.4f}")
            print(f"F1 점수: {metrics['f1']:.4f}")

            # 커스텀 임계값 사용
            metrics = model.evaluate(X_test, y_test, threshold=0.6)

            # 회귀
            metrics = model.evaluate(X_test, y_test)
            print(f"RMSE: {metrics['rmse']:.4f}")
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
        """학습된 모델에서 특성 중요도 점수를 가져옵니다.

        모델이 지원하는 경우 특성 중요도 점수를 추출하여 반환합니다.
        결과는 중요도 내림차순으로 정렬됩니다.

        Args:
            top_n (Optional[int], optional): 반환할 상위 특성 개수.
                None인 경우 모든 특성을 반환합니다. 기본값은 20.

        Returns:
            pd.DataFrame: 'feature'와 'importance' 열이 있는 DataFrame,
                중요도 내림차순으로 정렬됨. 모델이 특성 중요도를 지원하지
                않으면 빈 DataFrame 반환.

        Raises:
            ValueError: 모델이 학습되지 않은 경우.

        사용 예시:
            importance_df = model.get_feature_importance(top_n=10)
            print(importance_df)
            # 특성 중요도 시각화
            importance_df.plot(x='feature', y='importance', kind='barh')
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
        """학습된 모델을 디스크에 저장합니다.

        joblib을 사용하여 모델과 메타데이터를 파일로 직렬화합니다.
        상위 디렉토리가 없으면 생성합니다.

        Args:
            path (str): 모델을 저장할 파일 경로.

        Raises:
            ValueError: 모델이 학습되지 않은 경우.

        사용 예시:
            model.save('/path/to/model.pkl')
            # 디렉토리 자동 생성과 함께
            model.save('/new/directory/model.pkl')
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
        """디스크에서 학습된 모델을 로드합니다.

        이전에 저장된 모델을 역직렬화하고 상태를 복원합니다.

        Args:
            path (str): 저장된 모델 파일 경로.

        Raises:
            FileNotFoundError: 모델 파일이 존재하지 않는 경우.

        사용 예시:
            model = XGBoostModel()
            model.load('/path/to/saved_model.pkl')
            predictions = model.predict(X_test)
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
        """모델의 하이퍼파라미터를 가져옵니다.

        Returns:
            Dict[str, Any]: 하이퍼파라미터 이름과 값의 딕셔너리.
                모델이 빌드되지 않은 경우 빈 딕셔너리 반환.

        사용 예시:
            params = model.get_params()
            print(f"학습률: {params.get('learning_rate')}")
        """
        if self.model is None:
            return {}
        return self.model.get_params()

    def set_params(self, **params: Any) -> None:
        """모델의 하이퍼파라미터를 설정합니다.

        Args:
            **params: 하이퍼파라미터 이름과 값을 나타내는 키워드 인자.

        Raises:
            ValueError: 모델이 빌드되지 않은 경우.

        사용 예시:
            model.set_params(learning_rate=0.01, max_depth=10)
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
        """모델에 대해 교차 검증을 수행합니다.

        날짜가 제공되면 시계열 인식 교차 검증을 사용하고, 그렇지 않으면
        표준 k-fold 교차 검증을 사용합니다. 이는 모델 성능과
        일반화 능력을 평가하는 데 도움이 됩니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 특성 데이터.
            y (Union[pd.Series, np.ndarray]): 타겟 데이터.
            dates (Optional[pd.Series], optional): 시계열 교차 검증을 위한
                날짜 정보. None인 경우 표준 k-fold CV 사용. 기본값은 None.
            cv_splits (int, optional): 교차 검증 폴드 수. 기본값은 5.
            verbose (bool, optional): 상세 로그 출력 여부. 기본값은 True.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 다음을 포함하는 튜플:
                - 모든 폴드에 대한 평균 점수 (메트릭 이름과 값의 딕셔너리)
                - 각 폴드별 점수 리스트

        사용 예시:
            # 표준 교차 검증
            avg_scores, all_scores = model.cross_validate(X, y, cv_splits=5)
            print(f"평균 정확도: {avg_scores['accuracy']:.4f}")

            # 시계열 교차 검증
            avg_scores, all_scores = model.cross_validate(
                X, y, dates=date_series, cv_splits=5
            )
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
        """교차 검증을 수행한 다음 전체 데이터셋에서 학습합니다.

        이 메서드는 먼저 교차 검증을 사용하여 모델을 평가하여 성능을 추정한 다음,
        사용 가능한 모든 데이터로 최종 모델을 학습합니다.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): 특성 데이터.
            y (Union[pd.Series, np.ndarray]): 타겟 데이터.
            dates (Optional[pd.Series], optional): 시계열 교차 검증을 위한
                날짜 정보. 기본값은 None.
            cv_splits (int, optional): 교차 검증 폴드 수. 기본값은 5.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 다음을 포함하는 튜플:
                - 모든 폴드에 대한 평균 점수
                - 각 폴드별 점수 리스트

        사용 예시:
            # 교차 검증 후 최종 모델 학습
            avg_scores, all_scores = model.fit_with_cv(X, y, cv_splits=5)
            print(f"CV 정확도: {avg_scores['accuracy']:.4f}")
            # 이제 모델은 전체 데이터셋에서 학습됨
            predictions = model.predict(X_test)
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
        """모델의 문자열 표현을 반환합니다.

        Returns:
            str: 모델 유형, 작업, 학습 상태를 설명하는 문자열.

        사용 예시:
            print(model)
            XGBOOST classification model (trained)
        """
        status = "trained" if self.is_trained else "not trained"
        return f"{self.model_type.upper()} {self.task} model ({status})"
