"""
Base model class for all ML models
"""

from abc import ABC, abstractmethod
import joblib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report


class BaseModel(ABC):
    """
    모든 ML 모델의 기본 클래스

    Provides:
    - Consistent interface for all models
    - Model saving/loading
    - Evaluation metrics
    - Feature importance
    """

    def __init__(self, model_type: str, task: str = 'classification'):
        """
        Initialize base model

        Args:
            model_type: 모델 타입 ('xgboost', 'lightgbm', 'catboost')
            task: 'classification' or 'regression'
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.feature_names = None
        self.is_trained = False

    @abstractmethod
    def build_model(self, params: Dict[str, Any]):
        """모델 생성 (서브클래스에서 구현)"""
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        모델 학습

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터 (선택)
            y_val: 검증 레이블 (선택)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None

        logging.info(f"Training {self.model_type} {self.task} model...")
        logging.info(f"  Training samples: {len(X_train):,}")
        if X_val is not None:
            logging.info(f"  Validation samples: {len(X_val):,}")

        # 학습
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, **kwargs)
        else:
            self.model.fit(X_train, y_train, **kwargs)

        self.is_trained = True
        logging.info(f"✅ Training completed")

        return self

    def predict(self, X):
        """예측"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X):
        """확률 예측 (분류 모델만)"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict_proba(X)

    def evaluate(self, X, y, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        모델 평가

        Args:
            X: 평가 데이터
            y: 평가 레이블
            threshold: 분류 임계값 (선택, 분류 모델만)

        Returns:
            평가 메트릭 딕셔너리
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        metrics = {}

        if self.task == 'classification':
            y_pred_proba = self.predict_proba(X)[:, 1]

            if threshold is not None:
                y_pred = (y_pred_proba >= threshold).astype(int)
            else:
                y_pred = self.predict(X)

            metrics['accuracy'] = accuracy_score(y, y_pred)

            # Precision, Recall, F1
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
            metrics['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics['mae'] = np.mean(np.abs(y - y_pred))

            logging.info(f"Evaluation - RMSE: {metrics['rmse']:.4f}, "
                        f"MAE: {metrics['mae']:.4f}")

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        특징 중요도 반환

        Args:
            top_n: 상위 N개 특징

        Returns:
            특징 중요도 DataFrame
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            logging.warning("Model does not support feature importance")
            return pd.DataFrame()

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

    def save(self, path: str):
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task': self.task,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, path)
        logging.info(f"💾 Model saved to: {path}")

    def load(self, path: str):
        """
        모델 로드

        Args:
            path: 모델 경로
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        self.feature_names = model_data.get('feature_names')
        self.is_trained = True

        logging.info(f"📂 Model loaded from: {path}")

    def get_params(self) -> Dict:
        """모델 파라미터 반환"""
        if self.model is None:
            return {}
        return self.model.get_params()

    def set_params(self, **params):
        """모델 파라미터 설정"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.set_params(**params)

    def cross_validate(self, X, y, dates=None, cv_splits=5, verbose=True):
        """
        교차 검증을 사용한 모델 평가

        Args:
            X: 특징 데이터
            y: 타겟 데이터
            dates: 날짜 정보 (시계열 교차검증에 사용, 선택)
            cv_splits: Cross-validation fold 수
            verbose: 상세 로그 출력 여부

        Returns:
            (평균 점수, 각 fold 점수 리스트)
        """
        from validation.time_series_cv import TimeSeriesCV

        cv = TimeSeriesCV(n_splits=cv_splits)
        avg_scores, all_scores = cv.cross_validate_model(
            self, X, y, dates=dates, verbose=verbose
        )

        return avg_scores, all_scores

    def fit_with_cv(self, X, y, dates=None, cv_splits=5):
        """
        교차 검증 후 전체 데이터로 학습

        Args:
            X: 특징 데이터
            y: 타겟 데이터
            dates: 날짜 정보 (선택)
            cv_splits: Cross-validation fold 수

        Returns:
            (평균 점수, 각 fold 점수 리스트)
        """
        # 교차 검증
        avg_scores, all_scores = self.cross_validate(X, y, dates, cv_splits)

        # 전체 데이터로 최종 학습
        logging.info(f"\n{'='*60}")
        logging.info("전체 데이터로 최종 모델 학습 중...")
        logging.info(f"{'='*60}")
        self.fit(X, y)

        return avg_scores, all_scores

    def __repr__(self):
        status = "trained" if self.is_trained else "not trained"
        return f"{self.model_type.upper()} {self.task} model ({status})"
