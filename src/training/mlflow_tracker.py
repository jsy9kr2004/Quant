"""
MLflow experiment tracking for model management
"""

import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class MLflowTracker:
    """
    MLflow를 사용한 실험 추적

    Features:
    - Automatic experiment tracking
    - Parameter and metric logging
    - Model versioning
    - Artifact management
    - Easy comparison between runs
    """

    def __init__(self,
                 experiment_name: str = "quant_trading",
                 tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker

        Args:
            experiment_name: 실험 이름
            tracking_uri: MLflow 서버 URI (None이면 로컬)
        """
        self.experiment_name = experiment_name

        # MLflow 설정
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # 로컬 디렉토리에 저장
            mlflow.set_tracking_uri("file:./mlruns")

        # 실험 설정 (있으면 가져오고, 없으면 생성)
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        logging.info(f"📊 MLflow tracking initialized")
        logging.info(f"   Experiment: {experiment_name}")
        logging.info(f"   Tracking URI: {mlflow.get_tracking_uri()}")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        새로운 실험 Run 시작

        Args:
            run_name: Run 이름
            tags: 태그 딕셔너리
        """
        self.active_run = mlflow.start_run(run_name=run_name)

        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        logging.info(f"🚀 MLflow run started: {run_name or 'unnamed'}")

        return self.active_run

    def log_params(self, params: Dict[str, Any]):
        """파라미터 로깅"""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """메트릭 로깅"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, model_type: str, artifact_path: str = "model"):
        """
        모델 로깅

        Args:
            model: 모델 객체
            model_type: 'xgboost', 'lightgbm', 'catboost', 'sklearn'
            artifact_path: 아티팩트 경로
        """
        if model_type == 'xgboost':
            mlflow.xgboost.log_model(model, artifact_path)
        elif model_type == 'lightgbm':
            mlflow.lightgbm.log_model(model, artifact_path)
        elif model_type == 'catboost':
            mlflow.catboost.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)

        logging.info(f"💾 Model logged to MLflow")

    def log_artifact(self, file_path: str):
        """파일 아티팩트 로깅"""
        mlflow.log_artifact(file_path)
        logging.info(f"📎 Artifact logged: {file_path}")

    def log_dataframe(self, df: pd.DataFrame, name: str):
        """DataFrame 로깅 (CSV로 저장)"""
        temp_file = f"/tmp/{name}.csv"
        df.to_csv(temp_file, index=False)
        mlflow.log_artifact(temp_file)
        logging.info(f"📊 DataFrame logged: {name}")

    def end_run(self):
        """Run 종료"""
        mlflow.end_run()
        logging.info("✅ MLflow run ended")

    def log_training_run(self,
                        model_name: str,
                        model,
                        model_type: str,
                        params: Dict[str, Any],
                        train_metrics: Dict[str, float],
                        test_metrics: Optional[Dict[str, float]] = None,
                        feature_importance: Optional[pd.DataFrame] = None,
                        tags: Optional[Dict] = None):
        """
        학습 Run 통합 로깅 (편의 함수)

        Args:
            model_name: 모델 이름
            model: 모델 객체
            model_type: 모델 타입
            params: 파라미터
            train_metrics: 학습 메트릭
            test_metrics: 테스트 메트릭
            feature_importance: 특징 중요도 DataFrame
            tags: 태그
        """
        with mlflow.start_run(run_name=model_name) as run:
            # 태그
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            mlflow.set_tag("model_type", model_type)

            # 파라미터
            self.log_params(params)

            # 학습 메트릭
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)

            # 테스트 메트릭
            if test_metrics:
                for key, value in test_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)

            # 모델
            self.log_model(model, model_type)

            # 특징 중요도
            if feature_importance is not None:
                self.log_dataframe(feature_importance, "feature_importance")

            logging.info(f"✅ Training run logged: {model_name}")

            return run.info.run_id

    def load_best_model(self, metric: str = "test_accuracy", model_type: str = "sklearn"):
        """
        최고 성능 모델 로드

        Args:
            metric: 기준 메트릭
            model_type: 모델 타입

        Returns:
            로드된 모델
        """
        # 실험의 모든 runs 가져오기
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )

        if runs.empty:
            raise ValueError("No runs found in experiment")

        best_run_id = runs.iloc[0]['run_id']

        logging.info(f"📂 Loading best model (run_id: {best_run_id})")
        logging.info(f"   Best {metric}: {runs.iloc[0][f'metrics.{metric}']:.4f}")

        # 모델 로드
        model_uri = f"runs:/{best_run_id}/model"

        if model_type == 'xgboost':
            model = mlflow.xgboost.load_model(model_uri)
        elif model_type == 'lightgbm':
            model = mlflow.lightgbm.load_model(model_uri)
        elif model_type == 'catboost':
            model = mlflow.catboost.load_model(model_uri)
        else:
            model = mlflow.sklearn.load_model(model_uri)

        return model

    def compare_runs(self, metric: str = "test_accuracy", top_n: int = 10) -> pd.DataFrame:
        """
        Run 비교

        Args:
            metric: 비교 메트릭
            top_n: 상위 N개

        Returns:
            비교 DataFrame
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n
        )

        if runs.empty:
            logging.warning("No runs found")
            return pd.DataFrame()

        # 관련 컬럼만 선택
        cols = ['run_id', 'start_time', 'tags.model_type'] + \
               [col for col in runs.columns if col.startswith('metrics.') or col.startswith('params.')]

        comparison_df = runs[cols]

        logging.info(f"📊 Top {len(comparison_df)} runs by {metric}:")
        for idx, row in comparison_df.head(5).iterrows():
            logging.info(f"   {idx+1}. {row.get('tags.model_type', 'unknown')}: {row[f'metrics.{metric}']:.4f}")

        return comparison_df

    def delete_experiment(self):
        """실험 삭제 (주의!)"""
        mlflow.delete_experiment(self.experiment_id)
        logging.warning(f"🗑️  Experiment deleted: {self.experiment_name}")
