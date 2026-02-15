"""
Optuna 하이퍼파라미터 최적화
"""

import logging
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Callable
import numpy as np


class OptunaOptimizer:
    """
    Optuna를 사용한 자동 하이퍼파라미터 튜닝

    기능:
    - 베이지안 최적화 (TPE)
    - 빠른 탐색을 위한 가지치기(Pruning)
    - 교차 검증 지원
    - 최적 모델 추적
    """

    def __init__(self,
                 model_class,
                 search_space: Dict[str, tuple],
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 direction: str = 'maximize',
                 pruning: bool = True):
        """
        Optuna 옵티마이저를 초기화합니다.

        Args:
            model_class: 모델 클래스 (XGBoostModel, LightGBMModel 등)
            search_space: 탐색 공간 딕셔너리
            n_trials: 시도 횟수
            cv_folds: 교차 검증 fold 수
            direction: 'maximize' 또는 'minimize'
            pruning: 가지치기(Pruning) 사용 여부
        """
        self.model_class = model_class
        self.search_space = search_space
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.direction = direction
        self.pruning = pruning

        self.study = None
        self.best_params = None
        self.best_score = None

    def _objective(self, trial: optuna.Trial, X, y, task: str, scoring: str) -> float:
        """
        Optuna 목적 함수입니다.

        Args:
            trial: Optuna trial
            X: 학습 데이터
            y: 학습 레이블
            task: 'classification' 또는 'regression'
            scoring: sklearn 평가 메트릭

        Returns:
            교차 검증 점수
        """
        # 파라미터 샘플링
        params = {}

        for param_name, (min_val, max_val) in self.search_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif isinstance(min_val, float) and isinstance(max_val, float):
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            else:
                # 범주형
                params[param_name] = trial.suggest_categorical(param_name, [min_val, max_val])

        # 모델 생성 및 학습
        try:
            model = self.model_class(task=task)
            model.build_model(params)

            # 교차 검증
            scores = cross_val_score(
                model.model,
                X, y,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=-1
            )

            mean_score = scores.mean()

            # 가지치기
            if self.pruning:
                trial.report(mean_score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return mean_score

        except Exception as e:
            logging.warning(f"Trial failed: {e}")
            return float('-inf') if self.direction == 'maximize' else float('inf')

    def optimize(self,
                 X, y,
                 task: str = 'classification',
                 scoring: str = None,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화를 실행합니다.

        Args:
            X: 학습 데이터
            y: 학습 레이블
            task: 'classification' 또는 'regression'
            scoring: sklearn 평가 메트릭 (None이면 자동 선택)
            verbose: 로그 출력 여부

        Returns:
            최적 파라미터 딕셔너리
        """
        if scoring is None:
            scoring = 'accuracy' if task == 'classification' else 'neg_mean_squared_error'

        logging.info("="*80)
        logging.info("Starting Optuna Hyperparameter Optimization")
        logging.info("="*80)
        logging.info(f"Model: {self.model_class.__name__}")
        logging.info(f"Trials: {self.n_trials}")
        logging.info(f"CV folds: {self.cv_folds}")
        logging.info(f"Scoring: {scoring}")
        logging.info(f"Direction: {self.direction}")
        logging.info("="*80)

        # Optuna study 생성
        sampler = TPESampler(seed=42)
        pruner = MedianPruner() if self.pruning else None

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner
        )

        # 최적화 실행
        objective_func = lambda trial: self._objective(trial, X, y, task, scoring)

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study.optimize(objective_func, n_trials=self.n_trials, show_progress_bar=verbose)

        # 결과
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logging.info("\n" + "="*80)
        logging.info("Optimization Completed")
        logging.info("="*80)
        logging.info(f"Best score: {self.best_score:.4f}")
        logging.info(f"Best parameters:")
        for param, value in self.best_params.items():
            logging.info(f"  {param}: {value}")
        logging.info("="*80)

        return self.best_params

    def get_best_model(self, task: str = 'classification'):
        """최적 파라미터로 모델 생성"""
        if self.best_params is None:
            raise ValueError("Optimization not run. Call optimize() first.")

        model = self.model_class(task=task)
        model.build_model(self.best_params)

        return model

    def plot_optimization_history(self, save_path: str = None):
        """최적화 히스토리 플롯"""
        if self.study is None:
            raise ValueError("Optimization not run. Call optimize() first.")

        import matplotlib.pyplot as plt

        fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"📊 Optimization history saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_param_importances(self, save_path: str = None):
        """파라미터 중요도 플롯"""
        if self.study is None:
            raise ValueError("Optimization not run. Call optimize() first.")

        import matplotlib.pyplot as plt

        fig = optuna.visualization.matplotlib.plot_param_importances(self.study)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"📊 Parameter importances saved to: {save_path}")
        else:
            plt.show()

        plt.close()
