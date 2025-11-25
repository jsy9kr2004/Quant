"""
Optuna hyperparameter optimization utilities for regressor.py

이 모듈은 XGBoost/LightGBM 모델의 하이퍼파라미터 최적화를 위한 유틸리티를 제공합니다.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import plotly
    import kaleido
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


def optimize_xgboost_params(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: Dict[str, list],
    n_trials: int = 50,
    cv_folds: int = 5,
    timeout: Optional[int] = None,
    task: str = 'classification'
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Optuna를 사용하여 XGBoost 하이퍼파라미터 최적화

    Args:
        x_train: 학습 데이터
        y_train: 학습 레이블
        search_space: 탐색 공간 {'param': [min, max], ...}
        n_trials: 최적화 trial 수
        cv_folds: Cross-validation folds
        timeout: 최대 실행 시간 (초)
        task: 'classification' or 'regression'

    Returns:
        (study, best_params): Optuna study 객체와 최적 파라미터
    """
    if not OPTUNA_AVAILABLE:
        logging.error("❌ Optuna not installed. Cannot optimize hyperparameters.")
        return None, {}

    import xgboost as xgb
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        """Optuna objective function"""
        # 파라미터 샘플링
        params = {}
        for param_name, (min_val, max_val) in search_space.items():
            if param_name in ['n_estimators', 'max_depth']:
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            else:
                params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=(param_name=='learning_rate'))

        # 모델 생성
        if task == 'classification':
            model = xgb.XGBClassifier(
                **params,
                tree_method='hist',
                device='cpu',
                objective='binary:logistic',
                eval_metric='logloss',
                missing=np.nan,
                random_state=42
            )
            scoring = 'accuracy'
        else:
            model = xgb.XGBRegressor(
                **params,
                tree_method='hist',
                device='cpu',
                objective='reg:squarederror',
                eval_metric='rmse',
                missing=np.nan,
                random_state=42
            )
            scoring = 'r2'

        # Cross-validation
        try:
            scores = cross_val_score(
                model, x_train, y_train,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1
            )
            mean_score = scores.mean()

            # Pruning
            trial.report(mean_score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return mean_score
        except Exception as e:
            logging.warning(f"⚠️  Trial {trial.number} failed: {e}")
            return 0.0 if task == 'classification' else -999999.0

    # Optuna study 생성
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    # 최적화 실행
    logging.info(f"🔧 Starting Optuna optimization ({n_trials} trials, {cv_folds}-fold CV)")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        callbacks=[
            lambda study, trial: logging.info(
                f"  Trial {trial.number}/{n_trials}: "
                f"score={trial.value:.4f}, "
                f"params={trial.params}"
            ) if trial.value is not None else None
        ]
    )

    logging.info(f"✅ Optimization completed!")
    logging.info(f"  Best score: {study.best_value:.4f}")
    logging.info(f"  Best params: {study.best_params}")

    return study, study.best_params


def save_optuna_report(
    study: optuna.Study,
    baseline_params: Dict[str, Any],
    baseline_score: float,
    model_name: str,
    output_dir: str = 'reports'
) -> str:
    """
    Optuna 최적화 결과를 상세 리포트로 저장

    Args:
        study: Optuna study 객체
        baseline_params: 기존 파라미터
        baseline_score: 기존 성능
        model_name: 모델 이름 (예: 'clsmodel_0')
        output_dir: 출력 디렉토리

    Returns:
        리포트 파일 경로
    """
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # TXT 상세 리포트
    txt_path = f"{output_dir}/optuna_report_{model_name}_{timestamp}.txt"

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Hyperparameter Optimization Report\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Optimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Best Trial: #{study.best_trial.number}\n\n")

        # Baseline vs Best 비교
        f.write("-"*80 + "\n")
        f.write("BEFORE OPTIMIZATION (Baseline)\n")
        f.write("-"*80 + "\n")
        f.write("Parameters:\n")
        for k, v in baseline_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nPerformance:\n")
        f.write(f"  Score: {baseline_score:.4f}\n\n")

        f.write("-"*80 + "\n")
        f.write(f"AFTER OPTIMIZATION (Best Trial #{study.best_trial.number})\n")
        f.write("-"*80 + "\n")
        f.write("Parameters:\n")
        for k, v in study.best_params.items():
            old_v = baseline_params.get(k, 0)
            diff = v - old_v if isinstance(v, (int, float)) else 'N/A'
            pct = f"({diff/old_v*100:+.1f}%)" if isinstance(diff, (int, float)) and old_v != 0 else ""
            f.write(f"  {k}: {v} (was {old_v}) {pct}\n")
        f.write(f"\nPerformance:\n")
        improvement = study.best_value - baseline_score
        f.write(f"  Score: {study.best_value:.4f} ({improvement:+.4f}, {improvement/baseline_score*100:+.2f}%)\n\n")

        # Top 5 trials
        f.write("-"*80 + "\n")
        f.write("TOP 5 TRIALS\n")
        f.write("-"*80 + "\n")
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)[:5]
        for i, trial in enumerate(sorted_trials, 1):
            f.write(f"{i}. Trial #{trial.number}: {trial.value:.4f}\n")
            f.write(f"   {trial.params}\n")
        f.write("\n")

        # Parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(study)
            f.write("-"*80 + "\n")
            f.write("PARAMETER IMPORTANCE\n")
            f.write("-"*80 + "\n")
            for i, (param, imp) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True), 1):
                f.write(f"{i}. {param}: {imp*100:.1f}%\n")
            f.write("\n")
        except Exception as e:
            logging.warning(f"⚠️  Could not compute parameter importance: {e}")

    logging.info(f"✅ TXT report saved: {txt_path}")

    # CSV 전체 trials
    csv_path = f"{output_dir}/optuna_trials_{model_name}_{timestamp}.csv"
    trials_df = study.trials_dataframe()
    trials_df.to_csv(csv_path, index=False)
    logging.info(f"✅ CSV trials saved: {csv_path}")

    # JSON 최적 파라미터
    json_path = f"{output_dir}/optuna_best_params_{model_name}_{timestamp}.json"
    json_data = {
        'model': model_name,
        'optimization_date': datetime.now().isoformat(),
        'best_trial': study.best_trial.number,
        'best_score': study.best_value,
        'baseline_score': baseline_score,
        'improvement': study.best_value - baseline_score,
        'improvement_percent': (study.best_value - baseline_score) / baseline_score * 100,
        'best_params': study.best_params,
        'baseline_params': baseline_params
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logging.info(f"✅ JSON params saved: {json_path}")

    return txt_path


def save_optuna_plots(
    study: optuna.Study,
    model_name: str,
    output_dir: str = 'reports'
) -> None:
    """
    Optuna 최적화 결과를 PNG 차트로 저장

    Args:
        study: Optuna study 객체
        model_name: 모델 이름
        output_dir: 출력 디렉토리
    """
    if not PLOT_AVAILABLE:
        logging.warning("⚠️  plotly/kaleido not installed. Skipping plot generation.")
        return

    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    try:
        # 1. Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f"{output_dir}/optuna_history_{model_name}_{timestamp}.png")
        logging.info(f"✅ Saved: optuna_history_{model_name}_{timestamp}.png")
    except Exception as e:
        logging.warning(f"⚠️  Could not save optimization history plot: {e}")

    try:
        # 2. Parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f"{output_dir}/optuna_importance_{model_name}_{timestamp}.png")
        logging.info(f"✅ Saved: optuna_importance_{model_name}_{timestamp}.png")
    except Exception as e:
        logging.warning(f"⚠️  Could not save parameter importance plot: {e}")

    try:
        # 3. Parallel coordinate
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(f"{output_dir}/optuna_parallel_{model_name}_{timestamp}.png")
        logging.info(f"✅ Saved: optuna_parallel_{model_name}_{timestamp}.png")
    except Exception as e:
        logging.warning(f"⚠️  Could not save parallel coordinate plot: {e}")

    try:
        # 4. Slice plot
        fig = optuna.visualization.plot_slice(study)
        fig.write_image(f"{output_dir}/optuna_slice_{model_name}_{timestamp}.png")
        logging.info(f"✅ Saved: optuna_slice_{model_name}_{timestamp}.png")
    except Exception as e:
        logging.warning(f"⚠️  Could not save slice plot: {e}")
