#!/usr/bin/env python3
"""
Complete pipeline example: Training with all new features
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.context_loader import load_config, MainContext
from storage import ParquetStorage
from models import XGBoostModel, LightGBMModel, CatBoostModel, StackingEnsemble
from training import OptunaOptimizer, MLflowTracker
from models.config import OPTUNA_SEARCH_SPACE


def main():
    # 1. Load configuration
    print("="*80)
    print("Quant Trading System - Complete Pipeline Example")
    print("="*80)

    config = load_config('config/conf.yaml')
    ctx = MainContext(config)

    # 2. Initialize storage
    storage = ParquetStorage(
        root_path=ctx.root_path,
        auto_validate=True
    )

    print("\n📊 Storage initialized")
    stats = storage.get_statistics()
    print(f"   Tables: {stats['total_tables']}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")
    print(f"   Total rows: {stats['total_rows']:,}")

    # 3. Load ML training data (예제)
    print("\n📥 Loading training data...")
    # 실제로는 make_mldata.py에서 생성한 데이터를 로드
    # X_train, y_train = load_ml_data(...)
    print("   [데이터 로드는 실제 구현에서 수행]")

    # 4. Initialize MLflow tracking
    ml_config = config.get('ML', {})
    use_mlflow = ml_config.get('USE_MLFLOW', False)

    if use_mlflow:
        tracker = MLflowTracker(
            experiment_name=ml_config.get('MLFLOW_EXPERIMENT', 'quant_trading')
        )
        print(f"\n📊 MLflow tracking enabled")

    # 5. Train base models
    print("\n" + "="*80)
    print("Training Base Models")
    print("="*80)

    models_config = ml_config.get('MODELS', [])
    trained_models = []

    # Example: XGBoost
    if 'xgboost_default' in models_config:
        print("\n🤖 Training XGBoost (default)...")
        xgb = XGBoostModel(task='classification', config_name='default')
        xgb.build_model()
        # xgb.fit(X_train, y_train, X_val, y_val)

        if use_mlflow:
            # tracker.log_training_run(...)
            pass

        trained_models.append(('xgb_default', xgb))
        print("   ✅ XGBoost trained")

    # Example: LightGBM
    if 'lightgbm_default' in models_config:
        print("\n🤖 Training LightGBM...")
        lgb = LightGBMModel(task='classification')
        lgb.build_model()
        # lgb.fit(X_train, y_train, X_val, y_val)

        trained_models.append(('lgb_default', lgb))
        print("   ✅ LightGBM trained")

    # Example: CatBoost (NEW)
    if 'catboost_default' in models_config:
        print("\n🤖 Training CatBoost (NEW)...")
        cat = CatBoostModel(task='classification', config_name='default')
        cat.build_model()
        # cat.fit(X_train, y_train, X_val, y_val)

        trained_models.append(('cat_default', cat))
        print("   ✅ CatBoost trained")

    # 6. Optuna hyperparameter tuning (optional)
    use_optuna = ml_config.get('USE_OPTUNA', False)

    if use_optuna:
        print("\n" + "="*80)
        print("Optuna Hyperparameter Optimization")
        print("="*80)

        optimizer = OptunaOptimizer(
            model_class=CatBoostModel,
            search_space=OPTUNA_SEARCH_SPACE['catboost'],
            n_trials=ml_config.get('OPTUNA_TRIALS', 100),
            cv_folds=ml_config.get('OPTUNA_CV_FOLDS', 5)
        )

        # best_params = optimizer.optimize(X_train, y_train, task='classification')
        # best_model = optimizer.get_best_model(task='classification')
        # best_model.fit(X_train, y_train)

        print("   ✅ Optimization completed")

    # 7. Ensemble (Stacking)
    use_ensemble = ml_config.get('USE_ENSEMBLE', False)
    ensemble_type = ml_config.get('ENSEMBLE_TYPE', 'stacking')

    if use_ensemble and len(trained_models) >= 2:
        print("\n" + "="*80)
        print(f"Building {ensemble_type.upper()} Ensemble")
        print("="*80)

        base_models = [(name, model.model) for name, model in trained_models]

        if ensemble_type == 'stacking':
            ensemble = StackingEnsemble(
                base_models=base_models,
                task='classification',
                meta_learner='ridge',
                cv=5
            )

            ensemble.build_ensemble()
            # ensemble.fit(X_train, y_train)
            print("   ✅ Stacking ensemble created")

    # 8. Validation report
    print("\n" + "="*80)
    print("Data Validation Report")
    print("="*80)

    report_path = storage.generate_validation_report('validation_report.txt')
    print(f"   📄 Report saved: {report_path}")

    # 9. Compare MLflow runs
    if use_mlflow:
        print("\n" + "="*80)
        print("MLflow Runs Comparison")
        print("="*80)

        # comparison = tracker.compare_runs(metric='test_accuracy', top_n=10)
        print("   [실제 구현에서 runs 비교]")

    print("\n" + "="*80)
    print("Pipeline Completed!")
    print("="*80)


if __name__ == "__main__":
    main()
