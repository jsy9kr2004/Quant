"""
Quant Trading System - Main Pipeline (Refactored)
Integrates: Storage (Parquet), ML Models, Backtesting
"""

import logging
import os
import sys
import yaml
import pandas as pd
from pathlib import Path

# 현재 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.context_loader import load_config, MainContext
from storage import ParquetStorage
from data_collector.fmp import FMP
from training.make_mldata import AIDataMaker
from models import XGBoostModel, LightGBMModel, CatBoostModel, StackingEnsemble
from training import OptunaOptimizer, MLflowTracker
from backtest import Backtest, PlanHandler


class RegressorIntegrated:
    """
    Regressor를 새 모델 구조와 통합한 버전
    기존 regressor.py 로직을 유지하면서 새로운 모듈 사용
    """
    def __init__(self, conf, use_new_models=True):
        self.conf = conf
        self.use_new_models = use_new_models

        # 기존 Regressor import (fallback)
        try:
            from training.regressor import Regressor
            self.legacy_regressor = Regressor(conf)
        except ImportError:
            logging.warning("Legacy regressor not found, using new models only")
            self.legacy_regressor = None

    def dataload(self):
        """기존 regressor.dataload() 호출"""
        if self.legacy_regressor:
            self.legacy_regressor.dataload()
        else:
            logging.warning("Using new data loading method")
            # TODO: 새로운 데이터 로딩 구현

    def train(self):
        """기존 또는 새 모델로 학습"""
        if self.use_new_models and self.conf.get('ML', {}).get('USE_MLFLOW'):
            self._train_with_new_models()
        elif self.legacy_regressor:
            self.legacy_regressor.train()
        else:
            logging.error("No training method available")

    def _train_with_new_models(self):
        """새로운 모델 구조로 학습 (MLflow 포함)"""
        logging.info("Training with new model structure + MLflow")

        ml_config = self.conf.get('ML', {})

        # MLflow tracker
        if ml_config.get('USE_MLFLOW'):
            tracker = MLflowTracker(
                experiment_name=ml_config.get('MLFLOW_EXPERIMENT', 'quant_trading')
            )

        # TODO: X_train, y_train 로드 (기존 legacy_regressor에서 가져오기)
        # X_train = self.legacy_regressor.x_train
        # y_train = self.legacy_regressor.y_train

        logging.info("New model training completed (placeholder)")

    def evaluation(self):
        """기존 evaluation 호출"""
        if self.legacy_regressor:
            self.legacy_regressor.evaluation()

    def latest_prediction(self):
        """기존 latest_prediction 호출"""
        if self.legacy_regressor:
            self.legacy_regressor.latest_prediction()


def get_config_path():
    """설정 파일 경로 찾기"""
    possible_paths = [
        'config/conf.yaml',
        '../config/conf.yaml',
        'Quant-refactoring/config/conf.yaml'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Config file not found. Please create config/conf.yaml\n"
        "Use config/conf.yaml.template as reference"
    )


def conf_check(config):
    """
    설정 파일 검증
    """
    # REPORT_LIST 체크
    valid_reports = ["EVAL", "RANK", "AI", "AVG"]
    report_list = config.get('BACKTEST', {}).get('REPORT_LIST', [])

    for rep_type in report_list:
        if rep_type not in valid_reports:
            logging.critical(f"Invalid REPORT_LIST: {rep_type}. Valid: {valid_reports}")
            sys.exit(1)

    # 필수 설정 체크
    data_config = config.get('DATA', {})
    if not data_config.get('ROOT_PATH'):
        logging.critical("ROOT_PATH not set in config")
        sys.exit(1)

    logging.info("✅ Configuration validated")


def main():
    """메인 파이프라인"""

    print("="*80)
    print("Quant Trading System - Refactored Version")
    print("="*80)

    # 1. 설정 로드
    try:
        config_path = get_config_path()
        config = load_config(config_path)
        logging.info(f"✅ Configuration loaded from: {config_path}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease create config/conf.yaml with your settings.")
        print("See config/conf.yaml.template for reference.")
        sys.exit(1)

    # 2. 컨텍스트 초기화
    main_ctx = MainContext(config)
    conf_check(config)

    # 3. 디렉토리 생성
    main_ctx.create_dir("./reports")

    # 4. 데이터 설정
    data_config = config.get('DATA', {})
    storage_type = data_config.get('STORAGE_TYPE', 'PARQUET')

    # 5. FMP 데이터 수집 (선택적)
    if data_config.get('GET_FMP') == 'Y':
        logging.info("="*80)
        logging.info("Step 1: FMP Data Collection")
        logging.info("="*80)

        try:
            from data_collector.fmp import FMP
            fmp = FMP(main_ctx)  # FMP는 main_ctx만 받음
            fmp.collect()  # get_new() -> collect()로 변경

            # Parquet 저장소로 변환
            if storage_type == 'PARQUET':
                logging.info("Converting to Parquet format...")

                # 새 ParquetStorage 사용
                storage = ParquetStorage(
                    root_path=main_ctx.root_path,
                    auto_validate=True
                )

                # 기존 parquet.py의 로직 사용 (CSV → Parquet)
                from storage.parquet_converter import Parquet
                df_engine = Parquet(main_ctx)
                df_engine.insert_csv()
                df_engine.rebuild_table_view()

                logging.info("✅ Data saved in Parquet format")

            elif storage_type == 'DB':
                logging.warning("Database storage not recommended. Use PARQUET instead.")
                from database import Database
                db = Database(main_ctx)
                db.insert_csv()
                db.rebuild_table_view()

        except Exception as e:
            logging.error(f"FMP data collection failed: {e}")
            logging.info("Continuing with existing data...")

    # 6. ML 파이프라인 (선택적)
    ml_config = config.get('ML', {})
    if ml_config.get('RUN_REGRESSION') == 'Y':
        logging.info("="*80)
        logging.info("Step 2: ML Pipeline")
        logging.info("="*80)

        # 6.1 ML 데이터 준비
        logging.info("Preparing ML training data...")
        AIDataMaker(main_ctx, config)

        # 6.2 모델 학습
        logging.info("Training models...")
        regressor = RegressorIntegrated(
            config,
            use_new_models=ml_config.get('USE_NEW_MODELS', False)
        )
        regressor.dataload()
        regressor.train()
        regressor.evaluation()
        regressor.latest_prediction()

        logging.info("✅ ML pipeline completed")

        # ML만 실행하고 종료
        if ml_config.get('EXIT_AFTER_ML', True):
            logging.info("Exiting after ML (set EXIT_AFTER_ML=N to continue to backtest)")
            sys.exit(0)

    # 7. 백테스팅 (선택적)
    backtest_config = config.get('BACKTEST', {})
    if backtest_config.get('RUN_BACKTEST', True):
        logging.info("="*80)
        logging.info("Step 3: Backtesting")
        logging.info("="*80)

        # Plan 로드
        plan_handler = PlanHandler(
            backtest_config.get('TOP_K_NUM', 100),
            backtest_config.get('ABSOLUTE_SCORE', 500),
            main_ctx
        )

        plan = []
        plan_file = "plan.csv"

        if os.path.exists(plan_file):
            plan_df = pd.read_csv(plan_file)
            plan_info = plan_df.values.tolist()

            for i in range(len(plan_info)):
                plan.append({
                    "f_name": plan_handler.single_metric_plan_no_parallel,
                    "params": {
                        "key": plan_info[i][0],
                        "key_dir": plan_info[i][1],
                        "weight": plan_info[i][2],
                        "diff": plan_info[i][3],
                        "base": plan_info[i][4],
                        "base_dir": plan_info[i][5]
                    }
                })

            plan_handler.plan_list = plan
        else:
            logging.warning(f"Plan file not found: {plan_file}")
            logging.info("Using default plan (empty)")

        # 백테스트 실행
        bt = Backtest(
            main_ctx,
            config,
            plan_handler,
            rebalance_period=backtest_config.get('REBALANCE_PERIOD', 3)
        )

        logging.info("✅ Backtesting completed")

        del plan_handler
        del bt

    # 8. 종료
    logging.info("="*80)
    logging.info("Pipeline completed successfully!")
    logging.info("="*80)
    logging.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
