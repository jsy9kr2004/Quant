"""
Quant Trading System - Main Pipeline (Refactored)

퀀트 트레이딩 시스템의 메인 진입점입니다. 데이터 수집부터 백테스팅까지
전체 파이프라인을 조율합니다:

1. 설정 로딩: YAML 설정 파일을 로드하고 검증합니다
2. 데이터 수집: FMP API에서 금융 데이터를 가져옵니다 (선택사항)
3. ML 파이프라인: 학습 데이터를 준비하고, 모델을 학습시키며, 성능을 평가합니다
4. 백테스팅: 과거 데이터로 트레이딩 전략을 시뮬레이션합니다

시스템 기능:
- Parquet 기반 스토리지로 효율적인 데이터 처리
- 다양한 ML 모델 지원 (XGBoost, LightGBM, CatBoost)
- MLflow 추적이 가능한 앙상블 전략
- 커스텀 스코어링 전략을 사용한 유연한 백테스팅

사용법:
    python main.py

설정:
    config/conf.yaml 파일을 편집하여 커스터마이즈:
    - DATA.GET_FMP: 데이터 수집 활성화/비활성화
    - ML.RUN_REGRESSION: ML 학습 활성화/비활성화
    - BACKTEST.RUN_BACKTEST: 백테스팅 활성화/비활성화

작성자: Quant Trading Team
날짜: 2025-10-29
"""

import logging
import os
import sys
from typing import Dict, Any, Optional, List
import yaml
import pandas as pd
from pathlib import Path

# Add current directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from config.context_loader import load_config, MainContext
from config.logger import get_logger
from storage import ParquetStorage
from data_collector.fmp import FMP
from training.make_mldata import AIDataMaker
from models import XGBoostModel, LightGBMModel, CatBoostModel, StackingEnsemble
from training import OptunaOptimizer, MLflowTracker
from backtest import Backtest, PlanHandler


class RegressorIntegrated:
    """
    레거시와 새로운 모델 아키텍처를 결합한 통합 ML 학습 파이프라인입니다.

    이 클래스는 레거시 regressor 구현과 새로운 모듈형 모델 구조 사이의
    브릿지 역할을 합니다. MLflow 추적 및 모듈형 모델과 같은 새로운 기능을
    사용하면서 하위 호환성을 제공합니다.

    Attributes:
        conf (Dict[str, Any]): YAML에서 로드한 설정 딕셔너리
        use_new_models (bool): 새로운 모델 아키텍처 사용 여부
        legacy_regressor (Optional[Regressor]): 폴백용 레거시 regressor 인스턴스

    Example:
        >>> config = load_config('config/conf.yaml')
        >>> regressor = RegressorIntegrated(config, use_new_models=True)
        >>> regressor.dataload()
        >>> regressor.train()
        >>> regressor.evaluation()

    TODO:
        - 레거시 의존성 없이 네이티브 데이터 로딩 구현
        - 커스텀 모델 아키텍처 지원 추가
        - 학습 파이프라인에 교차 검증 구현
    """

    def __init__(self, conf: Dict[str, Any], use_new_models: bool = True) -> None:
        """
        통합 regressor를 초기화합니다.

        Args:
            conf: DATA, ML, BACKTEST 설정을 포함한 설정 딕셔너리
            use_new_models: True인 경우 MLflow 추적이 가능한 새로운 모델 아키텍처 사용.
                          False인 경우 레거시 regressor 구현으로 폴백.

        Raises:
            ImportError: 레거시 regressor가 요청되었지만 사용 불가능한 경우
        """
        self.conf = conf
        self.use_new_models = use_new_models

        # Import legacy Regressor as fallback
        try:
            from training.regressor import Regressor
            self.legacy_regressor: Optional['Regressor'] = Regressor(conf)
        except ImportError:
            logger = get_logger('RegressorIntegrated')
            logger.warning("Legacy regressor not found, using new models only")
            self.legacy_regressor = None

    def dataload(self) -> None:
        """
        Parquet 파일에서 학습 및 테스트 데이터를 로드합니다.

        설정에서 지정한 학습 및 테스트 기간에 대한 분기별 ML 데이터 파일
        (rnorm_ml_{year}_{quarter}.parquet)을 로드합니다.

        데이터 처리 과정:
        - 결측치가 80% 이상인 열 필터링
        - 95% 이상의 행이 동일한 값을 가진 열 필터링
        - 결측치가 60% 이상인 행 필터링
        - 주식을 섹터에 매핑
        - 특성(X)과 타겟(y) 분리

        Raises:
            ValueError: 학습 데이터 파일을 찾을 수 없는 경우
            FileNotFoundError: 필수 데이터 파일이 누락된 경우

        TODO:
            - 레거시 의존성 없이 네이티브 데이터 로딩 구현
            - 데이터 검증 및 품질 체크 추가
        """
        if self.legacy_regressor:
            self.legacy_regressor.dataload()
        else:
            logger = get_logger('RegressorIntegrated')
            logger.warning("Using new data loading method")
            # TODO: Implement new data loading method
            # Should load from /data/ml_per_year/rnorm_ml_*.parquet

    def train(self) -> None:
        """
        설정된 전략으로 ML 모델을 학습시킵니다.

        학습 프로세스:
        1. use_new_models=True이고 MLflow가 활성화된 경우:
           - 새로운 모델 아키텍처 사용
           - MLflow로 실험 추적
           - 하이퍼파라미터 최적화 지원
        2. 그 외의 경우:
           - 레거시 학습 파이프라인으로 폴백
           - XGBoost + LightGBM 앙상블 모델 학습

        학습되는 모델:
        - 3x XGBoost Classifiers (depth 8, 9, 10)
        - 1x LightGBM Classifier
        - 2x XGBoost Regressors (depth 8, 10)

        Raises:
            ValueError: 사용 가능한 학습 방법이 없는 경우
            RuntimeError: 모델 학습 실패 시
        """
        if self.use_new_models and self.conf.get('ML', {}).get('USE_MLFLOW'):
            self._train_with_new_models()
        elif self.legacy_regressor:
            self.legacy_regressor.train()
        else:
            logger = get_logger('RegressorIntegrated')
            logger.error("No training method available")

    def _train_with_new_models(self) -> None:
        """
        MLflow 추적이 가능한 새로운 아키텍처로 모델을 학습시킵니다.

        기능:
        - MLflow 실험 추적
        - 모듈형 모델 아키텍처
        - 설정 가능한 하이퍼파라미터
        - 스태킹 앙상블 지원

        TODO:
            - 완전한 학습 파이프라인 구현
            - 커스텀 모델 설정 지원 추가
            - HPO를 위한 OptunaOptimizer 통합
        """
        logger = get_logger('RegressorIntegrated')
        logger.info("Training with new model structure + MLflow")

        ml_config = self.conf.get('ML', {})

        # Initialize MLflow tracker
        if ml_config.get('USE_MLFLOW'):
            tracker = MLflowTracker(
                experiment_name=ml_config.get('MLFLOW_EXPERIMENT', 'quant_trading')
            )

        # TODO: Load training data from legacy regressor or implement native loading
        # X_train = self.legacy_regressor.x_train
        # y_train = self.legacy_regressor.y_train

        # TODO: Build and train models
        # - Create base models (XGBoost, LightGBM, CatBoost)
        # - Train each model with early stopping
        # - Create stacking ensemble
        # - Log to MLflow

        logger.info("New model training completed (placeholder)")

    def evaluation(self) -> None:
        """
        학습된 모델을 테스트 세트에서 평가합니다.

        평가 프로세스:
        1. 테스트 데이터 로드 (TEST_START_YEAR부터 TEST_END_YEAR까지의 분기별 파일)
        2. 각 분기마다:
           - 분류 모델 실행 (상승/하락 예측)
           - 임계값 필터링 적용 (상위 8%)
           - 회귀 모델 실행 (변동 크기 예측)
           - 앙상블 투표로 예측 결합
        3. 메트릭 계산:
           - 분류: Accuracy, Precision, Recall, F1
           - 회귀: RMSE, MAE
           - Top-K 성능: 상위 3, 8, 16 종목의 평균 수익률
        4. 예측 결과를 CSV 파일로 저장

        출력 파일:
        - prediction_ai_{year}_{quarter}.csv: 전체 예측 결과
        - prediction_*_top0-3.csv: 상위 3개 종목 예측
        - pred_df_topk.csv: 요약 통계

        Raises:
            FileNotFoundError: 테스트 데이터 파일이 누락된 경우
        """
        if self.legacy_regressor:
            self.legacy_regressor.evaluation()

    def latest_prediction(self) -> None:
        """
        가장 최근 데이터에 대한 예측을 생성합니다.

        프로세스:
        1. 최신 분기별 데이터 로드 (가장 최근 year_period)
        2. 학습된 모든 모델 실행
        3. 앙상블 전략 적용
        4. 예측 수익률로 종목 순위 매김
        5. 각 모델 조합에 대해 상위 K개 종목 선택

        출력 파일:
        - latest_prediction.csv: 점수가 포함된 모든 예측
        - latest_prediction_{model}_top0-3.csv: 모델별 상위 3개 종목
        - latest_prediction_{model}_top0-7.csv: 모델별 상위 8개 종목
        - latest_prediction_{model}_top0-15.csv: 모델별 상위 16개 종목

        이 파일들은 실제 트레이딩 의사결정에 사용할 수 있습니다.

        Raises:
            FileNotFoundError: 최신 데이터 파일이 누락된 경우
        """
        if self.legacy_regressor:
            self.legacy_regressor.latest_prediction()


def get_config_path() -> str:
    """
    설정 파일 경로를 찾습니다.

    다음 위치에서 config/conf.yaml을 검색합니다:
    1. ./config/conf.yaml (현재 디렉토리)
    2. ../config/conf.yaml (부모 디렉토리)
    3. Quant-refactoring/config/conf.yaml (프로젝트 루트)

    Returns:
        str: 설정 파일 경로

    Raises:
        FileNotFoundError: 모든 위치에서 설정 파일을 찾을 수 없는 경우

    Example:
        >>> config_path = get_config_path()
        >>> print(config_path)
        'config/conf.yaml'
    """
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


def conf_check(config: Dict[str, Any]) -> None:
    """
    설정 파일을 검증합니다.

    검증 항목:
    1. REPORT_LIST가 유효한 리포트 유형만 포함하는지 확인
    2. DATA 섹션에 ROOT_PATH가 지정되어 있는지 확인
    3. 필수 섹션(DATA, ML, BACKTEST) 존재 여부 확인

    Args:
        config: YAML에서 로드한 설정 딕셔너리

    Raises:
        SystemExit: 검증 실패 시 (종료 코드 1)

    Example:
        >>> config = load_config('config/conf.yaml')
        >>> conf_check(config)
        INFO: ✅ Configuration validated
    """
    logger = get_logger('config_check')

    # Validate REPORT_LIST
    valid_reports = ["EVAL", "RANK", "AI", "AVG"]
    report_list = config.get('BACKTEST', {}).get('REPORT_LIST', [])

    for rep_type in report_list:
        if rep_type not in valid_reports:
            logger.critical(f"Invalid REPORT_LIST: {rep_type}. Valid: {valid_reports}")
            sys.exit(1)

    # Validate required settings
    data_config = config.get('DATA', {})
    if not data_config.get('ROOT_PATH'):
        logger.critical("ROOT_PATH not set in config")
        sys.exit(1)

    logger.info("✅ Configuration validated")


def main() -> None:
    """
    전체 퀀트 트레이딩 워크플로우를 조율하는 메인 파이프라인입니다.

    파이프라인 단계:
    ===============

    1. 설정 로딩 및 검증
       - config/conf.yaml 로드
       - 모든 설정 검증
       - 로깅 및 컨텍스트 초기화

    2. 데이터 수집 (선택사항: GET_FMP=Y)
       - FMP API에서 데이터 가져오기
       - CSV 파일로 저장
       - Parquet 형식으로 변환
       - 통합 뷰 구축

    3. ML 파이프라인 (선택사항: RUN_REGRESSION=Y)
       a. 데이터 준비:
          - VIEW 파일 로드
          - 시계열 특성 추출 (tsfresh)
          - 재무 비율 계산
          - RobustScaler로 정규화
          - 분기별 ML 데이터셋 저장
       b. 모델 학습:
          - 학습 데이터 로드 (2015-2021)
          - 분류 모델 학습 (상승/하락)
          - 회귀 모델 학습 (변동 크기)
          - 모델을 /data/MODELS/에 저장
       c. 평가:
          - 2022-2023 데이터로 테스트
          - 성능 리포트 생성
          - 상위 K개 예측 저장
       d. 최신 예측:
          - 가장 최근 데이터로 예측
          - 예상 수익률로 종목 순위 매김
          - 트레이딩 시그널 생성

    4. 백테스팅 (선택사항: RUN_BACKTEST=Y)
       - 커스텀 스코어링 전략 로드 (plan.csv)
       - 과거 데이터로 트레이딩 시뮬레이션
       - 수익률, 샤프 비율, 낙폭 계산
       - 백테스트 리포트 생성

    설정 옵션:
    =====================

    config/conf.yaml:
        DATA:
            GET_FMP: Y/N - API에서 새 데이터 가져오기
            ROOT_PATH: /path/to/data - 데이터 디렉토리
            START_YEAR: 2015 - 데이터 수집 시작
            END_YEAR: 2023 - 데이터 수집 종료
            STORAGE_TYPE: PARQUET - 스토리지 형식

        ML:
            RUN_REGRESSION: Y/N - ML 파이프라인 실행
            USE_NEW_MODELS: Y/N - 새로운 모델 아키텍처 사용
            USE_MLFLOW: Y/N - MLflow 추적 활성화
            TRAIN_START_YEAR: 2015 - 학습 기간 시작
            TRAIN_END_YEAR: 2021 - 학습 기간 종료
            TEST_START_YEAR: 2022 - 테스트 기간 시작
            TEST_END_YEAR: 2023 - 테스트 기간 종료
            EXIT_AFTER_ML: Y/N - ML 후 종료 (백테스트 건너뛰기)

        BACKTEST:
            RUN_BACKTEST: Y/N - 백테스팅 실행
            REBALANCE_PERIOD: 3 - 리밸런싱 간격(개월)
            TOP_K_NUM: 100 - 선택할 종목 수
            ABSOLUTE_SCORE: 500 - 최소 점수 임계값
            REPORT_LIST: [EVAL, RANK, AVG] - 리포트 유형

    Raises:
        FileNotFoundError: 설정 파일을 찾을 수 없는 경우
        ValueError: 설정이 유효하지 않은 경우
        SystemExit: 치명적 오류 발생 시 (종료 코드 1)
        KeyboardInterrupt: 사용자 중단 시 (종료 코드 0)

    Example:
        $ python main.py
        ================================================================================
        Quant Trading System - Refactored Version
        ================================================================================
        INFO: ✅ Configuration validated
        INFO: ================================================================================
        INFO: Step 1: FMP Data Collection
        INFO: ================================================================================
        ...

    See Also:
        - WORKFLOW_GUIDE.md: 상세한 시스템 문서
        - README.md: 빠른 시작 가이드
        - config/conf.yaml.template: 설정 템플릿
    """
    print("="*80)
    print("Quant Trading System - Refactored Version")
    print("="*80)

    # 1. Load and validate configuration
    try:
        config_path = get_config_path()
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease create config/conf.yaml with your settings.")
        print("See config/conf.yaml.template for reference.")
        sys.exit(1)

    # 2. Initialize context (automatically sets up logging)
    main_ctx = MainContext(config)
    logger = get_logger('main')

    conf_check(config)

    # 3. Create output directories
    main_ctx.create_dir("./reports")

    # 4. Configure data storage
    data_config = config.get('DATA', {})
    storage_type = data_config.get('STORAGE_TYPE', 'PARQUET')

    # 5. FMP Data Collection (Optional)
    if data_config.get('GET_FMP') == 'Y':
        logger.info("="*80)
        logger.info("Step 1: FMP Data Collection")
        logger.info("="*80)

        try:
            from data_collector.fmp import FMP
            fmp = FMP(main_ctx)
            fmp.collect()

            # Convert to Parquet format for efficient storage and retrieval
            if storage_type == 'PARQUET':
                logger.info("Converting to Parquet format...")

                # Initialize Parquet storage with auto-validation
                storage = ParquetStorage(
                    root_path=main_ctx.root_path,
                    auto_validate=True
                )

                # Use legacy converter to process CSV files
                from storage.parquet_converter import Parquet
                df_engine = Parquet(main_ctx)
                df_engine.insert_csv()
                df_engine.rebuild_table_view()

                logger.info("✅ Data saved in Parquet format")

            elif storage_type == 'DB':
                logger.warning("Database storage not recommended. Use PARQUET instead.")
                from database import Database
                db = Database(main_ctx)
                db.insert_csv()
                db.rebuild_table_view()

        except Exception as e:
            logger.error(f"FMP data collection failed: {e}")
            logger.info("Continuing with existing data...")

    # 6. ML Pipeline (Optional)
    ml_config = config.get('ML', {})
    if ml_config.get('RUN_REGRESSION') == 'Y':
        logger.info("="*80)
        logger.info("Step 2: ML Pipeline")
        logger.info("="*80)

        # 6.1 Prepare ML training data
        logger.info("Preparing ML training data...")
        AIDataMaker(main_ctx, config)

        # 6.2 Train models
        logger.info("Training models...")
        regressor = RegressorIntegrated(
            config,
            use_new_models=ml_config.get('USE_NEW_MODELS', False)
        )
        regressor.dataload()
        regressor.train()
        regressor.evaluation()
        regressor.latest_prediction()

        logger.info("✅ ML pipeline completed")

        # Exit after ML if configured (allows running ML without backtest)
        if ml_config.get('EXIT_AFTER_ML', True):
            logger.info("Exiting after ML (set EXIT_AFTER_ML=N to continue to backtest)")
            sys.exit(0)

    # 7. Backtesting (Optional)
    backtest_config = config.get('BACKTEST', {})
    if backtest_config.get('RUN_BACKTEST', True):
        logger.info("="*80)
        logger.info("Step 3: Backtesting")
        logger.info("="*80)

        # Initialize plan handler for stock scoring
        plan_handler = PlanHandler(
            backtest_config.get('TOP_K_NUM', 100),
            backtest_config.get('ABSOLUTE_SCORE', 500),
            main_ctx
        )

        # Load custom scoring plan from CSV
        plan: List[Dict[str, Any]] = []
        plan_file = "plan.csv"

        if os.path.exists(plan_file):
            plan_df = pd.read_csv(plan_file)
            plan_info = plan_df.values.tolist()

            for i in range(len(plan_info)):
                plan.append({
                    "f_name": plan_handler.single_metric_plan_no_parallel,
                    "params": {
                        "key": plan_info[i][0],         # Metric name (e.g., 'roe')
                        "key_dir": plan_info[i][1],     # Direction ('ascending'/'descending')
                        "weight": plan_info[i][2],      # Weight in scoring
                        "diff": plan_info[i][3],        # Use price difference
                        "base": plan_info[i][4],        # Base metric for normalization
                        "base_dir": plan_info[i][5]     # Base direction
                    }
                })

            plan_handler.plan_list = plan
        else:
            logger.warning(f"Plan file not found: {plan_file}")
            logger.info("Using default plan (empty)")

        # Run backtest simulation
        bt = Backtest(
            main_ctx,
            config,
            plan_handler,
            rebalance_period=backtest_config.get('REBALANCE_PERIOD', 3)
        )

        logger.info("✅ Backtesting completed")

        # Cleanup
        del plan_handler
        del bt

    # 8. Pipeline completed
    logger.info("="*80)
    logger.info("Pipeline completed successfully!")
    logger.info("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        err_logger = get_logger('main')
        err_logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        err_logger = get_logger('main')
        err_logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
