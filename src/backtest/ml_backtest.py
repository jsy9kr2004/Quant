"""
ML 예측 기반 Walk-Forward 백테스트 시스템

이 모듈은 머신러닝 모델을 사용한 실제 백테스트를 수행합니다.
미래 유출(Future Leakage)을 방지하기 위해 Walk-Forward Analysis를 구현합니다.

핵심 원칙:
1. 각 리밸런싱 시점마다 그 시점까지의 데이터만으로 모델을 학습
2. Filing Date(공시일)를 엄격히 준수
3. 미래 데이터는 절대 사용하지 않음

작성자: Quant Trading Team
날짜: 2025-11-20
"""

import logging
import os
import joblib
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

# ✨ REFACTORED: Import unified data schema and preprocessing
# DataProcessor now handles all scaling (RobustScaler, StandardScaler)
from src.constants.data_schema import DataSchema
from src.training.data_processor import DataProcessor


class MLBacktest:
    """
    머신러닝 모델을 사용한 Walk-Forward 백테스트

    백테스트 플로우:
    ---------------
    1. 리밸런싱 날짜 목록 생성 (예: 2023-03-13, 2023-06-13, ...)
    2. 각 리밸런싱 날짜마다:
       a. 그 날짜까지 사용 가능한 데이터 로드 (Filing Date 고려)
       b. 모델 학습 (또는 기존 모델 로드)
       c. 예측 수행
       d. 상위 K개 종목 선택
       e. 실제 수익률 계산
    3. 전체 성과 리포트 생성

    미래 유출 방지:
    -------------
    - Filing Date: 재무제표 공시 전에는 해당 데이터 사용 불가
    - Expanding Window: 시작~현재까지 데이터로 학습 (점점 증가)
    - 또는 Rolling Window: 최근 N년 데이터만 사용 (고정 크기)

    Parameters:
    ----------
    config : Dict[str, Any]
        설정 딕셔너리 (conf.yaml에서 로드)
    main_ctx : MainContext
        메인 컨텍스트 (root_path 등 포함)
    rebalance_period : int
        리밸런싱 주기 (개월), 기본값 3
    top_k : int
        선택할 종목 수, 기본값 20
    retrain_frequency : str
        모델 재학습 주기:
        - 'every': 매 리밸런싱마다 재학습 (가장 정확, 느림)
        - 'quarterly': 분기마다 재학습
        - 'yearly': 연도마다 재학습
        - 'once': 한 번만 학습 (현재 방식, 비추천)
    window_type : str
        학습 윈도우 타입:
        - 'expanding': 시작~현재 (점점 증가)
        - 'rolling': 최근 N년만 (고정)
    window_size : Optional[int]
        Rolling window 사용 시 윈도우 크기 (년), 기본값 3
    """

    def __init__(
        self,
        config: Dict[str, Any],
        main_ctx: Any,
        rebalance_period: int = 3,
        top_k: int = 20,
        retrain_frequency: str = 'quarterly',  # 'every', 'quarterly', 'yearly', 'once'
        window_type: str = 'expanding',  # 'expanding' or 'rolling'
        window_size: Optional[int] = 3,  # rolling window size in years
        use_sector_model: Optional[bool] = None  # None = use config, True/False = override
    ):
        self.config = config
        self.main_ctx = main_ctx
        self.rebalance_period = rebalance_period
        self.top_k = top_k
        self.retrain_frequency = retrain_frequency
        self.window_type = window_type
        self.window_size = window_size

        # 로깅 설정
        self.logger = logging.getLogger('MLBacktest')

        # 경로 설정
        ml_config = config.get('ML', {})
        self.data_path = Path(main_ctx.root_path) / 'processed' / 'ml_data' / 'per_year'
        self.model_path = Path(main_ctx.root_path) / 'MODELS_WALKFORWARD'
        self.model_path.mkdir(parents=True, exist_ok=True)

        # 섹터별 모델 사용 여부
        if use_sector_model is None:
            self.use_sector_model = ml_config.get('USE_SECTOR_MODEL', 'N') == 'Y'
        else:
            self.use_sector_model = use_sector_model

        self.sector_config = ml_config.get('SECTOR_CONFIG', {}) if self.use_sector_model else {}

        # Optuna 사용 여부 (regressor.py와 동일한 방식)
        self.use_optuna = ml_config.get('USE_OPTUNA', 'N') == 'Y'
        if self.use_optuna:
            self.logger.info("🔧 USE_OPTUNA=Y: Will load Optuna-optimized parameters from regressor.py")

        # 결과 저장용
        self.backtest_results = []
        self.detailed_results = []  # 각 종목별 상세 거래 내역
        self.predictions_history = []

        # Phase 4: 캐시된 예측 사용 설정
        eval_config = config.get('EVALUATION', {})
        self.use_cached_predictions = eval_config.get('USE_CACHED_PREDICTIONS', 'N') == 'Y'
        self.predictions_cache = None

        if self.use_cached_predictions:
            cache_file = eval_config.get('PREDICTIONS_CACHE_FILE', 'regressor_predictions.pkl')
            cache_path = Path(main_ctx.root_path) / 'MODELS' / cache_file

            if cache_path.exists():
                import joblib
                self.predictions_cache = joblib.load(cache_path)
                self.logger.info(f"✅ Loaded predictions cache: {cache_path}")
                self.logger.info(f"   Available periods: {len(self.predictions_cache)}")
                self.logger.info(f"   Dates: {list(self.predictions_cache.keys())}")
            else:
                # 캐시 파일이 없으면 에러 - 일원화 원칙 강제
                # fallback으로 직접 학습하면 regressor와 ml_backtest 간 예측이 달라질 수 있음
                raise FileNotFoundError(
                    f"\n{'='*60}\n"
                    f"❌ Predictions cache not found!\n"
                    f"   Path: {cache_path}\n"
                    f"\n"
                    f"   USE_CACHED_PREDICTIONS=Y requires regressor.py to run first.\n"
                    f"\n"
                    f"   Solutions:\n"
                    f"   1. Run regressor.py first to generate predictions cache\n"
                    f"   2. Or set USE_CACHED_PREDICTIONS=N in config to train directly\n"
                    f"{'='*60}"
                )

        # 거래 비용 설정 (Trading Costs)
        backtest_config = config.get('BACKTEST', {})
        trading_costs = backtest_config.get('TRADING_COSTS', {})
        self.trading_costs_enabled = trading_costs.get('ENABLED', 'N') == 'Y'
        self.commission = trading_costs.get('COMMISSION', 0.001)  # 0.1% 기본값
        self.slippage = trading_costs.get('SLIPPAGE', 0.001)      # 0.1% 기본값

        if self.trading_costs_enabled:
            total_cost = 2 * (self.commission + self.slippage)
            self.logger.info(f"💰 Trading costs enabled: commission={self.commission*100:.2f}%, "
                           f"slippage={self.slippage*100:.2f}%, total={total_cost*100:.2f}%/trade")
        else:
            self.logger.info("💰 Trading costs disabled (pure returns)")

    def _get_available_data_until(self, cutoff_date: datetime) -> pd.DataFrame:
        """
        특정 날짜까지 사용 가능한 데이터 로드 (Filing Date 고려)

        중요: 미래 유출 방지의 핵심 함수!

        Parameters:
        ----------
        cutoff_date : datetime
            기준 날짜 (이 날짜까지만 사용 가능)

        Returns:
        -------
        pd.DataFrame
            사용 가능한 학습 데이터

        예시:
        ----
        cutoff_date = 2023-06-13이라면:
        - 2023 Q1 실적이 2023-05-15에 공시되었다면 → 사용 가능 ✓
        - 2023 Q2 실적이 2023-08-15에 공시되었다면 → 사용 불가 ✗
        """
        # ✅ Use unified data loading (DataProcessor.load_quarterly_data)
        # Load data for all years from start_year to cutoff_date.year
        all_data = []

        for year in range(self.main_ctx.start_year, cutoff_date.year + 1):
            try:
                yearly_data = DataProcessor.load_quarterly_data(
                    data_dir=str(self.data_path),
                    year=year,
                    file_prefix='rnorm_ml',
                    cutoff_date=cutoff_date,
                    logger=self.logger
                )
                all_data.append(yearly_data)
            except ValueError:
                # No data for this year, skip
                self.logger.debug(f"No data found for year {year}")
                continue

        if not all_data:
            raise ValueError(f"No data available until {cutoff_date}")

        # Combine all years
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"📊 Available data until {cutoff_date.date()}: {len(combined_data)} rows")

        return combined_data

    def _calculate_threshold_config(
        self,
        y_probs: np.ndarray,
        classifier_mode: str = 'positive_screen',
        remove_pct: int = 8
    ) -> Dict[str, Any]:
        """
        Threshold 설정을 계산하는 공통 함수.

        학습 시 classifier의 예측 확률에서 threshold를 계산하여
        예측 시 hard filtering에 사용합니다.

        Parameters:
        ----------
        y_probs : np.ndarray
            분류기의 Class 1 예측 확률 (BAD 또는 GOOD)
        classifier_mode : str
            분류 모드:
            - 'negative_screen': BAD 확률 상위 N% 제거
            - 'positive_screen': GOOD 확률 하위 N% 제거
        remove_pct : int
            제거할 비율 (%)

        Returns:
        -------
        Dict[str, Any]
            threshold 설정 딕셔너리:
            - mode: classifier_mode
            - threshold_value: 계산된 threshold
            - remove_pct: 제거 비율
            - percentile: 사용된 percentile
        """
        if classifier_mode == "negative_screen":
            # BAD 확률 상위 N% 제거 → percentile = 100 - remove_pct
            percentile = 100 - remove_pct
        else:  # positive_screen
            # GOOD 확률 하위 N% 제거 → percentile = remove_pct
            percentile = remove_pct

        threshold_value = np.percentile(y_probs, percentile)

        return {
            'mode': classifier_mode,
            'threshold_value': threshold_value,
            'remove_pct': remove_pct,
            'percentile': percentile
        }

    def _should_retrain(self, current_date: datetime, last_train_date: Optional[datetime]) -> bool:
        """
        모델을 재학습해야 하는지 판단

        Parameters:
        ----------
        current_date : datetime
            현재 리밸런싱 날짜
        last_train_date : Optional[datetime]
            마지막 학습 날짜

        Returns:
        -------
        bool
            재학습 필요 여부
        """
        if last_train_date is None:
            return True  # 첫 학습

        if self.retrain_frequency == 'every':
            return True  # 매번 재학습

        if self.retrain_frequency == 'quarterly':
            # 분기가 바뀌었는지 확인
            return (current_date.year, (current_date.month - 1) // 3) != \
                   (last_train_date.year, (last_train_date.month - 1) // 3)

        if self.retrain_frequency == 'yearly':
            return current_date.year != last_train_date.year

        if self.retrain_frequency == 'once':
            return False  # 한 번만 학습

        return True

    def _load_optuna_params(self) -> Optional[Dict[str, Any]]:
        """
        Load Optuna-optimized parameters from regressor.py results.

        **Logic Unification**: Loads the same parameters that regressor.py saved,
        ensuring both systems use identical model hyperparameters.

        **Storage Location**: {ROOT_PATH}/models/optuna/ (portable, backup-friendly)

        **Important**: This method ALWAYS attempts to load existing Optuna parameters,
        regardless of the USE_OPTUNA setting. This allows reusing previously optimized
        parameters without re-running expensive optimization.

        Returns:
        -------
        Optional[Dict[str, Any]]
            Best parameters dictionary, or None if not found
        """
        # ✅ ALWAYS attempt to load existing Optuna parameters (regardless of USE_OPTUNA)
        # This allows reusing previously optimized parameters
        optuna_dir = Path(self.main_ctx.root_path) / 'models' / 'optuna'
        if not optuna_dir.exists():
            return None

        # Find latest optuna_best_params_clsmodel_0_*.json
        pattern = str(optuna_dir / 'optuna_best_params_clsmodel_0_*.json')
        json_files = glob.glob(pattern)

        if not json_files:
            return None

        # Get the latest file (by filename timestamp or modification time)
        latest_file = max(json_files, key=os.path.getmtime)

        try:
            with open(latest_file, 'r') as f:
                json_data = json.load(f)

            best_params = json_data.get('best_params', {})
            if not best_params:
                self.logger.warning(f"⚠️  Found {latest_file} but 'best_params' is empty")
                return None

            self.logger.info(f"✅ Loaded Optuna params from: {Path(latest_file).name}")
            self.logger.info(f"   Optimization date: {json_data.get('optimization_date', 'unknown')}")
            self.logger.info(f"   Best score: {json_data.get('best_score', 'unknown')}")
            self.logger.info(f"   Params: {best_params}")

            return best_params

        except Exception as e:
            self.logger.error(f"❌ Failed to load Optuna params from {latest_file}: {e}")
            return None

    def _load_sector_optuna_params(self, sectors: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Load sector-specific Optuna-optimized parameters from regressor.py results.

        **Logic Unification**: Loads the same sector-specific parameters that regressor.py saved,
        ensuring both systems use identical model hyperparameters for each sector.

        **Storage Location**: {ROOT_PATH}/models/optuna/ (portable, backup-friendly)

        **Important**: This method ALWAYS attempts to load existing sector-specific
        Optuna parameters, regardless of USE_OPTUNA or OPTUNA_OPTIMIZE_SECTORS settings.
        This allows reusing previously optimized parameters.

        Parameters:
        ----------
        sectors : List[str]
            List of sector names to load parameters for

        Returns:
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping sector name to best parameters
            Format: {'Technology': {...}, 'Financial': {...}, ...}
        """
        sector_params = {}

        # ✅ ALWAYS attempt to load existing sector-specific Optuna parameters
        # (regardless of USE_OPTUNA or OPTUNA_OPTIMIZE_SECTORS)
        optuna_dir = Path(self.main_ctx.root_path) / 'models' / 'optuna'
        if not optuna_dir.exists():
            return sector_params

        loaded_count = 0
        for sector in sectors:
            # Find latest optuna_best_params_sector_{sector}_*.json
            pattern = str(optuna_dir / f'optuna_best_params_sector_{sector}_*.json')
            json_files = glob.glob(pattern)

            if not json_files:
                self.logger.debug(f"   No Optuna results for sector: {sector}")
                continue

            # Get the latest file
            latest_file = max(json_files, key=os.path.getmtime)

            try:
                with open(latest_file, 'r') as f:
                    json_data = json.load(f)

                best_params = json_data.get('best_params', {})
                if best_params:
                    sector_params[sector] = best_params
                    loaded_count += 1
                    self.logger.debug(f"   ✅ Loaded Optuna params for {sector}: {best_params}")
                else:
                    self.logger.warning(f"   ⚠️  Found {latest_file} but 'best_params' is empty")

            except Exception as e:
                self.logger.error(f"   ❌ Failed to load sector Optuna params from {latest_file}: {e}")

        if loaded_count > 0:
            self.logger.info(f"✅ Loaded sector Optuna params: {loaded_count}/{len(sectors)} sectors")
        # No warning if no params found - will use SECTOR_CONFIG or default params

        return sector_params

    def _train_model(self, train_data: pd.DataFrame, cutoff_date: datetime) -> Dict[str, Any]:
        """
        모델 학습 (섹터별 또는 통합)

        Parameters:
        ----------
        train_data : pd.DataFrame
            학습 데이터
        cutoff_date : datetime
            학습 기준 날짜

        Returns:
        -------
        Dict[str, Any]
            학습된 모델 딕셔너리
        """
        if self.use_sector_model:
            self.logger.info(f"🔧 Training SECTOR-BASED models with data until {cutoff_date.date()}")
            return self._train_model_sector(train_data, cutoff_date)
        else:
            self.logger.info(f"🔧 Training UNIFIED model with data until {cutoff_date.date()}")
            return self._train_model_unified(train_data, cutoff_date)

    def _train_model_unified(self, train_data: pd.DataFrame, cutoff_date: datetime) -> Dict[str, Any]:
        """
        통합 모델 학습 (전체 데이터를 하나의 모델로)

        ✨ REFACTORED: Now uses ModelFactory for consistent model creation!

        Parameters:
        ----------
        train_data : pd.DataFrame
            학습 데이터
        cutoff_date : datetime
            학습 기준 날짜

        Returns:
        -------
        Dict[str, Any]
            학습된 모델 딕셔너리
        """
        from src.models.model_factory import create_models_for_backtest

        self.logger.info(f"   Training samples: {len(train_data)}")
        self.logger.info("   🔧 Using ModelFactory (same models as regressor.py)")

        # ✨ REFACTORED: Use DataSchema for column definitions
        # This ensures ml_backtest.py and regressor.py use IDENTICAL column exclusions
        # Prevents bugs from column name mismatches
        exclude_cols = DataSchema.get_excluded_cols()
        feature_cols = DataSchema.get_feature_cols(train_data)

        X = train_data[feature_cols].copy()
        # ✅ UNIFIED: Use DataFrame for y (same structure as regressor.py)
        y = train_data[[DataSchema.REGRESSION_TARGET]].copy()  # DataFrame with column name preserved

        # ========================================
        # 🎯 UNIFIED PREPROCESSING (Single Source of Truth)
        # ========================================
        # Replaces ALL scattered preprocessing with ONE unified method
        # SAME method used in regressor.py - GUARANTEED IDENTICAL preprocessing
        #
        # Steps performed (in order):
        # 1. Remove infinite values from X and y
        # 2. Replace remaining infinite with NaN
        # 3. Remove rows with infinite in y labels (CRITICAL)
        # 4. Log transformation (extreme value compression)
        # 5. Remove columns with >50% NaN
        # 6. Remove rows with NaN in y labels
        # 7. (Optional) Winsorization - NOW CONFIG-DRIVEN (not hardcoded!)
        # 8. (Optional) Feature selection - NOW CONFIG-DRIVEN (not hardcoded!)

        X, y, _, selected_features = DataProcessor.preprocess_training_data(
            X,
            y,
            y_cls=None,  # No classification for unified regressor
            config=self.config,
            logger=self.logger
        )

        # Update feature_cols if feature selection was applied
        if selected_features is not None:
            feature_cols = selected_features
            self.selected_features_unified = selected_features
        else:
            self.selected_features_unified = None

        # ✅ Create binary target for classifier after preprocessing
        y_binary = DataProcessor.create_binary_target(y, config=self.config, logger=self.logger)

        # ✨ Load Optuna parameters (if USE_OPTUNA=Y)
        # This ensures ml_backtest.py uses SAME parameters as regressor.py!
        optuna_params = self._load_optuna_params()

        # ✨ Create models using ModelFactory (same as regressor.py!)
        use_gpu = self._is_gpu_available()
        clf, reg = create_models_for_backtest(self.config, optuna_params=optuna_params, use_gpu=use_gpu)

        # 모델 학습
        models = {}

        # 분류 모델 (상승/하락 예측) - only if USE_CLASSIFIER=Y
        if clf is not None:
            self.logger.info("   Training classifier...")
            clf.fit(X, y_binary)
            models['classifier'] = clf

            # ========================================
            # 🎯 THRESHOLD CONFIG 계산 및 저장 (공통 함수 사용)
            # ========================================
            ml_config = self.config.get('ML', {})
            classifier_mode = ml_config.get('CLASSIFIER_MODE', 'positive_screen')
            remove_pct = ml_config.get('CLASSIFIER_REMOVE_PCT', 8)

            # 학습 데이터에서 확률 계산
            y_probs = clf.predict_proba(X)[:, 1]

            # 공통 함수로 threshold config 계산
            models['threshold_config'] = self._calculate_threshold_config(
                y_probs, classifier_mode, remove_pct
            )

            threshold_value = models['threshold_config']['threshold_value']
            self.logger.info(f"   Threshold config: mode={classifier_mode}, "
                           f"remove_pct={remove_pct}%, threshold={threshold_value:.3f}")
        else:
            self.logger.info("   USE_CLASSIFIER=N: Skipping classifier training")
            models['classifier'] = None
            models['threshold_config'] = {}

        # 회귀 모델 (수익률 크기 예측)
        self.logger.info("   Training regressor...")
        reg.fit(X, y)
        models['regressor'] = reg

        models['features'] = feature_cols
        # No scaler needed for tree-based models

        # 모델 저장
        model_file = self.model_path / f'model_{cutoff_date.strftime("%Y%m%d")}.pkl'
        joblib.dump(models, model_file)
        self.logger.info(f"   ✅ Model saved: {model_file}")

        return models

    def _train_model_sector(self, train_data: pd.DataFrame, cutoff_date: datetime) -> Dict[str, Any]:
        """
        섹터별 모델 학습 (각 섹터마다 별도 모델)

        ✨ REFACTORED: Uses ModelFactory.create_sector_models() - SAME LOGIC AS REGRESSOR.PY!

        Logic Unification:
        - Unified classifier (same as regressor.py)
        - Sector-specific regressors (2 variants per sector, same as regressor.py)
        - SECTOR_CONFIG applied via ModelFactory

        Parameters:
        ----------
        train_data : pd.DataFrame
            학습 데이터 (sector 컬럼 필요)
        cutoff_date : datetime
            학습 기준 날짜

        Returns:
        -------
        Dict[str, Any]
            섹터별 학습된 모델 딕셔너리
            {
                'type': 'sector',
                'classifier': unified_clf,
                'sectors': {
                    'Technology': {'regressor_0': ..., 'regressor_1': ..., 'features': ...},
                    'Financial': {...},
                    ...
                }
            }
        """
        from src.models.model_factory import ModelFactory, create_models_for_backtest

        if 'sector' not in train_data.columns:
            self.logger.warning("⚠️ 'sector' column not found! Falling back to unified model.")
            return self._train_model_unified(train_data, cutoff_date)

        # ✅ 섹터 카테고리화 적용 (config 기반)
        train_data = train_data.copy()  # 원본 보호
        train_data = DataProcessor.map_sectors_to_categories(
            train_data,
            self.config,
            sector_column='sector',
            logger=self.logger
        )

        # sector를 sector_category로 교체 (원본은 sector_original로 백업)
        train_data['sector_original'] = train_data['sector']
        train_data['sector'] = train_data['sector_category']

        self.logger.info(f"   Training samples: {len(train_data)}")
        self.logger.info("   🔧 Using ModelFactory.create_sector_models() (SAME LOGIC as regressor.py)")

        # ✨ REFACTORED: Use DataSchema for column definitions (unified with regressor.py)
        exclude_cols = DataSchema.get_excluded_cols()

        # 각 섹터별로 학습 (카테고리 사용)
        sectors = train_data['sector'].unique()
        sectors = [s for s in sectors if str(s) != 'nan']  # Remove NaN sectors
        self.logger.info(f"   Sectors/Categories found: {list(sectors)}")

        # ✨ Load Optuna parameters (if USE_OPTUNA=Y)
        optuna_params = self._load_optuna_params()
        sector_optuna_params = self._load_sector_optuna_params(list(sectors))

        # ✨ Create sector models using ModelFactory (SAME as regressor.py!)
        use_gpu = self._is_gpu_available()
        factory = ModelFactory(self.config, optuna_params=optuna_params, use_ensemble=False)
        sector_classifiers, sector_regressors = factory.create_sector_models(
            sector_list=list(sectors),
            num_regressor_variants=2,
            sector_optuna_params=sector_optuna_params
        )

        # Check if we're using classifiers
        use_classifier = len(sector_classifiers) > 0
        if use_classifier:
            self.logger.info(f"   Created sector models: {len(sectors)} sectors x 4 classifiers x 2 regressors")
        else:
            self.logger.info(f"   Created sector models: {len(sectors)} sectors x 2 regressors (USE_CLASSIFIER=N)")

        # 섹터별 데이터로 모델 학습
        sector_results = {}

        for sector in sectors:
            sector_data = train_data[train_data['sector'] == sector]

            if len(sector_data) < 50:  # 최소 샘플 수
                self.logger.warning(f"   ⚠️ {sector}: Too few samples ({len(sector_data)}), skipping")
                continue

            self.logger.info(f"   Training {sector} sector ({len(sector_data)} samples)")

            # 특성과 타겟 분리
            feature_cols = DataSchema.get_feature_cols(sector_data)
            X = sector_data[feature_cols].copy()
            # ✅ UNIFIED: Use DataFrame for y (same structure as regressor.py)
            y = sector_data[[DataSchema.REGRESSION_TARGET]].copy()  # DataFrame with column name preserved

            # Extract classification target if USE_CLASSIFIER=Y
            y_cls = sector_data[[DataSchema.CLASSIFICATION_TARGET]].copy() if use_classifier else None

            # 🎯 UNIFIED PREPROCESSING (SAME method as regressor.py sector models)
            X, y, y_cls_clean, _ = DataProcessor.preprocess_training_data(
                X,
                y,
                y_cls=y_cls,
                config=self.config,
                logger=self.logger
            )

            try:
                sector_models = {}

                # Train sector classifiers (if USE_CLASSIFIER=Y)
                if use_classifier and y_cls_clean is not None:
                    y_binary = DataProcessor.create_binary_target(y_cls_clean, config=self.config, logger=self.logger)
                    for clf_idx in range(4):
                        clf = sector_classifiers[(sector, clf_idx)]
                        clf.fit(X, y_binary)
                        sector_models[f'classifier_{clf_idx}'] = clf
                    self.logger.info(f"      ✅ {sector} classifiers trained (4 variants)")

                    # ========================================
                    # 🎯 SECTOR THRESHOLD CONFIG 계산 및 저장 (4개 앙상블 평균)
                    # ========================================
                    ml_config = self.config.get('ML', {})
                    classifier_mode = ml_config.get('CLASSIFIER_MODE', 'positive_screen')
                    remove_pct = ml_config.get('CLASSIFIER_REMOVE_PCT', 8)

                    # 4개 classifier 앙상블 평균으로 threshold 계산 (더 robust)
                    y_probs_list = [
                        sector_classifiers[(sector, i)].predict_proba(X)[:, 1]
                        for i in range(4)
                    ]
                    y_probs = np.mean(y_probs_list, axis=0)

                    # 공통 함수로 threshold config 계산
                    sector_models['threshold_config'] = self._calculate_threshold_config(
                        y_probs, classifier_mode, remove_pct
                    )

                    threshold_value = sector_models['threshold_config']['threshold_value']
                    self.logger.info(f"      Threshold: mode={classifier_mode}, threshold={threshold_value:.3f} (4-clf ensemble)")

                # Train sector-specific regressors (2 variants, same as regressor.py)
                for variant_idx in range(2):
                    reg = sector_regressors[(sector, variant_idx)]
                    reg.fit(X, y)
                    sector_models[f'regressor_{variant_idx}'] = reg

                    self.logger.info(f"      ✅ {sector} regressor_{variant_idx} trained (R²={reg.score(X, y):.4f})")

                sector_results[sector] = {
                    **sector_models,
                    'features': feature_cols,
                    'train_samples': len(sector_data)
                }

                self.logger.info(f"      ✅ {sector} training complete")

            except Exception as e:
                self.logger.error(f"      ❌ {sector} training failed: {str(e)}")
                continue

        if not sector_results:
            self.logger.warning("⚠️ No sector models trained! Falling back to unified model.")
            return self._train_model_unified(train_data, cutoff_date)

        # Create unified classifier (same as regressor.py)
        # Use the same factory that created sector models
        self.logger.info("   Creating unified classifier...")
        if use_classifier:
            unified_clf, _ = factory.create_single_models(use_gpu=use_gpu)
        else:
            unified_clf = None
            self.logger.info("   USE_CLASSIFIER=N: Skipping unified classifier")

        # Train unified classifier on all data (if USE_CLASSIFIER=Y)
        if unified_clf is not None:
            self.logger.info("   Training unified classifier on all data...")
            all_features = DataSchema.get_feature_cols(train_data)
            X_all = train_data[all_features].copy()
            # ✅ UNIFIED: Use DataFrame for y_all (same structure as regressor.py)
            y_all = train_data[[DataSchema.REGRESSION_TARGET]].copy()  # DataFrame with column name preserved

            # 🎯 UNIFIED PREPROCESSING (SAME method as regressor.py)
            X_all, y_all, _, _ = DataProcessor.preprocess_training_data(
                X_all,
                y_all,
                y_cls=None,  # No classification for unified classifier
                config=self.config,
                logger=self.logger
            )

            y_all_binary = DataProcessor.create_binary_target(y_all, config=self.config, logger=self.logger)

            unified_clf.fit(X_all, y_all_binary)
            self.logger.info(f"   ✅ Unified classifier trained (Acc={unified_clf.score(X_all, y_all_binary):.4f})")

        # 모델 저장
        models = {
            'type': 'sector',
            'classifier': unified_clf,  # Unified classifier (same as regressor.py)
            'sectors': sector_results
        }

        model_file = self.model_path / f'model_sector_{cutoff_date.strftime("%Y%m%d")}.pkl'
        joblib.dump(models, model_file)
        self.logger.info(f"   ✅ Sector models saved: {model_file}")
        self.logger.info(f"   Trained {len(sector_results)} sectors: {list(sector_results.keys())}")

        return models

    def _is_gpu_available(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import cupy
            return True
        except:
            return False

    def _predict(self, models: Dict[str, Any], test_data: pd.DataFrame) -> pd.DataFrame:
        """
        예측 수행 (섹터별 또는 통합)

        Parameters:
        ----------
        models : Dict[str, Any]
            학습된 모델
        test_data : pd.DataFrame
            예측할 데이터

        Returns:
        -------
        pd.DataFrame
            예측 결과 포함 데이터프레임
        """
        if models.get('type') == 'sector':
            return self._predict_sector(models, test_data)
        else:
            return self._predict_unified(models, test_data)

    def _predict_unified(self, models: Dict[str, Any], test_data: pd.DataFrame) -> pd.DataFrame:
        """
        통합 모델 예측

        Parameters:
        ----------
        models : Dict[str, Any]
            학습된 통합 모델
        test_data : pd.DataFrame
            예측할 데이터

        Returns:
        -------
        pd.DataFrame
            예측 결과 포함 데이터프레임
        """
        # ===== Feature Alignment (Unified via DataProcessor) =====
        # Uses DataProcessor.align_features_to_model() to ensure consistency
        # with regressor.py and eliminate code duplication

        # Choose model for feature extraction (prefer classifier if available)
        reference_model = models['classifier'] if models['classifier'] is not None else models['regressor']

        X = test_data.copy()
        X = DataProcessor.align_features_to_model(X, reference_model, self.logger)

        # ✅ NaN handling: Let XGBoost/LightGBM handle NaN during prediction
        # Don't fillna(0) - models trained with NaN can handle NaN in test data

        # 분류 예측 (상승 확률) - only if USE_CLASSIFIER=Y
        if models['classifier'] is not None:
            y_pred_proba = models['classifier'].predict_proba(X)[:, 1]
        else:
            # USE_CLASSIFIER=N: Use regressor prediction sign as proxy
            y_pred_proba = np.ones(len(X))  # Default to 1.0 (no filtering)

        # 회귀 예측 (예상 수익률)
        y_pred_return = models['regressor'].predict(X)

        # 결과 추가
        result = test_data.copy()
        result['pred_up_proba'] = y_pred_proba
        result['pred_return'] = y_pred_return

        # ========================================
        # 🎯 NEGATIVE SCREEN HARD FILTERING (REFACTORED)
        # ========================================
        # 철학: 분류기는 "Gate" 역할만 수행
        # - 학습 시: threshold 기준으로 데이터 필터링
        # - 예측 시: 동일한 threshold로 hard filtering 후, 회귀 예측값만으로 순위
        #
        # negative_screen: BAD 확률 >= threshold → 제외, 나머지는 y_pred_return으로 순위
        # positive_screen: GOOD 확률 < threshold → 제외, 나머지는 y_pred_return으로 순위
        threshold_config = models.get('threshold_config', {})

        # Classifier 비활성화 또는 threshold_config 없음: 회귀값만 사용
        if models['classifier'] is None or not threshold_config:
            result['ml_score'] = y_pred_return
            if models['classifier'] is None:
                self.logger.debug("   USE_CLASSIFIER=N: Using regressor predictions only")
            else:
                self.logger.warning("   ⚠️ Classifier exists but threshold_config is empty! Using regressor predictions only")
            return result

        # Classifier 활성화: Hard filtering 적용
        classifier_mode = threshold_config['mode']
        threshold_value = threshold_config['threshold_value']

        if classifier_mode == "negative_screen":
            # BAD 확률이 threshold 이상이면 제외 (위험 종목)
            pass_mask = y_pred_proba < threshold_value
        else:  # positive_screen
            # GOOD 확률이 threshold 이하면 제외 (비유망 종목)
            pass_mask = y_pred_proba > threshold_value

        # Hard filtering: 통과한 종목은 회귀 예측값으로만 순위, 탈락은 -inf
        result['ml_score'] = np.where(pass_mask, y_pred_return, -np.inf)

        n_passed = pass_mask.sum()
        n_total = len(pass_mask)
        self.logger.debug(
            f"   Classifier filter ({classifier_mode}): "
            f"{n_passed}/{n_total} passed (threshold={threshold_value:.3f})"
        )

        return result

    def _predict_sector(self, models: Dict[str, Any], test_data: pd.DataFrame) -> pd.DataFrame:
        """
        섹터별 모델 예측

        ✨ REFACTORED: Uses unified classifier + sector-specific regressors (SAME as regressor.py)

        Parameters:
        ----------
        models : Dict[str, Any]
            학습된 섹터별 모델
            {
                'type': 'sector',
                'classifier': unified_clf,
                'sectors': {
                    'Technology': {'regressor_0': ..., 'regressor_1': ..., 'features': ...},
                    ...
                }
            }
        test_data : pd.DataFrame
            예측할 데이터 (sector 컬럼 필요)

        Returns:
        -------
        pd.DataFrame
            예측 결과 포함 데이터프레임
        """
        if 'sector' not in test_data.columns:
            self.logger.warning("⚠️ 'sector' column not found! Cannot use sector models.")
            return test_data.copy()

        # ✅ 섹터 카테고리화 적용 (학습 시와 동일하게)
        result = test_data.copy()
        result = DataProcessor.map_sectors_to_categories(
            result,
            self.config,
            sector_column='sector',
            logger=None  # 예측 시에는 로깅 생략
        )

        # sector를 sector_category로 교체 (원본은 sector_original로 백업)
        result['sector_original'] = result['sector']
        result['sector'] = result['sector_category']

        sector_models = models['sectors']

        # 초기화
        result['pred_up_proba'] = 0.0
        result['pred_return'] = 0.0
        result['ml_score'] = 0.0

        # Check if sector models have classifiers (USE_CLASSIFIER=Y architecture)
        first_sector = list(sector_models.keys())[0] if sector_models else None
        has_sector_classifiers = first_sector and 'classifier_0' in sector_models[first_sector]

        # y 컬럼 리스트 (전처리에서 제외)
        y_col_list = ['price_dev', 'sec_price_dev_subavg', 'symbol', 'date',
                      'year_period', 'industry', 'rebalance_date', 'report_date',
                      'fillingDate', 'fillingDate_x']

        # Sector-specific predictions (both classifiers and regressors if USE_CLASSIFIER=Y)
        for sector, sector_model in sector_models.items():
            sector_mask = result['sector'] == sector
            sector_data = result[sector_mask]

            if len(sector_data) == 0:
                continue

            try:
                # ✅ Use unified preprocessing (DataProcessor.prepare_sector_data)
                # Ensures consistency with regressor.py
                first_regressor = sector_model['regressor_0']
                X = DataProcessor.prepare_sector_data(
                    sector_df=sector_data,
                    sector_model=first_regressor,
                    y_col_list=y_col_list,
                    use_winsorization=True,  # Match regressor.py
                    logger=self.logger
                )

                # Sector classifier prediction (if USE_CLASSIFIER=Y)
                if has_sector_classifiers:
                    # Average classifier predictions from 4 variants
                    clf_probas = []
                    for clf_idx in range(4):
                        clf = sector_model[f'classifier_{clf_idx}']
                        clf_probas.append(clf.predict_proba(X)[:, 1])
                    y_pred_proba = np.mean(clf_probas, axis=0)
                    result.loc[sector_mask, 'pred_up_proba'] = y_pred_proba
                else:
                    # USE_CLASSIFIER=N: Default to 1.0 (no filtering)
                    result.loc[sector_mask, 'pred_up_proba'] = 1.0
                    y_pred_proba = np.ones(len(sector_data))

                # Average prediction from 2 regressor variants (same as regressor.py ensemble)
                reg_0 = sector_model['regressor_0']
                reg_1 = sector_model['regressor_1']

                y_pred_0 = reg_0.predict(X)
                y_pred_1 = reg_1.predict(X)
                y_pred_return = (y_pred_0 + y_pred_1) / 2  # Average of 2 variants

                # 결과 저장
                result.loc[sector_mask, 'pred_return'] = y_pred_return

                # ========================================
                # 🎯 NEGATIVE SCREEN HARD FILTERING (SECTOR)
                # ========================================
                # 철학: 분류기는 "Gate" 역할만 수행 (섹터별 모델도 동일)
                threshold_config = sector_model.get('threshold_config', {})

                # Classifier 비활성화 또는 threshold_config 없음: 회귀값만 사용
                if not has_sector_classifiers or not threshold_config:
                    result.loc[sector_mask, 'ml_score'] = y_pred_return
                    if has_sector_classifiers and not threshold_config:
                        self.logger.warning(f"   ⚠️ {sector}: Classifiers exist but threshold_config is empty!")
                else:
                    # Classifier 활성화: Hard filtering 적용
                    classifier_mode = threshold_config['mode']
                    threshold_value = threshold_config['threshold_value']

                    if classifier_mode == "negative_screen":
                        # BAD 확률이 threshold 이상이면 제외
                        pass_mask = y_pred_proba < threshold_value
                    else:  # positive_screen
                        # GOOD 확률이 threshold 이하면 제외
                        pass_mask = y_pred_proba > threshold_value

                    # Hard filtering: 통과한 종목은 회귀 예측값으로만 순위
                    ml_scores = np.where(pass_mask, y_pred_return, -np.inf)
                    result.loc[sector_mask, 'ml_score'] = ml_scores

                clf_msg = "classifiers + regressors" if has_sector_classifiers else "regressors only"
                self.logger.info(f"   Predicted {sector} ({clf_msg}): {len(sector_data)} stocks")

            except Exception as e:
                self.logger.error(f"   ❌ {sector} prediction failed: {str(e)}")
                continue

        # ✅ 원본 sector 복원 (레포트용)
        if 'sector_original' in result.columns:
            result['sector'] = result['sector_original']
            result.drop(columns=['sector_original', 'sector_category'], inplace=True, errors='ignore')

        return result

    def _select_top_k(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        상위 K개 종목 선택

        Parameters:
        ----------
        predictions : pd.DataFrame
            예측 결과 (symbol, sector, ml_score 등 포함)

        Returns:
        -------
        pd.DataFrame
            선택된 종목 정보 (symbol, sector 등 포함)
        """
        # ml_score 기준으로 정렬
        sorted_df = predictions.sort_values('ml_score', ascending=False)

        # 상위 K개 선택 (symbol과 sector 모두 포함)
        top_k_df = sorted_df.head(self.top_k)

        return top_k_df[['symbol', 'sector']].copy()

    def _calculate_period_return(
        self,
        selected_stocks: pd.DataFrame,
        buy_date: datetime,
        sell_date: datetime,
        price_table: pd.DataFrame
    ) -> dict:
        """
        기간 수익률 계산 (상세 정보 포함)

        Parameters:
        ----------
        selected_stocks : pd.DataFrame
            선택된 종목 정보 (symbol, sector 컬럼 포함)
        buy_date : datetime
            매수 날짜
        sell_date : datetime
            매도 날짜
        price_table : pd.DataFrame
            가격 데이터

        Returns:
        -------
        dict
            {
                'avg_return': float,  # 평균 수익률
                'details': list[dict]  # 각 종목별 상세 정보
            }
        """
        # 실제 거래일 찾기 (주말/휴일 처리)
        # ✅ 일원화: DataProcessor.get_trade_date() 사용 (make_mldata.py, regressor.py 공통)
        actual_buy_date = DataProcessor.get_trade_date(pd.Timestamp(buy_date), price_table)
        actual_sell_date = DataProcessor.get_trade_date(pd.Timestamp(sell_date), price_table)

        if actual_buy_date is None or actual_sell_date is None:
            self.logger.warning(
                f"Trading date not found: buy={buy_date.date()}, sell={sell_date.date()}"
            )
            return {'avg_return': 0.0, 'details': [], 'actual_buy_date': None, 'actual_sell_date': None}

        returns = []
        details = []

        # ✅ 섹터 정보 포함하여 반복
        for _, stock in selected_stocks.iterrows():
            symbol = stock['symbol']
            sector = stock.get('sector', 'Unknown')  # sector가 없으면 'Unknown'

            symbol_prices = price_table[price_table['symbol'] == symbol]

            # 매수 가격 (실제 거래일)
            buy_price_rows = symbol_prices[symbol_prices['date'] == actual_buy_date]
            if buy_price_rows.empty:
                # 매수일에 가격 없음 = 데이터 오류 (거래 불가능)
                self.logger.warning(f"   ⚠️  {symbol}: No price at buy date {actual_buy_date.date()} - skipping")
                continue
            buy_price = buy_price_rows.iloc[0]['close']

            # 매도 가격 (실제 거래일)
            sell_price_rows = symbol_prices[symbol_prices['date'] == actual_sell_date]
            if sell_price_rows.empty:
                # ✅ 상장폐지: -100% 수익률로 처리
                gross_ret = -1.0
                net_ret = -1.0
                sell_price = 0.0
                is_delisted = True
                trading_cost = 0.0  # 상장폐지 시 거래 비용 없음
                self.logger.warning(
                    f"   ⚠️  {symbol}: DELISTED (no price at {actual_sell_date.date()}) "
                    f"→ -100% return"
                )
            else:
                sell_price = sell_price_rows.iloc[0]['close']

                # 순수 수익률 (거래 비용 미반영)
                gross_ret = (sell_price - buy_price) / buy_price

                # 거래 비용 반영
                if self.trading_costs_enabled:
                    # 슬리피지: 매수 시 높게, 매도 시 낮게
                    effective_buy_price = buy_price * (1 + self.slippage)
                    effective_sell_price = sell_price * (1 - self.slippage)

                    # 슬리피지 적용 후 수익률
                    ret_after_slippage = (effective_sell_price - effective_buy_price) / effective_buy_price

                    # 거래 수수료 (매수 + 매도)
                    net_ret = ret_after_slippage - 2 * self.commission
                    trading_cost = gross_ret - net_ret
                else:
                    net_ret = gross_ret
                    trading_cost = 0.0

                is_delisted = False

            # 수익률 리스트에 추가 (상장폐지 포함) - 순수익률 사용
            returns.append(net_ret)

            # 상세 정보 저장 (섹터 정보 + 상장폐지 여부 + 거래 비용)
            details.append({
                'symbol': symbol,
                'sector': sector,  # ✅ 섹터 정보 추가
                'buy_price': buy_price,
                'sell_price': sell_price,
                'gross_return': gross_ret,         # 순수 수익률
                'trading_cost': trading_cost,       # 거래 비용
                'return': net_ret,                  # 순수익률 (거래 비용 차감)
                'return_pct': net_ret * 100,
                'delisted': is_delisted  # ✅ 상장폐지 여부 추가
            })

        if not returns:
            return {'avg_return': 0.0, 'details': [], 'actual_buy_date': actual_buy_date, 'actual_sell_date': actual_sell_date}

        return {
            'avg_return': np.mean(returns),
            'details': details,
            'actual_buy_date': actual_buy_date,
            'actual_sell_date': actual_sell_date
        }

    def _calculate_benchmark_returns(
        self,
        start_date: datetime,
        end_date: datetime,
        price_table: pd.DataFrame
    ) -> pd.DataFrame:
        """
        벤치마크 Buy-and-Hold 수익률 계산

        ETF와 주식 데이터를 분리하여 처리합니다:
        - 주식: 기존 price_table 사용
        - ETF: etf_price.parquet 별도 로드 (ETFDataLoader 사용)

        이를 통해 ETF가 ML 학습 데이터에 오염되는 것을 방지합니다.

        Parameters:
        ----------
        start_date : datetime
            백테스트 시작 날짜
        end_date : datetime
            백테스트 종료 날짜
        price_table : pd.DataFrame
            가격 데이터 (주식 only)

        Returns:
        -------
        pd.DataFrame
            벤치마크 결과 (symbol, return, sharpe, mdd 등)
        """
        benchmark_config = self.config.get('BENCHMARK', {})

        if not benchmark_config.get('ENABLED', 'N') == 'Y':
            self.logger.info("📊 Benchmark comparison disabled (BENCHMARK.ENABLED=N)")
            return pd.DataFrame()

        benchmark_symbols = benchmark_config.get('SYMBOLS', [])
        if not benchmark_symbols:
            self.logger.warning("⚠️  No benchmark symbols configured")
            return pd.DataFrame()

        self.logger.info(f"\n📊 Calculating benchmark returns for {len(benchmark_symbols)} symbols...")

        # ✅ ETF 가격 데이터 로드 (별도 파일)
        # 주식 데이터(price_table)와 완전히 분리하여 ETF 오염 방지
        from src.backtest.etf_data_loader import ETFDataLoader

        try:
            etf_loader = ETFDataLoader(
                config=self.config,
                root_path=self.main_ctx.root_path,
                logger=self.logger
            )

            etf_price_table = etf_loader.load_etf_prices(
                symbols=benchmark_symbols,
                start_date=start_date,
                end_date=end_date
            )

            # 주식 + ETF 통합 (벤치마크 계산용만)
            # 주의: 이 통합 테이블은 벤치마크 계산에만 사용되며,
            #      ML 학습 데이터에는 절대 사용되지 않음
            if not etf_price_table.empty:
                combined_price_table = pd.concat([
                    price_table,
                    etf_price_table
                ], ignore_index=True)
                self.logger.info(f"   Combined: {len(price_table)} stock prices + {len(etf_price_table)} ETF prices")
            else:
                self.logger.warning("   ⚠️  No ETF data loaded, using stock data only")
                combined_price_table = price_table

        except Exception as e:
            self.logger.error(f"❌ ETF data loading failed: {e}")
            self.logger.warning("   Falling back to stock price table only")
            combined_price_table = price_table

        results = []

        # 실제 거래일 찾기 (통합 테이블 사용)
        actual_start = DataProcessor.get_trade_date(pd.Timestamp(start_date), combined_price_table)
        actual_end = DataProcessor.get_trade_date(pd.Timestamp(end_date), combined_price_table)

        if actual_start is None or actual_end is None:
            self.logger.error(f"❌ Cannot find trading dates for benchmark period")
            return pd.DataFrame()

        for symbol in benchmark_symbols:
            try:
                # 심볼 가격 데이터 가져오기 (통합 테이블에서)
                # 주식이면 price_table에서, ETF면 etf_price_table에서 자동으로 찾음
                symbol_prices = combined_price_table[combined_price_table['symbol'] == symbol]

                if symbol_prices.empty:
                    self.logger.warning(f"   ⚠️  {symbol}: No price data found (symbol may not exist or typo)")
                    continue

                # 시작 가격
                start_prices = symbol_prices[symbol_prices['date'] == actual_start]
                if start_prices.empty:
                    self.logger.warning(f"   ⚠️  {symbol}: No price at start date {actual_start.date()}")
                    continue
                start_price = start_prices.iloc[0]['close']

                # 종료 가격
                end_prices = symbol_prices[symbol_prices['date'] == actual_end]
                if end_prices.empty:
                    self.logger.warning(f"   ⚠️  {symbol}: No price at end date {actual_end.date()}")
                    continue
                end_price = end_prices.iloc[0]['close']

                # 총 수익률
                total_return = (end_price - start_price) / start_price

                # 기간 내 모든 가격 데이터 (MDD, Sharpe 계산용)
                period_prices = symbol_prices[
                    (symbol_prices['date'] >= actual_start) &
                    (symbol_prices['date'] <= actual_end)
                ].sort_values('date')

                if len(period_prices) < 2:
                    self.logger.warning(f"   ⚠️  {symbol}: Insufficient price data in period")
                    continue

                # 일별 수익률
                period_prices = period_prices.copy()
                period_prices['daily_return'] = period_prices['close'].pct_change()

                # Sharpe Ratio (연율화: √252)
                daily_returns = period_prices['daily_return'].dropna()
                if len(daily_returns) > 0:
                    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
                else:
                    sharpe = 0.0

                # Maximum Drawdown
                cumulative = (1 + period_prices['daily_return'].fillna(0)).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()

                # Win Rate (상승일 비율)
                win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0.0

                # 결과 저장
                results.append({
                    'strategy': symbol,
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'max_drawdown_pct': max_drawdown * 100,
                    'win_rate': win_rate,
                    'win_rate_pct': win_rate * 100,
                    'start_price': start_price,
                    'end_price': end_price,
                    'num_days': len(period_prices)
                })

                self.logger.info(
                    f"   ✅ {symbol}: {total_return*100:+.2f}% | "
                    f"Sharpe: {sharpe:.2f} | MDD: {max_drawdown*100:.2f}%"
                )

            except Exception as e:
                self.logger.warning(f"   ⚠️  {symbol}: Calculation failed - {str(e)}")
                continue

        if not results:
            self.logger.warning("⚠️  No valid benchmark results calculated")
            return pd.DataFrame()

        return pd.DataFrame(results)

    def run(self) -> pd.DataFrame:
        """
        Walk-Forward 백테스트 실행

        Returns:
        -------
        pd.DataFrame
            백테스트 결과
        """
        self.logger.info("="*80)
        self.logger.info("ML Walk-Forward Backtest Starting")
        self.logger.info("="*80)
        self.logger.info(f"Model type: {'SECTOR-BASED' if self.use_sector_model else 'UNIFIED'}")
        self.logger.info(f"Rebalance period: {self.rebalance_period} months")
        self.logger.info(f"Top K: {self.top_k}")
        self.logger.info(f"Retrain frequency: {self.retrain_frequency}")
        self.logger.info(f"Window type: {self.window_type}")

        # 가격 데이터 로드 (수익률 계산용)
        price_table = pd.read_parquet(self.main_ctx.root_path + "/processed/views/price.parquet")
        price_table['date'] = pd.to_datetime(price_table['date'])

        # 리밸런싱 날짜 생성
        # 우선순위: EVALUATION > BACKTEST (하위 호환성)
        eval_config = self.config.get('EVALUATION', {})
        backtest_config = self.config.get('BACKTEST', {})

        # PERIODS 읽기 (EVALUATION 우선, BACKTEST 폴백)
        periods = eval_config.get('PERIODS', backtest_config.get('PERIODS', []))

        if periods:
            # 여러 구간 모드
            self.logger.info(f"📅 Multiple backtest periods configured: {len(periods)} periods")
            source = "EVALUATION" if eval_config.get('PERIODS') else "BACKTEST"
            self.logger.info(f"   Config source: {source}.PERIODS")
            all_rebalance_dates = []

            for i, period in enumerate(periods):
                start_year = period.get('START_YEAR')
                end_year = period.get('END_YEAR')
                # 우선순위:
                # 1) PERIODS[*].START_MONTH/START_DATE
                # 2) BACKTEST.START_MONTH/START_DATE
                # 3) 최상위 START_MONTH/START_DATE (레거시 호환 앵커)
                # 4) 기본값 3/13
                start_month = int(period.get('START_MONTH', backtest_config.get('START_MONTH', self.config.get('START_MONTH', 3))))
                start_date_day = int(period.get('START_DATE', backtest_config.get('START_DATE', self.config.get('START_DATE', 13))))

                if not start_year or not end_year:
                    raise ValueError(f"Period {i+1} missing START_YEAR or END_YEAR")

                if isinstance(start_year, str):
                    start_year = int(start_year)
                if isinstance(end_year, str):
                    end_year = int(end_year)

                self.logger.info(f"  Period {i+1}: {start_year}/{start_month}/{start_date_day} ~ {end_year}/12/31")

                start_date = datetime(start_year, start_month, start_date_day)
                end_date = datetime(end_year, 12, 31)

                # 이 구간의 리밸런싱 날짜 생성
                current = start_date
                while current <= end_date:
                    all_rebalance_dates.append(current)
                    current += relativedelta(months=self.rebalance_period)

            rebalance_dates = sorted(all_rebalance_dates)

        else:
            # 단일 구간 모드 (하위 호환성)
            start_year = backtest_config.get('START_YEAR')
            end_year = backtest_config.get('END_YEAR')

            if not start_year or not end_year:
                raise ValueError(
                    "Backtest period configuration not found!\n\n"
                    "Please configure in one of these ways:\n"
                    "  Option 1 (Recommended): Use EVALUATION section\n"
                    "    EVALUATION:\n"
                    "      PERIODS:\n"
                    "        - START_YEAR: 2020\n"
                    "          END_YEAR: 2023\n"
                    "\n"
                    "  Option 2 (Legacy): Use BACKTEST section\n"
                    "    BACKTEST:\n"
                    "      START_YEAR: 2020\n"
                    "      END_YEAR: 2023\n"
                    "\n"
                    "See config/conf.yaml.template for examples."
                )

            if isinstance(start_year, str):
                start_year = int(start_year)
            if isinstance(end_year, str):
                end_year = int(end_year)

            start_month = int(backtest_config.get('START_MONTH', self.config.get('START_MONTH', 3)))
            start_date_day = int(backtest_config.get('START_DATE', self.config.get('START_DATE', 13)))

            self.logger.info(f"📅 Single backtest period: {start_year}/{start_month}/{start_date_day} ~ {end_year}/12/31")
            self.logger.info(f"   Config source: BACKTEST (legacy mode)")

            start_date = datetime(start_year, start_month, start_date_day)
            end_date = datetime(end_year, 12, 31)

            rebalance_dates = []
            current = start_date
            while current <= end_date:
                rebalance_dates.append(current)
                current += relativedelta(months=self.rebalance_period)

        # ✅ 거래일 조정 (regressor.py/make_mldata.py와 일원화)
        # 휴장일(주말, 공휴일)을 실제 거래 가능일로 조정
        self.logger.info(f"\n📅 Adjusting rebalance dates to actual trading days...")
        adjusted_dates = []

        for i, target_date in enumerate(rebalance_dates):
            # DataProcessor.get_trade_date()로 월초/월말 구분하여 거래일 찾기
            # - 월초(day <= 15): 미래 방향 (같은 분기 유지)
            # - 월말(day > 15): 과거 방향 (같은 분기 유지)
            actual_trade_date = DataProcessor.get_trade_date(pd.Timestamp(target_date), price_table)

            if actual_trade_date is None:
                self.logger.warning(
                    f"   ⚠️  Skipping {target_date.date()} - no trading day found within 10 days"
                )
                continue

            # 조정된 날짜가 원래 날짜와 다르면 로깅
            if actual_trade_date.date() != target_date.date():
                self.logger.info(
                    f"   {target_date.date()} → {actual_trade_date.date()} "
                    f"(adjusted to nearest trading day)"
                )
            else:
                self.logger.info(f"   {target_date.date()} (already a trading day)")

            # pd.Timestamp를 datetime으로 변환
            adjusted_dates.append(actual_trade_date.to_pydatetime())

        rebalance_dates = adjusted_dates
        self.logger.info(f"\n📅 Rebalance dates after adjustment: {len(rebalance_dates)}")
        for date in rebalance_dates:
            self.logger.info(f"   {date.date()}")

        # Walk-Forward 백테스트
        last_train_date = None
        current_models = None

        for i, rebalance_date in enumerate(rebalance_dates):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Rebalance #{i+1}: {rebalance_date.date()}")
            self.logger.info(f"{'='*80}")

            # 1. 사용 가능한 데이터 로드 (미래 유출 방지!)
            try:
                available_data = self._get_available_data_until(rebalance_date)
            except ValueError as e:
                self.logger.warning(f"⚠️ {e}, skipping this period")
                continue

            # 2. 캐시 사용 여부 확인 (Phase 4)
            cache_key = rebalance_date.strftime('%Y-%m-%d')
            use_cache_for_this_period = (
                self.use_cached_predictions and
                self.predictions_cache is not None and
                cache_key in self.predictions_cache
            )

            if use_cache_for_this_period:
                # 캐시에서 예측 로드
                self.logger.info(f"📦 Using cached predictions from regressor.py")
                cached_data = self.predictions_cache[cache_key]
                predictions = cached_data['predictions_df']

                if predictions is None or predictions.empty:
                    self.logger.warning(f"⚠️ Cache exists but predictions_df is empty, falling back to training")
                    use_cache_for_this_period = False

                if use_cache_for_this_period:
                    self.logger.info(f"   Loaded {len(predictions)} predictions from cache")
                    self.logger.info(f"   Top-K selected: {len(cached_data['top_k_selected'])} stocks")

            if not use_cache_for_this_period:
                # Normal training/prediction mode
                # 2. 모델 재학습 필요 여부 판단
                should_retrain = self._should_retrain(rebalance_date, last_train_date)

                if should_retrain:
                    # Window 타입에 따라 학습 데이터 필터링
                    if self.window_type == 'rolling' and self.window_size:
                        # Rolling: 최근 N년만
                        cutoff = rebalance_date - relativedelta(years=self.window_size)
                        train_data = available_data[
                            pd.to_datetime(available_data.get('fillingDate', available_data.index)) >= cutoff
                        ]
                    else:
                        # Expanding: 전체
                        train_data = available_data

                    # 모델 학습
                    current_models = self._train_model(train_data, rebalance_date)
                    last_train_date = rebalance_date
                else:
                    self.logger.info(f"📦 Reusing existing model from {last_train_date.date()}")

                # 3. 예측용 데이터 로드 (현재 시점의 최신 데이터)
                predict_file = self.data_path / f'rnorm_fs_{rebalance_date.year}_Q{(rebalance_date.month-1)//3 + 1}.parquet'
                if not predict_file.exists():
                    self.logger.warning(f"⚠️ Prediction file not found: {predict_file}")
                    continue

                predict_data = pd.read_parquet(predict_file)

                # 4. 예측 수행
                predictions = self._predict(current_models, predict_data)

            # 5. 상위 K개 선택 (symbol + sector 포함)
            selected_stocks = self._select_top_k(predictions)
            self.logger.info(f"📊 Selected {len(selected_stocks)} stocks")

            # 6. 수익률 계산 (다음 리밸런싱 날짜까지)
            if i < len(rebalance_dates) - 1:
                next_rebalance = rebalance_dates[i + 1]
                period_result = self._calculate_period_return(
                    selected_stocks,  # ✅ DataFrame (symbol, sector 포함)
                    rebalance_date,
                    next_rebalance,
                    price_table
                )

                avg_return = period_result['avg_return']
                self.logger.info(f"💰 Period return: {avg_return*100:.2f}%")

                # 결과 저장 (요약)
                self.backtest_results.append({
                    'rebalance_date': rebalance_date,
                    'actual_buy_date': period_result['actual_buy_date'],
                    'actual_sell_date': period_result['actual_sell_date'],
                    'num_stocks': len(selected_stocks),  # ✅ DataFrame 길이
                    'avg_return': avg_return,
                    'retrained': should_retrain
                })

                # 상세 정보 저장 (각 종목별 + 섹터)
                for detail in period_result['details']:
                    self.detailed_results.append({
                        'rebalance_date': rebalance_date,
                        'actual_buy_date': period_result['actual_buy_date'],
                        'actual_sell_date': period_result['actual_sell_date'],
                        'symbol': detail['symbol'],
                        'sector': detail.get('sector', 'Unknown'),  # ✅ 섹터 정보 추가
                        'buy_price': detail['buy_price'],
                        'sell_price': detail['sell_price'],
                        'return': detail['return'],
                        'return_pct': detail['return_pct']
                    })

        # 7. 최종 리포트
        results_df = pd.DataFrame(self.backtest_results)
        self._print_summary(results_df)

        # 8. 벤치마크 계산 (전체 백테스트 기간)
        benchmark_df = pd.DataFrame()
        if len(rebalance_dates) > 0:
            backtest_start = rebalance_dates[0]
            backtest_end = rebalance_dates[-1]
            benchmark_df = self._calculate_benchmark_returns(backtest_start, backtest_end, price_table)

            # ML 모델 성능 추가 (비교용)
            if not results_df.empty:
                total_return = (1 + results_df['avg_return']).prod() - 1
                avg_return = results_df['avg_return'].mean()
                std_return = results_df['avg_return'].std()
                sharpe = (avg_return / std_return) * np.sqrt(4) if std_return > 0 else 0.0  # Quarterly → Annual

                # MDD 계산
                cumulative_returns = (1 + results_df['avg_return']).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()

                # Win Rate
                win_rate = (results_df['avg_return'] > 0).sum() / len(results_df)

                ml_model_result = pd.DataFrame([{
                    'strategy': 'ML Model',
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'max_drawdown_pct': max_drawdown * 100,
                    'win_rate': win_rate,
                    'win_rate_pct': win_rate * 100,
                    'start_price': None,
                    'end_price': None,
                    'num_days': len(results_df)
                }])

                # ML Model을 첫 번째 행으로 추가
                if not benchmark_df.empty:
                    benchmark_df = pd.concat([ml_model_result, benchmark_df], ignore_index=True)
                else:
                    benchmark_df = ml_model_result

        # 결과 저장 - Excel 통합 레포트
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path('outputs/reports')
        report_dir.mkdir(parents=True, exist_ok=True)

        excel_file = report_dir / f'ml_backtest_report_{timestamp}.xlsx'

        # Excel Writer로 여러 시트 저장
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Summary (요약)
            results_df.to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 2: Detailed (상세)
            detailed_df = pd.DataFrame(self.detailed_results)
            if not detailed_df.empty:
                detailed_df.to_excel(writer, sheet_name='Detailed', index=False)

            # Sheet 3: Benchmark (벤치마크 비교)
            if not benchmark_df.empty:
                benchmark_df.to_excel(writer, sheet_name='Benchmark', index=False)

        self.logger.info(f"\n✅ Backtest report saved: {excel_file}")
        if not detailed_df.empty:
            self.logger.info(f"   Total trades: {len(detailed_df)}")
        if not benchmark_df.empty:
            self.logger.info(f"   Benchmark comparisons: {len(benchmark_df)}")

        return results_df

    def _print_summary(self, results: pd.DataFrame):
        """백테스트 요약 출력"""
        self.logger.info("\n" + "="*80)
        self.logger.info("BACKTEST SUMMARY")
        self.logger.info("="*80)

        total_return = (1 + results['avg_return']).prod() - 1
        avg_return = results['avg_return'].mean()
        std_return = results['avg_return'].std()
        sharpe = avg_return / std_return * np.sqrt(12/self.rebalance_period) if std_return > 0 else 0

        # MDD 계산
        cumulative = (1 + results['avg_return']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        mdd = drawdown.min()

        win_rate = (results['avg_return'] > 0).sum() / len(results)

        self.logger.info(f"Total Periods: {len(results)}")
        self.logger.info(f"Total Return: {total_return*100:.2f}%")
        self.logger.info(f"Average Return: {avg_return*100:.2f}%")
        self.logger.info(f"Std Dev: {std_return*100:.2f}%")
        self.logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        self.logger.info(f"Max Drawdown: {mdd*100:.2f}%")
        self.logger.info(f"Win Rate: {win_rate*100:.1f}%")
        self.logger.info(f"Models Retrained: {results['retrained'].sum()} times")
