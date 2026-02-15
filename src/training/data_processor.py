"""
통합 데이터 전처리 파이프라인

이 모듈은 regressor.py와 ml_backtest.py가 동일한 전처리 단계를 사용하도록
보장하는 중앙화된 DataProcessor를 제공합니다.

Key Features:
- 희소 행/열 제거
- 이상치 클리핑 (분위수 기반)
- Feature 스케일링 (RobustScaler 또는 StandardScaler)
- NaN 처리
- Feature/타겟 분리
- 학습과 테스트 간 일관된 전처리

Critical: 이 모듈은 코드 중복을 제거하고, 한 곳에서 수정된 버그가
regressor.py와 ml_backtest.py 양쪽에 자동으로 적용되도록 합니다.

Author: Quant Trading Team
Date: 2025-12-01
"""

from typing import Tuple, Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.constants.data_schema import DataSchema
from src.training.data_quality import (
    DataQualityReport,
    DataCleaningValidator,
    generate_data_quality_report
)


class DataProcessor:
    """
    통합 데이터 전처리 파이프라인입니다.

    이 클래스는 regressor.py와 ml_backtest.py의 모든 데이터 전처리 로직을
    단일하고 재사용 가능한 구현으로 통합합니다.

    Purpose:
    --------
    - 전처리의 단일 진실 공급원 (Single source of truth)
    - 학습과 백테스팅에서 동일한 전처리 보장
    - 전체 시스템에 걸친 쉬운 파라미터 튜닝

    사용 예시 (학습):
    -----------------
    processor = DataProcessor(config)
    result = processor.full_pipeline(train_df, test_df)
    X_train, y_train = result['X_train'], result['y_train']

    사용 예시 (백테스팅):
    --------------------
    processor = DataProcessor(config)
    processor.fit(train_df)
    X_test = processor.transform(test_df)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        DataProcessor를 초기화합니다.

        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            설정 딕셔너리 (conf.yaml에서 로드)
        logger : Optional[logging.Logger]
            Logger 인스턴스
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger('DataProcessor')

        # 상태 변수 (fit/transform 패턴용)
        self.scaler = None
        self.clip_bounds = {}  # {col: (lower, upper)}
        self.dropped_cols = []
        self.feature_names = []
        self.is_fitted = False

    # ========================================================================
    # 데이터 로딩 (분기별 Parquet 파일)
    # ========================================================================

    @staticmethod
    def load_quarterly_data(
        data_dir: str,
        year: int,
        file_prefix: str = 'rnorm_ml',
        quarters: Optional[List[str]] = None,
        cutoff_date: Optional[pd.Timestamp] = None,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:
        """
        분기별 parquet 파일을 로드하고 연결합니다.

        이 메서드는 regressor.py (최신 예측)와 ml_backtest.py (백테스팅) 모두를 위한
        통합된 분기별 데이터 로딩 방법을 제공합니다.

        **Key Features**:
        - 여러 분기별 parquet 파일 로드 (Q1, Q2, Q3, Q4)
        - ignore_index=True로 연결 (중복 인덱스 방지)
        - 백테스팅을 위한 선택적 cutoff_date 필터링
        - 일관된 에러 처리

        Parameters:
        -----------
        data_dir : str
            parquet 파일이 있는 디렉토리
        year : int
            로드할 연도 (예: 2025)
        file_prefix : str
            파일 접두사 (기본값: 'rnorm_ml')
            - 백테스팅: 'rnorm_ml' 사용
            - 최신 예측: 'rnorm_fs' 사용
        quarters : Optional[List[str]]
            로드할 분기 (기본값: ['Q1', 'Q2', 'Q3', 'Q4'])
        cutoff_date : Optional[pd.Timestamp]
            제공 시 fillingDate <= cutoff_date인 데이터만 필터링
            백테스팅에서 과거 시점의 정보만 사용하기 위해 사용
        logger : Optional[logging.Logger]
            Logger 인스턴스

        Returns:
        --------
        pd.DataFrame
            깨끗한 순차 인덱스를 가진 결합된 분기별 데이터

        Raises:
        -------
        ValueError
            지정된 연도에 대한 데이터 파일을 찾을 수 없는 경우

        사용 예시 (regressor.py):
        -----------------------
        ldf = DataProcessor.load_quarterly_data(
            data_dir='../data_parquet/ml_input',
            year=2025,
            file_prefix='rnorm_fs',
            logger=logging.getLogger()
        )

        사용 예시 (ml_backtest.py):
        --------------------------
        df = DataProcessor.load_quarterly_data(
            data_dir=self.data_path,
            year=2023,
            file_prefix='rnorm_ml',
            cutoff_date=pd.Timestamp('2023-06-15'),
            logger=self.logger
        )
        """
        if logger is None:
            logger = logging.getLogger('DataProcessor')

        if quarters is None:
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']

        all_data = []
        loaded_quarters = []

        for Q in quarters:
            file_path = Path(data_dir) / f'{file_prefix}_{year}_{Q}.parquet'

            if not file_path.exists():
                logger.debug(f"Quarterly file not found: {file_path.name}")
                continue

            try:
                df = pd.read_parquet(file_path)

                # 백테스팅을 위한 cutoff_date 필터 적용 (제공된 경우)
                if cutoff_date is not None and 'fillingDate' in df.columns:
                    df['fillingDate'] = pd.to_datetime(df['fillingDate'])
                    df = df[df['fillingDate'] <= cutoff_date]

                if not df.empty:
                    all_data.append(df)
                    loaded_quarters.append(Q)
                    logger.debug(f"Loaded {year}_{Q}: {len(df)} rows")

            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
                continue

        if not all_data:
            raise ValueError(
                f"No data loaded for year {year} "
                f"(prefix='{file_prefix}', quarters={quarters})"
            )

        # ✅ ignore_index=True로 연결하여 중복 인덱스 방지
        # 섹터 예측에서 .loc[index] 사용 시 필수적
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"Loaded {year} data: {len(combined_df)} rows "
            f"from quarters {loaded_quarters}"
        )

        return combined_df

    # ========================================================================
    # Feature 이름 정규화 (공통 로직)
    # ========================================================================

    @staticmethod
    def normalize_feature_names(
        df: pd.DataFrame,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:
        """
        Feature 이름에서 특수 JSON 문자를 제거하여 정규화합니다.

        **모델 학습에 필수**: XGBoost, LightGBM, CatBoost 모델은
        괄호 (), 대괄호 [], 중괄호 {}, 쉼표 등의 특수 JSON 문자를
        feature 이름에서 처리할 수 없습니다.

        이 함수는 다음 모듈 간 일관된 feature 이름을 보장합니다:
        - make_mldata.py (ML 데이터 준비)
        - regressor.py (모델 학습)
        - ml_backtest.py (walk-forward 백테스팅)

        tsfresh에서 자주 문제되는 feature 이름:
        - 'feature__param_(value1, value2)' → 'feature_param_value1_value2'
        - 'feature[index]' → 'feature_index'
        - 'feature{key}' → 'feature_key'
        - 'feature:value' → 'feature_value'

        전략: 영숫자와 밑줄만 유지합니다.
        기타 모든 문자(마침표, 콜론, 슬래시 등)는 최대 호환성을 위해
        밑줄로 대체됩니다.

        Parameters:
        -----------
        df : pd.DataFrame
            문제가 될 수 있는 feature 이름을 가진 DataFrame
        logger : Optional[logging.Logger]
            정규화 추적용 Logger

        Returns:
        --------
        pd.DataFrame
            정규화된 컬럼 이름을 가진 DataFrame

        사용 예시:
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'price_(2, 5)': [1, 2, 3],
        ...     'volume[0]': [10, 20, 30],
        ...     'ratio{a}': [0.1, 0.2, 0.3]
        ... })
        >>> normalized_df = DataProcessor.normalize_feature_names(df)
        >>> print(normalized_df.columns.tolist())
        ['price_2_5', 'volume_0', 'ratio_a']
        """
        import re

        original_cols = df.columns.tolist()

        # 특수 JSON 문자를 밑줄로 대체
        # 전략: 영숫자와 밑줄만 유지, 나머지는 모두 대체
        normalized_cols = []
        for col in original_cols:
            # 영숫자가 아닌 모든 문자(밑줄 제외)를 밑줄로 대체
            # 더 공격적이지만 JSON/XGBoost/LightGBM 호환성을 위해 안전
            new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
            # 연속 밑줄 제거
            new_col = re.sub(r'_+', '_', new_col)
            # 앞뒤 밑줄 제거
            new_col = new_col.strip('_')
            normalized_cols.append(new_col)

        # 변경 사항이 있는지 확인
        changed_count = sum(1 for orig, norm in zip(original_cols, normalized_cols) if orig != norm)

        # CRITICAL FIX: 정규화 후 중복 이름 처리
        # 다른 원본 이름이 같은 이름으로 정규화될 수 있음:
        # 'feature(a, b)' → 'feature_a_b'
        # 'feature[a, b]' → 'feature_a_b'  (중복!)
        # 고유성 보장을 위해 중복에 숫자 접미사 추가
        if len(normalized_cols) != len(set(normalized_cols)):
            seen = {}
            unique_cols = []
            for col in normalized_cols:
                if col not in seen:
                    seen[col] = 0
                    unique_cols.append(col)
                else:
                    seen[col] += 1
                    unique_cols.append(f"{col}_{seen[col]}")

            duplicate_count = sum(1 for count in seen.values() if count > 0)
            if logger:
                logger.warning(f"⚠️  Found {duplicate_count} duplicate normalized names, added numeric suffixes")

            normalized_cols = unique_cols

        if changed_count > 0:
            if logger:
                logger.info(f"   🔧 Normalized {changed_count} feature names (removed special characters)")
                # 몇 가지 예시 표시
                examples = [
                    f"      - '{orig}' → '{norm}'"
                    for orig, norm in zip(original_cols, normalized_cols)
                    if orig != norm
                ][:3]  # Show first 3 examples
                if examples:
                    for example in examples:
                        logger.info(example)
                    if changed_count > 3:
                        logger.info(f"      ... and {changed_count - 3} more")

        # 정규화된 컬럼 이름으로 새 DataFrame 생성
        df_normalized = df.copy()
        df_normalized.columns = normalized_cols

        return df_normalized

    # ========================================================================
    # 섹터 예측 (공통 로직)
    # ========================================================================

    @staticmethod
    def prepare_sector_data(
        sector_df: pd.DataFrame,
        sector_model: Any,
        y_col_list: List[str],
        use_winsorization: bool = True,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:
        """
        섹터 데이터 전처리 (regressor.py와 ml_backtest.py 공통 로직)

        이 함수는 섹터별 예측을 위한 데이터 전처리를 통합합니다.
        regressor.py와 ml_backtest.py가 동일한 전처리를 사용하도록 보장합니다.

        **Critical**: 백테스트 결과가 실제 예측과 일치하려면 동일한 전처리 필수

        Parameters:
        -----------
        sector_df : pd.DataFrame
            섹터 필터링된 원본 데이터 (sector 컬럼 포함)
        sector_model : Any
            섹터 모델 (feature list 추출용)
        y_col_list : List[str]
            제외할 타겟 컬럼 리스트
        use_winsorization : bool
            Winsorization 적용 여부
        logger : Optional[logging.Logger]
            Logger instance

        Returns:
        --------
        pd.DataFrame
            전처리된 feature 데이터 (X)

        Steps:
        ------
        1. sector 컬럼 제거
        2. y 컬럼 제외하고 feature만 추출
        3. Feature name cleaning
        4. Feature alignment (모델이 학습한 피처만 유지)
        5. Winsorization 적용

        Example:
        --------
        X = DataProcessor.prepare_sector_data(
            sector_df=df[df['sector'] == 'Healthcare'],
            sector_model=sector_models[('Healthcare', 0)],
            y_col_list=['price_dev', 'sec_price_dev_subavg'],
            use_winsorization=True
        )
        y_pred = sector_model.predict(X)
        """
        if logger is None:
            logger = logging.getLogger('DataProcessor')

        # 1. 섹터 컬럼 제거 (명시적 복사로 SettingWithCopyWarning 방지)
        df_no_sector = sector_df.drop('sector', axis=1, errors='ignore').copy()

        # 2. y 컬럼 제외하고 feature만 추출 (.copy()로 SettingWithCopyWarning 방지)
        X = df_no_sector[df_no_sector.columns.difference(y_col_list)].copy()

        # 3. Feature name cleaning (특수문자 제거)
        X = DataProcessor.clean_feature_names(X)

        # 4. Feature alignment (모델이 학습한 피처만 유지)
        if hasattr(sector_model, 'get_booster'):
            model_features = sector_model.get_booster().feature_names
            if model_features is not None:
                # ✅ FIX: Ensure X is a copy to avoid SettingWithCopyWarning
                # Direct assignment handles copy internally if needed
                missing_features = set(model_features) - set(X.columns)
                for col in missing_features:
                    X[col] = np.nan  # ✅ Changed from X.loc[:, col] = np.nan

                # 모델 피처만 선택 (순서 맞춤)
                X = X[model_features]

        # 5. Winsorization 적용
        if use_winsorization:
            X = DataProcessor.winsorize_features(
                X,
                lower_percentile=0.01,
                upper_percentile=0.99,
                enabled=True
            )

        return X

    @staticmethod
    def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature 이름에서 특수문자 제거

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            Cleaned feature names
        """
        # 컬럼명에서 특수문자 제거
        df.columns = df.columns.str.replace('[', '_', regex=False)
        df.columns = df.columns.str.replace(']', '_', regex=False)
        df.columns = df.columns.str.replace('<', '_', regex=False)
        df.columns = df.columns.str.replace('>', '_', regex=False)
        df.columns = df.columns.str.replace(',', '_', regex=False)
        return df

    @staticmethod
    def align_features_to_model(
        X: pd.DataFrame,
        model: Any,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:
        """
        모델 예측을 위한 Feature 정렬입니다.

        **regressor.py와 ml_backtest.py 일관성에 필수적**

        이 함수는 테스트 데이터가 학습된 모델이 기대하는 정확한 feature를
        올바른 순서로 갖도록 보장합니다. 이것이 필수적인 이유:
        1. 학습 데이터는 전처리 중 feature가 삭제되었을 수 있음
        2. 테스트 데이터는 데이터 가용성으로 인해 feature가 누락될 수 있음
        3. XGBoost/LightGBM은 정확한 feature 일치 필요 (이름 + 순서)

        Parameters:
        -----------
        X : pd.DataFrame
            테스트 데이터 feature (모델이 기대하는 것과 다른 컬럼을 가질 수 있음)
        model : Any
            학습된 모델 (XGBoost/LightGBM용 get_booster() 메서드 필요)
        logger : Optional[logging.Logger]
            정보/경고 메시지용 Logger

        Returns:
        --------
        pd.DataFrame
            모델이 기대하는 feature와 일치하도록 정렬된 feature

        Process:
        --------
        1. model.get_booster().feature_names에서 기대 feature 추출
        2. 누락된 feature 추가 (NaN으로 채움 - XGBoost가 처리 가능)
        3. 모델에 없는 추가 feature 제거
        4. 학습 시 순서에 맞게 컬럼 재정렬 (필수!)

        사용 예시:
        --------
        >>> # regressor.py
        >>> x_test_full = DataProcessor.align_features_to_model(
        ...     x_test_full, self.clsmodels[2], logger
        ... )
        >>> y_probs = model.predict_proba(x_test_full)

        >>> # ml_backtest.py
        >>> X = DataProcessor.align_features_to_model(
        ...     X, models['classifier'], self.logger
        ... )
        >>> y_pred = model.predict(X)

        Notes:
        ------
        - regressor.py와 ml_backtest.py의 중복된 정렬 코드를 대체합니다
        - 일관성 보장: 여기서 수정된 버그가 학습과 백테스팅 모두에 적용됩니다
        - 누락된 feature는 NaN으로 채움 (트리 모델은 NaN을 네이티브로 처리)
        """
        if logger is None:
            logger = logging.getLogger('DataProcessor')

        # 명시적 복사로 SettingWithCopyWarning 방지
        X = X.copy()

        # 모델에서 기대 feature 추출
        if hasattr(model, 'get_booster'):
            model_features = model.get_booster().feature_names
            if model_features is not None:
                logger.info(f"   Aligning features to model...")
                logger.info(f"   Model expects {len(model_features)} features")
                logger.info(f"   Test data has {len(X.columns)} features")

                # 누락된 feature를 NaN으로 추가
                missing_features = set(model_features) - set(X.columns)
                if missing_features:
                    logger.warning(f"   ⚠️  {len(missing_features)} features missing in test data, filling with NaN")
                    for col in missing_features:
                        X.loc[:, col] = np.nan

                # 추가 feature 제거
                extra_features = set(X.columns) - set(model_features)
                if extra_features:
                    logger.info(f"   Removing {len(extra_features)} extra features from test data")
                    X = X.drop(columns=list(extra_features))

                # 학습 시 순서에 맞게 feature 재정렬 (필수!)
                X = X[model_features]
                logger.info(f"   ✅ Feature alignment complete: {len(X.columns)} features")
        else:
            logger.warning("   Model does not have get_booster() method, skipping alignment")

        return X

    # ========================================================================
    # 희소 데이터 처리
    # ========================================================================

    @staticmethod
    def drop_sparse_rows(
        df: pd.DataFrame,
        threshold: float = 0.9
    ) -> pd.DataFrame:
        """
        NaN 값이 너무 많은 행을 제거합니다.

        NaN 비율이 임계값을 초과하는 행을 제거합니다.

        Parameters:
        -----------
        df : pd.DataFrame
            입력 DataFrame
        threshold : float
            NaN 허용 임계값 (기본값: 0.9)
            예: 0.9는 NaN이 90% 미만인 행만 유지

        Returns:
        --------
        pd.DataFrame
            정리된 DataFrame

        사용 예시:
        --------
        # NaN이 90% 초과인 행 제거
        df_clean = DataProcessor.drop_sparse_rows(df, threshold=0.9)
        """
        nan_count_per_row = df.isnull().sum(axis=1)
        max_allowed_nan = int(len(df.columns) * (1 - threshold))
        mask = nan_count_per_row < max_allowed_nan

        dropped_count = len(df) - mask.sum()
        if dropped_count > 0:
            logging.info(f"   Dropped {dropped_count} sparse rows (threshold={threshold})")

        return df[mask].copy()

    def drop_sparse_cols(
        self,
        df: pd.DataFrame,
        missing_threshold: Optional[float] = None,
        same_value_threshold: Optional[float] = None,
        protect_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        결측값이 너무 많거나 동일 값이 너무 많은 컬럼을 제거합니다.

        정보가 없는 컬럼을 식별하고 제거합니다:
        1. 결측값이 80% 초과인 컬럼
        2. 값의 98% 이상이 동일한 컬럼

        **설정 기반**: FEATURES 설정에서 임계값을 읽습니다 (단일 진실 공급원)

        Parameters:
        -----------
        df : pd.DataFrame
            입력 DataFrame
        missing_threshold : Optional[float]
            최대 허용 결측 비율 (기본값: FEATURES.MISSING_THRESHOLD 또는 0.8에서 읽음)
        same_value_threshold : Optional[float]
            최대 허용 동일값 비율 (기본값: FEATURES.SAME_VALUE_THRESHOLD 또는 0.95에서 읽음)
        protect_cols : Optional[List[str]]
            절대 삭제하지 않을 컬럼 (예: 메타데이터)

        Returns:
        --------
        df_clean : pd.DataFrame
            희소 컬럼이 제거된 DataFrame
        dropped_cols : List[str]
            삭제된 컬럼 이름 리스트

        사용 예시:
        --------
        processor = DataProcessor(config)
        df_clean, dropped = processor.drop_sparse_cols(df)  # 설정 값 사용
        print(f"Dropped {len(dropped)} uninformative columns")
        """
        # 제공되지 않으면 config에서 읽기 (regressor와 ml_backtest 동일 로직)
        if missing_threshold is None:
            features_config = self.config.get('FEATURES', {})
            missing_threshold = float(features_config.get('MISSING_THRESHOLD', 0.8))

        if same_value_threshold is None:
            features_config = self.config.get('FEATURES', {})
            same_value_threshold = float(features_config.get('SAME_VALUE_THRESHOLD', 0.95))

        if protect_cols is None:
            protect_cols = DataSchema.get_excluded_cols()

        cols_to_drop = []
        cols_to_drop_missing = []
        cols_to_drop_same = []

        for col in df.columns:
            # 보호된 컬럼 건너뛰기
            if col in protect_cols:
                continue

            # 결측 비율 확인
            missing_ratio = df[col].isna().mean()
            if missing_ratio > missing_threshold:
                cols_to_drop.append(col)
                cols_to_drop_missing.append(col)
                continue

            # 동일값 비율 확인
            value_counts = df[col].value_counts(normalize=True, dropna=False)
            if len(value_counts) > 0:
                top_value_ratio = value_counts.iloc[0]
                if top_value_ratio > same_value_threshold:
                    cols_to_drop.append(col)
                    cols_to_drop_same.append(col)

        logging.info(f"   Dropping {len(cols_to_drop)} sparse columns:")
        logging.info(f"     - {len(cols_to_drop_missing)} due to missing values")
        logging.info(f"     - {len(cols_to_drop_same)} due to same values")

        df_clean = df.drop(columns=cols_to_drop)

        return df_clean, cols_to_drop

    # ========================================================================
    # 무한값 처리
    # ========================================================================

    @staticmethod
    def remove_infinite_values(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        무한값이 있는 행을 제거합니다 (XGBoost 호환성).

        XGBoost는 무한값을 처리할 수 없으며 에러를 발생시킵니다.
        이 메서드는 inf 또는 -inf를 포함하는 모든 행을 제거합니다.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature DataFrame
        y : Optional[pd.Series]
            타겟 Series (제공 시 X와 정렬됨)

        Returns:
        --------
        X_clean : pd.DataFrame
            무한값이 없는 DataFrame
        y_clean : Optional[pd.Series]
            무한값이 없는 타겟 Series (제공된 경우)

        사용 예시:
        --------
        X_clean, y_clean = DataProcessor.remove_infinite_values(X, y)
        """
        # 무한값 확인 (숫자 컬럼에서만)
        # 문자열/객체 컬럼은 np.isinf()에서 TypeError 발생
        # 메모리 효율적: 대규모 데이터에서 메모리 급증을 유발하는 select_dtypes 사용 회피
        numeric_cols = [col for col in X.columns
                       if pd.api.types.is_numeric_dtype(X[col])]

        if len(numeric_cols) == 0:
            # 숫자 컬럼이 없으므로 확인할 필요 없음
            return X, y

        inf_mask = np.isinf(X[numeric_cols])
        rows_with_inf = inf_mask.any(axis=1)

        if rows_with_inf.sum() > 0:
            logging.warning(f"⚠️  Found {rows_with_inf.sum()} rows with infinite values, removing...")
            X_clean = X[~rows_with_inf].copy()

            if y is not None:
                y_clean = y[~rows_with_inf].copy()
                logging.info(f"   After infinite removal: {len(X_clean)} rows remaining")
                return X_clean, y_clean
            else:
                logging.info(f"   After infinite removal: {len(X_clean)} rows remaining")
                return X_clean, None

        return X, y

    @staticmethod
    def replace_infinite_with_nan(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        무한값을 NaN으로 대체합니다.

        행을 제거하는 것보다 inf를 NaN으로 대체하는 것이 나은 경우가 있습니다.
        이렇게 하면 fillna()로 처리할 수 있습니다.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature DataFrame
        y : Optional[pd.Series]
            타겟 Series

        Returns:
        --------
        X_clean : pd.DataFrame
            inf가 NaN으로 대체된 DataFrame
        y_clean : Optional[pd.Series]
            inf가 NaN으로 대체된 타겟 (제공된 경우)
        """
        X_clean = X.replace([np.inf, -np.inf], np.nan)

        if y is not None:
            y_clean = y.replace([np.inf, -np.inf], np.nan)
            return X_clean, y_clean

        return X_clean, None

    @staticmethod
    def check_duplicate_index(
        df: pd.DataFrame,
        stage_name: str,
        logger: Optional[logging.Logger] = None
    ) -> bool:
        """
        DataFrame의 중복 인덱스를 확인하고 상세 정보를 로깅합니다.

        ✨ UNIFIED: regressor.py와 ml_backtest.py 모두에서 디버깅에 사용됩니다.

        이 함수는 중복 인덱스 에러의 원인을 추적하는 데 도움을 줍니다:
        - 중복 존재 여부 감지
        - 어떤 인덱스가 중복되었는지 로깅
        - 각 중복이 몇 번 나타나는지 표시
        - 실행 가능한 디버깅 정보 제공

        Parameters:
        -----------
        df : pd.DataFrame
            중복 인덱스를 확인할 DataFrame
        stage_name : str
            현재 처리 단계 설명 (예: "after train/test split")
        logger : Optional[logging.Logger]
            사용할 Logger 인스턴스. None이면 루트 Logger 사용.

        Returns:
        --------
        bool
            중복이 발견되면 True, 아니면 False

        사용 예시:
        --------
        >>> if DataProcessor.check_duplicate_index(x_train, "after split", logger):
        >>>     # 중복 처리
        >>>     x_train = x_train[~x_train.index.duplicated(keep='first')]
        """
        if logger is None:
            logger = logging.getLogger()

        logger.info("=" * 80)
        logger.info(f"🔍 DUPLICATE INDEX CHECK: {stage_name}")

        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.error(f"❌ FOUND {dup_count} DUPLICATE INDICES!")

            # 처음 10개의 중복 인덱스 표시
            dup_indices = df.index[df.index.duplicated()].unique()[:10]
            logger.error(f"   First 10 duplicate indices: {list(dup_indices)}")

            # Shape 표시
            logger.error(f"   DataFrame shape: {df.shape}")

            # 가장 흔한 중복 표시
            dup_mask = df.index.duplicated(keep=False)
            dup_values = df.index[dup_mask].value_counts().head(5)
            logger.error(f"   Top 5 most common duplicate values:")
            for idx, count in dup_values.items():
                logger.error(f"      Index {idx} appears {count} times")

            logger.error(f"   ⚠️  This may cause 'cannot reindex on an axis with duplicate labels' error!")
            logger.info("=" * 80)
            return True
        else:
            logger.info(f"✅ No duplicates ({len(df)} rows)")
            logger.info("=" * 80)
            return False

    @staticmethod
    def create_clean_dataframe(
        series: pd.Series,
        columns: pd.Index,
        index: pd.Index
    ) -> pd.DataFrame:
        """
        깨끗한 인덱스로 Series에서 DataFrame을 생성합니다 (인덱스 충돌 방지).

        ✨ UNIFIED: regressor.py의 여러 곳에서 "cannot reindex on an axis with
        duplicate labels" 에러를 방지하기 위해 사용됩니다.

        이미 인덱스가 있는 Series에서 DataFrame을 생성할 때, pandas가 Series의
        원본 인덱스를 새 인덱스와 정렬하려고 시도하여 충돌이 발생할 수 있습니다.
        이 메서드는 .values를 사용하여 원본 인덱스를 제거하고 깨끗한 새 인덱스를 할당합니다.

        Parameters:
        -----------
        series : pd.Series
            데이터가 있는 Series (인덱스가 제거됨)
        columns : pd.Index
            DataFrame의 컬럼 이름
        index : pd.Index
            할당할 새 깨끗한 인덱스

        Returns:
        --------
        pd.DataFrame
            충돌 없는 깨끗한 인덱스를 가진 DataFrame

        사용 예시:
        --------
        >>> y_clean = pd.Series([1, 2, 3], index=[10, 20, 30])
        >>> new_index = pd.Index([0, 1, 2])
        >>> df = DataProcessor.create_clean_dataframe(y_clean, ['target'], new_index)
        >>> df.index  # [0, 1, 2] not [10, 20, 30]

        이것이 중요한 이유:
        -----------------
        ❌ 잘못된 방법:
        pd.DataFrame({col: series}, index=new_index)
        → pandas가 series.index를 new_index와 정렬 시도 → 충돌!

        ✅ 올바른 방법:
        pd.DataFrame(series.values, columns=[col], index=new_index)
        → 정렬 없이 깨끗한 할당
        """
        return pd.DataFrame(series.values, columns=columns, index=index)

    @staticmethod
    def log_transform_features(
        X: pd.DataFrame,
        inverse: bool = False
    ) -> pd.DataFrame:
        """
        극단값 처리를 위한 부호 보존 로그 변환입니다.

        ✨ UNIFIED: 하드코딩된 임계값 클리핑을 적응형 로그 스케일링으로 대체합니다.
        regressor.py와 ml_backtest.py 모두에서 사용됩니다:
        - 통합 모델 (x_train)
        - 섹터 모델 (sector_x_train)

        로그 변환이 clip_extreme_values(threshold=1e10)보다 나은 이유:
        ========================================================================
        1. ✅ 실제 값 순서 보존 (Apple $3T > Microsoft $2T)
        2. ✅ 모든 스케일에 적응 (시가총액은 무한히 성장 가능)
        3. ✅ 자연스러운 압축 (1e12 → 27.6, 1e10 → 23.0)
        4. ✅ 역변환 가능 (필요시 원래 스케일 복원)
        5. ✅ 임의 임계값 없음 (1e10은 대형주에 너무 작음)

        하드 클리핑의 문제점:
        ========================================================================
        - Apple 시가총액: $3조 = 3e12 → 1e10으로 클리핑 (300배 압축!)
        - NVIDIA 시가총액: $2조 = 2e12 → 1e10으로 클리핑 (200배 압축!)
        - 모든 대형주가 구분 불가능해짐

        Formula:
        ========================================================================
        - Forward: sign(x) * log(1 + |x|)
          → Handles positive, negative, and zero values
          → log1p(x) = log(1 + x) is numerically stable near 0

        - Inverse: sign(x) * (exp(|x|) - 1)
          → Recovers original scale (useful for feature importance interpretation)

        Example transformation:
        ========================================================================
        Original Value    →  Log Transformed  →  Interpretation
        ---------------      ----------------     ---------------
        3e12 (Apple)     →  28.7              →  Preserved distinction
        2e12 (NVIDIA)    →  28.3              →  Preserved distinction
        1e12             →  27.6              →  Preserved distinction
        1e10             →  23.0              →
        1e8              →  18.4              →
        0                →  0                 →  Zero stays zero
        -1e8             →  -18.4             →  Sign preserved

        Parameters:
        -----------
        X : pd.DataFrame
            변환할 Feature
        inverse : bool
            True이면 역변환 (exp)으로 원래 스케일 복원
            기본값: False (정방향 변환)

        Returns:
        --------
        X_transformed : pd.DataFrame
            로그 변환된 Feature (입력과 동일한 shape 및 컬럼)

        사용 예시:
        ------
        # 학습 (정방향 변환)
        X_train_log = DataProcessor.log_transform_features(X_train)

        # 예측 (동일한 변환 사용)
        X_test_log = DataProcessor.log_transform_features(X_test)

        # 해석 (역변환, 선택 사항)
        X_original = DataProcessor.log_transform_features(X_train_log, inverse=True)

        Notes:
        ------
        - 모든 feature에 적용됨 (트리 기반 모델은 스케일에 무관)
        - XGBoost/LightGBM은 압축된 스케일에서 이점 ("value too large" 에러 방지)
        - Feature 이름과 DataFrame 구조 보존
        """
        # 숫자 컬럼만 변환, 문자열/객체 컬럼은 그대로 유지
        # 메모리 효율적: 대규모 데이터에서 메모리 급증을 유발하는 select_dtypes 사용 회피
        numeric_cols = [col for col in X.columns
                       if pd.api.types.is_numeric_dtype(X[col])]
        object_cols = [col for col in X.columns
                      if not pd.api.types.is_numeric_dtype(X[col])]

        if len(numeric_cols) == 0:
            # 변환할 숫자 컬럼이 없음
            return X.copy()

        X_transformed = X.copy()

        # 메모리 효율적 처리: 메모리 폭발을 피하기 위해 청크 단위로 처리
        # 10,972개 feature를 한 번에 처리하면 12GB 이상 필요
        chunk_size = 500  # Process 500 columns at a time

        if inverse:
            # 역변환: sign(x) * (exp(|x|) - 1)
            # 로그 변환된 값에서 원래 스케일 복원
            # ✅ CRITICAL FIX: NaN 값 건너뛰기 (그대로 보존)
            for i in range(0, len(numeric_cols), chunk_size):
                chunk_cols = numeric_cols[i:i+chunk_size]
                X_transformed[chunk_cols] = X[chunk_cols].apply(
                    lambda col: np.where(
                        pd.isna(col),
                        np.nan,  # NaN stays NaN
                        np.sign(col) * (np.exp(np.abs(col)) - 1)
                    )
                )
        else:
            # 정방향: sign(x) * log(1 + |x|)
            # 순서를 보존하면서 극단값 압축
            # ✅ CRITICAL FIX: NaN 값 건너뛰기 (그대로 보존)
            for i in range(0, len(numeric_cols), chunk_size):
                chunk_cols = numeric_cols[i:i+chunk_size]
                X_transformed[chunk_cols] = X[chunk_cols].apply(
                    lambda col: np.where(
                        pd.isna(col),
                        np.nan,  # NaN stays NaN (XGBoost will handle it)
                        np.sign(col) * np.log1p(np.abs(col))
                    )
                )

        # 객체 컬럼은 변경되지 않음
        return X_transformed

    @staticmethod
    def preprocess_training_data(
        X: pd.DataFrame,
        y: pd.DataFrame,
        y_cls: Optional[pd.DataFrame] = None,
        config: Optional[Dict] = None,
        logger=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], List[str]]:
        """
        모든 학습 데이터 전처리의 단일 진실 공급원 (SINGLE SOURCE OF TRUTH)입니다.

        regressor.py와 ml_backtest.py 모두에서 동일한 전처리를 보장하기 위해 사용됩니다.

        이 메서드는 산재된 모든 전처리 코드를 하나의 통합된 흐름으로 대체합니다.

        전처리 단계 (순서대로):
        ================================
        0. Feature 이름 정규화 (특수 JSON 문자 제거)
        1. X와 y에서 무한값 제거
        2. 남은 무한값을 NaN으로 대체 (안전장치)
        3. y 레이블에서 무한값이 있는 행 제거 (필수)
        4. 로그 변환 (극단값 압축)
        5. NaN이 50% 초과인 컬럼 제거
        6. y 레이블에 NaN이 있는 행 제거
        7. (선택) Winsorization
        8. (선택) Feature 선택

        Parameters:
        -----------
        X : pd.DataFrame
            Feature 행렬
        y : pd.DataFrame
            회귀 타겟 (컬럼이 있는 DataFrame이어야 함)
        y_cls : Optional[pd.DataFrame]
            분류 타겟 (선택, regressor.py용)
        config : Optional[Dict]
            설정 딕셔너리 (winsorization, feature 선택용)
        logger : Optional
            진행 메시지용 Logger

        Returns:
        --------
        X_clean : pd.DataFrame
            전처리된 Feature
        y_clean : pd.DataFrame
            전처리된 회귀 타겟
        y_cls_clean : Optional[pd.DataFrame]
            전처리된 분류 타겟 (제공된 경우)
        dropped_cols : List[str]
            삭제된 컬럼 이름 리스트

        사용 예시:
        --------
        >>> X, y, y_cls, dropped = DataProcessor.preprocess_training_data(
        >>>     X_train, y_train, y_train_cls, config, logger
        >>> )

        이것이 중요한 이유:
        -----------------
        통합 전:
        - regressor.py: 200줄에 걸쳐 산재된 10개 전처리 단계
        - ml_backtest.py: 150줄에 걸쳐 산재된 7개 전처리 단계
        - 다른 순서, 다른 로직, 다른 버그

        통합 후:
        - 하나의 메서드, 하나의 로직, 하나의 진실 공급원
        - 동일한 전처리 보장
        - 백테스트가 학습을 정확하게 검증
        """
        if logger:
            logger.info("=" * 80)
            logger.info("🔧 UNIFIED PREPROCESSING (DataProcessor.preprocess_training_data)")
            logger.info("=" * 80)

        rows_before = len(X)
        cols_before = len(X.columns)

        # 1단계: 데이터 품질 검사 (선택적 레포트)
        DataProcessor._preprocess_quality_checks(X, y, config, logger)

        # 2단계: Feature 이름 정규화 및 비feature 컬럼 제거
        X = DataProcessor._preprocess_normalize_columns(X, logger)

        # 3단계: 무한값 제거 및 타겟 정리 (Steps 1-3)
        X_clean, y_clean, y_cls_clean = DataProcessor._preprocess_remove_infinities(
            X, y, y_cls, rows_before, logger
        )

        # 4단계: Feature 변환 및 NaN 필터링 (Steps 4-6)
        X_clean, y_clean, y_cls_clean = DataProcessor._preprocess_transform_and_filter(
            X_clean, y_clean, y_cls_clean, y.columns, logger
        )

        # 5단계: 선택적 처리 및 Feature 선택 (Steps 7-8)
        X_clean, selected_features = DataProcessor._preprocess_optional_steps(
            X_clean, y_clean, config, logger
        )

        # ===== 안전 검사: y_clean이 None이 아닌지 확인 =====
        if y_clean is None:
            if logger:
                logger.error("❌ CRITICAL: y_clean is None after preprocessing!")
                logger.error(f"   X_clean type: {type(X_clean)}, shape: {X_clean.shape if hasattr(X_clean, 'shape') else 'N/A'}")
                logger.error(f"   y_cls_clean type: {type(y_cls_clean)}")
            raise ValueError("preprocess_training_data: y_clean is None after preprocessing pipeline")

        # ===== 최종 요약 =====
        rows_after = len(X_clean)
        cols_after = len(X_clean.columns)

        if logger:
            logger.info("=" * 80)
            logger.info("✅ PREPROCESSING COMPLETE")
            if rows_before > 0:
                logger.info(f"   Rows: {rows_before} → {rows_after} ({rows_after/rows_before*100:.1f}% retained)")
            else:
                logger.info(f"   Rows: {rows_before} → {rows_after}")
            logger.info(f"   Cols: {cols_before} → {cols_after} ({cols_after/cols_before*100:.1f}% retained)" if cols_before > 0 else f"   Cols: {cols_before} → {cols_after}")
            logger.info(f"   Remaining NaN: {X_clean.isna().sum().sum()}")
            logger.info(f"   y_clean type: {type(y_clean).__name__}, shape: {y_clean.shape}")

            # 데이터 품질 설정 요약
            if config:
                dq_config = config.get('DATA_QUALITY', {})
                if dq_config:
                    logger.info(f"   Data Quality Report: {'Enabled' if dq_config.get('GENERATE_REPORT', 'N') == 'Y' else 'Disabled'}")
                    logger.info(f"   A/B Cleaning Test: {'Enabled' if dq_config.get('VALIDATE_CLEANING_EFFECT', 'N') == 'Y' else 'Disabled'}")

            logger.info("=" * 80)

        return X_clean, y_clean, y_cls_clean, selected_features

    @staticmethod
    def _preprocess_quality_checks(X: pd.DataFrame, y: pd.DataFrame,
                                    config: Optional[Dict], logger) -> None:
        """Phase 1: Data Quality Report 및 A/B Cleaning Test (optional)"""
        # ===== Data Quality Report (BEFORE preprocessing) =====
        # Generate report to capture original data state
        data_quality_config = config.get('DATA_QUALITY', {}) if config else {}
        generate_report = data_quality_config.get('GENERATE_REPORT', 'N') == 'Y'

        if generate_report:
            if logger:
                logger.info("Generating Data Quality Report (before preprocessing)...")

            y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
            report = DataQualityReport(X, y_series, config, logger)
            report.generate()

            # Save report if path is configured
            report_path = data_quality_config.get('REPORT_OUTPUT_PATH')
            if report_path:
                try:
                    report.save(report_path)
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to save data quality report: {e}")

        # ===== A/B Test for Cleaning Effect (optional, config-controlled) =====
        validate_cleaning = data_quality_config.get('VALIDATE_CLEANING_EFFECT', 'N') == 'Y'

        if validate_cleaning:
            if logger:
                logger.info("Running A/B test for cleaning effect validation...")

            try:
                y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
                validator = DataCleaningValidator(X, y_series, config, logger)
                validation_results = validator.validate_cleaning_effect()

                # Log recommendation
                rec = validation_results.get('recommendation', {})
                if logger:
                    logger.info(f"A/B Test Result: {rec.get('preferred', 'N/A').upper()} preferred")
                    logger.info(f"   Reason: {rec.get('reason', 'N/A')}")
            except Exception as e:
                if logger:
                    logger.warning(f"A/B test validation failed: {e}")

    @staticmethod
    def _preprocess_normalize_columns(X: pd.DataFrame, logger) -> pd.DataFrame:
        """Phase 2: Feature name 정규화, metadata 컬럼 제거, non-numeric 컬럼 제거"""
        # ===== Step 0: Normalize feature names (Critical for model training) =====
        # Remove special JSON characters from feature names to prevent errors
        # in XGBoost, LightGBM, and CatBoost
        if logger:
            logger.info("Step 0/8: Normalizing feature names...")
        X = DataProcessor.normalize_feature_names(X, logger=logger)

        # ===== Step 0.5: Remove metadata columns =====
        # 🚨 CRITICAL: Remove non-feature columns that are used for grouping/filtering
        # These should never be used as ML features
        METADATA_COLUMNS = [
            'sector_category',      # Sector categorization metadata
            'sector_original',      # Original sector before categorization
            'sector',               # Sector information (if still present)
            'symbol',               # Stock ticker
            'date',                 # Date information
            'rebalance_date',       # Rebalancing date
            'report_date',          # Financial report date
            'filingDate',           # SEC filing date
        ]

        metadata_found = [col for col in METADATA_COLUMNS if col in X.columns]
        if metadata_found:
            if logger:
                logger.warning(f"⚠️  Found {len(metadata_found)} metadata columns (removing before training)")
                logger.warning(f"   Columns: {metadata_found}")
            X = X.drop(columns=metadata_found, errors='ignore')
            if logger:
                logger.info(f"   ✅ Removed metadata columns. Remaining: {len(X.columns)} columns")

        # ===== DIAGNOSTIC: Check for remaining object/string columns =====
        # 🚨 CRITICAL: ML models require numeric input only
        # Object/string columns cause errors in np.isinf(), np.abs(), log transform
        if logger:
            # Memory-efficient check: avoid select_dtypes which causes 10GB memory spike
            object_cols = [col for col in X.columns
                          if not pd.api.types.is_numeric_dtype(X[col])]
            if len(object_cols) > 0:
                logger.warning(f"⚠️  Found {len(object_cols)} non-numeric columns (object/string type)")
                logger.warning(f"   Columns: {list(object_cols[:10])}")
                if len(object_cols) > 10:
                    logger.warning(f"   ... and {len(object_cols) - 10} more")
                # Show sample values to diagnose root cause
                for col in object_cols[:3]:
                    sample_vals = X[col].dropna().unique()[:5]
                    logger.warning(f"   '{col}' sample values: {sample_vals}")

                # 🔧 AUTO-FIX: Remove remaining object columns
                logger.warning(f"   🔧 Removing {len(object_cols)} object columns...")
                # Memory-efficient drop: avoid select_dtypes
                X = X.drop(columns=object_cols)
                logger.warning(f"   ✅ Removed. Remaining columns: {len(X.columns)}")
            else:
                logger.info("✅ All columns are numeric")

        return X

    @staticmethod
    def _preprocess_remove_infinities(X: pd.DataFrame, y: pd.DataFrame,
                                       y_cls: Optional[pd.DataFrame],
                                       rows_before: int,
                                       logger) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
        """Phase 3: 무한값 제거 및 타겟 변수 정리 (Steps 1-3)"""
        # ===== Step 1: Remove infinite values from X and y =====
        if logger:
            logger.info("Step 1/8: Removing infinite values from X and y...")

        X_clean, y_clean = DataProcessor.remove_infinite_values(X, y.iloc[:, 0])

        rows_removed_inf = rows_before - len(X_clean)
        if rows_removed_inf > 0 and logger:
            logger.warning(f"   ⚠️  Removed {rows_removed_inf} rows with infinite values in X or y")

        # ===== Step 2: Replace remaining infinite with NaN (safety) =====
        if logger:
            logger.info("Step 2/8: Replacing remaining infinite with NaN...")

        X_clean, y_clean = DataProcessor.replace_infinite_with_nan(X_clean, y_clean)

        # ===== Step 3: Remove rows with infinite in y_cls (if provided) =====
        y_cls_clean = None
        if y_cls is not None:
            if logger:
                logger.info("Step 3/8: Removing infinite values from y_cls...")

            y_cls_aligned = y_cls.loc[X_clean.index]
            X_clean, y_cls_series = DataProcessor.remove_infinite_values(
                X_clean, y_cls_aligned.iloc[:, 0]
            )

            rows_removed_y_cls = len(y_clean) - len(X_clean)
            if rows_removed_y_cls > 0 and logger:
                logger.warning(f"   ⚠️  Removed {rows_removed_y_cls} rows with infinite in y_cls")

            # Align y with final index
            y_clean = y_clean.loc[X_clean.index]

            # Convert y_cls back to DataFrame
            y_cls_clean = DataProcessor.create_clean_dataframe(
                y_cls_series, y_cls.columns, X_clean.index
            )

        return X_clean, y_clean, y_cls_clean

    @staticmethod
    def _preprocess_transform_and_filter(X_clean: pd.DataFrame, y_clean,
                                          y_cls_clean: Optional[pd.DataFrame],
                                          y_columns, logger) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Phase 4: Log 변환, 고NaN 컬럼 제거, NaN 라벨 행 제거 (Steps 4-6)"""
        # ===== Step 4: Log transformation =====
        if logger:
            logger.info("Step 4/8: Applying log transformation...")
            # Only check numeric columns for max value
            # Memory-efficient: avoid select_dtypes which causes 10GB memory spike
            numeric_cols = [col for col in X_clean.columns
                           if pd.api.types.is_numeric_dtype(X_clean[col])]
            if len(numeric_cols) > 0:
                max_before = np.nanmax(np.abs(X_clean[numeric_cols].values))
                logger.info(f"   Before: max abs value = {max_before:.2e}")
            else:
                logger.info("   No numeric columns to transform")

        X_clean = DataProcessor.log_transform_features(X_clean)

        if logger:
            # Only check numeric columns for max value
            # Memory-efficient: avoid select_dtypes which causes 10GB memory spike
            numeric_cols = [col for col in X_clean.columns
                           if pd.api.types.is_numeric_dtype(X_clean[col])]
            if len(numeric_cols) > 0:
                max_after = np.nanmax(np.abs(X_clean[numeric_cols].values))
                logger.info(f"   After: max abs value = {max_after:.2f}")

        # ===== Step 5: Remove columns with >50% NaN =====
        if logger:
            logger.info("Step 5/8: Removing columns with >50% NaN...")

        nan_threshold = 0.5
        nan_ratio_per_col = X_clean.isna().sum() / len(X_clean)
        high_nan_cols = nan_ratio_per_col[nan_ratio_per_col > nan_threshold].index.tolist()

        if high_nan_cols:
            if logger:
                logger.warning(f"   ⚠️  Removing {len(high_nan_cols)} columns with >{nan_threshold*100}% NaN")
                for col in high_nan_cols[:5]:
                    logger.warning(f"      - {col}: {nan_ratio_per_col[col]*100:.1f}% NaN")
                if len(high_nan_cols) > 5:
                    logger.warning(f"      ... and {len(high_nan_cols)-5} more columns")

            X_clean = X_clean.drop(columns=high_nan_cols)

        # ===== Step 6: Remove rows with NaN in y labels =====
        if logger:
            logger.info("Step 6/8: Removing rows with NaN in y labels...")

        # Initialize y_df to handle both Series and DataFrame cases safely
        y_df = None

        if isinstance(y_clean, pd.Series):
            nan_mask_y = y_clean.isna()
        elif isinstance(y_clean, pd.DataFrame):
            # Reconstruct y as DataFrame
            y_df = DataProcessor.create_clean_dataframe(y_clean, y_columns, X_clean.index)
            nan_mask_y = y_df.isna().any(axis=1)
        else:
            nan_mask_y = False

        nan_mask_y_cls = False
        if y_cls_clean is not None:
            nan_mask_y_cls = y_cls_clean.isna().any(axis=1)

        nan_mask_labels = nan_mask_y | nan_mask_y_cls

        if isinstance(nan_mask_labels, pd.Series) and nan_mask_labels.sum() > 0:
            if logger:
                logger.warning(f"   ⚠️  Removing {nan_mask_labels.sum()} rows with NaN labels")

            X_clean = X_clean[~nan_mask_labels]

            if isinstance(y_clean, pd.Series):
                y_clean = y_clean[~nan_mask_labels]
            elif y_df is not None:
                y_df = y_df[~nan_mask_labels]

            if y_cls_clean is not None:
                y_cls_clean = y_cls_clean[~nan_mask_labels]

        # Reconstruct y as DataFrame
        if isinstance(y_clean, pd.Series):
            y_clean = DataProcessor.create_clean_dataframe(y_clean, y_columns, X_clean.index)
        elif y_df is not None:
            y_clean = y_df
        # else: y_clean stays as-is (should not happen in normal flow)

        return X_clean, y_clean, y_cls_clean

    @staticmethod
    def _preprocess_optional_steps(X_clean: pd.DataFrame, y_clean: pd.DataFrame,
                                    config: Optional[Dict],
                                    logger) -> Tuple[pd.DataFrame, List[str]]:
        """Phase 5: Winsorization 및 Feature Selection (Steps 7-8, optional)"""
        # ===== Step 7: Winsorization (optional, config-driven) =====
        use_winsorization = False
        if config:
            features_config = config.get('FEATURES', {})
            use_winsorization = features_config.get('USE_WINSORIZATION', 'N') == 'Y'

        if use_winsorization:
            if logger:
                logger.info("Step 7/8: Applying winsorization (percentiles: 1-99%)...")

            X_clean = DataProcessor.winsorize_features(
                X_clean,
                lower_percentile=0.01,
                upper_percentile=0.99,
                enabled=True
            )

            if logger:
                logger.info("   ✅ Winsorization applied")
        else:
            if logger:
                logger.info("Step 7/8: Skipping winsorization (disabled in config)")

        # ===== Step 8: Feature selection (optional, config-driven) =====
        selected_features = None
        use_feature_selection = False
        target_features = 1000

        if config:
            features_config = config.get('FEATURES', {})
            use_feature_selection = features_config.get('USE_FEATURE_SELECTION', 'N') == 'Y'
            target_features = int(features_config.get('TARGET_FEATURES', 1000))

        if use_feature_selection:
            if logger:
                logger.info(f"Step 8/8: Selecting top {target_features} features...")

            y_for_selection = y_clean.iloc[:, 0] if isinstance(y_clean, pd.DataFrame) else y_clean

            X_clean, selected_features = DataProcessor.select_features_by_importance(
                X_clean,
                y_for_selection,
                n_features=target_features,
                task='regression',
                enabled=True
            )

            if logger:
                logger.info(f"   ✅ Selected {len(selected_features)} features")
        else:
            if logger:
                logger.info("Step 8/8: Skipping feature selection (disabled in config)")
            selected_features = X_clean.columns.tolist()

        return X_clean, selected_features


    @staticmethod
    def clip_extreme_values(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        threshold: float = 1e10,
        enabled: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], int]:
        """
        Clip extreme values for XGBoost/LightGBM compatibility.

        XGBoost errors with "Input data contains inf or a value too large"
        when values exceed ~1e10, even if not strictly infinite.

        This method provides unified extreme value handling across
        regressor.py and ml_backtest.py.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : Optional[pd.Series]
            Target series (unchanged, only X is clipped)
        threshold : float
            Maximum absolute value allowed (default: 1e10)
        enabled : bool
            Whether to apply clipping (default: True)

        Returns:
        --------
        X_clipped : pd.DataFrame
            Dataframe with extreme values clipped to [-threshold, threshold]
        y : Optional[pd.Series]
            Target series (unchanged)
        n_clipped : int
            Number of values that were clipped

        Example:
        --------
        >>> X_safe, y, n = DataProcessor.clip_extreme_values(X, y, threshold=1e10)
        >>> if n > 0:
        >>>     logging.warning(f"Clipped {n} extreme values")

        Note:
        -----
        - Tree-based models (XGBoost/LightGBM) are robust to outliers
        - BUT they cannot handle values > 1e10 (XGBoost internal limit)
        - Clipping preserves information (direction + relative magnitude)
        - Removing rows would lose valuable data
        """
        if not enabled:
            return X, y, 0

        # Count extreme values before clipping
        if isinstance(X, pd.DataFrame):
            extreme_mask = (X.abs() > threshold)
            n_extreme = extreme_mask.sum().sum()
        else:
            extreme_mask = (np.abs(X) > threshold)
            n_extreme = int(extreme_mask.sum())

        if n_extreme > 0:
            X_clipped = X.clip(-threshold, threshold)
            return X_clipped, y, n_extreme
        else:
            return X, y, 0

    # ========================================================================
    # NaN Handling
    # ========================================================================

    @staticmethod
    def handle_nan(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        method: str = 'fillna',
        fill_value: Any = 0
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Unified NaN handling.

        Provides consistent NaN handling across regressor.py and ml_backtest.py.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : Optional[pd.Series]
            Target series
        method : str
            'fillna' (default), 'drop', or 'forward_fill'
        fill_value : Any
            Value to fill NaN with (default: 0)

        Returns:
        --------
        X_clean : pd.DataFrame
            Dataframe with NaN handled
        y_clean : Optional[pd.Series]
            Target with NaN handled (if provided)

        Example:
        --------
        X, y = DataProcessor.handle_nan(X, y, method='fillna', fill_value=0)
        """
        if method == 'fillna':
            X_clean = X.fillna(fill_value)
            y_clean = y.fillna(fill_value) if y is not None else None
        elif method == 'drop':
            nan_mask = X.isna().any(axis=1)
            X_clean = X[~nan_mask]
            y_clean = y[~nan_mask] if y is not None else None
        elif method == 'forward_fill':
            X_clean = X.fillna(method='ffill')
            y_clean = y.fillna(method='ffill') if y is not None else None
        else:
            raise ValueError(f"Unknown method: {method}. Use 'fillna', 'drop', or 'forward_fill'")

        return X_clean, y_clean

    # ========================================================================
    # Outlier Handling
    # ========================================================================

    def clip_outliers(
        self,
        df: pd.DataFrame,
        lower_percentile: float = 0.02,
        upper_percentile: float = 0.98,
        save_bounds: bool = True,
        apply_saved_bounds: bool = False
    ) -> pd.DataFrame:
        """
        Clip outliers using quantile-based bounds.

        This method clips extreme values to prevent model distortion.

        Critical: This was present in regressor.py but MISSING in ml_backtest.py,
        causing inconsistent preprocessing! Now unified.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        lower_percentile : float
            Lower quantile (default: 0.02 = 2nd percentile)
        upper_percentile : float
            Upper quantile (default: 0.98 = 98th percentile)
        save_bounds : bool
            If True, save bounds for later use on test data
        apply_saved_bounds : bool
            If True, use previously saved bounds (for test data)

        Returns:
        --------
        pd.DataFrame
            Dataframe with clipped values

        Example:
        --------
        # Training: clip and save bounds
        df_train_clipped = processor.clip_outliers(df_train, save_bounds=True)

        # Testing: apply saved bounds
        df_test_clipped = processor.clip_outliers(df_test, apply_saved_bounds=True)
        """
        feature_cols = DataSchema.get_feature_cols(df)
        df_clipped = df.copy()

        # Exclude non-numeric columns (like 'sector')
        numeric_features = [
            col for col in feature_cols
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        if apply_saved_bounds:
            # Use saved bounds (for test data)
            if not self.clip_bounds:
                self.logger.warning("No saved clip bounds found! Skipping clipping.")
                return df_clipped

            for col, (lower, upper) in self.clip_bounds.items():
                if col in df_clipped.columns:
                    df_clipped[col] = df_clipped[col].clip(lower, upper)

            self.logger.info(f"   Applied saved clip bounds to {len(self.clip_bounds)} columns")

        else:
            # Calculate new bounds (for training data)
            for col in numeric_features:
                lower, upper = df[col].quantile([lower_percentile, upper_percentile])
                df_clipped[col] = df[col].clip(lower, upper)

                if save_bounds:
                    self.clip_bounds[col] = (float(lower), float(upper))

            self.logger.info(f"   Clipped {len(numeric_features)} columns (percentiles: {lower_percentile}, {upper_percentile})")

        return df_clipped

    # ========================================================================
    # Feature Scaling
    # ========================================================================

    @staticmethod
    def scale_features(
        X: pd.DataFrame,
        scaler_type: str = 'robust',
        fitted_scaler: Optional[Any] = None
    ) -> Tuple[np.ndarray, Any]:
        """
        Unified feature scaling.

        Provides consistent scaling across regressor.py and ml_backtest.py.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        scaler_type : str
            'robust' (default) or 'standard'
        fitted_scaler : Optional[Any]
            Pre-fitted scaler for transform mode
            If None, creates new scaler and fits (train mode)

        Returns:
        --------
        X_scaled : np.ndarray
            Scaled features
        scaler : Any
            Fitted scaler object (for use in transform mode)

        Example:
        --------
        # Train mode (fit new scaler)
        X_train_scaled, scaler = DataProcessor.scale_features(X_train, 'robust')

        # Test mode (use fitted scaler)
        X_test_scaled, _ = DataProcessor.scale_features(X_test, fitted_scaler=scaler)
        """
        if fitted_scaler is not None:
            # Transform mode (use existing scaler)
            X_scaled = fitted_scaler.transform(X)
            return X_scaled, fitted_scaler
        else:
            # Fit mode (create and fit new scaler)
            if scaler_type == 'robust':
                scaler = RobustScaler()
            elif scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler: {scaler_type}. Use 'robust' or 'standard'")

            X_scaled = scaler.fit_transform(X)
            logging.info(f"   Fitted {scaler_type.upper()} scaler on {X.shape[1]} features")

            return X_scaled, scaler

    # Backward compatibility methods
    def fit_scaler(
        self,
        X: pd.DataFrame,
        scaler_type: str = 'robust'
    ) -> np.ndarray:
        """
        Fit scaler and transform features (backward compatibility).

        Use DataProcessor.scale_features() for new code.
        """
        X_scaled, scaler = self.scale_features(X, scaler_type)
        self.scaler = scaler
        return X_scaled

    def transform_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler (backward compatibility).

        Use DataProcessor.scale_features(X, fitted_scaler=scaler) for new code.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted! Call fit_scaler() first.")

        X_scaled, _ = self.scale_features(X, fitted_scaler=self.scaler)
        return X_scaled

    @staticmethod
    def create_binary_target(
        y: Union[pd.Series, pd.DataFrame],
        mode: str = "positive_screen",
        threshold: float = 0.0,
        config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None,
        analyze: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Convert regression target to binary classification target.

        This method ensures consistent binary classification across both
        regressor.py and ml_backtest.py.

        Args:
            y: Regression target values (price_dev or price_dev_subavg)
            mode: Classification mode
                  - "negative_screen": Identify BAD stocks (extreme losses)
                    → 1 = BAD (should avoid), 0 = OK (safe to consider)
                  - "positive_screen": Identify GOOD stocks (gains)
                    → 1 = GOOD (likely gain), 0 = LOSS (likely loss)
            threshold: Threshold value
                      - For negative_screen: loss threshold (e.g., -0.3 = -30% loss)
                      - For positive_screen: gain threshold (e.g., 0.0 = breakeven)
            config: Optional config dict to read mode/threshold from
            logger: Optional logger for distribution analysis
            analyze: If True, print distribution analysis for multiple thresholds

        Returns:
            Binary target (1d Series)

        Example (Negative Screening):
            >>> y = pd.Series([0.05, -0.10, -0.35, 0.02, -0.45])
            >>> binary = DataProcessor.create_binary_target(
            ...     y, mode="negative_screen", threshold=-0.3
            ... )
            >>> # Result: [0, 0, 1, 0, 1]  (1 = BAD: -35%, -45%)
            >>> # Meaning: Remove stocks with >30% loss

        Example (Positive Screening):
            >>> y = pd.Series([0.05, -0.01, -0.03, 0.02])
            >>> binary = DataProcessor.create_binary_target(
            ...     y, mode="positive_screen", threshold=0.0
            ... )
            >>> # Result: [1, 0, 0, 1]  (1 = GOOD: +5%, +2%)

        Note:
            Negative screening is more robust:
            - Easier to identify "obvious losers" (bankruptcy, extreme losses)
            - Conservative strategy: avoid bad > find good
            - Uses more training data (remove bottom 5~10%, keep rest 90~95%)
        """
        if logger is None:
            logger = logging.getLogger('DataProcessor')

        # Extract Series from DataFrame
        if isinstance(y, pd.DataFrame):
            y_values = y.iloc[:, 0]
        else:
            y_values = y

        # Read mode and threshold from config if provided
        if config is not None:
            ml_config = config.get('ML', {})
            mode = ml_config.get('CLASSIFIER_MODE', mode)

            if mode == "negative_screen":
                neg_config = ml_config.get('NEGATIVE_SCREEN', {})
                threshold = neg_config.get('LOSS_THRESHOLD', threshold)

                # Analyze threshold candidates if requested
                if analyze and neg_config.get('ANALYZE_THRESHOLDS', False):
                    candidates = neg_config.get('THRESHOLD_CANDIDATES', [-0.2, -0.3, -0.4, -0.5])
                    DataProcessor._analyze_threshold_candidates(y_values, candidates, logger)

        # Create binary target based on mode
        if mode == "negative_screen":
            # 1 = BAD (extreme loss), 0 = OK
            binary = (y_values < threshold).astype(int)

            bad_count = binary.sum()
            bad_pct = bad_count / len(binary) * 100
            avg_bad_loss = y_values[binary == 1].mean() if bad_count > 0 else 0
            avg_ok_return = y_values[binary == 0].mean() if len(binary) - bad_count > 0 else 0

            logger.info("")
            logger.info("="*80)
            logger.info("📊 NEGATIVE SCREENING: Binary Target Distribution")
            logger.info("="*80)
            logger.info(f"   Mode: {mode}")
            logger.info(f"   Threshold: {threshold:.2f} ({threshold*100:.0f}% loss)")
            logger.info(f"   BAD stocks (label=1): {bad_count:,} ({bad_pct:.1f}%)")
            logger.info(f"   OK stocks (label=0): {len(binary) - bad_count:,} ({100-bad_pct:.1f}%)")
            logger.info(f"   Avg loss of BAD stocks: {avg_bad_loss:.4f} ({avg_bad_loss*100:.1f}%)")
            logger.info(f"   Avg return of OK stocks: {avg_ok_return:.4f} ({avg_ok_return*100:.1f}%)")
            logger.info("="*80)
            logger.info("")

        else:  # positive_screen
            # 1 = GOOD (gain), 0 = LOSS
            binary = (y_values > threshold).astype(int)

            good_count = binary.sum()
            good_pct = good_count / len(binary) * 100

            logger.info(f"   Positive screening: {good_count:,} GOOD ({good_pct:.1f}%), "
                       f"{len(binary) - good_count:,} LOSS ({100-good_pct:.1f}%)")

        return binary

    @staticmethod
    def _analyze_threshold_candidates(
        y: pd.Series,
        candidates: List[float],
        logger: logging.Logger
    ) -> None:
        """
        Analyze distribution for multiple threshold candidates.

        Helps find the optimal threshold by showing:
        - How many stocks are labeled as BAD
        - Average loss of BAD stocks
        - Average return of OK stocks
        """
        logger.info("")
        logger.info("="*80)
        logger.info("🔍 THRESHOLD ANALYSIS: Finding Optimal Negative Screening Threshold")
        logger.info("="*80)
        logger.info(f"   Total stocks: {len(y):,}")
        logger.info(f"   Avg return: {y.mean():.4f} ({y.mean()*100:.1f}%)")
        logger.info(f"   Std dev: {y.std():.4f}")
        logger.info("")
        logger.info("   Candidate thresholds:")
        logger.info("   " + "-"*76)
        logger.info(f"   {'Threshold':<12} {'BAD Count':<12} {'BAD %':<10} {'Avg BAD Loss':<15} {'Avg OK Return':<15}")
        logger.info("   " + "-"*76)

        for threshold in sorted(candidates):
            is_bad = y < threshold
            bad_count = is_bad.sum()
            bad_pct = bad_count / len(y) * 100
            avg_bad_loss = y[is_bad].mean() if bad_count > 0 else 0
            avg_ok_return = y[~is_bad].mean() if len(y) - bad_count > 0 else 0

            logger.info(f"   {threshold:>10.2f}   {bad_count:>10,}   {bad_pct:>8.1f}%   "
                       f"{avg_bad_loss:>13.4f}   {avg_ok_return:>13.4f}")

        logger.info("   " + "-"*76)
        logger.info("")
        logger.info("   💡 Recommendation:")
        logger.info("      - Lower threshold (e.g., -0.5): Very conservative (remove only extreme cases)")
        logger.info("      - Higher threshold (e.g., -0.2): More aggressive (remove more risky stocks)")
        logger.info("      - Sweet spot: Usually -0.3 to -0.4 (5~10% of stocks)")
        logger.info("="*80)
        logger.info("")

    @staticmethod
    def fit_outlier_clipper(
        df: pd.DataFrame,
        lower_percentile: float = 0.02,
        upper_percentile: float = 0.98,
        exclude_cols: Optional[List[str]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute clipping bounds for outlier removal based on percentiles.

        This method ensures consistent outlier handling across both
        regressor.py and ml_backtest.py. Extreme values can negatively
        impact model training and predictions.

        Args:
            df: Training data to compute bounds from
            lower_percentile: Lower percentile (default 0.02 = 2nd percentile)
            upper_percentile: Upper percentile (default 0.98 = 98th percentile)
            exclude_cols: Columns to skip (e.g., ['sector'])

        Returns:
            Dictionary mapping column names to (lower, upper) bounds

        Example:
            >>> # Fit on training data
            >>> clip_bounds = DataProcessor.fit_outlier_clipper(X_train)
            >>> # Apply to both train and test
            >>> X_train_clipped = DataProcessor.apply_outlier_clipper(X_train, clip_bounds)
            >>> X_test_clipped = DataProcessor.apply_outlier_clipper(X_test, clip_bounds)

        Note:
            - Default 2-98 percentile removes top/bottom 2% extreme values
            - For targets, use 1-97 percentile (more aggressive)
            - Saves ~50 lines of duplicated clipping code
        """
        exclude_cols = exclude_cols or []
        clip_bounds = {}

        # Memory-efficient: avoid select_dtypes which causes memory spike on large data
        numeric_cols = [col for col in df.columns
                       if pd.api.types.is_numeric_dtype(df[col])]

        for col in numeric_cols:
            if col in exclude_cols:
                continue

            lower = df[col].quantile(lower_percentile)
            upper = df[col].quantile(upper_percentile)
            clip_bounds[col] = (float(lower), float(upper))

        logging.info(f"Computed clip bounds for {len(clip_bounds)} features "
                    f"(percentiles: {lower_percentile*100:.0f}-{upper_percentile*100:.0f})")

        return clip_bounds

    @staticmethod
    def apply_outlier_clipper(
        df: pd.DataFrame,
        clip_bounds: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Apply pre-computed clipping bounds to remove outliers.

        Args:
            df: Data to clip (train, test, or new data)
            clip_bounds: Dictionary from fit_outlier_clipper()

        Returns:
            Clipped DataFrame (copy, original unchanged)

        Example:
            >>> # First fit on training data
            >>> clip_bounds = DataProcessor.fit_outlier_clipper(X_train)
            >>> # Then apply to test data
            >>> X_test_clipped = DataProcessor.apply_outlier_clipper(X_test, clip_bounds)

        Note:
            - Only clips columns present in both df and clip_bounds
            - Silently skips columns not in df (for flexibility)
        """
        df = df.copy()

        clipped_count = 0
        for col, (lower, upper) in clip_bounds.items():
            if col in df.columns:
                original_values = df[col].copy()
                df[col] = df[col].clip(lower, upper)

                # Count how many values were clipped
                clipped = ((original_values < lower) | (original_values > upper)).sum()
                if clipped > 0:
                    clipped_count += 1

        if clipped_count > 0:
            logging.info(f"Applied outlier clipping to {clipped_count} columns")

        return df

    @staticmethod
    def winsorize_features(
        df: pd.DataFrame,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
        exclude_cols: Optional[List[str]] = None,
        enabled: bool = True
    ) -> pd.DataFrame:
        """
        Apply Winsorization to handle outliers by capping at percentiles.

        Winsorization replaces extreme values with percentile values rather than
        removing them. This is gentler than clipping and preserves information.

        DIFFERENCE from clipping:
        - Clipping: [1, 2, 5, 10, 50, 100, 1000] → [2, 2, 5, 10, 50, 100, 100]
          (cuts off extremes, hard boundary)
        - Winsorization: [1, 2, 5, 10, 50, 100, 1000] → [2, 2, 5, 10, 50, 100, 100]
          (replaces with percentile values, soft cap)

        Args:
            df: Input DataFrame
            lower_percentile: Lower percentile (default 0.01 = 1%)
            upper_percentile: Upper percentile (default 0.99 = 99%)
            exclude_cols: Columns to skip (e.g., ['sector'])
            enabled: If False, returns df unchanged (for easy on/off toggle)

        Returns:
            Winsorized DataFrame

        Example:
            >>> # Enable Winsorization
            >>> df_clean = DataProcessor.winsorize_features(df, enabled=True)
            >>>
            >>> # Disable for comparison
            >>> df_raw = DataProcessor.winsorize_features(df, enabled=False)
            >>>
            >>> # More aggressive (0.5-99.5 percentile)
            >>> df_gentle = DataProcessor.winsorize_features(
            ...     df, lower_percentile=0.005, upper_percentile=0.995
            ... )

        Note:
            - enabled=False: Easy way to disable without changing code
            - Recommended for tree-based models: Try enabled=False first
            - Use Winsorization if raw data has too many extreme outliers
            - Default 1-99% is gentler than clipping's 2-98%
        """
        if not enabled:
            logging.info("Winsorization disabled (enabled=False)")
            return df.copy()

        exclude_cols = exclude_cols or []
        df = df.copy()

        # CRITICAL FIX: Remove duplicate columns first
        # Duplicate columns cause df[col] to return DataFrame instead of Series
        # which makes quantile() return Series instead of scalar → ambiguous boolean error
        if df.columns.duplicated().any():
            duplicated_cols = df.columns[df.columns.duplicated()].tolist()
            logging.warning(f"⚠️  Found {len(duplicated_cols)} duplicate columns, keeping first occurrence")
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Memory-efficient: avoid select_dtypes which causes memory spike on large data
        numeric_cols = [col for col in df.columns
                       if pd.api.types.is_numeric_dtype(df[col])]
        winsorized_count = 0

        for col in numeric_cols:
            if col in exclude_cols:
                continue

            # Get column data (should be Series after duplicate removal)
            col_data = df[col]

            # Skip if not a Series (shouldn't happen after duplicate removal, but defensive)
            if not isinstance(col_data, pd.Series):
                logging.warning(f"⚠️  Column '{col}' is not a Series, skipping winsorization")
                continue

            # Get percentile values (should be scalar)
            lower_val = col_data.quantile(lower_percentile)
            upper_val = col_data.quantile(upper_percentile)

            # Skip if quantile values are NaN (all NaN column) or not scalar
            # Use np.isscalar for safe scalar check
            if not np.isscalar(lower_val) or not np.isscalar(upper_val):
                continue
            if pd.isna(lower_val) or pd.isna(upper_val):
                continue

            # Count how many will be winsorized
            lower_mask = col_data < lower_val
            upper_mask = col_data > upper_val

            # Explicit boolean conversion for safety (handles edge cases)
            has_lower_outliers = bool(lower_mask.any())
            has_upper_outliers = bool(upper_mask.any())

            if has_lower_outliers or has_upper_outliers:
                # Replace extreme values with percentile values
                df.loc[lower_mask, col] = lower_val
                df.loc[upper_mask, col] = upper_val
                winsorized_count += 1

        if winsorized_count > 0:
            logging.info(f"Winsorized {winsorized_count} columns "
                        f"(percentiles: {lower_percentile*100:.1f}-{upper_percentile*100:.1f}%)")

        return df

    @staticmethod
    def select_features_by_importance(
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        n_features: Optional[int] = None,
        top_pct: Optional[float] = None,
        task: str = 'regression',
        enabled: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using model-based importance (LightGBM).

        Uses LightGBM's built-in feature importance to identify the most
        predictive features. This is more robust than correlation-based
        methods and handles multicollinearity automatically.

        STRATEGY: Model-based selection (강력한 방법)
        - Trains a LightGBM model to compute feature importances
        - Selects top N features or top percentage
        - Handles interactions and non-linear relationships
        - No need for feature scaling

        Args:
            X: Feature dataframe
            y: Target variable
            n_features: Number of top features to select (e.g., 1000)
            top_pct: Percentage of top features (e.g., 0.3 for top 30%)
            task: 'regression' or 'classification'
            enabled: If False, returns all features unchanged
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (selected feature dataframe, list of selected feature names)

        Example:
            >>> # Reduce from 4,279 to 1,000 features
            >>> X_selected, selected_cols = DataProcessor.select_features_by_importance(
            ...     X, y, n_features=1000, task='regression'
            ... )
            >>> # Apply same selection to test set
            >>> X_test_selected = X_test[selected_cols]

        Note:
            - Default disabled (enabled=False) - user can test with/without
            - Must specify either n_features or top_pct (not both)
            - Selected feature names must be saved for test set application
            - Target ratio recommendation: samples/features >= 10:1
        """
        if not enabled:
            logging.info("Feature selection disabled (enabled=False)")
            return X.copy(), list(X.columns)

        # Validate parameters
        if n_features is None and top_pct is None:
            raise ValueError("Must specify either n_features or top_pct")
        if n_features is not None and top_pct is not None:
            raise ValueError("Cannot specify both n_features and top_pct")

        # Import LightGBM
        try:
            import lightgbm as lgb
        except ImportError:
            logging.warning("LightGBM not available, returning all features")
            return X.copy(), list(X.columns)

        # Calculate target number of features
        if top_pct is not None:
            n_features = max(1, int(len(X.columns) * top_pct))

        n_features = min(n_features, len(X.columns))

        logging.info(f"Selecting top {n_features} features from {len(X.columns)} "
                    f"using LightGBM importance ({task})")

        # Prepare data
        X_np = X.values
        y_np = y.values.ravel() if hasattr(y, 'values') else y

        # Train LightGBM model for feature importance
        if task == 'regression':
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                verbose=-1,
                force_col_wise=True  # Suppress warning
            )
        else:  # classification
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                verbose=-1,
                force_col_wise=True
            )

        # Fit model
        model.fit(X_np, y_np)

        # Get feature importances (gain-based)
        importances = model.feature_importances_

        # Select top N features by importance
        top_indices = np.argsort(importances)[-n_features:]
        selected_cols = [X.columns[i] for i in top_indices]

        # Create selected dataframe
        X_selected = X[selected_cols].copy()

        # Calculate reduction stats
        reduction_pct = (1 - n_features / len(X.columns)) * 100
        new_ratio = len(X) / n_features

        logging.info(f"✓ Feature selection complete: "
                    f"{len(X.columns)} → {n_features} features "
                    f"({reduction_pct:.1f}% reduction)")
        logging.info(f"  New sample/feature ratio: {new_ratio:.1f}:1 "
                    f"(recommended >= 10:1)")

        return X_selected, selected_cols

    @staticmethod
    def filter_by_liquidity(
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        top_pct: float = 0.50
    ) -> Tuple[pd.DataFrame, float]:
        """
        Filter stocks by volume*price liquidity metric to select liquid stocks.

        This method ensures consistent liquidity filtering across both
        regressor.py and ml_backtest.py. Trading illiquid stocks can lead
        to poor execution and slippage.

        Args:
            df: DataFrame with 'symbol' and 'volume_mul_price' columns
            threshold: Minimum liquidity threshold (if None, compute from top_pct)
            top_pct: Top percentage to keep (default 0.50 for top 50%)

        Returns:
            (filtered_df, threshold_used)

        Example:
            >>> # Training: compute threshold and filter
            >>> df_filtered, threshold = DataProcessor.filter_by_liquidity(train_df)
            >>> # Save threshold for test data
            >>> # Test: use saved threshold
            >>> test_filtered, _ = DataProcessor.filter_by_liquidity(test_df, threshold=threshold)

        Note:
            - Default keeps top 50% most liquid stocks
            - Prevents model from learning on illiquid stocks
            - Eliminates 4 duplications in regressor.py (~40 lines)
        """
        if 'symbol' not in df.columns or 'volume_mul_price' not in df.columns:
            logging.warning("Missing required columns for liquidity filtering, returning original df")
            return df, 0.0

        # Compute mean volume*price per symbol
        symbol_means = df.groupby('symbol')['volume_mul_price'].mean().reset_index()

        if threshold is None:
            # Compute threshold from top_pct
            threshold = symbol_means['volume_mul_price'].quantile(1.0 - top_pct)

        # Filter symbols above threshold
        top_symbols = symbol_means[symbol_means['volume_mul_price'] >= threshold]
        filtered_df = df[df['symbol'].isin(top_symbols['symbol'])].copy()

        logging.info(f"Liquidity filter: {len(symbol_means)} symbols → {len(top_symbols)} symbols "
                    f"(top {top_pct*100:.0f}%, threshold={threshold:.2e})")

        return filtered_df, float(threshold)

    @staticmethod
    def clip_large_values(
        df: pd.DataFrame,
        threshold: float = 1e9,
        replacement: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Replace extremely large values to prevent numerical overflow.

        This method ensures consistent large value handling. Very large
        numbers can cause overflow in XGBoost and other models.

        Args:
            df: Input DataFrame
            threshold: Values above this (absolute) are considered too large
            replacement: Value to use (default: np.finfo(np.float32).max)

        Returns:
            DataFrame with large values clipped

        Example:
            >>> df_clean = DataProcessor.clip_large_values(df, threshold=1e9)

        Note:
            - Prevents XGBoost overflow errors
            - Eliminates 3 duplicate implementations (~18 lines)
        """
        if replacement is None:
            replacement = np.finfo(np.float32).max

        df = df.copy()
        # Memory-efficient: avoid select_dtypes which causes memory spike on large data
        numeric_cols = [col for col in df.columns
                       if pd.api.types.is_numeric_dtype(df[col])]

        clipped_cols = 0
        for col in numeric_cols:
            mask = df[col].abs() > threshold
            if mask.any():
                df.loc[mask, col] = np.sign(df.loc[mask, col]) * replacement
                clipped_cols += 1

        if clipped_cols > 0:
            logging.info(f"Clipped large values in {clipped_cols} columns (threshold={threshold:.0e})")

        return df

    @staticmethod
    def drop_many_nan_row(
        df: pd.DataFrame,
        threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Drop rows with excessive NaN values.

        This method ensures consistent NaN row filtering. Rows with too
        many missing values provide little training signal.

        Args:
            df: Input DataFrame
            threshold: Drop rows with NaN ratio > threshold
                      (0.6 = drop if >60% NaN, keep if <=60% NaN)

        Returns:
            DataFrame with excessive-NaN rows removed

        Example:
            >>> # Drop rows with >40% NaN (keep rows with <=40% NaN)
            >>> df_clean = DataProcessor.drop_many_nan_row(df, threshold=0.6)
            >>> # Drop rows with >5% NaN (stricter)
            >>> df_clean = DataProcessor.drop_many_nan_row(df, threshold=0.95)

        Note:
            - threshold=0.6 means "drop if NaN% > 60%", keep if <=60%
            - Eliminates duplicate in regressor.py (~15 lines)
        """
        if df.empty:
            return df

        # Calculate NaN ratio per row
        nan_ratio = df.isna().sum(axis=1) / len(df.columns)

        # Keep rows where NaN ratio <= threshold
        mask = nan_ratio <= threshold
        df_clean = df[mask].copy()

        removed = len(df) - len(df_clean)
        if removed > 0:
            logging.info(f"Removed {removed} rows with >{threshold*100:.0f}% NaN "
                        f"({len(df_clean)}/{len(df)} rows remaining)")

        return df_clean

    # ========================================================================
    # Feature/Target Separation
    # ========================================================================

    def prepare_features_and_target(
        self,
        df: pd.DataFrame,
        target_type: str = 'regression',
        drop_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_type : str
            'regression', 'classification', or 'sector'
        drop_cols : Optional[List[str]]
            Additional columns to drop (e.g., previously identified sparse cols)

        Returns:
        --------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target series

        Raises:
        -------
        KeyError
            If target column not found

        Example:
        --------
        X, y = processor.prepare_features_and_target(df, target_type='regression')
        """
        # Get feature columns
        feature_cols = DataSchema.get_feature_cols(df)

        # Remove additional dropped columns
        if drop_cols:
            feature_cols = [col for col in feature_cols if col not in drop_cols]

        # Extract features
        X = df[feature_cols].copy()

        # Extract target
        target_col = DataSchema.get_target_column(target_type)

        if target_col not in df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in dataframe. "
                f"Available columns: {df.columns.tolist()}"
            )

        y = df[target_col].copy()

        self.logger.info(f"   Separated features ({X.shape[1]} cols) and target ('{target_col}')")

        return X, y

    # ========================================================================
    # Full Pipeline
    # ========================================================================

    def full_pipeline(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        sparse_row_threshold: float = 0.6,
        final_sparse_row_threshold: float = 0.95,
        missing_col_threshold: Optional[float] = None,
        same_value_col_threshold: Optional[float] = None,
        clip_outliers: bool = True,
        clip_percentiles: Tuple[float, float] = (0.02, 0.98),
        scaler_type: str = 'robust',
        target_type: str = 'regression',
        save_artifacts: bool = False,
        artifact_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full preprocessing pipeline.

        This method runs all preprocessing steps in the correct order:
        1. Drop sparse rows (first pass)
        2. Drop sparse columns
        3. Separate features and target
        4. Clip outliers (optional)
        5. Drop sparse rows (final pass)
        6. Scale features

        **Config-driven**: Reads missing/same-value thresholds from FEATURES config

        Parameters:
        -----------
        train_df : pd.DataFrame
            Training dataframe
        test_df : Optional[pd.DataFrame]
            Testing dataframe (will use same preprocessing)
        sparse_row_threshold : float
            Initial sparse row threshold (default: 0.6)
        final_sparse_row_threshold : float
            Final sparse row threshold (default: 0.95)
        missing_col_threshold : Optional[float]
            Column missing value threshold (default: read from FEATURES.MISSING_THRESHOLD or 0.8)
        same_value_col_threshold : Optional[float]
            Column same-value threshold (default: read from FEATURES.SAME_VALUE_THRESHOLD or 0.95)
        clip_outliers : bool
            Whether to clip outliers (default: True)
        clip_percentiles : Tuple[float, float]
            Clipping percentiles (default: (0.02, 0.98))
        scaler_type : str
            'robust' or 'standard' (default: 'robust')
        target_type : str
            'regression', 'classification', or 'sector'
        save_artifacts : bool
            Save preprocessing artifacts (clip bounds, dropped cols, etc.)
        artifact_dir : Optional[str]
            Directory to save artifacts

        Returns:
        --------
        dict
            {
                'X_train': np.ndarray,
                'y_train': np.ndarray,
                'X_test': np.ndarray (if test_df provided),
                'y_test': np.ndarray (if test_df provided),
                'feature_names': List[str],
                'dropped_cols': List[str],
                'clip_bounds': Dict[str, Tuple[float, float]],
                'scaler': StandardScaler or RobustScaler
            }

        Example:
        --------
        processor = DataProcessor()
        result = processor.full_pipeline(train_df, test_df, clip_outliers=True)
        X_train, y_train = result['X_train'], result['y_train']
        """
        self.logger.info("=" * 80)
        self.logger.info("DATA PREPROCESSING PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Train shape: {train_df.shape}")
        if test_df is not None:
            self.logger.info(f"Test shape: {test_df.shape}")

        # Read thresholds from config if not provided (same logic for regressor and ml_backtest)
        if missing_col_threshold is None:
            features_config = self.config.get('FEATURES', {})
            missing_col_threshold = float(features_config.get('MISSING_THRESHOLD', 0.8))

        if same_value_col_threshold is None:
            features_config = self.config.get('FEATURES', {})
            same_value_col_threshold = float(features_config.get('SAME_VALUE_THRESHOLD', 0.95))

        # Step 1: Drop sparse rows (first pass - aggressive)
        self.logger.info("\n[1/7] Dropping sparse rows (first pass)...")
        train_clean = self.drop_sparse_rows(train_df, threshold=sparse_row_threshold)

        # Step 2: Drop sparse columns
        self.logger.info("\n[2/7] Dropping sparse columns...")
        train_clean, dropped_cols = self.drop_sparse_cols(
            train_clean,
            missing_threshold=missing_col_threshold,
            same_value_threshold=same_value_col_threshold
        )
        self.dropped_cols = dropped_cols

        # Step 3: Separate features and target
        self.logger.info("\n[3/7] Separating features and target...")
        X_train, y_train = self.prepare_features_and_target(
            train_clean,
            target_type=target_type,
            drop_cols=dropped_cols
        )

        # Step 4: Clip outliers (optional)
        if clip_outliers:
            self.logger.info("\n[4/7] Clipping outliers...")
            X_train = self.clip_outliers(
                X_train,
                lower_percentile=clip_percentiles[0],
                upper_percentile=clip_percentiles[1],
                save_bounds=True
            )
        else:
            self.logger.info("\n[4/7] Skipping outlier clipping")

        # Step 5: Drop sparse rows (final pass - conservative)
        self.logger.info("\n[5/7] Dropping sparse rows (final pass)...")
        # Recombine for final sparse row check
        temp_df = X_train.copy()
        temp_df['_target_'] = y_train
        temp_df = self.drop_sparse_rows(temp_df, threshold=final_sparse_row_threshold)
        y_train = temp_df['_target_']
        X_train = temp_df.drop(columns=['_target_'])

        # Step 6: Scale features
        self.logger.info("\n[6/7] Scaling features...")
        X_train_scaled = self.fit_scaler(X_train, scaler_type=scaler_type)
        self.feature_names = X_train.columns.tolist()
        self.is_fitted = True

        # ✅ FIX: Remove rows with NaN in target (CRITICAL BUG FIX)
        # NaN in target means no future price data (delisting, bankruptcy, etc.)
        # Filling with 0 is WRONG - it teaches model: "missing data = 0% return"
        nan_mask = y_train.isna()
        if nan_mask.any():
            n_nan = nan_mask.sum()
            n_total = len(y_train)
            pct_nan = (n_nan / n_total) * 100

            self.logger.warning(f"⚠️  Found {n_nan} NaN values in target ({pct_nan:.2f}%)")
            self.logger.warning(f"   Removing these rows (NaN target = invalid training data)")

            # Remove rows with NaN targets
            valid_mask = ~nan_mask
            X_train_scaled = X_train_scaled[valid_mask]
            y_train = y_train[valid_mask]

            self.logger.info(f"   After NaN removal: {len(y_train)} samples ({n_total - n_nan} valid)")

        # Prepare result
        result = {
            'X_train': X_train_scaled,
            'y_train': y_train.values,  # ✅ FIX: No fillna(0)!
            'feature_names': self.feature_names,
            'dropped_cols': self.dropped_cols,
            'clip_bounds': self.clip_bounds,
            'scaler': self.scaler,
            'target_type': target_type
        }

        # Step 7: Process test data (if provided)
        if test_df is not None:
            self.logger.info("\n[7/7] Processing test data...")

            # Apply same column drops
            test_clean = test_df.drop(columns=self.dropped_cols, errors='ignore')

            # Separate features and target
            X_test, y_test = self.prepare_features_and_target(
                test_clean,
                target_type=target_type,
                drop_cols=self.dropped_cols
            )

            # Apply saved clip bounds
            if clip_outliers and self.clip_bounds:
                X_test = self.clip_outliers(X_test, apply_saved_bounds=True)

            # Scale using fitted scaler
            X_test_scaled = self.transform_scaler(X_test)

            # ✅ FIX: Remove rows with NaN in test target (same as train)
            nan_mask_test = y_test.isna()
            if nan_mask_test.any():
                n_nan_test = nan_mask_test.sum()
                n_total_test = len(y_test)
                pct_nan_test = (n_nan_test / n_total_test) * 100

                self.logger.warning(f"⚠️  Found {n_nan_test} NaN values in test target ({pct_nan_test:.2f}%)")
                self.logger.warning(f"   Removing these rows (NaN target = invalid test data)")

                # Remove rows with NaN targets
                valid_mask_test = ~nan_mask_test
                X_test_scaled = X_test_scaled[valid_mask_test]
                y_test = y_test[valid_mask_test]

                self.logger.info(f"   After NaN removal: {len(y_test)} test samples")

            result['X_test'] = X_test_scaled
            result['y_test'] = y_test.values  # ✅ FIX: No fillna(0)!

            self.logger.info(f"   Test processed: {X_test_scaled.shape}")

        # Save artifacts (optional)
        if save_artifacts and artifact_dir:
            self._save_artifacts(artifact_dir, result)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PREPROCESSING COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Final train shape: {X_train_scaled.shape}")
        self.logger.info(f"Feature count: {len(self.feature_names)}")
        self.logger.info(f"Dropped columns: {len(self.dropped_cols)}")

        return result

    # ========================================================================
    # Artifact Management
    # ========================================================================

    def _save_artifacts(self, artifact_dir: str, result: Dict[str, Any]):
        """Save preprocessing artifacts for reproducibility."""
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)

        # Save dropped columns
        with open(artifact_path / 'dropped_cols.json', 'w') as f:
            json.dump(self.dropped_cols, f, indent=2)

        # Save clip bounds
        if self.clip_bounds:
            with open(artifact_path / 'clip_bounds.json', 'w') as f:
                json.dump(self.clip_bounds, f, indent=2)

        # Save feature names
        with open(artifact_path / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)

        self.logger.info(f"   Saved artifacts to {artifact_path}")

    def load_artifacts(self, artifact_dir: str):
        """Load preprocessing artifacts from directory."""
        artifact_path = Path(artifact_dir)

        # Load dropped columns
        with open(artifact_path / 'dropped_cols.json', 'r') as f:
            self.dropped_cols = json.load(f)

        # Load clip bounds
        clip_bounds_file = artifact_path / 'clip_bounds.json'
        if clip_bounds_file.exists():
            with open(clip_bounds_file, 'r') as f:
                self.clip_bounds = json.load(f)

        # Load feature names
        with open(artifact_path / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)

        self.logger.info(f"   Loaded artifacts from {artifact_path}")

    # ========================================================================
    # Trading Date Utilities
    # ========================================================================

    @staticmethod
    def get_trade_date(
        pdate: pd.Timestamp,
        price_table: pd.DataFrame,
        date_column: str = 'date',
        min_stocks: int = 100,
        debug: bool = True
    ) -> Optional[pd.Timestamp]:
        """
        달력 날짜를 실제 거래 날짜로 변환합니다.

        월초(1~15일)는 미래 방향, 월말(16일~)은 과거 방향에서 가장 가까운 거래일을 찾습니다.
        이는 분기 초(1/1, 4/1 등)가 공휴일일 때 같은 분기 내의 거래일을 찾기 위함입니다.

        Parameters:
        ----------
        pdate : pd.Timestamp
            목표 달력 날짜
        price_table : pd.DataFrame
            가격 데이터 (date 컬럼 필요)
        date_column : str
            날짜 컬럼명 (기본값: 'date')
        min_stocks : int
            거래일로 인정하기 위한 최소 종목 수 (기본값: 100)
            휴장일에 일부 종목만 데이터가 있는 경우를 필터링
        debug : bool
            디버그 로깅 활성화 여부 (기본값: True)

        Returns:
        -------
        pd.Timestamp or None
            거래 날짜를 찾으면 반환, 그렇지 않으면 None

        Examples:
        --------
        >>> # 1998-01-01 (월초) → 1998-01-02 반환 (미래 방향, 같은 분기)
        >>> # 1998-12-31 (월말) → 1998-12-30 반환 (과거 방향, 같은 분기)
        >>> trade_date = DataProcessor.get_trade_date(
        ...     pd.Timestamp('1998-01-01'),
        ...     price_table
        ... )

        Notes:
        -----
        - 월초/월말 구분: 15일 기준
        - 월초: pdate 이후 10일 내 첫 거래일
        - 월말: pdate 이전 10일 내 마지막 거래일
        - 최소 min_stocks개 이상 종목이 거래되어야 유효한 거래일로 인정
        - regressor.py와 ml_backtest.py에서 동일하게 사용
        """
        from dateutil.relativedelta import relativedelta

        # 월초/월말 구분: 15일 기준
        is_month_start = pdate.day <= 15

        if is_month_start:
            # 월초: 날짜 이후 10일 내에서 가장 가까운 거래일 찾기
            future_date = pdate + relativedelta(days=10)
            res = price_table.query(f"{date_column} >= @pdate and {date_column} <= @future_date")
        else:
            # 월말: 날짜 이전 10일 내에서 가장 가까운 거래일 찾기
            past_date = pdate - relativedelta(days=10)
            res = price_table.query(f"{date_column} >= @past_date and {date_column} <= @pdate")

        if res.empty:
            return None

        # 날짜별 종목 수 계산
        date_counts = res.groupby(date_column).size()

        # 디버그 로깅: 날짜별 종목 수와 샘플 종목 출력
        if debug:
            logging.info(f"      [DEBUG] Checking dates around {pdate.date()} (direction: {'future' if is_month_start else 'past'}):")
            # 날짜 정렬 (월초: 오름차순, 월말: 내림차순)
            sorted_dates = date_counts.sort_index(ascending=is_month_start)
            for date_val, count in sorted_dates.head(5).items():
                sample_symbols = res[res[date_column] == date_val]['symbol'].head(5).tolist()
                status = "✓" if count >= min_stocks else "✗"
                logging.info(f"      [DEBUG]   {date_val.date()}: {count:4d} stocks {status} - samples: {sample_symbols}")

        # 최소 종목 수 이상인 날짜만 유효한 거래일로 인정
        valid_dates = date_counts[date_counts >= min_stocks]

        if valid_dates.empty:
            logging.warning(f"      [DEBUG] No valid trading day found (min_stocks={min_stocks})")
            return None

        # 가장 가까운 유효 거래일 반환
        if is_month_start:
            return valid_dates.index.min()  # 미래 방향: 가장 작은 날짜
        else:
            return valid_dates.index.max()  # 과거 방향: 가장 큰 날짜

    # ========================================================================
    # Sector Categorization Utilities
    # ========================================================================

    @staticmethod
    def map_sectors_to_categories(
        df: pd.DataFrame,
        config: Dict[str, Any],
        sector_column: str = 'sector',
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:
        """
        섹터를 카테고리로 매핑합니다.

        작은 섹터들을 경제적 특성에 따라 카테고리로 통합하여 샘플 수 부족 문제를 해결합니다.

        Parameters:
        ----------
        df : pd.DataFrame
            섹터 정보를 포함한 데이터프레임
        config : Dict[str, Any]
            전체 설정 딕셔너리 (SECTOR_CATEGORIZATION 섹션 포함)
        sector_column : str
            원본 섹터 컬럼명 (기본값: 'sector')
        logger : Optional[logging.Logger]
            로거 (선택)

        Returns:
        -------
        pd.DataFrame
            'sector_category' 컬럼이 추가된 데이터프레임

        Examples:
        --------
        >>> df_with_category = DataProcessor.map_sectors_to_categories(
        ...     df, config, sector_column='sector'
        ... )
        >>> df_with_category[['sector', 'sector_category']].drop_duplicates()
           sector                    sector_category
        0  Financials                Financial
        1  Real Estate               Financial
        2  Information Technology    Technology
        3  Conglomerates             Others

        Notes:
        -----
        - ENABLED=N이면 원본 섹터를 그대로 사용 (sector_category = sector)
        - ENABLED=Y이면 카테고리로 매핑
        - 정의되지 않은 섹터는 자동으로 "Others" 카테고리에 할당
        - 원본 sector 컬럼은 유지됨 (백업용)
        """
        cat_config = config.get('ML', {}).get('SECTOR_CATEGORIZATION', {})

        # 카테고리화 비활성화 시 원본 섹터 사용
        if cat_config.get('ENABLED', 'N') != 'Y':
            if logger:
                logger.info("📊 Sector categorization disabled (ENABLED=N), using original sectors")
            df['sector_category'] = df[sector_column]
            return df

        categories_def = cat_config.get('CATEGORIES', {})

        if not categories_def:
            if logger:
                logger.warning("⚠️  No categories defined, using original sectors")
            df['sector_category'] = df[sector_column]
            return df

        # 섹터 → 카테고리 매핑 딕셔너리 생성
        sector_to_category = {}
        for category_name, category_info in categories_def.items():
            sectors = category_info.get('sectors', [])
            for sector in sectors:
                sector_to_category[sector] = category_name

        # 매핑 적용 (정의되지 않은 섹터는 "Others")
        df['sector_category'] = df[sector_column].map(sector_to_category).fillna('Others')

        if logger:
            logger.info("✅ Sector categorization applied")
            logger.info(f"   Categories defined: {list(categories_def.keys())}")

            # 카테고리별 샘플 수 로깅
            category_counts = df['sector_category'].value_counts()
            for cat, count in category_counts.items():
                logger.info(f"   {cat}: {count} samples")

            # Others에 포함된 섹터 로깅
            others_sectors = df[df['sector_category'] == 'Others'][sector_column].unique()
            if len(others_sectors) > 0:
                logger.info(f"   'Others' category includes: {list(others_sectors)}")

        return df


if __name__ == "__main__":
    # Self-test
    print("DataProcessor module loaded successfully")
    print("\nKey features:")
    print("  - Sparse row/column removal")
    print("  - Outlier clipping")
    print("  - Feature scaling (Robust/Standard)")
    print("  - NaN handling")
    print("  - Artifact saving/loading")
