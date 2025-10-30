"""Stock Price Movement Prediction using Two-Stage Classification and Regression.

This module implements a sophisticated machine learning pipeline that predicts stock price
movements using a two-stage approach:
    1. **Classification Stage**: Multiple binary classifiers predict whether a stock will
       increase or decrease in value.
    2. **Regression Stage**: Multiple regressors predict the magnitude of price change.
    3. **Ensemble Voting**: Combines predictions from multiple classifiers to filter stocks
       before applying regression predictions.

The two-stage strategy reduces false positives by requiring consensus from multiple
classifiers before trusting the regression model's predictions.

Key Features:
    - GPU-accelerated XGBoost and LightGBM models
    - Multiple model ensembles with different hyperparameters
    - Quarterly data processing with parquet format for efficiency
    - Robust handling of missing data and feature selection
    - Sector-based predictions (optional)
    - Comprehensive evaluation with top-K stock selection

Usage Example:
    >>> from config.config_loader import load_config
    >>> conf = load_config('config/config.yaml')
    >>>
    >>> # Initialize regressor
    >>> regressor = Regressor(conf)
    >>>
    >>> # Load and prepare data
    >>> regressor.dataload()
    >>>
    >>> # Train models
    >>> regressor.train()
    >>>
    >>> # Evaluate on test data
    >>> regressor.evaluation()
    >>>
    >>> # Make predictions on latest data
    >>> regressor.latest_prediction()

Model Combinations:
    The module trains and evaluates multiple model variants:
    - Classification models (4 variants):
        * clsmodel_0, 1, 2: XGBoost with max_depth 8, 9, 10
        * clsmodel_3: LightGBM with max_depth 8
    - Regression models (2 variants):
        * model_0: XGBoost with max_depth 8
        * model_1: XGBoost with max_depth 10
    - Ensemble predictions (per regression model):
        * prediction: Raw regression output
        * prediction_wbinary_0-3: Filtered by each classifier
        * prediction_wbinary_ensemble: Filtered by classifiers 1 AND 3
        * prediction_wbinary_ensemble2: Filtered by classifiers 1 AND 2
        * prediction_wbinary_ensemble3: Filtered by majority vote (2+ of 3)

TODO:
    - Implement PER_SECTOR=True functionality for sector-specific predictions
    - Remove sector mapping from this file (should be in make_mldata.py)
    - Migrate GridSearchCV code to optimizer.py or remove if unused
"""

import glob
import joblib
import logging
import torch
import os
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as nn_f
import torch.optim as optim
from dateutil.relativedelta import relativedelta
import datetime
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any

# datasets library is not used, import removed
# from datasets import Dataset
from config.g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map, sparse_col_list
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from torch.utils.data import DataLoader
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Global Configuration
# TODO: Implement PER_SECTOR=True functionality for sector-based predictions
PER_SECTOR = False  # Whether to train separate models per sector
MODEL_SAVE_PATH = ""  # Path for saving trained models (set in methods)
THRESHOLD = 92  # Percentile threshold for classification (92 = top 8% predicted as positive)

# Columns to exclude from model input (metadata and target variables)
y_col_list = [
    "symbol",
    "exchangeShortName",
    "type",
    "delistedDate",
    "industry",
    "ipoDate",
    "rebalance_date",
    "price",
    "volume",
    "marketCap",
    "price_diff",
    "volume_mul_price",
    "price_dev",
    "report_date",
    "fillingDate_x",
    "sector",
    "price_dev_subavg",
    "sec_price_dev_subavg"
]


class Regressor:
    """Two-stage stock price prediction model using classification and regression.

    This class implements a comprehensive machine learning pipeline for predicting stock
    price movements. It uses a two-stage approach where classification models first
    identify stocks likely to increase, then regression models predict the magnitude
    of price change. Multiple models are trained and combined using ensemble voting.

    The training pipeline includes:
        - Automatic data loading from parquet files (quarterly data)
        - Feature selection based on missing data and variance
        - Training multiple XGBoost and LightGBM models
        - Evaluation with various ensemble strategies
        - Generation of top-K stock recommendations

    Attributes:
        conf (Dict): Configuration dictionary from YAML file
        x_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training labels (price_dev_subavg - price deviation minus mean)
        y_train_cls (pd.DataFrame): Training classification labels (binary: up/down)
        x_test (pd.DataFrame): Test features
        y_test (pd.DataFrame): Test labels
        y_test_cls (pd.DataFrame): Test classification labels
        train_df (pd.DataFrame): Complete training dataset
        test_df (pd.DataFrame): Complete test dataset
        test_df_list (List[Tuple[str, pd.DataFrame]]): List of (filepath, dataframe) for each test period
        train_files (List[str]): Paths to training data files
        test_files (List[str]): Paths to test data files
        root_path (str): Root directory for data and models
        clsmodels (Dict[int, Any]): Dictionary of trained classification models
        models (Dict[int, Any]): Dictionary of trained regression models
        drop_col_list (List[str]): Features dropped due to low variance or high missing rate
        n_sector (int): Number of sectors (for PER_SECTOR mode)
        sector_list (List[str]): List of sector names (for PER_SECTOR mode)
        sector_train_dfs (Dict[str, pd.DataFrame]): Training data per sector
        sector_test_dfs (Dict[str, pd.DataFrame]): Test data per sector
        sector_test_df_lists (List): Test data list per sector
        sector_models (Dict[Tuple[str, int], Any]): Regression models per sector
        sector_cls_models (Dict): Classification models per sector
        sector_x_train (Dict[str, pd.DataFrame]): Training features per sector
        sector_y_train (Dict[str, pd.DataFrame]): Training labels per sector

    Usage Example:
        >>> from config.config_loader import load_config
        >>> conf = load_config('config/config.yaml')
        >>>
        >>> # Create regressor
        >>> regressor = Regressor(conf)
        >>>
        >>> # Load data (automatically loads quarterly parquet files)
        >>> regressor.dataload()
        >>>
        >>> # Train all models (4 classifiers + 2 regressors)
        >>> regressor.train()
        >>>
        >>> # Evaluate on test periods
        >>> regressor.evaluation()
        >>>
        >>> # Make predictions on latest data
        >>> regressor.latest_prediction()
    """

    def __init__(self, conf: Dict[str, Any]) -> None:
        """Initialize the Regressor with configuration.

        Sets up paths, file lists, and empty containers for models and data.
        Automatically discovers quarterly data files based on configured year ranges.

        Args:
            conf: Configuration dictionary with structure:
                {
                    'DATA': {
                        'ROOT_PATH': '/path/to/data'
                    },
                    'ML': {
                        'TRAIN_START_YEAR': 2015,
                        'TRAIN_END_YEAR': 2021,
                        'TEST_START_YEAR': 2022,
                        'TEST_END_YEAR': 2023
                    }
                }

        Raises:
            ValueError: If no training data directory exists at ROOT_PATH/ml_per_year/
        """
        self.conf = conf
        self.x_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.x_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        print(self.conf)

        # Extract configuration values from nested structure
        data_config = conf.get('DATA', {})
        ml_config = conf.get('ML', {})
        self.root_path: str = data_config.get('ROOT_PATH', '/home/user/Quant/data')

        aidata_dir = self.root_path + '/ml_per_year/'
        print("aidata path : " + aidata_dir)
        if not os.path.exists(aidata_dir):
            print("there is no ai data : " + aidata_dir)
            return

        # Build training file list (quarterly parquet files)
        self.train_files: List[str] = []
        train_start = int(ml_config.get('TRAIN_START_YEAR', 2015))
        train_end = int(ml_config.get('TRAIN_END_YEAR', 2021))
        for year in range(train_start, train_end + 1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # Parquet format for 5-10x faster reading
                path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.parquet"
                self.train_files.append(path)

        # Build test file list (quarterly parquet files)
        self.test_files: List[str] = []
        test_start = int(ml_config.get('TEST_START_YEAR', 2022))
        test_end = int(ml_config.get('TEST_END_YEAR', 2023))
        for year in range(test_start, test_end + 1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # Parquet format for 5-10x faster reading
                path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.parquet"
                self.test_files.append(path)

        print("train file list : ", self.train_files)
        print("test file list : ", self.test_files)

        # Initialize data containers
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.test_df_list: List[Tuple[str, pd.DataFrame]] = []

        # Sector-based prediction attributes (for PER_SECTOR mode)
        self.n_sector: int = 0
        self.sector_list: List[str] = []
        self.sector_train_dfs: Dict[str, pd.DataFrame] = dict()
        self.sector_test_dfs: Dict[str, pd.DataFrame] = dict()
        self.sector_test_df_lists: List = []

        # Model containers
        self.clsmodels: Dict[int, Any] = dict()  # Classification models
        self.models: Dict[int, Any] = dict()  # Regression models
        self.sector_models: Dict[Tuple[str, int], Any] = dict()  # Per-sector models
        self.sector_cls_models: Dict = dict()

        # Training data per sector
        self.sector_x_train: Dict[str, pd.DataFrame] = dict()
        self.sector_y_train: Dict[str, pd.DataFrame] = dict()

        # Feature selection tracking
        self.drop_col_list: List[str] = []

    def clean_feature_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean feature names to be compatible with LightGBM.

        LightGBM does not support special JSON characters in feature names and
        requires unique feature names. This method:
            1. Removes special characters (keeps only alphanumeric and underscore)
            2. Handles duplicate names by appending indices

        Args:
            df: DataFrame with potentially problematic feature names

        Returns:
            DataFrame with cleaned column names

        Example:
            >>> df = pd.DataFrame({'price@2023': [1, 2], 'price@2024': [3, 4]})
            >>> df = regressor.clean_feature_names(df)
            >>> print(df.columns)
            Index(['price2023', 'price2024'], dtype='object')
        """
        # Remove special characters from column names
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
        new_n_list = list(new_names.values())

        # Handle duplicate names by appending index
        # [LightGBM] Feature appears more than one time.
        new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col
                     for i, (col, new_col) in enumerate(new_names.items())}
        df = df.rename(columns=new_names)
        return df

    def dataload(self) -> None:
        """Load training and test data from parquet files and prepare features.

        This method performs comprehensive data loading and preprocessing:
            1. Loads all training files and concatenates them
            2. Removes meaningless features (>80% missing or >95% same value)
            3. Filters rows with excessive missing data (>60% NaN)
            4. Computes sector-based price deviation (price_dev minus sector average)
            5. Loads test files separately for per-period evaluation
            6. Splits features (X) and labels (y) for training and testing

        The method handles missing files gracefully by logging warnings and continuing.

        Raises:
            ValueError: If no training data files are found (fatal error)

        Side Effects:
            - Sets self.train_df, self.test_df, self.test_df_list
            - Sets self.x_train, self.y_train, self.y_train_cls
            - Sets self.x_test, self.y_test, self.y_test_cls
            - Sets self.drop_col_list with removed features
            - Logs warnings for missing files
            - Logs info about data shapes and class distribution

        Note:
            - Missing files are skipped with warnings (not fatal)
            - Empty test data is allowed (evaluation will be skipped)
            - Sector mapping should be done in make_mldata.py (see TODO in code)
        """
        # Load all training files and concatenate
        for fpath in self.train_files:
            print(fpath)
            # Skip missing files with warning
            if not os.path.exists(fpath):
                logging.warning(f"Train file not found, skipping: {fpath}")
                print(f"WARNING: Train file not found, skipping: {fpath}")
                continue
            # Parquet reading is 5-10x faster than CSV, 70-90% compressed
            df = pd.read_parquet(fpath, engine='pyarrow')
            df = df.dropna(axis=0, subset=['price_diff'])
            self.train_df = pd.concat([self.train_df, df], axis=0)

        # Remove meaningless columns (high missing rate or low variance)
        # Thresholds: >80% missing OR >95% same value
        missing_threshold = 0.8
        same_value_threshold = 0.95
        columns_to_drop = []

        for col in self.train_df.columns:
            # Check missing rate
            missing_ratio = self.train_df[col].isna().mean()
            if missing_ratio > missing_threshold:
                columns_to_drop.append(col)
            else:
                # Check if dominated by single value (low variance)
                top_value_ratio = self.train_df[col].value_counts(normalize=True, dropna=False).iloc[0]
                if top_value_ratio > same_value_threshold:
                    columns_to_drop.append(col)

        # Don't drop metadata columns (y_col_list)
        columns_to_drop = [col for col in columns_to_drop if col not in y_col_list]

        # Apply feature removal
        self.train_df = self.train_df.drop(columns=columns_to_drop)
        self.drop_col_list = columns_to_drop
        print(f'Removed columns # : {len(columns_to_drop)}')
        print(f'Cleaned DataFrame shape: {self.train_df.shape}')

        # Remove rows with excessive missing data (>60% NaN)
        print("in train set before dtable len : ", len(self.train_df))
        self.train_df['nan_count_per_row'] = self.train_df.isnull().sum(axis=1)
        filtered_row = self.train_df['nan_count_per_row'] < int(len(self.train_df.columns)*0.6)
        self.train_df = self.train_df.loc[filtered_row,:]
        print("in train set after dtable len : ", len(self.train_df))

        # TODO: This should be handled in make_mldata.py, not here
        # Compute sector-based price deviation (price_dev minus sector mean)
        self.train_df["sector"] = self.train_df["industry"].map(sector_map)
        sector_list = list(self.train_df['sector'].unique())
        sector_list = [x for x in sector_list if str(x) != 'nan']
        for sec in sector_list:
            sec_mask = self.train_df['sector'] == sec
            sec_mean = self.train_df.loc[sec_mask, 'price_dev'].mean()
            self.train_df.loc[sec_mask, 'sec_price_dev_subavg'] = self.train_df.loc[sec_mask, 'price_dev'] - sec_mean

        # PER_SECTOR mode: separate training data by sector
        if PER_SECTOR == True:
            print(self.train_df['sector'].value_counts())
            self.sector_list = list(self.train_df['sector'].unique())
            self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
            for sec in self.sector_list:
                self.sector_train_dfs[sec] = self.train_df[self.train_df['sector']==sec].copy()
                print(self.sector_train_dfs[sec])

        # Load test files separately for per-period evaluation
        self.test_df_list = []
        for fpath in self.test_files:
            print(fpath)
            # Skip missing files with warning
            if not os.path.exists(fpath):
                logging.warning(f"Test file not found, skipping: {fpath}")
                print(f"WARNING: Test file not found, skipping: {fpath}")
                continue
            # Parquet reading is 5-10x faster than CSV
            df = pd.read_parquet(fpath, engine='pyarrow')
            df = df.dropna(axis=0, subset=['price_diff'])
            # Remove same features as training
            df = df.drop(columns=columns_to_drop, errors='ignore')

            # Remove rows with excessive missing data
            print("in test set before dtable len : ", len(df))
            df['nan_count_per_row'] = df.isnull().sum(axis=1)
            filtered_row = df['nan_count_per_row'] < int(len(df.columns)*0.6)
            df = df.loc[filtered_row,:]
            print("in test set after dtable len : ", len(df))

            # TODO: This should be handled in make_mldata.py
            # Compute sector-based price deviation
            df["sector"] = df["industry"].map(sector_map)
            sector_list = list(df['sector'].unique())
            sector_list = [x for x in sector_list if str(x) != 'nan']
            for sec in sector_list:
                sec_mask = df['sector'] == sec
                sec_mean = df.loc[sec_mask, 'price_dev'].mean()
                df.loc[sec_mask, 'sec_price_dev_subavg'] = df.loc[sec_mask, 'price_dev'] - sec_mean

            # Concatenate all test data and keep per-period list
            self.test_df = pd.concat([self.test_df, df], axis=0)
            self.test_df_list.append([fpath, df])

            # PER_SECTOR mode: separate test data by sector
            if PER_SECTOR == True:
                for sec in self.sector_list:
                    self.sector_test_df_lists.append([fpath, df[df['sector']==sec].copy(), sec])

        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)
        logging.debug("test_df shape : ")
        logging.debug(self.test_df.shape)

        # Optionally save for debugging
        # self.train_df.to_csv(self.root_path + '/train_df.csv', index=False)
        # self.test_df.to_csv(self.root_path + '/test_df.csv', index=False)

        # Check for critical error: no training data
        if self.train_df.empty:
            error_msg = "❌ FATAL ERROR: No training data available! Cannot train models without data."
            logging.error(error_msg)
            print(f"\n{error_msg}\n")
            raise ValueError("No training data files found. Please check your data directory and configuration.")

        # Log class distribution
        positive_count = (self.train_df['price_dev'] > 0).sum()
        negative_count = (self.train_df['price_dev'] < 0).sum()
        logging.info("positive # : {}, negative # : {}".format(positive_count, negative_count))

        # Split features (X) and labels (y) for training
        self.x_train = self.train_df[self.train_df.columns.difference(y_col_list)]
        self.y_train = self.train_df[['price_dev_subavg']]  # Regression target (price change - mean)
        self.y_train_cls = self.train_df[['price_dev']]  # Classification target (binary: up/down)

        # Prepare sector-specific training data
        for sec in self.sector_list:
            print("sector : ", sec)
            self.sector_x_train[sec] = self.sector_train_dfs[sec][self.sector_train_dfs[sec].columns.difference(y_col_list)]
            self.sector_y_train[sec] = self.sector_train_dfs[sec][['sec_price_dev_subavg']]

        # Handle case of no test data (not fatal, just skip evaluation)
        if self.test_df.empty:
            logging.warning("=" * 80)
            logging.warning("⚠️  No test data available!")
            logging.warning("All test files were missing. Creating empty test datasets.")
            logging.warning("Model evaluation and testing will be skipped.")
            logging.warning("=" * 80)
            print("\n⚠️  WARNING: No test data available. Creating empty test datasets.\n")
            # Create empty test sets with same structure as training
            self.x_test = pd.DataFrame(columns=self.x_train.columns)
            self.y_test = pd.DataFrame(columns=['price_dev_subavg'])
            self.y_test_cls = pd.DataFrame(columns=['price_dev'])
        else:
            # Split features and labels for testing
            self.x_test = self.test_df[self.test_df.columns.difference(y_col_list)]
            self.y_test = self.test_df[['price_dev_subavg']]
            self.y_test_cls = self.test_df[['price_dev']]

    def def_model(self) -> None:
        """Define and initialize classification and regression models.

        Creates multiple model variants with different hyperparameters for ensemble prediction:

        Classification Models (4 variants):
            - clsmodels[0]: XGBClassifier, max_depth=8, GPU-accelerated
            - clsmodels[1]: XGBClassifier, max_depth=9, GPU-accelerated
            - clsmodels[2]: XGBClassifier, max_depth=10, GPU-accelerated
            - clsmodels[3]: LGBMClassifier, max_depth=8, GPU-accelerated

        Regression Models (2 variants):
            - models[0]: XGBRegressor, max_depth=8, GPU-accelerated
            - models[1]: XGBRegressor, max_depth=10, GPU-accelerated

        All models use GPU acceleration (tree_method='gpu_hist' for XGBoost, device='gpu' for LightGBM).

        Note:
            - LGBMRegressor was tested but disabled due to poor accuracy
            - Grid search found optimal LGB params: learning_rate=0.01, max_depth=6,
              min_child_samples=30, n_estimators=1000, num_leaves=31
            - For PER_SECTOR mode, also creates sector-specific regression models

        Side Effects:
            - Populates self.clsmodels with 4 classification models
            - Populates self.models with 2 regression models
            - Populates self.sector_models if PER_SECTOR=True
        """
        # Classification Models: Predict binary up/down
        # Using multiple models with different depths for ensemble diversity
        self.clsmodels[0] = xgboost.XGBClassifier(
            tree_method='gpu_hist', gpu_id=0, n_estimators=500, learning_rate=0.1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=8,
            objective='binary:logistic', eval_metric='logloss')
        self.clsmodels[1] = xgboost.XGBClassifier(
            tree_method='gpu_hist', gpu_id=0, n_estimators=500, learning_rate=0.1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=9,
            objective='binary:logistic', eval_metric='logloss')
        self.clsmodels[2] = xgboost.XGBClassifier(
            tree_method='gpu_hist', gpu_id=0, n_estimators=500, learning_rate=0.1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=10,
            objective='binary:logistic', eval_metric='logloss')

        # LightGBM Classification Model
        # Grid search found optimal parameters:
        # {'learning_rate': 0.01, 'max_depth': 6, 'min_child_samples': 30,
        #  'n_estimators': 1000, 'num_leaves': 31}
        self.clsmodels[3] = lgb.LGBMClassifier(
            boosting_type='gbdt', objective='binary', n_estimators=1000,
            max_depth=8, learning_rate=0.1, device='gpu', boost_from_average=False)

        # Regression Models: Predict magnitude of price change
        self.models[0] = xgboost.XGBRegressor(
            tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=8,
            objective='reg:squarederror', eval_metric='rmse')
        self.models[1] = xgboost.XGBRegressor(
            tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=10,
            objective='reg:squarederror', eval_metric='rmse')

        # LightGBM Regression Model (disabled - poor accuracy)
        # self.models[1] = lgb.LGBMRegressor(
        #     boosting_type='gbdt', objective='regression', max_depth=8,
        #     learning_rate=0.1, n_estimators=1000, subsample=0.8,
        #     colsample_bytree=0.8, device='gpu')

        # Sector-specific models (for PER_SECTOR mode)
        if PER_SECTOR:
            for sec in self.sector_list:
                # Two variants per sector with different hyperparameters
                cur_key = (sec, 0)
                self.sector_models[cur_key] = xgboost.XGBRegressor(
                    tree_method='gpu_hist', gpu_id=0, n_estimators=1000,
                    learning_rate=0.05, gamma=0.01, subsample=0.8,
                    colsample_bytree=0.7, max_depth=7)  # BEST hyperparameters
                cur_key = (sec, 1)
                self.sector_models[cur_key] = xgboost.XGBRegressor(
                    tree_method='gpu_hist', gpu_id=0, n_estimators=1000,
                    learning_rate=0.05, gamma=0.01, subsample=0.8,
                    colsample_bytree=0.7, max_depth=8)

    def train(self) -> None:
        """Train all classification and regression models and save to disk.

        Training pipeline:
            1. Initialize models with def_model()
            2. Clean feature names for LightGBM compatibility
            3. Convert regression target to binary labels (price_dev > 0)
            4. Train 4 classification models
            5. Train 2 regression models
            6. Save all models to MODEL_SAVE_PATH
            7. If PER_SECTOR=True, train sector-specific models

        All models are saved as .sav files using joblib for later loading.
        Training scores (accuracy for classification, R² for regression) are logged.

        Side Effects:
            - Creates MODEL_SAVE_PATH directory if it doesn't exist
            - Saves models to disk as .sav files:
                * clsmodel_0.sav, clsmodel_1.sav, clsmodel_2.sav, clsmodel_3.sav
                * model_0.sav, model_1.sav
                * {sector}_model_0.sav, {sector}_model_1.sav (if PER_SECTOR=True)
            - Logs training scores for all models

        Note:
            - Feature importance analysis code is commented out (uncomment if needed)
            - Grid search / random search code for hyperparameter tuning is commented out
            - Models are trained on GPU for speed (requires CUDA-enabled GPU)
        """
        # Commented out: Grid search for LightGBM hyperparameter tuning
        # param_grid = {
        #     'n_estimators': [1000],
        #     'max_depth': [6, 8, 10, 12],
        #     'learning_rate': [0.01, 0.05, 0.1],
        #     'num_leaves': [31, 50, 70],
        #     'min_child_samples': [20, 30, 40]
        # }
        # lgbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',
        #                           device='gpu', boost_from_average=False)
        # grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid,
        #                            cv=5, scoring='accuracy', n_jobs=-1)
        # self.x_train = self.clean_feature_names(self.x_train)
        # y_train_binary = (self.y_train_cls > 0).astype(int)
        # grid_search.fit(self.x_train, y_train_binary)
        # print("Best parameters found: ", grid_search.best_params_)
        # print("Best accuracy: ", grid_search.best_score_)
        # exit()

        # Commented out: Random search for XGBoost hyperparameter tuning
        # params = {
        #     'learning_rate': np.arange(0.05, 0.3, 0.05),
        #     'max_depth': range(3, 10),
        #     'n_estimators': range(50, 500, 50),
        #     'colsample_bytree': np.arange(0.3, 1.0, 0.1),
        #     'subsample': np.arange(0.5, 1.0, 0.1),
        #     'gamma': [0, 1, 5]
        # }
        # xgb = xgboost.XGBRegressor()
        # cv = KFold(n_splits=5, shuffle=True)
        # search = RandomizedSearchCV(xgb, params, n_iter=100, cv=cv,
        #                             scoring='neg_mean_squared_error', random_state=42)
        # search.fit(self.x_train, self.y_train.values.ravel())
        # print(search.best_params_)
        # exit()

        # Set model save path
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'
        self.def_model()

        # Create save directory if needed
        if not os.path.exists(MODEL_SAVE_PATH):
            print("creating MODELS path : " + MODEL_SAVE_PATH)
            os.makedirs(MODEL_SAVE_PATH)

        # Clean feature names for LightGBM compatibility
        self.x_train = self.clean_feature_names(self.x_train)

        # Convert regression labels to binary classification labels (0/1)
        y_train_binary = (self.y_train_cls > 0).astype(int)

        # Train all classification models
        for i, model in self.clsmodels.items():
            logging.info("start fitting classifier")
            model.fit(self.x_train, y_train_binary)
            filename = MODEL_SAVE_PATH + 'clsmodel_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, y_train_binary))

        # Train all regression models
        for i, model in self.models.items():
            logging.info("start fitting XGBRegressor")
            model.fit(self.x_train, self.y_train.values.ravel())
            filename = MODEL_SAVE_PATH + 'model_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, self.y_train))

            # Commented out: Feature importance analysis
            # logging.info("end fitting RandomForestRegressor")
            # ftr_importances_values = model.feature_importances_
            # ftr_importances = pd.Series(ftr_importances_values, index=self.x_train.columns)
            # ftr_importances.to_csv(MODEL_SAVE_PATH+'model_importances.csv')
            # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            # logging.info(ftr_top20)

        # Train sector-specific models (if PER_SECTOR=True)
        if PER_SECTOR == True:
            for sec_idx, sec in enumerate(self.sector_list):
                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    model.fit(self.sector_x_train[sec], self.sector_y_train[sec].values.ravel())
                    filename = MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))

                    joblib.dump(model, filename)
                    logging.info("model {} score : ".format(str(i)))
                    logging.info(model.score(self.sector_x_train[sec], self.sector_y_train[sec]))
                    logging.info("end fitting per sector XGBRegressor")

                    # Commented out: Feature importance analysis per sector
                    # ftr_importances_values = model.feature_importances_
                    # ftr_importances = pd.Series(ftr_importances_values,
                    #                            index=self.sector_x_train[sec].columns)
                    # ftr_importances.to_csv(MODEL_SAVE_PATH + sec + '_model_importances.csv')
                    # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
                    # logging.info(ftr_top20)


    def evaluation(self) -> None:
        """Evaluate trained models on test data and generate comprehensive reports.

        This method implements the two-stage prediction and ensemble voting strategy:

        Evaluation Pipeline:
            1. Load trained models from disk
            2. For each test period (quarterly):
                a. Run all 4 classifiers to get binary predictions
                b. Run all 2 regressors to get price change magnitude
                c. Create ensemble predictions by combining classifiers:
                   - prediction_wbinary_0-3: Filtered by each classifier individually
                   - prediction_wbinary_ensemble: Filtered by classifiers 1 AND 3
                   - prediction_wbinary_ensemble2: Filtered by classifiers 1 AND 2
                   - prediction_wbinary_ensemble3: Majority vote (2+ of 3 agree)
                d. Compute losses for all prediction variants
                e. Select top-K stocks (K=3, 7, 15) for each prediction method
                f. Compute average earnings per stock for top-K selections
            3. Save results to CSV files
            4. Generate comprehensive evaluation report

        Model Combinations:
            For each of 2 regression models, generates 8 prediction variants:
                1. model_i_prediction: Raw regression output
                2-5. model_i_prediction_wbinary_0-3: Filtered by each classifier
                   (sets prediction to -1 if classifier predicts down)
                6. model_i_prediction_wbinary_ensemble: Filtered by cls1 AND cls3
                7. model_i_prediction_wbinary_ensemble2: Filtered by cls1 AND cls2
                8. model_i_prediction_wbinary_ensemble3: Majority vote filter

            Total: 2 regressors × 8 variants = 16 prediction methods

        Ensemble Voting Logic:
            - prediction_wbinary_0: Use classifier 0's prediction
              If cls0 predicts down (0), set regression output to -1
            - prediction_wbinary_1: Use classifier 1's prediction
            - prediction_wbinary_2: Use classifier 2's prediction
            - prediction_wbinary_3: Use classifier 3's prediction
            - prediction_wbinary_ensemble: Require BOTH cls1 AND cls3 to predict up
              If either predicts down, set regression output to -1
            - prediction_wbinary_ensemble2: Require BOTH cls1 AND cls2 to predict up
            - prediction_wbinary_ensemble3: Majority vote - at least 2 of 3 must predict up
              Uses cls1, cls2, cls3 for voting

        Output Files (saved to MODEL_SAVE_PATH):
            - prediction_ai_{date}.csv: Predictions for each test period
            - prediction_ai.csv: All predictions concatenated
            - pred_df_topk.csv: Top-K evaluation metrics for all models
            - prediction_{date}_{model}_{col}_top{s}-{e}.csv: Top-K stocks per model

        Side Effects:
            - Loads models from MODEL_SAVE_PATH/*.sav
            - Creates evaluation CSV files in MODEL_SAVE_PATH
            - Logs classification reports and metrics
            - Logs top-K earnings for each prediction method

        Note:
            - Uses THRESHOLD (default 92) to convert classifier probabilities to binary
            - Top 8% of stocks (100-92=8%) are predicted as positive
            - Evaluation includes both per-period and accumulated metrics
            - If PER_SECTOR=True, also evaluates sector-specific models
        """
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # Load trained classification models
        self.models = dict()
        self.clsmodels = dict()
        self.clsmodels[0] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_2.sav')
        self.clsmodels[3] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_3.sav')

        # Load trained regression models
        self.models[0] = joblib.load(MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(MODEL_SAVE_PATH + 'model_1.sav')

        # List of all prediction column names (for top-K evaluation)
        pred_col_list = ['ai_pred_avg']  # Average of all regression models

        # Build prediction column names for all model combinations
        for i in range(2):
            pred_col_name = 'model_' + str(i) + '_prediction'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_0'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_1'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_2'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_3'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble2'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble3'
            pred_col_list.append(pred_col_name)

        model_eval_hist = []  # Stores evaluation results for all periods
        full_df = pd.DataFrame()  # Accumulates all test data with predictions

        # Evaluate each test period separately
        for test_idx, (testdate, df) in enumerate(self.test_df_list):

            logging.info("evaluation date : ")
            # Extract date from file path
            tmp = testdate.split('\\')
            tmp = [v for v in tmp if v.endswith('.csv')]
            print(f"in test loop tmp : {tmp}")
            tdate = "_".join(tmp[0].split('_')[4:6])
            print(f"in test loop tdate : {tdate}")

            # Prepare features and labels for this test period
            x_test = df[df.columns.difference(y_col_list)]
            y_test = df[['price_dev_subavg']]
            y_test_cls = df[['price_dev']]
            y_test_binary = (y_test_cls > 0).astype(int)

            preds = np.empty((0, x_test.shape[0]))  # Store raw regression predictions

            df['label'] = y_test  # True price change
            df['label_binary'] = y_test_binary  # True binary label

            # Clean feature names for LightGBM
            x_test = self.clean_feature_names(x_test)

            # === CLASSIFICATION STAGE ===
            # Run all 4 classifiers and evaluate their performance
            for i, model in self.clsmodels.items():
                logging.info(f"classification model # {i}")
                pred_col_name = 'clsmodel_' + str(i) + '_prediction'
                correct_col_name = 'clsmodel_' + str(i) + '_correct'

                # Get predicted probabilities (probability of class 1 = price increase)
                y_probs = model.predict_proba(x_test)[:, 1]

                # Convert probabilities to binary predictions using percentile threshold
                # THRESHOLD=92 means top 8% are predicted as positive
                threshold = np.percentile(y_probs, THRESHOLD)
                y_predict_binary = (y_probs > threshold).astype(int)

                logging.info(f"20% positive threshold == {threshold}")
                logging.info(classification_report(y_test_binary, y_predict_binary))

                # Store classifier predictions
                df[pred_col_name] = y_predict_binary
                df[correct_col_name] = (y_test_binary.values.ravel() == y_predict_binary).astype(int)

                acc = accuracy_score(df['label_binary'], df[pred_col_name])
                logging.info(f"Accuracy for {pred_col_name}: {acc:.4f}")


            # === REGRESSION STAGE ===
            # Run all 2 regressors and create ensemble predictions
            for i, model in self.models.items():
                # Column names for classifier outputs
                pred_bin_col_name_0 = 'clsmodel_0_prediction'
                pred_bin_col_name_1 = 'clsmodel_1_prediction'
                pred_bin_col_name_2 = 'clsmodel_2_prediction'
                pred_bin_col_name_3 = 'clsmodel_3_prediction'

                # Column names for regression outputs
                pred_col_name = 'model_' + str(i) + '_prediction'
                correct_col_name = 'clsmodel_' + str(i) + '_correct'
                pred_col_name_wbinary_0 = 'model_' + str(i) + '_prediction_wbinary_0'
                pred_col_name_wbinary_1 = 'model_' + str(i) + '_prediction_wbinary_1'
                pred_col_name_wbinary_2 = 'model_' + str(i) + '_prediction_wbinary_2'
                pred_col_name_wbinary_3 = 'model_' + str(i) + '_prediction_wbinary_3'
                pred_col_name_wbinary_ensemble = 'model_' + str(i) + '_prediction_wbinary_ensemble'
                pred_col_name_wbinary_ensemble2 = 'model_' + str(i) + '_prediction_wbinary_ensemble2'
                pred_col_name_wbinary_ensemble3 = 'model_' + str(i) + '_prediction_wbinary_ensemble3'

                # Column names for prediction errors (losses)
                loss_col_name = 'model_' + str(i) + '_prediction_loss'
                loss_bin_col_name_0 = 'model_' + str(i) + '_prediction_wbinary_loss_0'
                loss_bin_col_name_1 = 'model_' + str(i) + '_prediction_wbinary_loss_1'
                loss_bin_col_name_2 = 'model_' + str(i) + '_prediction_wbinary_loss_2'
                loss_bin_col_name_3 = 'model_' + str(i) + '_prediction_wbinary_loss_3'

                # Get raw regression predictions
                y_predict = model.predict(x_test)

                # Store raw regression predictions
                df[pred_col_name] = y_predict

                # Create filtered predictions by combining with classifiers
                # If classifier predicts down (0), replace regression output with -1
                df[pred_col_name_wbinary_0] = np.where(df[pred_bin_col_name_0] == 0, -1, y_predict)
                df[pred_col_name_wbinary_1] = np.where(df[pred_bin_col_name_1] == 0, -1, y_predict)
                df[pred_col_name_wbinary_2] = np.where(df[pred_bin_col_name_2] == 0, -1, y_predict)
                df[pred_col_name_wbinary_3] = np.where(df[pred_bin_col_name_3] == 0, -1, y_predict)

                # Ensemble 1: Require BOTH classifier 1 AND 3 to predict up
                df[pred_col_name_wbinary_ensemble] = np.where(
                    ((df[pred_bin_col_name_1] == 0) | (df[pred_bin_col_name_3] == 0)),
                    -1, y_predict)

                # Ensemble 2: Require BOTH classifier 1 AND 2 to predict up
                df[pred_col_name_wbinary_ensemble2] = np.where(
                    ((df[pred_bin_col_name_1] == 0) | (df[pred_bin_col_name_2] == 0)),
                    -1, y_predict)

                # Ensemble 3: Majority vote - at least 2 of 3 classifiers must predict up
                # If 2 or more classifiers predict down (0), replace regression output with -1
                condition = (
                    (df[[pred_bin_col_name_1, pred_bin_col_name_2, pred_bin_col_name_3]] == 0).sum(axis=1) >= 2
                )
                df[pred_col_name_wbinary_ensemble3] = np.where(condition, -1, y_predict)

                # Store raw predictions for averaging
                preds = np.vstack((preds, y_predict[None,:]))

                # Compute prediction errors (losses) for all variants
                df[loss_col_name] = abs(df['label'] - y_predict)
                df[loss_bin_col_name_0] = abs(df['label'] - df[pred_col_name_wbinary_0])
                df[loss_bin_col_name_1] = abs(df['label'] - df[pred_col_name_wbinary_1])
                df[loss_bin_col_name_2] = abs(df['label'] - df[pred_col_name_wbinary_2])
                df[loss_bin_col_name_3] = abs(df['label'] - df[pred_col_name_wbinary_3])

                # Log evaluation metrics for this period
                logging.info(f"eval : model i : {i} loss : {df[loss_col_name].mean()} "
                           f"loss_wbin_0 {df[loss_bin_col_name_0].mean()} "
                           f"loss_wbin_1 {df[loss_bin_col_name_1].mean()} "
                           f"loss_wbin_2 {df[loss_bin_col_name_2].mean()} "
                           f"loss_wbin_3 {df[loss_bin_col_name_3].mean()}")

                # Log accumulated metrics (all periods so far)
                if test_idx != 0:
                    logging.info(f"accumulated eval : model i : {i} "
                               f"loss : {full_df[loss_col_name].mean()} "
                               f"loss_wbin_0 {full_df[loss_bin_col_name_0].mean()} "
                               f"loss_wbin_1 {full_df[loss_bin_col_name_1].mean()} "
                               f"loss_wbin_2 {full_df[loss_bin_col_name_2].mean()} "
                               f"loss_wbin_3 {full_df[loss_bin_col_name_3].mean()}")

            # Compute average prediction across all regression models
            df['ai_pred_avg'] = np.average(preds, axis=0)
            df['ai_pred_avg_loss'] = abs(df['label']-df['ai_pred_avg'])

            # Accumulate results
            full_df = pd.concat([full_df, df], ignore_index=True)
            df.to_csv(MODEL_SAVE_PATH + "prediction_ai_{}.csv".format(tdate))

            # === TOP-K STOCK SELECTION ===
            # For each prediction method, select top-K stocks and compute average earnings
            topk_period_earning_sums = []
            topk_list = [(0,3), (0,7), (0,15)]  # Top 3, 7, 15 stocks

            for s, e in topk_list:
                logging.info("top" + str(s) + " ~ "  + str(e) )
                k = str(s) + '~' + str(e)

                # Evaluate each prediction method
                for col in pred_col_list:
                    # Select top-K stocks based on prediction
                    top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]

                    logging.info("")
                    logging.info(col)
                    logging.info(("label"))
                    logging.info((top_k_df['price_dev'].sum()/(e-s+1)))
                    logging.info(("pred"))
                    logging.info((top_k_df[col].sum()/(e-s+1)))
                    topk_period_earning_sums.append(top_k_df['price_dev'].sum())

                    # Save top-K stocks to CSV
                    top_k_df.to_csv(MODEL_SAVE_PATH+'prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))

                    # Record evaluation metrics for this model and top-K range
                    model_eval_hist.append([
                        tdate, col, k,
                        top_k_df['price_dev'].sum()/(e-s+1),  # Avg true earnings per stock
                        top_k_df[col].sum()/(e-s+1),  # Avg predicted earnings per stock
                        abs(top_k_df[col].sum()/(e-s+1) - top_k_df['price_dev'].sum()/(e-s+1)),  # Loss
                        int(top_k_df[col].sum()/(e-s+1) > 0),  # Is prediction positive?
                        top_k_df['ai_pred_avg'].sum()/(e-s+1),
                        top_k_df['model_0_prediction'].sum()/(e-s+1),
                        top_k_df['model_1_prediction'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_0'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_0'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_1'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_1'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_2'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_2'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_3'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_3'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_ensemble'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_ensemble'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_ensemble2'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_ensemble2'].sum()/(e-s+1),
                        top_k_df['model_0_prediction_wbinary_ensemble3'].sum()/(e-s+1),
                        top_k_df['model_1_prediction_wbinary_ensemble3'].sum()/(e-s+1)
                    ])

        # Create comprehensive evaluation report
        col_name = ['start_date', 'model', 'krange', 'avg_earning_per_stock', 'cur_model_pred',
                   'loss_y_and_pred', 'cur_model_pred_ispositive', 'avg_pred', 'model0_pred',
                   'model1_pred', 'model0_pred_wbinary_0', 'model1_pred_wbinary_0',
                   'model0_pred_wbinary_1', 'model1_pred_wbinary_1', 'model0_pred_wbinary_2',
                   'model1_pred_wbinary_2', 'model0_pred_wbinary_3', 'model1_pred_wbinary_3',
                   'model0_pred_wbinary_ensemble', 'model1_pred_wbinary_ensemble',
                   'model0_pred_wbinary_ensemble2', 'model1_pred_wbinary_ensemble2',
                   'model0_pred_wbinary_ensemble3', 'model1_pred_wbinary_ensemble3']

        pred_df = pd.DataFrame(model_eval_hist, columns=col_name)
        logging.info(pred_df)
        pred_df.to_csv(MODEL_SAVE_PATH+'pred_df_topk.csv', index=False)
        full_df.to_csv(MODEL_SAVE_PATH+'prediction_ai.csv', index=False)

        # === SECTOR-BASED EVALUATION (if PER_SECTOR=True) ===
        if PER_SECTOR == True:
            testdates = set()
            allsector_topk_df = pd.DataFrame()
            self.sector_models = dict()

            # Load sector-specific models
            for sec in self.sector_list:
                for i in range(2):
                    filename = MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))
                    k = (sec, i)
                    self.sector_models[k] = joblib.load(MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))

            sector_model_eval_hist = []

            # Evaluate each sector and test period
            for test_idx, (testdate, df, sec) in enumerate(self.sector_test_df_lists):
                print("sec evaluation date : ")
                tmp = testdate.split('\\')
                tmp = [v for v in tmp if v.endswith('.csv')]
                tdate = "_".join(tmp[0].split('_')[0:2])
                print(tdate)
                print(sec)
                testdates.add(tdate)

                x_test = df[df.columns.difference(y_col_list)]
                y_test = df[['price_dev_subavg']]
                y_test_2 = df[['price_dev_subavg']]

                if len(x_test) == 0:
                    continue

                sector_preds = np.empty((0, x_test.shape[0]))
                df['label'] = y_test

                # Use classifier 2 for sector-based filtering
                y_probs = self.clsmodels[2].predict_proba(x_test)[:, 1]
                threshold = np.percentile(y_probs, THRESHOLD)
                y_predict_binary = (y_probs > threshold).astype(int)

                # Run sector-specific models
                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    pred_col_name_wbin = 'model_' + str(i) + '_prediction_wbinary_2'
                    y_predict = model.predict(x_test)
                    df[pred_col_name] = y_predict

                    df[pred_col_name_wbin] = np.where(y_predict_binary == 0, -1, y_predict)
                    print(f"i{i} sec {sec}")
                    print(x_test.shape)
                    print(sector_preds.shape)
                    print(y_predict[None,:].shape)
                    sector_preds = np.vstack((sector_preds, y_predict[None,:]))

                df['ai_pred_avg'] = np.average(sector_preds, axis=0)
                df.to_csv(MODEL_SAVE_PATH+ "sec_{}_prediction_ai_{}.csv".format(sec, tdate))

                # Top-K evaluation for sector-specific predictions
                topk_period_earning_sums = []
                topk_list = [(0,3), (0,7)]
                for s, e in topk_list:
                    logging.info("top" + str(s) + " ~ "  + str(e) )
                    k = str(s) + '~' + str(e)
                    for col in pred_col_list:
                        top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                        logging.info(col)
                        logging.info(("label"))
                        logging.info((top_k_df['price_dev'].sum()/(e-s+1)))
                        logging.info(("pred"))
                        logging.info((top_k_df[col].sum()/(e-s+1)))
                        topk_period_earning_sums.append(top_k_df['price_dev'].sum())
                        top_k_df.to_csv(MODEL_SAVE_PATH+'prediction_{}_{}_{}_top{}-{}.csv'.format(tdate, sec, col, s, e))
                        top_k_df['start_date'] = tdate
                        top_k_df['col'] = col
                        allsector_topk_df = pd.concat([allsector_topk_df, top_k_df])
                        sector_model_eval_hist.append([
                            tdate, sec, col, k,
                            top_k_df['price_dev'].sum()/(e-s+1),
                            top_k_df[col].sum()/(e-s+1),
                            abs(top_k_df[col].sum()/(e-s+1) - top_k_df['price_dev'].sum()/(e-s+1)),
                            int(top_k_df[col].sum()/(e-s+1) > 0),
                            top_k_df['ai_pred_avg'].sum()/(e-s+1),
                            top_k_df['model_0_prediction'].sum()/(e-s+1),
                            top_k_df['model_1_prediction'].sum()/(e-s+1),
                            top_k_df['model_0_prediction_wbinary_2'].sum()/(e-s+1),
                            top_k_df['model_1_prediction_wbinary_2'].sum()/(e-s+1)
                        ])

            col_name = ['start_date', 'sector', 'model', 'krange', 'avg_earning_per_stock',
                       'cur_model_pred', 'loss_y_and_pred', 'cur_model_pred_ispositive',
                       'avg_pred', 'model0_pred', 'model1_pred',
                       'model0_pred_wbinary_2', 'model1_pred_wbinary_2']
            pred_df = pd.DataFrame(sector_model_eval_hist, columns=col_name)
            print(pred_df)
            pred_df.to_csv(MODEL_SAVE_PATH+'allsector_pred_df.csv'.format(sec), index=False)


    def latest_prediction(self) -> None:
        """Make predictions on the most recent data for stock selection.

        This method loads the latest quarterly data and generates predictions for
        current stock selection. It follows the same two-stage prediction and
        ensemble voting strategy as evaluation(), but works with the most recent
        data only (not historical test data).

        Pipeline:
            1. Load all trained classification and regression models
            2. Read latest year's data (all quarters) and keep most recent per symbol
            3. Filter to stocks with sufficient data (>60% non-NaN)
            4. Run 4 classification models to get binary predictions
            5. Run 2 regression models to get price change magnitude
            6. Create ensemble predictions using various voting strategies
            7. Generate top-K stock recommendations (K=3, 7, 15)
            8. Save predictions to CSV files

        Output Files (saved to MODEL_SAVE_PATH):
            - latest_prediction.csv: All predictions for latest data
            - latest_prediction_{model}_{col}_top{s}-{e}.csv: Top-K stocks per model
            - sec_{sector}_latest_prediction.csv: Sector-specific predictions (if PER_SECTOR=True)
            - allsector_latest_pred_df.csv: Sector-based top-K summary (if PER_SECTOR=True)

        Side Effects:
            - Loads models from MODEL_SAVE_PATH/*.sav
            - Creates prediction CSV files in MODEL_SAVE_PATH
            - Logs prediction thresholds and top-K ranges

        Note:
            - Uses year_period to keep only the most recent data per symbol
            - Applies same THRESHOLD (92) as evaluation() for classification
            - Sector-specific predictions available if PER_SECTOR=True
            - FIXME: Hardcoded to read 2024 data (should be configurable)
        """
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # Load trained models
        self.clsmodels = dict()
        self.clsmodels[0] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_2.sav')
        self.clsmodels[3] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_3.sav')
        self.models = dict()
        self.models[0] = joblib.load(MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(MODEL_SAVE_PATH + 'model_1.sav')

        aidata_dir = self.root_path + '/ml_per_year/'

        # Build prediction column list (same as evaluation)
        pred_col_list = ['ai_pred_avg']
        for i in range(2):
            pred_col_name = 'model_' + str(i) + '_prediction'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_0'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_1'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_2'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_3'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble2'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble3'
            pred_col_list.append(pred_col_name)

        # Load latest year's data (all quarters) and keep most recent per symbol
        # FIXME: Hardcoded to 2024 - should be configurable
        ldf = pd.DataFrame()
        for i in [1,2,3,4]:
            latest_data_path = aidata_dir + f'rnorm_fs_2024_Q{i}.csv'
            df = pd.read_csv(latest_data_path)
            ldf = pd.concat([ldf, df], axis=0)

        # Sort by year_period descending and keep first (most recent) per symbol
        ldf = ldf.sort_values(by='year_period', ascending=False)
        ldf = ldf.drop_duplicates(subset='symbol', keep='first')
        ldf = ldf.drop(columns=self.drop_col_list, errors='ignore')

        # Remove first column (index column from CSV)
        # FIXME: rnorm_fs*.csv files have index in first column
        ldf = ldf.drop(df.columns[0], axis=1)

        # Extract sector list
        self.sector_list = list(ldf['sector'].unique())
        self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
        ldf = ldf.drop('sector', axis=1)

        # Filter rows with excessive missing data (>60% NaN)
        print("before dtable len : ", len(ldf))
        ldf['nan_count_per_row'] = ldf.isnull().sum(axis=1)
        filtered_row = ldf['nan_count_per_row'] < int(len(ldf.columns)*0.6)
        ldf = ldf.loc[filtered_row,:]
        print("after dtable len : ", len(ldf))

        # Prepare input features
        input = ldf[ldf.columns.difference(y_col_list)]
        input = self.clean_feature_names(input)
        preds = np.empty((0, input.shape[0]))

        # === CLASSIFICATION STAGE ===
        # Run all classifiers
        for i, model in self.clsmodels.items():
            logging.info(f"classification model # {i}")
            pred_col_name = 'clsmodel_' + str(i) + '_prediction'
            y_probs = model.predict_proba(input)[:, 1]
            # Convert to binary using percentile threshold
            threshold = np.percentile(y_probs, THRESHOLD)
            y_predict_binary = (y_probs > threshold).astype(int)
            logging.info(f"20% positive threshold == {threshold}")
            ldf[pred_col_name] = y_predict_binary

        # === REGRESSION STAGE ===
        # Run all regressors and create ensemble predictions
        for i, model in self.models.items():
            pred_bin_col_name_0 = 'clsmodel_0_prediction'
            pred_bin_col_name_1 = 'clsmodel_1_prediction'
            pred_bin_col_name_2 = 'clsmodel_2_prediction'
            pred_bin_col_name_3 = 'clsmodel_3_prediction'
            pred_col_name = 'model_' + str(i) + '_prediction'
            correct_col_name = 'clsmodel_' + str(i) + '_correct'
            pred_col_name_wbinary_0 = 'model_' + str(i) + '_prediction_wbinary_0'
            pred_col_name_wbinary_1 = 'model_' + str(i) + '_prediction_wbinary_1'
            pred_col_name_wbinary_2 = 'model_' + str(i) + '_prediction_wbinary_2'
            pred_col_name_wbinary_3 = 'model_' + str(i) + '_prediction_wbinary_3'
            pred_col_name_wbinary_ensemble = 'model_' + str(i) + '_prediction_wbinary_ensemble'
            pred_col_name_wbinary_ensemble2 = 'model_' + str(i) + '_prediction_wbinary_ensemble2'
            pred_col_name_wbinary_ensemble3 = 'model_' + str(i) + '_prediction_wbinary_ensemble3'

            loss_col_name = 'model_' + str(i) + '_prediction_loss'
            loss_bin_col_name_0 = 'model_' + str(i) + '_prediction_wbinary_loss_0'
            loss_bin_col_name_1 = 'model_' + str(i) + '_prediction_wbinary_loss_1'
            loss_bin_col_name_2 = 'model_' + str(i) + '_prediction_wbinary_loss_2'
            loss_bin_col_name_3 = 'model_' + str(i) + '_prediction_wbinary_loss_3'

            # Get raw regression predictions
            y_predict = model.predict(input)

            # Store raw predictions
            ldf[pred_col_name] = y_predict

            # Create filtered predictions using classifier outputs
            ldf[pred_col_name_wbinary_0] = np.where(ldf[pred_bin_col_name_0] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_1] = np.where(ldf[pred_bin_col_name_1] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_2] = np.where(ldf[pred_bin_col_name_2] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_3] = np.where(ldf[pred_bin_col_name_3] == 0, -1, y_predict)

            # Ensemble predictions using various voting strategies
            ldf[pred_col_name_wbinary_ensemble] = np.where(
                ((ldf[pred_bin_col_name_1] == 0) | (ldf[pred_bin_col_name_3] == 0)),
                -1, y_predict)
            ldf[pred_col_name_wbinary_ensemble2] = np.where(
                ((ldf[pred_bin_col_name_1] == 0) | (ldf[pred_bin_col_name_2] == 0)),
                -1, y_predict)

            # Majority vote: at least 2 of 3 must predict up
            condition = (
                (ldf[[pred_bin_col_name_1, pred_bin_col_name_2, pred_bin_col_name_3]] == 0).sum(axis=1) >= 2
            )
            ldf[pred_col_name_wbinary_ensemble3] = np.where(condition, -1, y_predict)
            preds = np.vstack((preds, y_predict[None,:]))

        # Compute average prediction
        ldf['ai_pred_avg'] = np.average(preds, axis=0)
        ldf.to_csv(MODEL_SAVE_PATH+"latest_prediction.csv")

        # Generate top-K stock recommendations
        topk_list = [(0,3), (0,7), (0, 15)]
        for s, e in topk_list:
            logging.info("top" + str(s) + " ~ " + str(e))
            for col in pred_col_list:
                top_k_df = ldf.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                top_k_df.to_csv(MODEL_SAVE_PATH+'latest_prediction_{}_top{}-{}.csv'.format(col, s, e))

        # === SECTOR-SPECIFIC PREDICTIONS (if PER_SECTOR=True) ===
        if PER_SECTOR == True:
            self.sector_models = dict()
            ldf = pd.read_csv(latest_data_path)

            # Load sector-specific models
            for sec in self.sector_list:
                for i in range(2):
                    filename = MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))
                    k = (sec, i)
                    print("model path : ", MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))
                    self.sector_models[k] = joblib.load(MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))

            all_preds = []

            # Make predictions per sector
            for sec in self.sector_list:
                sec_df = ldf[ldf['sector']==sec]
                sec_df = sec_df.drop('sector', axis=1)
                indata = sec_df[sec_df.columns.difference(['symbol'])]
                print(indata)
                preds = np.empty((0, indata.shape[0]))

                # Run sector-specific models
                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    y_predict3 = model.predict(indata)
                    sec_df[pred_col_name] = y_predict3
                    preds = np.vstack((preds, y_predict3[None,:]))

                sec_df['ai_pred_avg'] = np.average(preds, axis=0)
                sec_df.to_csv(MODEL_SAVE_PATH+"sec_{}_latest_prediction.csv".format(sec))

                # Top-K per sector
                topk_list = [(0,3), (0,7), (0, 15)]
                for s, e in topk_list:
                    logging.info("top" + str(s) + " ~ " + str(e))
                    for col in pred_col_list:
                        top_k_df = sec_df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                        top_k_df.to_csv(MODEL_SAVE_PATH+'latest_prediction_{}_{}_top{}-{}.csv'.format(col, sec, s, e))
                        symbols = top_k_df['symbol'].to_list()
                        preds = top_k_df[col].to_list()
                        for i, sym in enumerate(symbols):
                            all_preds.append([(e-s), sec, col, i, sym, preds[i]])

            # Save sector-based summary
            col_name = ['k', 'sector', 'model', 'i', 'symbol', 'pred']
            pred_df = pd.DataFrame(all_preds, columns=col_name)
            pred_df.to_csv(MODEL_SAVE_PATH+'allsector_latest_pred_df.csv', index=False)
