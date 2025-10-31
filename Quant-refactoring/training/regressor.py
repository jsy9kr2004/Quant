"""2단계 분류 및 회귀를 사용한 주식 가격 변동 예측입니다.

이 모듈은 2단계 접근 방식을 사용하여 주식 가격 변동을 예측하는 정교한 머신러닝
파이프라인을 구현합니다:
    1. **분류 단계**: 여러 이진 분류기가 주식 가격의 상승 또는 하락 여부를 예측합니다.
    2. **회귀 단계**: 여러 회귀 모델이 가격 변동의 크기를 예측합니다.
    3. **앙상블 투표**: 회귀 예측을 적용하기 전에 여러 분류기의 예측을 결합하여
       주식을 필터링합니다.

2단계 전략은 회귀 모델의 예측을 신뢰하기 전에 여러 분류기의 합의를 요구하여
거짓 양성을 줄입니다.

주요 기능:
    - GPU 가속 XGBoost 및 LightGBM 모델
    - 다양한 하이퍼파라미터를 가진 여러 모델 앙상블
    - 효율성을 위한 parquet 형식의 분기별 데이터 처리
    - 누락된 데이터 및 특성 선택의 강력한 처리
    - 섹터 기반 예측 (선택 사항)
    - 상위 K개 주식 선택을 통한 종합적인 평가

사용 예시:
    from config.config_loader import load_config
    conf = load_config('config/config.yaml')

    # 회귀 모델 초기화
    regressor = Regressor(conf)

    # 데이터 로드 및 준비
    regressor.dataload()

    # 모델 학습
    regressor.train()

    # 테스트 데이터로 평가
    regressor.evaluation()

    # 최신 데이터로 예측
    regressor.latest_prediction()

모델 조합:
    이 모듈은 여러 모델 변형을 학습하고 평가합니다:
    - 분류 모델 (4개 변형):
        * clsmodel_0, 1, 2: max_depth 8, 9, 10을 사용하는 XGBoost
        * clsmodel_3: max_depth 8을 사용하는 LightGBM
    - 회귀 모델 (2개 변형):
        * model_0: max_depth 8을 사용하는 XGBoost
        * model_1: max_depth 10을 사용하는 XGBoost
    - 앙상블 예측 (회귀 모델당):
        * prediction: 원시 회귀 출력
        * prediction_wbinary_0-3: 각 분류기로 필터링
        * prediction_wbinary_ensemble: 분류기 1 AND 3으로 필터링
        * prediction_wbinary_ensemble2: 분류기 1 AND 2로 필터링
        * prediction_wbinary_ensemble3: 다수결 투표 (3개 중 2개 이상)

TODO:
    - 섹터별 예측을 위한 PER_SECTOR=True 기능 구현
    - 이 파일에서 섹터 매핑 제거 (make_mldata.py에 있어야 함)
    - GridSearchCV 코드를 optimizer.py로 마이그레이션하거나 사용하지 않으면 제거
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

# datasets 라이브러리는 사용되지 않으므로 import 제거됨
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

# 전역 설정
# TODO: 섹터 기반 예측을 위한 PER_SECTOR=True 기능 구현
PER_SECTOR = False  # 섹터별로 개별 모델을 학습할지 여부
MODEL_SAVE_PATH = ""  # 학습된 모델 저장 경로 (메서드에서 설정됨)
THRESHOLD = 92  # 분류를 위한 백분위수 임계값 (92 = 상위 8%가 양성으로 예측됨)

# 모델 입력에서 제외할 컬럼 (메타데이터 및 타겟 변수)
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
    """분류와 회귀를 사용하는 2단계 주식 가격 예측 모델입니다.

    이 클래스는 주식 가격 변동을 예측하는 종합적인 머신러닝 파이프라인을 구현합니다.
    분류 모델이 먼저 상승할 가능성이 높은 주식을 식별한 다음, 회귀 모델이 가격 변동의
    크기를 예측하는 2단계 접근 방식을 사용합니다. 여러 모델을 학습하고 앙상블 투표를
    사용하여 결합합니다.

    학습 파이프라인은 다음을 포함합니다:
        - parquet 파일에서 자동 데이터 로드 (분기별 데이터)
        - 누락 데이터 및 분산을 기반으로 한 특성 선택
        - 여러 XGBoost 및 LightGBM 모델 학습
        - 다양한 앙상블 전략을 사용한 평가
        - 상위 K개 주식 추천 생성

    Attributes:
        conf (Dict): YAML 파일의 설정 딕셔너리
        x_train (pd.DataFrame): 학습 특성
        y_train (pd.DataFrame): 학습 레이블 (price_dev_subavg - 가격 편차에서 평균을 뺀 값)
        y_train_cls (pd.DataFrame): 학습 분류 레이블 (이진: 상승/하락)
        x_test (pd.DataFrame): 테스트 특성
        y_test (pd.DataFrame): 테스트 레이블
        y_test_cls (pd.DataFrame): 테스트 분류 레이블
        train_df (pd.DataFrame): 전체 학습 데이터셋
        test_df (pd.DataFrame): 전체 테스트 데이터셋
        test_df_list (List[Tuple[str, pd.DataFrame]]): 각 테스트 기간에 대한 (파일경로, 데이터프레임) 리스트
        train_files (List[str]): 학습 데이터 파일 경로
        test_files (List[str]): 테스트 데이터 파일 경로
        root_path (str): 데이터 및 모델의 루트 디렉토리
        clsmodels (Dict[int, Any]): 학습된 분류 모델 딕셔너리
        models (Dict[int, Any]): 학습된 회귀 모델 딕셔너리
        drop_col_list (List[str]): 낮은 분산 또는 높은 누락률로 인해 삭제된 특성
        n_sector (int): 섹터 수 (PER_SECTOR 모드용)
        sector_list (List[str]): 섹터 이름 리스트 (PER_SECTOR 모드용)
        sector_train_dfs (Dict[str, pd.DataFrame]): 섹터별 학습 데이터
        sector_test_dfs (Dict[str, pd.DataFrame]): 섹터별 테스트 데이터
        sector_test_df_lists (List): 섹터별 테스트 데이터 리스트
        sector_models (Dict[Tuple[str, int], Any]): 섹터별 회귀 모델
        sector_cls_models (Dict): 섹터별 분류 모델
        sector_x_train (Dict[str, pd.DataFrame]): 섹터별 학습 특성
        sector_y_train (Dict[str, pd.DataFrame]): 섹터별 학습 레이블

    사용 예시:
        from config.config_loader import load_config
        conf = load_config('config/config.yaml')

        # 회귀 모델 생성
        regressor = Regressor(conf)

        # 데이터 로드 (분기별 parquet 파일을 자동으로 로드)
        regressor.dataload()

        # 모든 모델 학습 (4개 분류기 + 2개 회귀 모델)
        regressor.train()

        # 테스트 기간에 대해 평가
        regressor.evaluation()

        # 최신 데이터로 예측
        regressor.latest_prediction()
    """

    def __init__(self, conf: Dict[str, Any]) -> None:
        """설정으로 Regressor를 초기화합니다.

        경로, 파일 리스트, 모델 및 데이터를 위한 빈 컨테이너를 설정합니다.
        설정된 연도 범위를 기반으로 분기별 데이터 파일을 자동으로 검색합니다.

        Args:
            conf: 다음 구조의 설정 딕셔너리:
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
            ValueError: ROOT_PATH/ml_per_year/에 학습 데이터 디렉토리가 없는 경우
        """
        self.conf = conf
        self.x_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.x_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        print(self.conf)

        # 중첩된 구조에서 설정 값 추출
        data_config = conf.get('DATA', {})
        ml_config = conf.get('ML', {})
        self.root_path: str = data_config.get('ROOT_PATH', '/home/user/Quant/data')

        aidata_dir = self.root_path + '/ml_per_year/'
        print("aidata path : " + aidata_dir)
        if not os.path.exists(aidata_dir):
            print("there is no ai data : " + aidata_dir)
            return

        # 학습 파일 리스트 생성 (분기별 parquet 파일)
        self.train_files: List[str] = []
        train_start = int(ml_config.get('TRAIN_START_YEAR', 2015))
        train_end = int(ml_config.get('TRAIN_END_YEAR', 2021))
        for year in range(train_start, train_end + 1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # 5-10배 빠른 읽기를 위한 Parquet 형식
                path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.parquet"
                self.train_files.append(path)

        # 테스트 파일 리스트 생성 (분기별 parquet 파일)
        self.test_files: List[str] = []
        test_start = int(ml_config.get('TEST_START_YEAR', 2022))
        test_end = int(ml_config.get('TEST_END_YEAR', 2023))
        for year in range(test_start, test_end + 1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # 5-10배 빠른 읽기를 위한 Parquet 형식
                path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.parquet"
                self.test_files.append(path)

        print("train file list : ", self.train_files)
        print("test file list : ", self.test_files)

        # 데이터 컨테이너 초기화
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.test_df_list: List[Tuple[str, pd.DataFrame]] = []

        # 섹터 기반 예측 속성 (PER_SECTOR 모드용)
        self.n_sector: int = 0
        self.sector_list: List[str] = []
        self.sector_train_dfs: Dict[str, pd.DataFrame] = dict()
        self.sector_test_dfs: Dict[str, pd.DataFrame] = dict()
        self.sector_test_df_lists: List = []

        # 모델 컨테이너
        self.clsmodels: Dict[int, Any] = dict()  # 분류 모델
        self.models: Dict[int, Any] = dict()  # 회귀 모델
        self.sector_models: Dict[Tuple[str, int], Any] = dict()  # 섹터별 모델
        self.sector_cls_models: Dict = dict()

        # 섹터별 학습 데이터
        self.sector_x_train: Dict[str, pd.DataFrame] = dict()
        self.sector_y_train: Dict[str, pd.DataFrame] = dict()

        # 특성 선택 추적
        self.drop_col_list: List[str] = []

    def clean_feature_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """LightGBM과 호환되도록 특성 이름을 정리합니다.

        LightGBM은 특성 이름에 특수 JSON 문자를 지원하지 않으며 고유한 특성 이름이
        필요합니다. 이 메서드는:
            1. 특수 문자를 제거합니다 (영숫자와 밑줄만 유지)
            2. 인덱스를 추가하여 중복 이름을 처리합니다

        Args:
            df: 문제가 있을 수 있는 특성 이름을 가진 DataFrame

        Returns:
            정리된 컬럼 이름을 가진 DataFrame

        사용 예시:
            df = pd.DataFrame({'price@2023': [1, 2], 'price@2024': [3, 4]})
            df = regressor.clean_feature_names(df)
            print(df.columns)
            Index(['price2023', 'price2024'], dtype='object')
        """
        # 컬럼 이름에서 특수 문자 제거
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
        new_n_list = list(new_names.values())

        # 인덱스를 추가하여 중복 이름 처리
        # [LightGBM] Feature appears more than one time.
        new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col
                     for i, (col, new_col) in enumerate(new_names.items())}
        df = df.rename(columns=new_names)
        return df

    def dataload(self) -> None:
        """parquet 파일에서 학습 및 테스트 데이터를 로드하고 특성을 준비합니다.

        이 메서드는 포괄적인 데이터 로드 및 전처리를 수행합니다:
            1. 모든 학습 파일을 로드하고 연결합니다
            2. 의미 없는 특성을 제거합니다 (>80% 누락 또는 >95% 동일한 값)
            3. 과도한 누락 데이터가 있는 행을 필터링합니다 (>60% NaN)
            4. 섹터 기반 가격 편차를 계산합니다 (price_dev에서 섹터 평균을 뺀 값)
            5. 기간별 평가를 위해 테스트 파일을 개별적으로 로드합니다
            6. 학습 및 테스트를 위해 특성(X)과 레이블(y)을 분할합니다

        이 메서드는 경고를 로깅하고 계속 진행하여 누락된 파일을 우아하게 처리합니다.

        Raises:
            ValueError: 학습 데이터 파일을 찾을 수 없는 경우 (치명적 오류)

        부작용:
            - self.train_df, self.test_df, self.test_df_list 설정
            - self.x_train, self.y_train, self.y_train_cls 설정
            - self.x_test, self.y_test, self.y_test_cls 설정
            - 제거된 특성으로 self.drop_col_list 설정
            - 누락된 파일에 대한 경고 로깅
            - 데이터 형상 및 클래스 분포에 대한 정보 로깅

        참고:
            - 누락된 파일은 경고와 함께 건너뜁니다 (치명적이지 않음)
            - 빈 테스트 데이터는 허용됩니다 (평가가 건너뜀)
            - 섹터 매핑은 make_mldata.py에서 수행되어야 합니다 (코드의 TODO 참조)
        """
        # 모든 학습 파일 로드 및 연결
        for fpath in self.train_files:
            print(fpath)
            # 경고와 함께 누락된 파일 건너뛰기
            if not os.path.exists(fpath):
                logging.warning(f"Train file not found, skipping: {fpath}")
                print(f"WARNING: Train file not found, skipping: {fpath}")
                continue
            # Parquet 읽기는 CSV보다 5-10배 빠르고, 70-90% 압축됨
            df = pd.read_parquet(fpath, engine='pyarrow')
            df = df.dropna(axis=0, subset=['price_diff'])
            self.train_df = pd.concat([self.train_df, df], axis=0)

        # 의미 없는 컬럼 제거 (높은 누락률 또는 낮은 분산)
        # 임계값: >80% 누락 OR >95% 동일한 값
        missing_threshold = 0.8
        same_value_threshold = 0.95
        columns_to_drop = []

        for col in self.train_df.columns:
            # 누락률 확인
            missing_ratio = self.train_df[col].isna().mean()
            if missing_ratio > missing_threshold:
                columns_to_drop.append(col)
            else:
                # 단일 값으로 지배되는지 확인 (낮은 분산)
                top_value_ratio = self.train_df[col].value_counts(normalize=True, dropna=False).iloc[0]
                if top_value_ratio > same_value_threshold:
                    columns_to_drop.append(col)

        # 메타데이터 컬럼 삭제 안 함 (y_col_list)
        columns_to_drop = [col for col in columns_to_drop if col not in y_col_list]

        # 특성 제거 적용
        self.train_df = self.train_df.drop(columns=columns_to_drop)
        self.drop_col_list = columns_to_drop
        print(f'Removed columns # : {len(columns_to_drop)}')
        print(f'Cleaned DataFrame shape: {self.train_df.shape}')

        # 과도한 누락 데이터가 있는 행 제거 (>60% NaN)
        print("in train set before dtable len : ", len(self.train_df))
        self.train_df['nan_count_per_row'] = self.train_df.isnull().sum(axis=1)
        filtered_row = self.train_df['nan_count_per_row'] < int(len(self.train_df.columns)*0.6)
        self.train_df = self.train_df.loc[filtered_row,:]
        print("in train set after dtable len : ", len(self.train_df))

        # TODO: 이것은 여기가 아니라 make_mldata.py에서 처리되어야 합니다
        # 섹터 기반 가격 편차 계산 (price_dev에서 섹터 평균을 뺀 값)
        self.train_df["sector"] = self.train_df["industry"].map(sector_map)
        sector_list = list(self.train_df['sector'].unique())
        sector_list = [x for x in sector_list if str(x) != 'nan']
        for sec in sector_list:
            sec_mask = self.train_df['sector'] == sec
            sec_mean = self.train_df.loc[sec_mask, 'price_dev'].mean()
            self.train_df.loc[sec_mask, 'sec_price_dev_subavg'] = self.train_df.loc[sec_mask, 'price_dev'] - sec_mean

        # PER_SECTOR 모드: 섹터별로 학습 데이터 분리
        if PER_SECTOR == True:
            print(self.train_df['sector'].value_counts())
            self.sector_list = list(self.train_df['sector'].unique())
            self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
            for sec in self.sector_list:
                self.sector_train_dfs[sec] = self.train_df[self.train_df['sector']==sec].copy()
                print(self.sector_train_dfs[sec])

        # 기간별 평가를 위해 테스트 파일을 개별적으로 로드
        self.test_df_list = []
        for fpath in self.test_files:
            print(fpath)
            # 경고와 함께 누락된 파일 건너뛰기
            if not os.path.exists(fpath):
                logging.warning(f"Test file not found, skipping: {fpath}")
                print(f"WARNING: Test file not found, skipping: {fpath}")
                continue
            # Parquet 읽기는 CSV보다 5-10배 빠름
            df = pd.read_parquet(fpath, engine='pyarrow')
            df = df.dropna(axis=0, subset=['price_diff'])
            # 학습과 동일한 특성 제거
            df = df.drop(columns=columns_to_drop, errors='ignore')

            # 과도한 누락 데이터가 있는 행 제거
            print("in test set before dtable len : ", len(df))
            df['nan_count_per_row'] = df.isnull().sum(axis=1)
            filtered_row = df['nan_count_per_row'] < int(len(df.columns)*0.6)
            df = df.loc[filtered_row,:]
            print("in test set after dtable len : ", len(df))

            # TODO: 이것은 make_mldata.py에서 처리되어야 합니다
            # 섹터 기반 가격 편차 계산
            df["sector"] = df["industry"].map(sector_map)
            sector_list = list(df['sector'].unique())
            sector_list = [x for x in sector_list if str(x) != 'nan']
            for sec in sector_list:
                sec_mask = df['sector'] == sec
                sec_mean = df.loc[sec_mask, 'price_dev'].mean()
                df.loc[sec_mask, 'sec_price_dev_subavg'] = df.loc[sec_mask, 'price_dev'] - sec_mean

            # 모든 테스트 데이터를 연결하고 기간별 리스트 유지
            self.test_df = pd.concat([self.test_df, df], axis=0)
            self.test_df_list.append([fpath, df])

            # PER_SECTOR 모드: 섹터별로 테스트 데이터 분리
            if PER_SECTOR == True:
                for sec in self.sector_list:
                    self.sector_test_df_lists.append([fpath, df[df['sector']==sec].copy(), sec])

        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)
        logging.debug("test_df shape : ")
        logging.debug(self.test_df.shape)

        # 디버깅을 위해 선택적으로 저장
        # self.train_df.to_csv(self.root_path + '/train_df.csv', index=False)
        # self.test_df.to_csv(self.root_path + '/test_df.csv', index=False)

        # 치명적 오류 확인: 학습 데이터 없음
        if self.train_df.empty:
            error_msg = "❌ FATAL ERROR: No training data available! Cannot train models without data."
            logging.error(error_msg)
            print(f"\n{error_msg}\n")
            raise ValueError("No training data files found. Please check your data directory and configuration.")

        # 클래스 분포 로깅
        positive_count = (self.train_df['price_dev'] > 0).sum()
        negative_count = (self.train_df['price_dev'] < 0).sum()
        logging.info("positive # : {}, negative # : {}".format(positive_count, negative_count))

        # 학습을 위한 특성(X)과 레이블(y) 분할
        self.x_train = self.train_df[self.train_df.columns.difference(y_col_list)]
        self.y_train = self.train_df[['price_dev_subavg']]  # 회귀 타겟 (가격 변동 - 평균)
        self.y_train_cls = self.train_df[['price_dev']]  # 분류 타겟 (이진: 상승/하락)

        # 섹터별 학습 데이터 준비
        for sec in self.sector_list:
            print("sector : ", sec)
            self.sector_x_train[sec] = self.sector_train_dfs[sec][self.sector_train_dfs[sec].columns.difference(y_col_list)]
            self.sector_y_train[sec] = self.sector_train_dfs[sec][['sec_price_dev_subavg']]

        # 테스트 데이터가 없는 경우 처리 (치명적이지 않음, 평가만 건너뜀)
        if self.test_df.empty:
            logging.warning("=" * 80)
            logging.warning("⚠️  No test data available!")
            logging.warning("All test files were missing. Creating empty test datasets.")
            logging.warning("Model evaluation and testing will be skipped.")
            logging.warning("=" * 80)
            print("\n⚠️  WARNING: No test data available. Creating empty test datasets.\n")
            # 학습과 동일한 구조로 빈 테스트 세트 생성
            self.x_test = pd.DataFrame(columns=self.x_train.columns)
            self.y_test = pd.DataFrame(columns=['price_dev_subavg'])
            self.y_test_cls = pd.DataFrame(columns=['price_dev'])
        else:
            # 테스트를 위한 특성과 레이블 분할
            self.x_test = self.test_df[self.test_df.columns.difference(y_col_list)]
            self.y_test = self.test_df[['price_dev_subavg']]
            self.y_test_cls = self.test_df[['price_dev']]

    def def_model(self) -> None:
        """분류 및 회귀 모델을 정의하고 초기화합니다.

        앙상블 예측을 위해 다양한 하이퍼파라미터를 가진 여러 모델 변형을 생성합니다:

        분류 모델 (4개 변형):
            - clsmodels[0]: XGBClassifier, max_depth=8, GPU 가속
            - clsmodels[1]: XGBClassifier, max_depth=9, GPU 가속
            - clsmodels[2]: XGBClassifier, max_depth=10, GPU 가속
            - clsmodels[3]: LGBMClassifier, max_depth=8, GPU 가속

        회귀 모델 (2개 변형):
            - models[0]: XGBRegressor, max_depth=8, GPU 가속
            - models[1]: XGBRegressor, max_depth=10, GPU 가속

        모든 모델은 GPU 가속을 사용합니다 (XGBoost는 tree_method='gpu_hist', LightGBM은 device='gpu').

        참고:
            - LGBMRegressor는 테스트되었지만 낮은 정확도로 인해 비활성화됨
            - Grid search로 최적의 LGB 파라미터를 찾았습니다: learning_rate=0.01, max_depth=6,
              min_child_samples=30, n_estimators=1000, num_leaves=31
            - PER_SECTOR 모드의 경우 섹터별 회귀 모델도 생성합니다

        부작용:
            - 4개의 분류 모델로 self.clsmodels 채우기
            - 2개의 회귀 모델로 self.models 채우기
            - PER_SECTOR=True인 경우 self.sector_models 채우기
        """
        # 분류 모델: 이진 상승/하락 예측
        # 앙상블 다양성을 위해 다양한 깊이를 가진 여러 모델 사용
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

        # LightGBM 분류 모델
        # Grid search로 최적의 파라미터를 찾았습니다:
        # {'learning_rate': 0.01, 'max_depth': 6, 'min_child_samples': 30,
        #  'n_estimators': 1000, 'num_leaves': 31}
        self.clsmodels[3] = lgb.LGBMClassifier(
            boosting_type='gbdt', objective='binary', n_estimators=1000,
            max_depth=8, learning_rate=0.1, device='gpu', boost_from_average=False)

        # 회귀 모델: 가격 변동의 크기 예측
        self.models[0] = xgboost.XGBRegressor(
            tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=8,
            objective='reg:squarederror', eval_metric='rmse')
        self.models[1] = xgboost.XGBRegressor(
            tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=10,
            objective='reg:squarederror', eval_metric='rmse')

        # LightGBM 회귀 모델 (비활성화됨 - 낮은 정확도)
        # self.models[1] = lgb.LGBMRegressor(
        #     boosting_type='gbdt', objective='regression', max_depth=8,
        #     learning_rate=0.1, n_estimators=1000, subsample=0.8,
        #     colsample_bytree=0.8, device='gpu')

        # 섹터별 모델 (PER_SECTOR 모드용)
        if PER_SECTOR:
            for sec in self.sector_list:
                # 섹터당 다양한 하이퍼파라미터를 가진 2개 변형
                cur_key = (sec, 0)
                self.sector_models[cur_key] = xgboost.XGBRegressor(
                    tree_method='gpu_hist', gpu_id=0, n_estimators=1000,
                    learning_rate=0.05, gamma=0.01, subsample=0.8,
                    colsample_bytree=0.7, max_depth=7)  # 최적 하이퍼파라미터
                cur_key = (sec, 1)
                self.sector_models[cur_key] = xgboost.XGBRegressor(
                    tree_method='gpu_hist', gpu_id=0, n_estimators=1000,
                    learning_rate=0.05, gamma=0.01, subsample=0.8,
                    colsample_bytree=0.7, max_depth=8)

    def train(self) -> None:
        """모든 분류 및 회귀 모델을 학습하고 디스크에 저장합니다.

        학습 파이프라인:
            1. def_model()로 모델 초기화
            2. LightGBM 호환성을 위해 특성 이름 정리
            3. 회귀 타겟을 이진 레이블로 변환 (price_dev > 0)
            4. 4개의 분류 모델 학습
            5. 2개의 회귀 모델 학습
            6. 모든 모델을 MODEL_SAVE_PATH에 저장
            7. PER_SECTOR=True인 경우 섹터별 모델 학습

        모든 모델은 나중에 로드하기 위해 joblib을 사용하여 .sav 파일로 저장됩니다.
        학습 점수(분류는 정확도, 회귀는 R²)가 로깅됩니다.

        부작용:
            - MODEL_SAVE_PATH 디렉토리가 없으면 생성
            - 모델을 디스크에 .sav 파일로 저장:
                * clsmodel_0.sav, clsmodel_1.sav, clsmodel_2.sav, clsmodel_3.sav
                * model_0.sav, model_1.sav
                * {sector}_model_0.sav, {sector}_model_1.sav (PER_SECTOR=True인 경우)
            - 모든 모델의 학습 점수 로깅

        참고:
            - 특성 중요도 분석 코드는 주석 처리됨 (필요시 주석 해제)
            - 하이퍼파라미터 튜닝을 위한 Grid search / random search 코드는 주석 처리됨
            - 모델은 속도를 위해 GPU에서 학습됩니다 (CUDA 지원 GPU 필요)
        """
        # 주석 처리됨: LightGBM 하이퍼파라미터 튜닝을 위한 Grid search
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

        # 주석 처리됨: XGBoost 하이퍼파라미터 튜닝을 위한 Random search
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

        # 모델 저장 경로 설정
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'
        self.def_model()

        # 필요시 저장 디렉토리 생성
        if not os.path.exists(MODEL_SAVE_PATH):
            print("creating MODELS path : " + MODEL_SAVE_PATH)
            os.makedirs(MODEL_SAVE_PATH)

        # LightGBM 호환성을 위해 특성 이름 정리
        self.x_train = self.clean_feature_names(self.x_train)

        # 회귀 레이블을 이진 분류 레이블로 변환 (0/1)
        y_train_binary = (self.y_train_cls > 0).astype(int)

        # 모든 분류 모델 학습
        for i, model in self.clsmodels.items():
            logging.info("start fitting classifier")
            model.fit(self.x_train, y_train_binary)
            filename = MODEL_SAVE_PATH + 'clsmodel_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, y_train_binary))

        # 모든 회귀 모델 학습
        for i, model in self.models.items():
            logging.info("start fitting XGBRegressor")
            model.fit(self.x_train, self.y_train.values.ravel())
            filename = MODEL_SAVE_PATH + 'model_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, self.y_train))

            # 주석 처리됨: 특성 중요도 분석
            # logging.info("end fitting RandomForestRegressor")
            # ftr_importances_values = model.feature_importances_
            # ftr_importances = pd.Series(ftr_importances_values, index=self.x_train.columns)
            # ftr_importances.to_csv(MODEL_SAVE_PATH+'model_importances.csv')
            # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            # logging.info(ftr_top20)

        # 섹터별 모델 학습 (PER_SECTOR=True인 경우)
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

                    # 주석 처리됨: 섹터별 특성 중요도 분석
                    # ftr_importances_values = model.feature_importances_
                    # ftr_importances = pd.Series(ftr_importances_values,
                    #                            index=self.sector_x_train[sec].columns)
                    # ftr_importances.to_csv(MODEL_SAVE_PATH + sec + '_model_importances.csv')
                    # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
                    # logging.info(ftr_top20)


    def evaluation(self) -> None:
        """학습된 모델을 테스트 데이터로 평가하고 종합 보고서를 생성합니다.

        이 메서드는 2단계 예측 및 앙상블 투표 전략을 구현합니다:

        평가 파이프라인:
            1. 디스크에서 학습된 모델 로드
            2. 각 테스트 기간(분기별):
                a. 4개의 모든 분류기를 실행하여 이진 예측 얻기
                b. 2개의 모든 회귀 모델을 실행하여 가격 변동 크기 얻기
                c. 분류기를 결합하여 앙상블 예측 생성:
                   - prediction_wbinary_0-3: 각 분류기로 개별적으로 필터링
                   - prediction_wbinary_ensemble: 분류기 1 AND 3으로 필터링
                   - prediction_wbinary_ensemble2: 분류기 1 AND 2로 필터링
                   - prediction_wbinary_ensemble3: 다수결 투표 (3개 중 2개 이상 동의)
                d. 모든 예측 변형에 대한 손실 계산
                e. 각 예측 방법에 대해 상위 K개 주식 선택 (K=3, 7, 15)
                f. 상위 K개 선택에 대한 주식당 평균 수익 계산
            3. 결과를 CSV 파일로 저장
            4. 종합 평가 보고서 생성

        모델 조합:
            2개의 회귀 모델 각각에 대해 8개의 예측 변형을 생성합니다:
                1. model_i_prediction: 원시 회귀 출력
                2-5. model_i_prediction_wbinary_0-3: 각 분류기로 필터링
                   (분류기가 하락을 예측하면 예측값을 -1로 설정)
                6. model_i_prediction_wbinary_ensemble: cls1 AND cls3으로 필터링
                7. model_i_prediction_wbinary_ensemble2: cls1 AND cls2로 필터링
                8. model_i_prediction_wbinary_ensemble3: 다수결 투표 필터

            총: 2개 회귀 모델 × 8개 변형 = 16개 예측 방법

        앙상블 투표 로직:
            - prediction_wbinary_0: 분류기 0의 예측 사용
              cls0이 하락(0)을 예측하면 회귀 출력을 -1로 설정
            - prediction_wbinary_1: 분류기 1의 예측 사용
            - prediction_wbinary_2: 분류기 2의 예측 사용
            - prediction_wbinary_3: 분류기 3의 예측 사용
            - prediction_wbinary_ensemble: cls1 AND cls3 모두 상승을 예측해야 함
              둘 중 하나라도 하락을 예측하면 회귀 출력을 -1로 설정
            - prediction_wbinary_ensemble2: cls1 AND cls2 모두 상승을 예측해야 함
            - prediction_wbinary_ensemble3: 다수결 투표 - 3개 중 최소 2개가 상승을 예측해야 함
              cls1, cls2, cls3를 투표에 사용

        출력 파일 (MODEL_SAVE_PATH에 저장):
            - prediction_ai_{date}.csv: 각 테스트 기간의 예측
            - prediction_ai.csv: 모든 예측 연결
            - pred_df_topk.csv: 모든 모델의 상위 K개 평가 메트릭
            - prediction_{date}_{model}_{col}_top{s}-{e}.csv: 모델당 상위 K개 주식

        부작용:
            - MODEL_SAVE_PATH/*.sav에서 모델 로드
            - MODEL_SAVE_PATH에 평가 CSV 파일 생성
            - 분류 보고서 및 메트릭 로깅
            - 각 예측 방법의 상위 K개 수익 로깅

        참고:
            - 분류기 확률을 이진으로 변환하기 위해 THRESHOLD (기본값 92) 사용
            - 상위 8%의 주식 (100-92=8%)이 양성으로 예측됨
            - 평가에는 기간별 및 누적 메트릭이 모두 포함됨
            - PER_SECTOR=True인 경우 섹터별 모델도 평가합니다
        """
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # 학습된 분류 모델 로드
        self.models = dict()
        self.clsmodels = dict()
        self.clsmodels[0] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_2.sav')
        self.clsmodels[3] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_3.sav')

        # 학습된 회귀 모델 로드
        self.models[0] = joblib.load(MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(MODEL_SAVE_PATH + 'model_1.sav')

        # 모든 예측 컬럼 이름 리스트 (상위 K개 평가용)
        pred_col_list = ['ai_pred_avg']  # 모든 회귀 모델의 평균

        # 모든 모델 조합에 대한 예측 컬럼 이름 생성
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

        model_eval_hist = []  # 모든 기간의 평가 결과 저장
        full_df = pd.DataFrame()  # 예측이 포함된 모든 테스트 데이터 누적

        # 각 테스트 기간을 개별적으로 평가
        for test_idx, (testdate, df) in enumerate(self.test_df_list):

            logging.info("evaluation date : ")
            # 파일 경로에서 날짜 추출
            tmp = testdate.split('\\')
            tmp = [v for v in tmp if v.endswith('.csv')]
            print(f"in test loop tmp : {tmp}")
            tdate = "_".join(tmp[0].split('_')[4:6])
            print(f"in test loop tdate : {tdate}")

            # 이 테스트 기간의 특성과 레이블 준비
            x_test = df[df.columns.difference(y_col_list)]
            y_test = df[['price_dev_subavg']]
            y_test_cls = df[['price_dev']]
            y_test_binary = (y_test_cls > 0).astype(int)

            preds = np.empty((0, x_test.shape[0]))  # 원시 회귀 예측 저장

            df['label'] = y_test  # 실제 가격 변동
            df['label_binary'] = y_test_binary  # 실제 이진 레이블

            # LightGBM을 위해 특성 이름 정리
            x_test = self.clean_feature_names(x_test)

            # === 분류 단계 ===
            # 4개의 모든 분류기를 실행하고 성능 평가
            for i, model in self.clsmodels.items():
                logging.info(f"classification model # {i}")
                pred_col_name = 'clsmodel_' + str(i) + '_prediction'
                correct_col_name = 'clsmodel_' + str(i) + '_correct'

                # 예측 확률 가져오기 (클래스 1의 확률 = 가격 상승)
                y_probs = model.predict_proba(x_test)[:, 1]

                # 백분위수 임계값을 사용하여 확률을 이진 예측으로 변환
                # THRESHOLD=92는 상위 8%가 양성으로 예측됨을 의미
                threshold = np.percentile(y_probs, THRESHOLD)
                y_predict_binary = (y_probs > threshold).astype(int)

                logging.info(f"20% positive threshold == {threshold}")
                logging.info(classification_report(y_test_binary, y_predict_binary))

                # 분류기 예측 저장
                df[pred_col_name] = y_predict_binary
                df[correct_col_name] = (y_test_binary.values.ravel() == y_predict_binary).astype(int)

                acc = accuracy_score(df['label_binary'], df[pred_col_name])
                logging.info(f"Accuracy for {pred_col_name}: {acc:.4f}")


            # === 회귀 단계 ===
            # 2개의 모든 회귀 모델을 실행하고 앙상블 예측 생성
            for i, model in self.models.items():
                # 분류기 출력의 컬럼 이름
                pred_bin_col_name_0 = 'clsmodel_0_prediction'
                pred_bin_col_name_1 = 'clsmodel_1_prediction'
                pred_bin_col_name_2 = 'clsmodel_2_prediction'
                pred_bin_col_name_3 = 'clsmodel_3_prediction'

                # 회귀 출력의 컬럼 이름
                pred_col_name = 'model_' + str(i) + '_prediction'
                correct_col_name = 'clsmodel_' + str(i) + '_correct'
                pred_col_name_wbinary_0 = 'model_' + str(i) + '_prediction_wbinary_0'
                pred_col_name_wbinary_1 = 'model_' + str(i) + '_prediction_wbinary_1'
                pred_col_name_wbinary_2 = 'model_' + str(i) + '_prediction_wbinary_2'
                pred_col_name_wbinary_3 = 'model_' + str(i) + '_prediction_wbinary_3'
                pred_col_name_wbinary_ensemble = 'model_' + str(i) + '_prediction_wbinary_ensemble'
                pred_col_name_wbinary_ensemble2 = 'model_' + str(i) + '_prediction_wbinary_ensemble2'
                pred_col_name_wbinary_ensemble3 = 'model_' + str(i) + '_prediction_wbinary_ensemble3'

                # 예측 오차(손실)의 컬럼 이름
                loss_col_name = 'model_' + str(i) + '_prediction_loss'
                loss_bin_col_name_0 = 'model_' + str(i) + '_prediction_wbinary_loss_0'
                loss_bin_col_name_1 = 'model_' + str(i) + '_prediction_wbinary_loss_1'
                loss_bin_col_name_2 = 'model_' + str(i) + '_prediction_wbinary_loss_2'
                loss_bin_col_name_3 = 'model_' + str(i) + '_prediction_wbinary_loss_3'

                # 원시 회귀 예측 가져오기
                y_predict = model.predict(x_test)

                # 원시 회귀 예측 저장
                df[pred_col_name] = y_predict

                # 분류기와 결합하여 필터링된 예측 생성
                # 분류기가 하락(0)을 예측하면 회귀 출력을 -1로 대체
                df[pred_col_name_wbinary_0] = np.where(df[pred_bin_col_name_0] == 0, -1, y_predict)
                df[pred_col_name_wbinary_1] = np.where(df[pred_bin_col_name_1] == 0, -1, y_predict)
                df[pred_col_name_wbinary_2] = np.where(df[pred_bin_col_name_2] == 0, -1, y_predict)
                df[pred_col_name_wbinary_3] = np.where(df[pred_bin_col_name_3] == 0, -1, y_predict)

                # 앙상블 1: 분류기 1 AND 3 모두 상승을 예측해야 함
                df[pred_col_name_wbinary_ensemble] = np.where(
                    ((df[pred_bin_col_name_1] == 0) | (df[pred_bin_col_name_3] == 0)),
                    -1, y_predict)

                # 앙상블 2: 분류기 1 AND 2 모두 상승을 예측해야 함
                df[pred_col_name_wbinary_ensemble2] = np.where(
                    ((df[pred_bin_col_name_1] == 0) | (df[pred_bin_col_name_2] == 0)),
                    -1, y_predict)

                # 앙상블 3: 다수결 투표 - 3개 중 최소 2개가 상승을 예측해야 함
                # 2개 이상의 분류기가 하락(0)을 예측하면 회귀 출력을 -1로 대체
                condition = (
                    (df[[pred_bin_col_name_1, pred_bin_col_name_2, pred_bin_col_name_3]] == 0).sum(axis=1) >= 2
                )
                df[pred_col_name_wbinary_ensemble3] = np.where(condition, -1, y_predict)

                # 평균화를 위한 원시 예측 저장
                preds = np.vstack((preds, y_predict[None,:]))

                # 모든 변형에 대한 예측 오차(손실) 계산
                df[loss_col_name] = abs(df['label'] - y_predict)
                df[loss_bin_col_name_0] = abs(df['label'] - df[pred_col_name_wbinary_0])
                df[loss_bin_col_name_1] = abs(df['label'] - df[pred_col_name_wbinary_1])
                df[loss_bin_col_name_2] = abs(df['label'] - df[pred_col_name_wbinary_2])
                df[loss_bin_col_name_3] = abs(df['label'] - df[pred_col_name_wbinary_3])

                # 이 기간의 평가 메트릭 로깅
                logging.info(f"eval : model i : {i} loss : {df[loss_col_name].mean()} "
                           f"loss_wbin_0 {df[loss_bin_col_name_0].mean()} "
                           f"loss_wbin_1 {df[loss_bin_col_name_1].mean()} "
                           f"loss_wbin_2 {df[loss_bin_col_name_2].mean()} "
                           f"loss_wbin_3 {df[loss_bin_col_name_3].mean()}")

                # 누적 메트릭 로깅 (지금까지의 모든 기간)
                if test_idx != 0:
                    logging.info(f"accumulated eval : model i : {i} "
                               f"loss : {full_df[loss_col_name].mean()} "
                               f"loss_wbin_0 {full_df[loss_bin_col_name_0].mean()} "
                               f"loss_wbin_1 {full_df[loss_bin_col_name_1].mean()} "
                               f"loss_wbin_2 {full_df[loss_bin_col_name_2].mean()} "
                               f"loss_wbin_3 {full_df[loss_bin_col_name_3].mean()}")

            # 모든 회귀 모델의 평균 예측 계산
            df['ai_pred_avg'] = np.average(preds, axis=0)
            df['ai_pred_avg_loss'] = abs(df['label']-df['ai_pred_avg'])

            # 결과 누적
            full_df = pd.concat([full_df, df], ignore_index=True)
            df.to_csv(MODEL_SAVE_PATH + "prediction_ai_{}.csv".format(tdate))

            # === 상위 K개 주식 선택 ===
            # 각 예측 방법에 대해 상위 K개 주식을 선택하고 평균 수익 계산
            topk_period_earning_sums = []
            topk_list = [(0,3), (0,7), (0,15)]  # 상위 3, 7, 15개 주식

            for s, e in topk_list:
                logging.info("top" + str(s) + " ~ "  + str(e) )
                k = str(s) + '~' + str(e)

                # 각 예측 방법 평가
                for col in pred_col_list:
                    # 예측을 기반으로 상위 K개 주식 선택
                    top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]

                    logging.info("")
                    logging.info(col)
                    logging.info(("label"))
                    logging.info((top_k_df['price_dev'].sum()/(e-s+1)))
                    logging.info(("pred"))
                    logging.info((top_k_df[col].sum()/(e-s+1)))
                    topk_period_earning_sums.append(top_k_df['price_dev'].sum())

                    # 상위 K개 주식을 CSV로 저장
                    top_k_df.to_csv(MODEL_SAVE_PATH+'prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))

                    # 이 모델 및 상위 K개 범위에 대한 평가 메트릭 기록
                    model_eval_hist.append([
                        tdate, col, k,
                        top_k_df['price_dev'].sum()/(e-s+1),  # 주식당 평균 실제 수익
                        top_k_df[col].sum()/(e-s+1),  # 주식당 평균 예측 수익
                        abs(top_k_df[col].sum()/(e-s+1) - top_k_df['price_dev'].sum()/(e-s+1)),  # 손실
                        int(top_k_df[col].sum()/(e-s+1) > 0),  # 예측이 양수인가?
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

        # 종합 평가 보고서 생성
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

        # === 섹터 기반 평가 (PER_SECTOR=True인 경우) ===
        if PER_SECTOR == True:
            testdates = set()
            allsector_topk_df = pd.DataFrame()
            self.sector_models = dict()

            # 섹터별 모델 로드
            for sec in self.sector_list:
                for i in range(2):
                    filename = MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))
                    k = (sec, i)
                    self.sector_models[k] = joblib.load(MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))

            sector_model_eval_hist = []

            # 각 섹터 및 테스트 기간 평가
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

                # 섹터 기반 필터링을 위해 분류기 2 사용
                y_probs = self.clsmodels[2].predict_proba(x_test)[:, 1]
                threshold = np.percentile(y_probs, THRESHOLD)
                y_predict_binary = (y_probs > threshold).astype(int)

                # 섹터별 모델 실행
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

                # 섹터별 예측의 상위 K개 평가
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
        """주식 선택을 위해 가장 최근 데이터로 예측합니다.

        이 메서드는 최신 분기별 데이터를 로드하고 현재 주식 선택을 위한 예측을
        생성합니다. evaluation()과 동일한 2단계 예측 및 앙상블 투표 전략을 따르지만
        과거 테스트 데이터가 아닌 가장 최근 데이터로만 작업합니다.

        파이프라인:
            1. 모든 학습된 분류 및 회귀 모델 로드
            2. 최신 연도 데이터(모든 분기)를 읽고 심볼당 가장 최근 것 유지
            3. 충분한 데이터가 있는 주식으로 필터링 (>60% non-NaN)
            4. 4개의 분류 모델을 실행하여 이진 예측 얻기
            5. 2개의 회귀 모델을 실행하여 가격 변동 크기 얻기
            6. 다양한 투표 전략을 사용하여 앙상블 예측 생성
            7. 상위 K개 주식 추천 생성 (K=3, 7, 15)
            8. 예측을 CSV 파일로 저장

        출력 파일 (MODEL_SAVE_PATH에 저장):
            - latest_prediction.csv: 최신 데이터의 모든 예측
            - latest_prediction_{model}_{col}_top{s}-{e}.csv: 모델당 상위 K개 주식
            - sec_{sector}_latest_prediction.csv: 섹터별 예측 (PER_SECTOR=True인 경우)
            - allsector_latest_pred_df.csv: 섹터 기반 상위 K개 요약 (PER_SECTOR=True인 경우)

        부작용:
            - MODEL_SAVE_PATH/*.sav에서 모델 로드
            - MODEL_SAVE_PATH에 예측 CSV 파일 생성
            - 예측 임계값 및 상위 K개 범위 로깅

        참고:
            - year_period를 사용하여 심볼당 가장 최근 데이터만 유지
            - 분류에는 evaluation()과 동일한 THRESHOLD (92) 적용
            - PER_SECTOR=True인 경우 섹터별 예측 사용 가능
            - FIXME: 2024 데이터를 읽도록 하드코딩됨 (설정 가능해야 함)
        """
        MODEL_SAVE_PATH = self.root_path + '/MODELS/'

        # 학습된 모델 로드
        self.clsmodels = dict()
        self.clsmodels[0] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_2.sav')
        self.clsmodels[3] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_3.sav')
        self.models = dict()
        self.models[0] = joblib.load(MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(MODEL_SAVE_PATH + 'model_1.sav')

        aidata_dir = self.root_path + '/ml_per_year/'

        # 예측 컬럼 리스트 생성 (evaluation과 동일)
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

        # 최신 연도 데이터(모든 분기)를 로드하고 심볼당 가장 최근 것 유지
        # FIXME: 2024로 하드코딩됨 - 설정 가능해야 함
        ldf = pd.DataFrame()
        for i in [1,2,3,4]:
            latest_data_path = aidata_dir + f'rnorm_fs_2024_Q{i}.csv'
            df = pd.read_csv(latest_data_path)
            ldf = pd.concat([ldf, df], axis=0)

        # year_period를 기준으로 내림차순 정렬하고 심볼당 첫 번째(가장 최근) 유지
        ldf = ldf.sort_values(by='year_period', ascending=False)
        ldf = ldf.drop_duplicates(subset='symbol', keep='first')
        ldf = ldf.drop(columns=self.drop_col_list, errors='ignore')

        # 첫 번째 컬럼 제거 (CSV의 인덱스 컬럼)
        # FIXME: rnorm_fs*.csv 파일은 첫 번째 컬럼에 인덱스가 있음
        ldf = ldf.drop(df.columns[0], axis=1)

        # 섹터 리스트 추출
        self.sector_list = list(ldf['sector'].unique())
        self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
        ldf = ldf.drop('sector', axis=1)

        # 과도한 누락 데이터가 있는 행 필터링 (>60% NaN)
        print("before dtable len : ", len(ldf))
        ldf['nan_count_per_row'] = ldf.isnull().sum(axis=1)
        filtered_row = ldf['nan_count_per_row'] < int(len(ldf.columns)*0.6)
        ldf = ldf.loc[filtered_row,:]
        print("after dtable len : ", len(ldf))

        # 입력 특성 준비
        input = ldf[ldf.columns.difference(y_col_list)]
        input = self.clean_feature_names(input)
        preds = np.empty((0, input.shape[0]))

        # === 분류 단계 ===
        # 모든 분류기 실행
        for i, model in self.clsmodels.items():
            logging.info(f"classification model # {i}")
            pred_col_name = 'clsmodel_' + str(i) + '_prediction'
            y_probs = model.predict_proba(input)[:, 1]
            # 백분위수 임계값을 사용하여 이진으로 변환
            threshold = np.percentile(y_probs, THRESHOLD)
            y_predict_binary = (y_probs > threshold).astype(int)
            logging.info(f"20% positive threshold == {threshold}")
            ldf[pred_col_name] = y_predict_binary

        # === 회귀 단계 ===
        # 모든 회귀 모델을 실행하고 앙상블 예측 생성
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

            # 원시 회귀 예측 가져오기
            y_predict = model.predict(input)

            # 원시 예측 저장
            ldf[pred_col_name] = y_predict

            # 분류기 출력을 사용하여 필터링된 예측 생성
            ldf[pred_col_name_wbinary_0] = np.where(ldf[pred_bin_col_name_0] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_1] = np.where(ldf[pred_bin_col_name_1] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_2] = np.where(ldf[pred_bin_col_name_2] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_3] = np.where(ldf[pred_bin_col_name_3] == 0, -1, y_predict)

            # 다양한 투표 전략을 사용한 앙상블 예측
            ldf[pred_col_name_wbinary_ensemble] = np.where(
                ((ldf[pred_bin_col_name_1] == 0) | (ldf[pred_bin_col_name_3] == 0)),
                -1, y_predict)
            ldf[pred_col_name_wbinary_ensemble2] = np.where(
                ((ldf[pred_bin_col_name_1] == 0) | (ldf[pred_bin_col_name_2] == 0)),
                -1, y_predict)

            # 다수결 투표: 3개 중 최소 2개가 상승을 예측해야 함
            condition = (
                (ldf[[pred_bin_col_name_1, pred_bin_col_name_2, pred_bin_col_name_3]] == 0).sum(axis=1) >= 2
            )
            ldf[pred_col_name_wbinary_ensemble3] = np.where(condition, -1, y_predict)
            preds = np.vstack((preds, y_predict[None,:]))

        # 평균 예측 계산
        ldf['ai_pred_avg'] = np.average(preds, axis=0)
        ldf.to_csv(MODEL_SAVE_PATH+"latest_prediction.csv")

        # 상위 K개 주식 추천 생성
        topk_list = [(0,3), (0,7), (0, 15)]
        for s, e in topk_list:
            logging.info("top" + str(s) + " ~ " + str(e))
            for col in pred_col_list:
                top_k_df = ldf.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                top_k_df.to_csv(MODEL_SAVE_PATH+'latest_prediction_{}_top{}-{}.csv'.format(col, s, e))

        # === 섹터별 예측 (PER_SECTOR=True인 경우) ===
        if PER_SECTOR == True:
            self.sector_models = dict()
            ldf = pd.read_csv(latest_data_path)

            # 섹터별 모델 로드
            for sec in self.sector_list:
                for i in range(2):
                    filename = MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))
                    k = (sec, i)
                    print("model path : ", MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))
                    self.sector_models[k] = joblib.load(MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))

            all_preds = []

            # 섹터별로 예측 수행
            for sec in self.sector_list:
                sec_df = ldf[ldf['sector']==sec]
                sec_df = sec_df.drop('sector', axis=1)
                indata = sec_df[sec_df.columns.difference(['symbol'])]
                print(indata)
                preds = np.empty((0, indata.shape[0]))

                # 섹터별 모델 실행
                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    y_predict3 = model.predict(indata)
                    sec_df[pred_col_name] = y_predict3
                    preds = np.vstack((preds, y_predict3[None,:]))

                sec_df['ai_pred_avg'] = np.average(preds, axis=0)
                sec_df.to_csv(MODEL_SAVE_PATH+"sec_{}_latest_prediction.csv".format(sec))

                # 섹터별 상위 K개
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

            # 섹터 기반 요약 저장
            col_name = ['k', 'sector', 'model', 'i', 'symbol', 'pred']
            pred_df = pd.DataFrame(all_preds, columns=col_name)
            pred_df.to_csv(MODEL_SAVE_PATH+'allsector_latest_pred_df.csv', index=False)
