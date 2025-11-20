import glob
import joblib
import logging
import os
import re
import pickle
import time
import datetime 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta
import lightgbm as lgb
import catboost
from catboost import CatBoostClassifier, CatBoostRegressor

from g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map, sparse_col_list
from sklearn.ensemble import RandomForestRegressor

import xgboost
from xgboost.callback import TrainingCallback
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from scipy.stats.mstats import winsorize
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# TODO: loss가 가장 작은 모델을 선택해야되나,..? 
# TODO: SECTOR 별로 결과가 나왔을 때, 이를 어떻게 합칠 것인가 -> SECTOR 중 TOP N만 뽑아서 그 수익률 ? 구현필요
# SECTOR 예측값합 SECTOR TOP 3~4 뽑아 사고 계산
PER_SECTOR=False
THRESHOLD=92
XGBOOST_MAX_VALUE = np.finfo(np.float32).max
DROP_NAN_ROW_THRESHOLD = 0.9


# PyTorch Dataset 정의
class FinancialDataset(Dataset):
    def __init__(self, X, y, sector):
        self.X = X  # 이미 변환된 데이터
        self.y = y
        self.sector = sector.astype(int)  # Ensure it's integer
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32), 
            torch.tensor(self.y[idx], dtype=torch.float32), 
            torch.tensor(self.sector[idx], dtype=torch.long)
        )

# 모델 정의
class FinancialClassifier(nn.Module):
    def __init__(self, input_dim, num_sectors, hidden_dim=4096):
        super(FinancialClassifier, self).__init__()
        self.embedding = nn.Embedding(num_sectors, 10)  # sector categorical feature embedding
        self.fc1 = nn.Linear(input_dim + 10, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, sector):        
        sector_embed = self.embedding(sector)
        x = torch.cat([x, sector_embed], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class FinancialRegressor(nn.Module):
    def __init__(self, input_dim, num_sectors, hidden_dim=4096):
        super(FinancialRegressor, self).__init__()
        self.embedding = nn.Embedding(num_sectors, 10)
        self.fc1 = nn.Linear(input_dim + 10, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x, sector):
        sector_embed = self.embedding(sector)
        x = torch.cat([x, sector_embed], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc3(x)

# Training Function
def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets, sectors in train_loader:
            inputs, targets, sectors = inputs.to(device), targets.to(device), sectors.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, sectors).squeeze()
            # print("Outputs min:", outputs.min().item(), "Outputs max:", outputs.max().item())
            
            outputs = outputs.view(-1, 1)  
            targets = targets.float()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(running_loss)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
    return model


# 입력으로 넣으면 안되는 colume들이라 중간에 drop하기 위한 list
y_col_list= [
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
    "price_diff_prediction",
    "price_dev_prediction",
    "volume_mul_price",
    "report_date",
    "fillingDate_x",
    # "sector",
    "price_dev_subavg",
    "sec_price_dev_subavg",
    "price_diff_3month",
    "price_diff_6month",
    "year_period",
    "date"
]
except_col_list = [
    # "dividendsperShareGrowth"
]

class TimeLogger(TrainingCallback):
    """ 100 Iteration마다 학습 진행 상황과 예상 남은 시간을 출력하는 Callback """
    def __init__(self, total_rounds, log_interval=100):
        self.total_rounds = total_rounds
        self.start_time = None
        self.log_interval = log_interval  # 몇 Iteration마다 로그를 찍을지 설정

    def before_training(self, model):
        self.start_time = time.time()
        print(f"🚀 Training started! Total {self.total_rounds} rounds")
        return model  # ✅ XGBoost가 요구하는 `Booster` 객체 반환

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.log_interval == 0:  # ✅ 100 Iteration 마다만 출력
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time / (epoch + 1) * self.total_rounds
            remaining_time = estimated_total_time - elapsed_time

            print(f"🔄 Iteration {epoch+1}/{self.total_rounds} - "
                  f"Elapsed: {elapsed_time:.2f}s, "
                  f"Remaining: {remaining_time:.2f}s")

        return False  # 학습을 계속 진행

    
class Regressor:   
    
    def __init__(self, conf):
        self.conf = conf
        print(self.conf)
        self.MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS/'
        
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.last_x_train = None
        self.last_y_train = None
        self.last_x_test = None
        self.last_y_test = None
        
        self.train_file_list = []
        self.last_train_file_list = []
        self.test_file_list = []
        
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.last_train_df = pd.DataFrame()
        self.test_df_list = []
    
        self.models = dict()  
        self.clsmodels = dict()              
        
        self.drop_col_list = []                
        
        self.final_top_40_percent_average = 0
        self.upper_bound = 0
        self.lower_bound = 0
        # for sector
        self.n_sector = 0
        self.sector_list = []
        
        self.sector_x_train = dict()
        self.sector_y_train = dict()
        self.sector_x_test = dict()
        self.sector_y_test = dict()        
        
        self.sector_train_dfs = dict()
        self.sector_test_dfs = dict()
        self.sector_test_df_lists = []
        
        self.sector_models = dict()
        self.sector_cls_models=dict()
        
        self.load_ai_data_list()

    def def_model(self):
    
        # 파라미터	역할	적절한 값 범위	현재 설정	추천 값
        # gamma	불필요한 분할 방지	0.1 ~ 0.5	0.1	0.2 ~ 0.3
        # subsample	데이터 샘플링 비율	0.7 ~ 0.9	0.8 ~ 0.9	✅ 유지 (0.8 ~ 0.9)
        # colsample_bynode	Feature 샘플링	0.4 ~ 0.7	0.4 ~ 0.8	0.5 ~ 0.7
        num_rounds = 1000

        self.clsmodels[0] = CatBoostClassifier(iterations=num_rounds, learning_rate=0.05, depth=6, 
                                                            loss_function='Logloss', eval_metric='Logloss', grow_policy="Lossguide",
                                                            random_strength=2, l2_leaf_reg=10, colsample_bylevel=0.5,
                                                            max_ctr_complexity=2,
                                                            allow_writing_files=True,  # 중간 결과를 캐싱하여 속도 최적화
                                                            verbose=100, task_type="CPU")           
        
        self.clsmodels[1] = xgboost.XGBClassifier(#tree_method='gpu_hist', gpu_id=0, 
                                                  device='cuda', n_estimators=num_rounds, learning_rate=0.05, 
                                        max_bin=128, max_depth=7, objective='binary:logistic', eval_metric=['logloss', 'aucpr'], 
                                        grow_policy="lossguide",
                                        colsample_bynode=0.8, min_child_weight=3, reg_lambda=3, gamma=0.2, subsample=0.9,
                                        enable_categorical=True, 
                                        callbacks=[TimeLogger(num_rounds, log_interval=100)])
        
        # self.clsmodels[2]  = DL
        

        self.models[0] = CatBoostRegressor(iterations=num_rounds, learning_rate=0.05, depth=8, max_bin=128, grow_policy="Lossguide",
                                                    loss_function='RMSE', eval_metric='RMSE', 
                                                    random_strength=3, l2_leaf_reg=10, colsample_bylevel=0.5,
                                                    max_ctr_complexity=2,
                                                    allow_writing_files=True,  # 중간 결과를 캐싱하여 속도 최적화
                                                    verbose=100, task_type="CPU")
        
        self.models[1] = xgboost.XGBRegressor(#tree_method='gpu_hist', gpu_id=0, 
                                              device='cuda', n_estimators=num_rounds, learning_rate=0.05, 
                                        max_depth=8, objective='reg:squarederror', eval_metric=['rmse', 'rmsle'], 
                                        # grow_policy="lossguide", # max_bin=256,
                                        min_child_weight=3, colsample_bynode=0.8, subsample=0.9, reg_lambda=2, # gamma=0.2, 
                                        enable_categorical=True, 
                                        callbacks=[TimeLogger(num_rounds, log_interval=100)])    
        
        # self.models[2]  = DL
        

        # self.clsmodels[2] = xgboost.XGBClassifier(#tree_method='gpu_hist', gpu_id=0, 
        #                                           device='cuda', n_estimators=num_rounds, learning_rate=0.05, 
        #                                 max_bin=128, max_depth=8, objective='binary:logistic', eval_metric=['logloss', 'aucpr'], 
        #                                 grow_policy="lossguide",
        #                                 colsample_bynode=0.7, min_child_weight=3, reg_lambda=3, gamma=0.2, subsample=0.9,
        #                                 enable_categorical=True, 
        #                                 callbacks=[TimeLogger(num_rounds, log_interval=100)])
        

        # self.clsmodels[0] = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5, boosting_type="Plain", 
        #                                                     loss_function='Logloss', eval_metric='Logloss', 
        #                                                     random_strength=2, l2_leaf_reg=10, colsample_bylevel=0.5,
        #                                                     early_stopping_rounds=100,  # 100번 동안 성능 향상 없으면 학습 종료    
        #                                                     max_ctr_complexity=2,	
        #                                                     allow_writing_files=True,  # 중간 결과를 캐싱하여 속도 최적화
        #                                                     verbose=100, task_type="CPU")
        
        # self.clsmodels[1] = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=5,  max_bin=128,  boosting_type="Ordered", 
        #                                                     loss_function='Logloss', eval_metric='Logloss', 
        #                                                     random_strength=2, l2_leaf_reg=10, colsample_bylevel=0.7,
        #                                                     early_stopping_rounds=100,  # 100번 동안 성능 향상 없으면 학습 종료
        #                                                     max_ctr_complexity=2,
        #                                                     allow_writing_files=True,  # 중간 결과를 캐싱하여 속도 최적화
        #                                                     verbose=100, task_type="CPU")            
        
        # self.clsmodels[2] = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, 
        #                                                     loss_function='Logloss', eval_metric='Logloss', grow_policy="Lossguide",
        #                                                     random_strength=2, l2_leaf_reg=10, colsample_bylevel=0.5,
        #                                                     early_stopping_rounds=100,  # 100번 동안 성능 향상 없으면 학습 종료
        #                                                     max_ctr_complexity=2,
        #                                                     allow_writing_files=True,  # 중간 결과를 캐싱하여 속도 최적화
        #                                                     verbose=100, task_type="CPU")           

        # self.models[0] = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=7, max_bin=128, boosting_type="Ordered", 
        #                                             loss_function='RMSE', eval_metric='RMSE', 
        #                                             random_strength=2, colsample_bylevel=0.9,
        #                                             early_stopping_rounds=100,  # 100번 동안 성능 향상 없으면 학습 종료
        #                                             max_ctr_complexity=2,
        #                                             allow_writing_files=True,  # 중간 결과를 캐싱하여 속도 최적화
        #                                             verbose=100, task_type="CPU")

        # self.models[1] = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8, max_bin=128, grow_policy="Lossguide",
        #                                             loss_function='RMSE', eval_metric='RMSE', 
        #                                             random_strength=3, l2_leaf_reg=10, colsample_bylevel=0.5,
        #                                             early_stopping_rounds=100,  # 100번 동안 성능 향상 없으면 학습 종료
        #                                             max_ctr_complexity=2,
        #                                             allow_writing_files=True,  # 중간 결과를 캐싱하여 속도 최적화
        #                                             verbose=100, task_type="CPU")


    
        # for sec in self.sector_list:
        #     cur_key = (sec, 0)
        #     self.sector_cls_models[cur_key] = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0.2, subsample=0.8,
        #                                 colsample_bynode=0.5, max_depth=6, objective='binary:logistic', eval_metric='logloss', min_child_weight=3)
        #     cur_key = (sec, 1)
        #     self.sector_cls_models[cur_key]  = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0.4, subsample=0.8,
        #                                 colsample_bynode=0.5, max_depth=7, objective='binary:logistic', eval_metric='logloss', min_child_weight=5)
        #     # cur_key = (sec, 2)
        #     # self.sector_cls_models[cur_key]  = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0.4, subsample=0.8,
        #     #                             colsample_bytree=0.8, max_depth=8, objective='binary:logistic', eval_metric='logloss', min_child_weight=5)
            
        #     cur_key = (sec, 0)
        #     self.sector_models[cur_key] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.1, gamma=0.01, subsample=0.8,
        #                                 colsample_bynode=0.5, max_depth=5) #BEST
        #     cur_key = (sec, 1)
        #     self.sector_models[cur_key]  = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.1, gamma=0.05, subsample=0.8,
        #                                 colsample_bynode=0.5, max_depth=6)


    def load_ai_data_list(self):
        conf = self.conf
        #TODO: conf로 빼기
        AIDATA_DIR_PATH = conf['ROOT_PATH'] + '/ml_per_year/'
        print("aidata path : " + AIDATA_DIR_PATH)
        if not os.path.exists(AIDATA_DIR_PATH):
            print("there is no ai data : " + AIDATA_DIR_PATH)
            return 
            
        for year in range(int(conf['TRAIN_START_YEAR']), int(conf['TRAIN_END_YEAR'])+1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # if year == 2024:
                #     if Q == 'Q1' or Q == 'Q2':
                #         path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.csv"
                #         self.train_file_list.append(path)
                # else:
                path = AIDATA_DIR_PATH + "rnorm_ml_" + str(year) + f"_{Q}.csv"
                self.train_file_list.append(path)


        for year in range(int(conf['TEST_START_YEAR']), int(conf['TEST_END_YEAR'])+1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # if year == 2024:
                #     if Q == 'Q1' or Q == 'Q2':
                #         path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.csv"
                #         self.train_file_list.append(path)
                # else:
                path = AIDATA_DIR_PATH + "rnorm_ml_" + str(year) + f"_{Q}.csv"
                self.last_train_file_list.append(path)
        
        for year in range(int(conf['TEST_START_YEAR']), int(conf['TEST_END_YEAR'])+1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                if year == 2024:
                    if Q != 'Q4':
                        path = AIDATA_DIR_PATH + "rnorm_ml_" + str(year) + f"_{Q}.csv"
                        self.test_file_list.append(path)
                else:
                    path = AIDATA_DIR_PATH + "rnorm_ml_" + str(year) + f"_{Q}.csv"
                    self.test_file_list.append(path)
                
        print("train file list : ", self.train_file_list)
        print("test file list : ", self.test_file_list)
        

    def except_col_df(self, df):
        filtered_columns = [col for col in df.columns if not any(except_str in col for except_str in except_col_list)]
        filtered_df = df[filtered_columns]
        return filtered_df

    def drop_many_nan_row(self, df, threshold = DROP_NAN_ROW_THRESHOLD):
        # 40%도 값이 없는 row 날리기(60%까지 nan이여도 남김)
        # 정리하면, threshold = 0.8이라는 것은 결과적으로 NaN 비율이 20%를 넘어가면 해당 row를 제거하는 로직입니다.  
        
        df['nan_count_per_row'] = df.isnull().sum(axis=1)        
        filtered_row = df['nan_count_per_row'] < int(len(df.columns)*(1-threshold)) 
        
        print(f"in drop_many_nan_row before df : {len(df)}, # filtered_row {filtered_row.sum()} # drop {len(df)- len(filtered_row)}")
        print(f"in drop_many_nan_row before df : {len(df)}, # filtered_row {filtered_row.sum()} # drop {len(df)- len(filtered_row)}")
        print(f"in drop_many_nan_row before df : {len(df)}, # filtered_row {filtered_row.sum()} # drop {len(df)- len(filtered_row)}")
        
        df = df.loc[filtered_row,:]
        return df
    
    def dataload(self):
        
        top_40_percent_values = []

        for fpath in self.train_file_list:
            print(fpath)
            df = pd.read_csv(fpath, low_memory=False)
            # y값 없는 row 버리기
            df = df.dropna(axis=0, subset=['price_diff_prediction'])             
            # 원래 데이터프레임에서 volume_mul_price 상위 20%(학습 데이터처음 생성시 조건 있음) * 50% (전체 10%) 에 해당하는 symbol의 데이터만 남김
            symbol_means = df.groupby('symbol')['volume_mul_price'].mean().reset_index()
            top_40_percent_value = symbol_means['volume_mul_price'].quantile(0.50)  # 상위 40%에 해당하는 경계값
            top_40_percent_values.append(top_40_percent_value)

            top_symbols = symbol_means.nlargest(int(len(symbol_means) * 0.50), 'volume_mul_price')
            df = df[df['symbol'].isin(top_symbols['symbol'])]

            self.train_df = pd.concat([self.train_df, df], axis=0)

        self.train_df = self.drop_many_nan_row(self.train_df, 0.6)

        # 이거랑 drop_col_list 랑 같이 json으로 저장해서 latest_prediction에서 불러오기
        self.final_top_40_percent_average = sum(top_40_percent_values) / len(top_40_percent_values) if top_40_percent_values else 0

        # Save the value to a text file
        with open("final_top_40_percent_average.txt", "w") as file:
            file.write(str(self.final_top_40_percent_average))
        # Read the value back from the text file
        with open("final_top_40_percent_average.txt", "r") as file:
            read_value = file.read()
            self.final_top_40_percent_average = float(read_value)
            
        # START 의미없는 column 날리기
        # TODO: 기준 설정
        missing_threshold = 0.8
        same_value_threshold = 0.98
        # 의미 없는 열을 찾기 위한 리스트 초기화
        columns_to_drop = []
        columns_to_drop_missing = []
        columns_to_drop_same = []
        # 각 열에 대해 비어 있는 값의 비율 계산 및 기준 초과하는 열 추가
        for col in self.train_df.columns:
            missing_ratio = self.train_df[col].isna().mean()
            if missing_ratio > missing_threshold:
                # 예를 들어 missing_threshold가 0.4라면, 결측치 비율이 40%를 넘는 컬럼은 제거 대상이 됩니다.
                columns_to_drop.append(col)
                columns_to_drop_missing.append(col)
            else:
                # 고유 값의 비율 계산
                top_value_ratio = self.train_df[col].value_counts(normalize=True, dropna=False).iloc[0]
                if top_value_ratio > same_value_threshold:
                    columns_to_drop.append(col)
                    columns_to_drop_same.append(col)
        columns_to_drop = [col for col in columns_to_drop if col not in y_col_list]
        print(f"missing ratio drop : {len(columns_to_drop_missing)}")
        print(f"same ratio drop : {len(columns_to_drop_same)}")
        self.train_df = self.train_df.drop(columns=columns_to_drop)
        # 날린 column은 test나 마지막 예측에서도 날려줘야함
        self.drop_col_list = columns_to_drop
        with open("./drop_col_list.json", 'w') as f:
            # indent=2 is not needed but makes the file human-readable 
            # if the data is nested
            json.dump(columns_to_drop, f, indent=2)    
        print(f'Removed columns # : {len(columns_to_drop)}')
        print(f'Cleaned DataFrame shape: {self.train_df.shape}')
        print("in train set before dtable len : ", len(self.train_df))
        # END 의미없는 column 날리기           
        self.train_df = self.drop_many_nan_row(self.train_df, 0.95)           
        # self.train_df = self.train_df.fillna(0)

        if PER_SECTOR == True:
            print(self.train_df['sector'].value_counts())
            self.sector_list = list(self.train_df['sector'].unique())
            self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
            for sec in self.sector_list:
                self.sector_train_dfs[sec] = self.train_df[self.train_df['sector']==sec].copy()
                print(self.sector_train_dfs[sec])

        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)

        positive_count = (self.train_df['price_dev_prediction'] > -0.02).sum()
        negative_count = (self.train_df['price_dev_prediction'] < -0.02).sum()
        logging.info("positive # : {}, negative # : {}".format(positive_count, negative_count))


        self.x_train = self.train_df[self.train_df.columns.difference(y_col_list)]
        self.x_train = self.except_col_df(self.x_train)

        self.y_train = self.train_df[['price_dev_prediction']]

        lower_bound = np.percentile(self.y_train.to_numpy(), 1)
        upper_bound = np.percentile(self.y_train.to_numpy(), 97)
        with open("y_lower_bound.txt", "w") as file:
            file.write(str(lower_bound))
        with open("y_upper_bound.txt", "w") as file:
            file.write(str(upper_bound))            

        print(f"Train 데이터에서 하위 1% y 값: {lower_bound}")
        print(f"Train 데이터에서 상위 3% y 값: {upper_bound}")
        # Train 데이터에 Winsorization 적용
        self.y_train = self.y_train.clip(lower=lower_bound, upper=upper_bound)

        # 각 열(column) 및 행(row)별로 inf 및 NaN 개수 확인
        numeric_x_train = self.x_train.select_dtypes(include=[np.number])
        
        column_inf_counts_train = np.isinf(numeric_x_train).sum()
        column_nan_counts_train = np.isnan(numeric_x_train).sum()
        column_large_counts_train = (numeric_x_train.abs() > 1e9).sum()
        row_inf_counts_train = np.isinf(numeric_x_train).sum(axis=1)
        row_nan_counts_train = np.isnan(numeric_x_train).sum(axis=1)
        row_large_counts_train = (numeric_x_train.abs() > 1e9).sum(axis=1)
  
        train_clip_bounds = {}  # 하위 2%, 상위 98% 값을 저장할 딕셔너리
        features = [col for col in self.x_train.columns if col != 'sector']
        for col in features:
            lower, upper = self.x_train[col].quantile([0.02, 0.98])
            self.x_train[col] = self.x_train[col].clip(lower, upper)
            train_clip_bounds[col] = (lower, upper)
        
        # JSON 파일로 저장
        with open("train_clip_bounds.json", "w") as f:
            json.dump(train_clip_bounds, f, indent=2)
                
        # 0이 아닌 컬럼 및 행만 필터링
        column_inf_counts_train = column_inf_counts_train[column_inf_counts_train > 0]
        column_nan_counts_train = column_nan_counts_train[column_nan_counts_train > 0]
        column_large_counts_train = column_large_counts_train[column_large_counts_train > 0]

        row_inf_counts_train = row_inf_counts_train[row_inf_counts_train > 0]
        row_nan_counts_train = row_nan_counts_train[row_nan_counts_train > 0]
        row_large_counts_train = row_large_counts_train[row_large_counts_train > 0]

        # 결과 출력
        print("=== Train Data - Columns with Issues ===")
        print("Inf Counts:")
        print(column_inf_counts_train)
        print("\nNaN Counts:")
        print(column_nan_counts_train)
        print("\nLarge Value Counts:")
        print(column_large_counts_train)

        print("\n=== Train Data - Rows with Issues ===")
        print("Inf Counts:")
        print(row_inf_counts_train)
        print("\nNaN Counts:")
        print(row_nan_counts_train)
        print("\nLarge Value Counts:")
        print(row_large_counts_train)

        def replace_large_values(x):
            if isinstance(x, (int, float)):  # 숫자형 데이터만 처리
                return XGBOOST_MAX_VALUE if x > 1e9 else x
            return x  # 문자열 등은 그대로 유지
                
        self.x_train = self.x_train.applymap(replace_large_values)
        self.x_train['sector'] = self.x_train['sector'].fillna("_missing_").astype(str)
        self.x_train['sector'] = self.x_train['sector'].astype('category')

        for sec in self.sector_list:
            print("sector : ", sec)
            self.sector_x_train[sec] = self.sector_train_dfs[sec][self.sector_train_dfs[sec].columns.difference(y_col_list)]
            self.sector_x_train[sec] = self.except_col_df(self.sector_x_train[sec])
            self.sector_y_train[sec] = self.sector_train_dfs[sec][['sec_price_dev_subavg']]
        
        
    def last_dataload(self):

        with open("final_top_40_percent_average.txt", "r") as file:
            read_value = file.read()
            self.final_top_40_percent_average = float(read_value)        

        with open("y_lower_bound.txt", "r") as file:
            read_value = file.read()
            self.lower_bound = float(read_value)        
            
        with open("y_upper_bound.txt", "r") as file:
            read_value = file.read()
            self.upper_bound = float(read_value)                
        
        with open("drop_col_list.json", 'r') as f:
            self.drop_col_list = json.load(f)
            
        for fpath in self.last_train_file_list:
            print(fpath)
            df = pd.read_csv(fpath, low_memory=False)
            df = df.dropna(axis=0, subset=['price_diff_prediction'])
            
            # 원래 데이터프레임에서 volume_mul_price 상위 50%(학습 데이터처음 생성시 조건 있음) * 40% (전체 20%) 에 해당하는 symbol의 데이터만 남김
            symbol_means = df.groupby('symbol')['volume_mul_price'].mean().reset_index()
            top_symbols = symbol_means[symbol_means['volume_mul_price'] >= self.final_top_40_percent_average]            
            df = df[df['symbol'].isin(top_symbols['symbol'])]            
            
            df = df.drop(columns=self.drop_col_list, errors='ignore')
            
            print("in last_train set before dtable len : ", len(df))
            df = self.drop_many_nan_row(df, 0.95)
            print("in last_train set after dtable len : ", len(df))
                
            if len(df) != 0:
                self.last_train_df = pd.concat([self.last_train_df, df], axis=0)


        logging.debug("last_train_df shape : ")
        logging.debug(self.last_train_df.shape)        
        self.last_x_train = self.last_train_df[self.last_train_df.columns.difference(y_col_list)]
        self.last_x_train = self.except_col_df(self.last_train_df)
     
        
        self.last_y_train = self.last_train_df[['price_dev_prediction']]
        
        self.last_y_train = self.last_y_train.clip(lower=self.lower_bound, upper=self.upper_bound)


        with open("train_clip_bounds.json", "r") as f:
            self.train_clip_bounds = json.load(f)

        # Validation 데이터 clip 적용
        for col, (lower, upper) in self.train_clip_bounds.items():
            self.last_x_train[col] = self.last_x_train[col].clip(lower, upper)

    
        # 각 열(column) 및 행(row)별로 inf 및 NaN 개수 확인
        numeric_x_train = self.last_x_train.select_dtypes(include=[np.number])
        
        column_inf_counts_train = np.isinf(numeric_x_train).sum()
        column_nan_counts_train = np.isnan(numeric_x_train).sum()
        column_large_counts_train = (numeric_x_train.abs() > 1e9).sum()

        row_inf_counts_train = np.isinf(numeric_x_train).sum(axis=1)
        row_nan_counts_train = np.isnan(numeric_x_train).sum(axis=1)
        row_large_counts_train = (numeric_x_train.abs() > 1e9).sum(axis=1)
        
        # 0이 아닌 컬럼 및 행만 필터링
        column_inf_counts_train = column_inf_counts_train[column_inf_counts_train > 0]
        column_nan_counts_train = column_nan_counts_train[column_nan_counts_train > 0]
        column_large_counts_train = column_large_counts_train[column_large_counts_train > 0]

        row_inf_counts_train = row_inf_counts_train[row_inf_counts_train > 0]
        row_nan_counts_train = row_nan_counts_train[row_nan_counts_train > 0]
        row_large_counts_train = row_large_counts_train[row_large_counts_train > 0]

        # 결과 출력
        print("=== Train Data - Columns with Issues ===")
        print("Inf Counts:")
        print(column_inf_counts_train)
        print("\nNaN Counts:")
        print(column_nan_counts_train)
        print("\nLarge Value Counts:")
        print(column_large_counts_train)

        print("\n=== Train Data - Rows with Issues ===")
        print("Inf Counts:")
        print(row_inf_counts_train)
        print("\nNaN Counts:")
        print(row_nan_counts_train)
        print("\nLarge Value Counts:")
        print(row_large_counts_train)

        def replace_large_values(x):
            if isinstance(x, (int, float)):  # 숫자형 데이터만 처리
                return XGBOOST_MAX_VALUE if x > 1e9 else x
            return x  # 문자열 등은 그대로 유지
                
        self.last_x_train = self.last_x_train.applymap(replace_large_values)
        self.last_x_train['sector'] = self.last_x_train['sector'].fillna("_missing_").astype(str)
        self.last_x_train['sector'] = self.last_x_train['sector'].astype('category')   


    def val_dataload(self):
        
        with open("final_top_40_percent_average.txt", "r") as file:
            read_value = file.read()
            self.final_top_40_percent_average = float(read_value)        
        
        with open("drop_col_list.json", 'r') as f:
            self.drop_col_list = json.load(f)
        self.test_df_list = []
        for fpath in self.test_file_list:
            print(fpath)
            df = pd.read_csv(fpath, low_memory=False)
            df = df.dropna(axis=0, subset=['price_diff_prediction'])
            
            # 원래 데이터프레임에서 volume_mul_price 상위 50%(학습 데이터처음 생성시 조건 있음) * 40% (전체 20%) 에 해당하는 symbol의 데이터만 남김
            symbol_means = df.groupby('symbol')['volume_mul_price'].mean().reset_index()
            top_symbols = symbol_means[symbol_means['volume_mul_price'] >= self.final_top_40_percent_average]            
            df = df[df['symbol'].isin(top_symbols['symbol'])]            
            
            df = df.drop(columns=self.drop_col_list, errors='ignore')
            
            print("in test set before dtable len : ", len(df))
            df = self.drop_many_nan_row(df, 0.95)
            print("in test set after dtable len : ", len(df))
                
            # df = df.fillna(0)    
            if len(df) != 0:
                self.test_df = pd.concat([self.test_df, df], axis=0)
                self.test_df_list.append([fpath, df.copy()])
            if PER_SECTOR == True:
                for sec in self.sector_list:
                    self.sector_test_df_lists.append([fpath, df[df['sector']==sec].copy(), sec])


        logging.debug("test_df shape : ")
        logging.debug(self.test_df.shape)        
        self.x_test = self.test_df[self.test_df.columns.difference(y_col_list)]
        self.x_test = self.except_col_df(self.x_test)

        self.y_test = self.test_df[['price_dev_prediction']]
        self.y_test_cls = self.test_df[['price_dev_prediction']]
        self.x_test.sample(10).to_csv(self.MODEL_SAVE_PATH + 'x_test.csv', index=False)

        with open("train_clip_bounds.json", "r") as f:
            self.train_clip_bounds = json.load(f)

        # Validation 데이터 clip 적용
        for col, (lower, upper) in self.train_clip_bounds.items():
            self.x_test[col] = self.x_test[col].clip(lower, upper)
        
        # 각 열(column) 및 행(row)별로 inf 및 NaN 개수 확인
        # 숫자형 데이터만 선택 (카테고리형, 문자열형 데이터 제외)
        numeric_x_train = self.x_test.select_dtypes(include=[np.number])
        
        column_inf_counts_train = np.isinf(numeric_x_train).sum()
        column_nan_counts_train = np.isnan(numeric_x_train).sum()
        column_large_counts_train = (numeric_x_train.abs() > 1e9).sum()
        row_inf_counts_train = np.isinf(numeric_x_train).sum(axis=1)
        row_nan_counts_train = np.isnan(numeric_x_train).sum(axis=1)
        row_large_counts_train = (numeric_x_train.abs() > 1e9).sum(axis=1)
        
        # 0이 아닌 컬럼 및 행만 필터링
        column_inf_counts_train = column_inf_counts_train[column_inf_counts_train > 0]
        column_nan_counts_train = column_nan_counts_train[column_nan_counts_train > 0]
        column_large_counts_train = column_large_counts_train[column_large_counts_train > 0]

        row_inf_counts_train = row_inf_counts_train[row_inf_counts_train > 0]
        row_nan_counts_train = row_nan_counts_train[row_nan_counts_train > 0]
        row_large_counts_train = row_large_counts_train[row_large_counts_train > 0]

        # 결과 출력
        print("=== val Data - Columns with Issues ===")
        print("Inf Counts:")
        print(column_inf_counts_train)
        print("\nNaN Counts:")
        print(column_nan_counts_train)
        print("\nLarge Value Counts:")
        print(column_large_counts_train)

        print("\n=== val Data - Rows with Issues ===")
        print("Inf Counts:")
        print(row_inf_counts_train)
        print("\nNaN Counts:")
        print(row_nan_counts_train)
        print("\nLarge Value Counts:")
        print(row_large_counts_train)
        
        # 1e9 이상인 값을 XGBoost 최대값으로 변경 (숫자형 데이터만 처리, 문자열은 패스)
        def replace_large_values(x):
            if isinstance(x, (int, float)):  # 숫자형 데이터만 처리
                return XGBOOST_MAX_VALUE if x > 1e9 else x
            return x  # 문자열 등은 그대로 유지
                
        self.x_test = self.x_test.applymap(replace_large_values)

        self.x_test['sector'] = self.x_test['sector'].fillna("_missing_").astype(str)
        self.x_test['sector'] = self.x_test['sector'].astype('category')        
                
    
    def train(self):

        self.def_model()
                
        if not os.path.exists(self.MODEL_SAVE_PATH):
            print("creating MODELS path : " + self.MODEL_SAVE_PATH)
            os.makedirs(self.MODEL_SAVE_PATH)

        y_train_binary = (self.y_train > -0.02).astype(int)        
        y_test_binary = (self.y_test > -0.02).astype(int)
        
        # for i, model in self.clsmodels.items():
        #     if i == 0: #catboost
        #         logging.info("start fitting CatBoostClassifier")
        #         cat_features = ['sector']  # 범주형 변수 리스트 추가
        #         model.fit(self.x_train, y_train_binary, cat_features=cat_features)            
        #         filename = self.MODEL_SAVE_PATH + 'clsmodel_{}.cbm'.format(str(i))
        #         model.save_model(filename)
                                
        #     elif i == 1: #xgboost
        #         logging.info("start fitting classifier")
        #         num_rounds = model.get_params()['n_estimators']
        #         model.set_params(predictor='gpu_predictor')
        #         model.fit(self.x_train, y_train_binary, verbose=100, eval_set=[(self.x_test, y_test_binary)])
        #         filename = self.MODEL_SAVE_PATH + 'clsmodel_{}.sav'.format(str(i))
        #         joblib.dump(model, filename)
                
        #     logging.info("model {} score : {}".format(str(i), model.score(self.x_train, y_train_binary)))
        #     ftr_importances_values = model.feature_importances_
        #     ftr_importances = pd.Series(ftr_importances_values, index = self.x_train.columns)
        #     ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
        #     logging.info(ftr_top20)   
        #     ftr_importances = ftr_importances.sort_values(ascending=False)  # 정렬 후 저장
        #     ftr_importances.to_csv(self.MODEL_SAVE_PATH + f'clsmodel_{str(i)}_importances.csv')

        for i, model in self.models.items():
            if i == 0: #catboost
                # logging.info("start fitting CatBoostRegressor")
                # cat_features = ['sector']  # 범주형 변수 리스트 추가
                # model.fit(self.x_train, self.y_train.values.ravel(), cat_features=cat_features)
                # filename = self.MODEL_SAVE_PATH + 'model_{}.cbm'.format(str(i))
                # model.save_model(filename)
                continue
                
            elif i == 1: #xgboost
                logging.info("start fitting XGBRegressor")
                num_rounds = model.get_params()['n_estimators']
                model.set_params(predictor='gpu_predictor')
                model.fit(self.x_train, self.y_train.values.ravel(), verbose=100, eval_set=[(self.x_test, self.y_test)])
                filename = self.MODEL_SAVE_PATH + 'model_{}.sav'.format(str(i))
                joblib.dump(model, filename)

            logging.info("model {} score : {}".format(str(i), model.score(self.x_train, self.y_train)))
            ftr_importances_values = model.feature_importances_
            ftr_importances = pd.Series(ftr_importances_values, index=self.x_train.columns)
            ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            ftr_importances = ftr_importances.sort_values(ascending=False)  # 정렬 후 저장
            ftr_importances.to_csv(self.MODEL_SAVE_PATH + f'model_{str(i)}_importances.csv')
            logging.info(ftr_top20)
        
        if PER_SECTOR == True:
            for sec_idx, sec in enumerate(self.sector_list):
                print(f"sec {sec} #{sec_idx}")
                for i in range(2):
                    k = (sec, i)
                    cls_model = self.sector_cls_models[k]
                    cur_y_train_binary = (self.sector_y_train[sec] > -0.02).astype(int)        
                    cls_model.fit(self.sector_x_train[sec], cur_y_train_binary, verbose=100)
                    filename = self.MODEL_SAVE_PATH + '{}_clsmodel_{}.sav'.format(sec, str(i))
                    joblib.dump(cls_model, filename)
                    logging.info("cls_model {} score : ".format(str(i)))
                    logging.info(cls_model.score(self.sector_x_train[sec], cur_y_train_binary))
                    logging.info("end fitting per sector XGBClssfication")
                    
                    model = self.sector_models[k]
                    model.fit(self.sector_x_train[sec], self.sector_y_train[sec].values.ravel(), verbose=100)
                    filename = self.MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))
                    joblib.dump(model, filename)
                    logging.info("model {} score : ".format(str(i)))
                    logging.info(model.score(self.sector_x_train[sec], self.sector_y_train[sec]))
                    logging.info("end fitting per sector XGBRegressor")
                    # ftr_importances_values = model.feature_importances_
                    # ftr_importances = pd.Series(ftr_importances_values, index = self.sector_x_train[sec].columns)
                    # ftr_importances.to_csv(self.MODEL_SAVE_PATH + sec + '_model_importances.csv')
                    # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
                    # logging.info(ftr_top20)

    def train_dl(self):
        label_encoder = LabelEncoder()
        scaler = StandardScaler()

        features = [col for col in self.x_train.columns if col != 'sector']
        # Feature 리스트 저장
        features_path = self.MODEL_SAVE_PATH + "features_list.pkl"
        joblib.dump(features, features_path)
        print(f"Feature list saved at {features_path}")
        

        self.x_train_dl = self.x_train.copy()
        self.x_train_dl['sector'] = label_encoder.fit_transform(self.x_train_dl['sector'])
        self.x_train_dl[features] = scaler.fit_transform(self.x_train_dl[features])
        self.x_train_dl = self.x_train_dl.fillna(0)

        label_encoder_path = self.MODEL_SAVE_PATH + "label_encoder.pkl"
        joblib.dump(label_encoder, label_encoder_path)
        print(f"LabelEncoder saved at {label_encoder_path}")
        
        scaler_path = self.MODEL_SAVE_PATH + "standard_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved at {scaler_path}")        
        
        self.x_test_dl = self.x_test.copy()
        self.x_test_dl['sector'] = label_encoder.transform(self.x_test_dl['sector'])
        self.x_test_dl[features] = scaler.transform(self.x_test_dl[features])
        self.x_test_dl = self.x_test_dl.fillna(0)
        

        print("x_train describe:\n", self.x_train_dl.describe())
        print("x_train isnull sum:\n", self.x_train_dl.isnull().sum())
        #---------------------#
        # 3) 분류(target) 전처리
        #---------------------#
        y_train_binary = (self.y_train > -0.02).astype(int)
        y_test_binary = (self.y_test > -0.02).astype(int)

        #---------------------#
        # 4) Embedding을 위한 sector 범위 확인
        #    - LabelEncoder된 결과의 최대값 + 1
        #---------------------#
        num_sectors = self.x_train_dl['sector'].nunique()
        input_dim = self.x_train_dl.shape[1] - 1
        print("Train shape:", self.x_train_dl.shape, "num_sectors:", num_sectors)  

        #---------------------#
        # 5) PyTorch Dataset/Dataloader
        #    - sector 컬럼 분리해서 dataset으로 전달
        #---------------------#
        train_dataset_cls = FinancialDataset(
            self.x_train_dl.drop(columns=['sector']).values, 
            y_train_binary.values, 
            self.x_train_dl['sector'].values
        )
        train_loader_cls = DataLoader(train_dataset_cls, batch_size=256, shuffle=True)

        # Device 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #---------------------#
        # 6) 분류 모델 학습
        #---------------------#
        classifier = FinancialClassifier(input_dim, num_sectors).to(device)
        # criterion_cls = nn.BCELoss()
        criterion_cls = nn.BCEWithLogitsLoss()  # <-- 변경

        optimizer_cls = optim.AdamW(classifier.parameters(), lr=0.001)
        scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_cls, mode='min', factor=0.5, patience=5
        )
        print("start train NN classifier")
        classifier = train_model(
            classifier, train_loader_cls, criterion_cls, 
            optimizer_cls, scheduler_cls, device
        )
        torch.save(classifier.state_dict(), self.MODEL_SAVE_PATH + 'torch_classifier.pth')
        print("end train NN classifier")
        # 또는 joblib.dump(classifier.state_dict(), ...)도 가능

        #---------------------#
        # 7) 회귀 모델 학습 (y_train)
        #---------------------#
        train_dataset_reg = FinancialDataset(
            self.x_train_dl.drop(columns=['sector']).values, 
            self.y_train.values, 
            self.x_train_dl['sector'].values
        )
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=256, shuffle=True)

        regressor = FinancialRegressor(input_dim, num_sectors).to(device)
        criterion_reg = nn.MSELoss()
        optimizer_reg = optim.AdamW(regressor.parameters(), lr=0.001)
        scheduler_reg = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_reg, mode='min', factor=0.5, patience=5
        )
        print("start train NN regressor")
        regressor = train_model(
            regressor, train_loader_reg, criterion_reg, 
            optimizer_reg, scheduler_reg, device
        )
        torch.save(regressor.state_dict(), self.MODEL_SAVE_PATH + 'torch_regressor.pth')
        # 또는 joblib.dump(regressor.state_dict(), ...)도 가능
        print("end train NN regressor")

    def evaluation(self):

        self.MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS/'
        self.models = dict()
        self.clsmodels = dict()
        
        classifier_path = self.MODEL_SAVE_PATH + 'torch_classifier.pth'
        regressor_path = self.MODEL_SAVE_PATH + 'torch_regressor.pth'
        

        features_path = self.MODEL_SAVE_PATH + "features_list.pkl"
        features = joblib.load(features_path)
        print(f"Feature list loaded from {features_path}")
        label_encoder_path = self.MODEL_SAVE_PATH + "label_encoder.pkl"
        label_encoder = joblib.load(label_encoder_path)
        print(f"LabelEncoder loaded from {label_encoder_path}")
        scaler_path = self.MODEL_SAVE_PATH + "standard_scaler.pkl"
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        
        self.x_test_dl = self.x_test.copy()
        self.x_test_dl['sector'] = label_encoder.transform(self.x_test_dl['sector'])        
        self.x_test_dl[features] = scaler.transform(self.x_test_dl[features])  # ✅ fit하지 않고 transform만 수행
        self.x_test_dl = self.x_test_dl.fillna(0)        


        # ✅ PyTorch 모델 인스턴스 생성
        input_dim = self.x_test_dl.shape[1] - 1  # feature 개수
        num_sectors = len(label_encoder.classes_) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ PyTorch 모델 불러오기
        classifier = FinancialClassifier(input_dim, num_sectors).to(device)
        classifier.load_state_dict(torch.load(classifier_path))
        classifier.eval()  # 평가 모드로 변경

        regressor = FinancialRegressor(input_dim, num_sectors).to(device)
        regressor.load_state_dict(torch.load(regressor_path))
        regressor.eval()  # 평가 모드로 변경

        # ✅ PyTorch 모델을 self.clsmodels[2], self.models[2]에 추가
        self.clsmodels[2] = classifier  # PyTorch 분류 모델
        self.models[2] = regressor  # PyTorch 회귀 모델

        self.clsmodels[0] = CatBoostClassifier()
        self.clsmodels[0].load_model(self.MODEL_SAVE_PATH + 'clsmodel_0.cbm')
        self.clsmodels[1] = joblib.load(self.MODEL_SAVE_PATH + 'clsmodel_1.sav')
        
        self.models[0] = CatBoostRegressor()
        self.models[0].load_model(self.MODEL_SAVE_PATH + 'model_0.cbm')
        self.models[1] = joblib.load(self.MODEL_SAVE_PATH + 'model_1.sav')
        

        pred_col_list = ['ai_pred_avg']
        for i in range(3):
            pred_col_list.extend([
                f'model_{i}_prediction',
                f'model_{i}_prediction_wbinary_0',
                f'model_{i}_prediction_wbinary_1',
                f'model_{i}_prediction_wbinary_2',
                f'model_{i}_prediction_wbinary_ensemble',
                f'model_{i}_prediction_wbinary_ensemble2',
                f'model_{i}_prediction_wbinary_ensemble3'
            ])

        model_eval_hist = []
        full_df = pd.DataFrame()
        for test_idx, (testdate, df) in enumerate(self.test_df_list):
            
            logging.info("evaluation date : ")
            tmp = testdate.split('\\')
            tmp = [v for v in tmp if v.endswith('.csv')]
            print(f"in test loop tmp : {tmp}")
            tdate = "_".join(tmp[0].split('_')[4:6])
            print(f"in test loop tdate : {tdate}")
            x_test = df[df.columns.difference(y_col_list)]
            x_test = self.except_col_df(x_test)
            x_test['sector'] = x_test['sector'].fillna("_missing_").astype(str)
            x_test['sector'] = x_test['sector'].astype('category') 
            
            x_test_dl = x_test.copy()
            x_test_dl['sector'] = label_encoder.transform(x_test_dl['sector'])        
            x_test_dl[features] = scaler.transform(x_test_dl[features])  # ✅ fit하지 않고 transform만 수행
            x_test_dl = x_test_dl.fillna(0) 

            y_test = df[['price_dev_prediction']]
            y_test_binary = (y_test > -0.02).astype(int)

            preds = np.empty((0,x_test.shape[0]))            
            df['label'] = y_test
            df['label_binary'] = y_test_binary

            for i, model in self.clsmodels.items():
                
                if i < 2:    
                    logging.info(f"classification model # {i}")
                    y_probs = model.predict_proba(x_test)[:, 1]
                elif i == 2:  # PyTorch 모델
                    print("PyTorch 분류 모델 예측 중...")
                    with torch.no_grad():
                        x_tensor = torch.tensor(x_test_dl.drop(columns=['sector']).values, dtype=torch.float32).to(device)
                        sector_tensor = torch.tensor(x_test_dl['sector'].values, dtype=torch.long).to(device)
                        y_probs = torch.sigmoid(model(x_tensor, sector_tensor)).cpu().numpy().flatten()                    
                        
                pred_col_name = 'clsmodel_' + str(i) + '_prediction'
                correct_col_name = 'clsmodel_' + str(i) + '_correct'
                # 상위 10%의 예측 확률값을 threshold로 설정
                threshold = np.percentile(y_probs, THRESHOLD)
                # 새로운 threshold로 예측값 생성
                y_predict_binary = (y_probs > threshold).astype(int)
                logging.info(f"20% positive threshold == {threshold}")
                logging.info(classification_report(y_test_binary, y_predict_binary))
                df[pred_col_name] = y_predict_binary
                df[correct_col_name] = (y_test_binary.values.ravel() == y_predict_binary).astype(int)
                acc = accuracy_score(df['label_binary'], df[pred_col_name])
                logging.info(f"Accuracy for {pred_col_name}: {acc:.4f}")
                
            for i, model in self.models.items():
                pred_bin_col_name_0 = 'clsmodel_0_prediction'
                pred_bin_col_name_1 = 'clsmodel_1_prediction'
                pred_bin_col_name_2 = 'clsmodel_2_prediction'
                
                correct_col_name = f'clsmodel_{i}_correct'

                pred_col_name = f'model_{i}_prediction'
                pred_col_name_wbinary_0 = f'model_{i}_prediction_wbinary_0'
                pred_col_name_wbinary_1 = f'model_{i}_prediction_wbinary_1'
                pred_col_name_wbinary_2 = f'model_{i}_prediction_wbinary_2'
                pred_col_name_wbinary_ensemble = f'model_{i}_prediction_wbinary_ensemble'
                pred_col_name_wbinary_ensemble2 = f'model_{i}_prediction_wbinary_ensemble2'
                pred_col_name_wbinary_ensemble3 = f'model_{i}_prediction_wbinary_ensemble3'
                
                loss_col_name = f'model_{i}_prediction_loss'
                loss_bin_col_name_0 = f'model_{i}_prediction_wbinary_loss_0'
                loss_bin_col_name_1 = f'model_{i}_prediction_wbinary_loss_1'
                loss_bin_col_name_2 = f'model_{i}_prediction_wbinary_loss_2'
                
                # ✅ XGBoost & CatBoost 회귀 모델 예측
                if i < 2:
                    y_predict = model.predict(x_test)
                elif i == 2:  # ✅ PyTorch 회귀 모델 예측
                    with torch.no_grad():
                        x_tensor = torch.tensor(x_test_dl.drop(columns=['sector']).values, dtype=torch.float32).to(device)
                        sector_tensor = torch.tensor(x_test_dl['sector'].values, dtype=torch.long).to(device)
                        y_predict = model(x_tensor, sector_tensor).cpu().numpy().flatten()

                # ✅ 예측값 저장
                df[pred_col_name] = y_predict

                # ✅ Binary Classification 결과를 이용한 Filtering 적용
                df[pred_col_name_wbinary_0] = np.where(df[pred_bin_col_name_0] == 0, -1, y_predict)
                df[pred_col_name_wbinary_1] = np.where(df[pred_bin_col_name_1] == 0, -1, y_predict)
                df[pred_col_name_wbinary_2] = np.where(df[pred_bin_col_name_2] == 0, -1, y_predict)

                # ✅ Ensemble Prediction 적용 (Binary Filtering)
                df[pred_col_name_wbinary_ensemble] = np.where(
                    ((df[pred_bin_col_name_0] == 0) | (df[pred_bin_col_name_1] == 0)), -1, y_predict
                )
                df[pred_col_name_wbinary_ensemble2] = np.where(
                    ((df[pred_bin_col_name_1] == 0) | (df[pred_bin_col_name_2] == 0)), -1, y_predict
                )

                # ✅ 2개 이상이 0인 경우 -1 처리
                condition = (df[[pred_bin_col_name_0, pred_bin_col_name_1, pred_bin_col_name_2]] == 0).sum(axis=1) >= 2
                df[pred_col_name_wbinary_ensemble3] = np.where(condition, -1, y_predict)

                # ✅ 예측값을 `preds`에 추가하여 `ai_pred_avg` 계산에 포함
                preds = np.vstack((preds, y_predict[None, :]))

                # ✅ 손실(Loss) 계산 및 저장
                df[loss_col_name] = abs(df['label'] - y_predict)
                df[loss_bin_col_name_0] = abs(df['label'] - df[pred_col_name_wbinary_0])
                df[loss_bin_col_name_1] = abs(df['label'] - df[pred_col_name_wbinary_1])
                df[loss_bin_col_name_2] = abs(df['label'] - df[pred_col_name_wbinary_2])

            
            df['ai_pred_avg'] = np.average(preds, axis=0)
            df['ai_pred_avg_loss'] = abs(df['label']-df['ai_pred_avg'])            
            
            # full_df = pd.concat([full_df, df], ignore_index=True)
            # df.to_csv(self.MODEL_SAVE_PATH + "prediction_ai_{}.csv".format(tdate))
            topk_period_earning_sums = []

            topk_list = [(0, 3), (0, 7), (0, 15)]
            for s, e in topk_list:
                logging.info(f"Top {s} ~ {e} evaluation")

                k = f"{s}~{e}"
                for col in pred_col_list:
                    top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]

                    # ✅ 평가 결과 저장
                    topk_period_earning_sums = top_k_df['price_dev_prediction'].sum()

                    # ✅ CSV로 저장
                    top_k_df.to_csv(f"{self.MODEL_SAVE_PATH}prediction_{tdate}_{col}_top{s}-{e}.csv", index=False)

                    # ✅ model_eval_hist에 PyTorch 결과도 포함하여 저장
                    model_eval_hist.append([
                        tdate, col, k, 
                        top_k_df['price_dev_prediction'].sum() / (e - s + 1), 
                        top_k_df[col].sum() / (e - s + 1),
                        abs(top_k_df[col].sum() / (e - s + 1) - top_k_df['price_dev_prediction'].sum() / (e - s + 1)),
                        int(top_k_df[col].sum() / (e - s + 1) > 0), 
                        top_k_df['ai_pred_avg'].sum() / (e - s + 1),
                        top_k_df['model_0_prediction'].sum() / (e - s + 1),
                        top_k_df['model_1_prediction'].sum() / (e - s + 1),
                        top_k_df['model_2_prediction'].sum() / (e - s + 1),  # ✅ PyTorch 모델 추가
                        top_k_df['model_0_prediction_wbinary_0'].sum() / (e - s + 1),
                        top_k_df['model_1_prediction_wbinary_0'].sum() / (e - s + 1),
                        top_k_df['model_2_prediction_wbinary_0'].sum() / (e - s + 1),  # ✅ PyTorch 모델 추가
                        top_k_df['model_0_prediction_wbinary_1'].sum() / (e - s + 1),
                        top_k_df['model_1_prediction_wbinary_1'].sum() / (e - s + 1),
                        top_k_df['model_2_prediction_wbinary_1'].sum() / (e - s + 1),  # ✅ PyTorch 모델 추가
                        top_k_df['model_0_prediction_wbinary_2'].sum() / (e - s + 1),
                        top_k_df['model_1_prediction_wbinary_2'].sum() / (e - s + 1),
                        top_k_df['model_2_prediction_wbinary_2'].sum() / (e - s + 1),  # ✅ PyTorch 모델 추가
                        top_k_df['model_0_prediction_wbinary_ensemble'].sum() / (e - s + 1),
                        top_k_df['model_1_prediction_wbinary_ensemble'].sum() / (e - s + 1),
                        top_k_df['model_2_prediction_wbinary_ensemble'].sum() / (e - s + 1),  # ✅ PyTorch 모델 추가
                        top_k_df['model_0_prediction_wbinary_ensemble2'].sum() / (e - s + 1),
                        top_k_df['model_1_prediction_wbinary_ensemble2'].sum() / (e - s + 1),
                        top_k_df['model_2_prediction_wbinary_ensemble2'].sum() / (e - s + 1),  # ✅ PyTorch 모델 추가
                        top_k_df['model_0_prediction_wbinary_ensemble3'].sum() / (e - s + 1),
                        top_k_df['model_1_prediction_wbinary_ensemble3'].sum() / (e - s + 1),
                        top_k_df['model_2_prediction_wbinary_ensemble3'].sum() / (e - s + 1)  # ✅ PyTorch 모델 추가
                    ])

        # ✅ 최종 평가 데이터 저장
        col_name = [
            'start_date', 'model', 'krange', 'avg_earning_per_stock', 'cur_model_pred', 'loss_y_and_pred',
            'cur_model_pred_ispositive', 'avg_pred', 'model0_pred', 'model1_pred', 'model2_pred',  # ✅ PyTorch 추가
            'model0_pred_wbinary_0', 'model1_pred_wbinary_0', 'model2_pred_wbinary_0',
            'model0_pred_wbinary_1', 'model1_pred_wbinary_1', 'model2_pred_wbinary_1',
            'model0_pred_wbinary_2', 'model1_pred_wbinary_2', 'model2_pred_wbinary_2',
            'model0_pred_wbinary_ensemble', 'model1_pred_wbinary_ensemble', 'model2_pred_wbinary_ensemble',
            'model0_pred_wbinary_ensemble2', 'model1_pred_wbinary_ensemble2', 'model2_pred_wbinary_ensemble2',
            'model0_pred_wbinary_ensemble3', 'model1_pred_wbinary_ensemble3', 'model2_pred_wbinary_ensemble3'
        ]

        pred_df = pd.DataFrame(model_eval_hist, columns=col_name)
        logging.info(pred_df)
        pred_df.to_csv(self.MODEL_SAVE_PATH + 'pred_df_topk.csv', index=False)

        
 
        # full_df.to_csv(self.MODEL_SAVE_PATH+'prediction_ai.csv', index=False)

        full_df = pd.DataFrame()
        if PER_SECTOR == True:
            pred_col_list = ['ai_pred_avg'] 
            for i in range(3):
                pred_col_name = 'model_' + str(i) + '_prediction'
                pred_col_list.append(pred_col_name)
                pred_col_name = 'model_' + str(i) + '_prediction_wbinary_0'
                pred_col_list.append(pred_col_name)
                pred_col_name = 'model_' + str(i) + '_prediction_wbinary_1'
                pred_col_list.append(pred_col_name)
                pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble'
                pred_col_list.append(pred_col_name)            
            
            testdates=set()
            allsector_topk_df=pd.DataFrame()
            self.sector_cls_models = dict()
            self.sector_models = dict()
            for sec in self.sector_list:
                for i in range(2):
                    k = (sec, i)
                    self.sector_cls_models[k] = joblib.load(self.MODEL_SAVE_PATH + '{}_clsmodel_{}.sav'.format(sec, str(i)))
                    self.sector_models[k] = joblib.load(self.MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))
            sector_model_eval_hist = []

            for test_idx, (testdate, df, sec) in enumerate(self.sector_test_df_lists):
                print("sec evaluation date : ")
                tmp = testdate.split('\\')
                tmp = [v for v in tmp if v.endswith('.csv')]
                tdate = "_".join(tmp[0].split('_')[4:6])
                print(tdate)
                print(sec)
                testdates.add(tdate)

                x_test = df[df.columns.difference(y_col_list)]
                x_test = self.except_col_df(x_test)
                y_test = df[['sec_price_dev_subavg']]
                y_test_cls = df[['sec_price_dev_subavg']]
                y_test_binary = (y_test_cls > -0.02).astype(int)


                
                if len(x_test) == 0:
                    continue
                
                sector_preds = np.empty((0,x_test.shape[0]))
                df['label'] = y_test
                df['label_binary'] = y_test_binary
                
                # x_test = self.clean_feature_names(x_test)

                for (cur_sec, i), model in self.sector_cls_models.items():    
                    if cur_sec == sec:
                        cur_model = model
                        cur_i = i
                        logging.info(f"classification model # {i}")
                        pred_col_name = 'clsmodel_' + str(i) + '_prediction'
                        correct_col_name = 'clsmodel_' + str(i) + '_correct'
                        y_probs = model.predict_proba(x_test)[:, 1]

                        # 상위 10%의 예측 확률값을 threshold로 설정
                        threshold = np.percentile(y_probs, THRESHOLD)
                        # 새로운 threshold로 예측값 생성
                        y_predict_binary = (y_probs > threshold).astype(int)
                        logging.info(f"20% positive threshold == {threshold}")
                        logging.info(classification_report(y_test_binary, y_predict_binary))
                        df[pred_col_name] = y_predict_binary
                        df[correct_col_name] = (y_test_binary.values.ravel() == y_predict_binary).astype(int)
                        acc = accuracy_score(df['label_binary'], df[pred_col_name])
                        logging.info(f"Accuracy for {pred_col_name}: {acc:.4f}")
                        
                
                    
                        
                for (cur_sec, i), model in self.sector_models.items():    
                    if cur_sec == sec:
                        pred_bin_col_name_0 = 'clsmodel_0_prediction'
                        pred_bin_col_name_1 = 'clsmodel_1_prediction'
                        pred_col_name = 'model_' + str(i) + '_prediction'
                        correct_col_name = 'clsmodel_' + str(i) + '_correct'
                        pred_col_name_wbinary_0 = 'model_' + str(i) + '_prediction_wbinary_0'
                        pred_col_name_wbinary_1 = 'model_' + str(i) + '_prediction_wbinary_1'
                        pred_col_name_wbinary_ensemble = 'model_' + str(i) + '_prediction_wbinary_ensemble'
                        loss_col_name = 'model_' + str(i) + '_prediction_loss'
                        loss_bin_col_name_0 = 'model_' + str(i) + '_prediction_wbinary_loss_0'
                        loss_bin_col_name_1 = 'model_' + str(i) + '_prediction_wbinary_loss_1'

                        y_predict = model.predict(x_test)
                        
                        df[pred_col_name] = y_predict
                        df[pred_col_name_wbinary_0] = np.where(df[pred_bin_col_name_0] == 0, -1, y_predict)
                        df[pred_col_name_wbinary_1] = np.where(df[pred_bin_col_name_1] == 0, -1, y_predict)
                        df[pred_col_name_wbinary_ensemble] = np.where(((df[pred_bin_col_name_0] == 0) | (df[pred_bin_col_name_1] == 0)), -1, y_predict)
                        
                        sector_preds = np.vstack((sector_preds, y_predict[None,:]))
                        df[loss_col_name] = abs(df['label'] - y_predict)
                        df[loss_bin_col_name_0] = abs(df['label'] - df[pred_col_name_wbinary_0])
                        df[loss_bin_col_name_1] = abs(df['label'] - df[pred_col_name_wbinary_1])
                        
                        logging.info(f"eval : model i : {i} loss : {df[loss_col_name].mean()} loss_wbin_0 {df[loss_bin_col_name_0].mean()} loss_wbin_1 {df[loss_bin_col_name_1].mean()}\
                            ")
                        if test_idx != 0:
                            logging.info(f"accumulated eval : model i : {i} loss : {full_df[loss_col_name].mean()} loss_wbin_0 {full_df[loss_bin_col_name_0].mean()} loss_wbin_1 {full_df[loss_bin_col_name_1].mean()}\
                            ")
                        
                df['ai_pred_avg'] = np.average(sector_preds, axis=0)
                df['ai_pred_avg_loss'] = abs(df['label']-df['ai_pred_avg'])
                full_df = pd.concat([full_df, df], ignore_index=True)
                df.to_csv(self.MODEL_SAVE_PATH + f"sector_{sec}_prediction_ai_{tdate}.csv")

                # 각 model의 top_k 종목의 period_price_diff 합을 구해서 model 최종 평가
                topk_period_earning_sums = []
                topk_list = [(0,3), (0,7), (0,15)]
                for s, e in topk_list:
                    # logging.info("top" + str(s) + " ~ "  + str(e) )
                    k = str(s) + '~' + str(e)
                    for col in pred_col_list:
                        top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                        # logging.info("")
                        # logging.info(col)
                        # logging.info(("label"))
                        # logging.info((top_k_df['price_dev_prediction'].sum()/(e-s+1)))
                        # logging.info(("pred"))
                        # logging.info((top_k_df[col].sum()/(e-s+1)))
                        topk_period_earning_sums.append(top_k_df['price_dev_prediction'].sum())
                        
                        top_k_df.to_csv(self.MODEL_SAVE_PATH+'prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))
                        sector_model_eval_hist.append([tdate, sec, col, k, top_k_df['price_dev_prediction'].sum()/(e-s+1), top_k_df[col].sum()/(e-s+1), 
                                                abs(top_k_df[col].sum()/(e-s+1) - top_k_df['price_dev_prediction'].sum()/(e-s+1)), int(top_k_df[col].sum()/(e-s+1) > 0), 
                                                top_k_df['ai_pred_avg'].sum()/(e-s+1), top_k_df['model_0_prediction'].sum()/(e-s+1), top_k_df['model_1_prediction'].sum()/(e-s+1),
                                                top_k_df['model_0_prediction_wbinary_0'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_0'].sum()/(e-s+1),
                                                top_k_df['model_0_prediction_wbinary_1'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_1'].sum()/(e-s+1),
                                                top_k_df['model_0_prediction_wbinary_ensemble'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_ensemble'].sum()/(e-s+1),
                                                ])

                

            col_name = ['start_date', 'sector', 'model', 'krange', 'avg_earning_per_stock', 'cur_model_pred', 'loss_y_and_pred', 
                'cur_model_pred_ispositive', 'avg_pred', 'model0_pred', 'model1_pred', 
                'model0_pred_wbinary_0', 'model1_pred_wbinary_0',
                'model0_pred_wbinary_1', 'model1_pred_wbinary_1', 
                'model0_pred_wbinary_ensemble', 'model1_pred_wbinary_ensemble',
                ]
    
            pred_df = pd.DataFrame(sector_model_eval_hist, columns=col_name)
            logging.info(pred_df)
            pred_df.to_csv(self.MODEL_SAVE_PATH+'sector_pred_df_topk.csv', index=False)
            full_df.to_csv(self.MODEL_SAVE_PATH+'sector_prediction_ai.csv', index=False)
            
            # 모델 리스트를 정의합니다.
            models_name = ['model0_pred', 'model1_pred', 'model0_pred_wbinary_0', 'model1_pred_wbinary_0',
                    'model0_pred_wbinary_1', 'model1_pred_wbinary_1', 
                    'model0_pred_wbinary_ensemble', 'model1_pred_wbinary_ensemble']

            # 결과를 저장할 데이터프레임을 생성합니다.
            results = []

            # 각 모델에 대해 작업을 수행합니다.
            for model in models_name:
                # start_date별로 그룹화
                grouped = pred_df.groupby('start_date')
                
                for start_date, group in grouped:
                    # 각 start_date별로 cur_model_pred 합계가 높은 상위 3개와 5개 sector 선택
                    top_3_sectors = group.nlargest(3, 'cur_model_pred')
                    top_5_sectors = group.nlargest(5, 'cur_model_pred')
                    
                    # 상위 3개 sector의 avg_earning_per_stock 합계 계산
                    top_3_avg_earning_per_stock = top_3_sectors['avg_earning_per_stock'].sum()
                    
                    # 상위 5개 sector의 avg_earning_per_stock 합계 계산
                    top_5_avg_earning_per_stock = top_5_sectors['avg_earning_per_stock'].sum()
                    
                            # 상위 3개, 5개 sector 이름을 쉼표로 구분된 문자열로 저장
                    top_3_sector_names = ', '.join(top_3_sectors['sector'].tolist())
                    top_5_sector_names = ', '.join(top_5_sectors['sector'].tolist())
                    
                    # 결과를 저장합니다.
                    results.append({
                        'start_date': start_date,
                        'model': model,
                        'top_3_avg_earning_per_stock': top_3_avg_earning_per_stock,
                        'top_5_avg_earning_per_stock': top_5_avg_earning_per_stock,
                        'top_3_sectors': top_3_sector_names,
                        'top_5_sectors': top_5_sector_names
                    })

            # 결과를 데이터프레임으로 변환
            results_df = pd.DataFrame(results)

            # 결과를 출력하거나 저장
            print(results_df)            
            results_df.to_csv(self.MODEL_SAVE_PATH+'0_top_K_sector_pred_df.csv', index=False)
            
        
        
    def latest_prediction(self):
        
        self.MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS/'
        self.clsmodels = dict()
        self.models = dict()
        

        # self.clsmodels[0] = CatBoostClassifier()
        # self.clsmodels[0].load_model(self.MODEL_SAVE_PATH + 'clsmodel_0.cbm')

        # self.clsmodels[1] = CatBoostClassifier()
        # self.clsmodels[1].load_model(self.MODEL_SAVE_PATH + 'clsmodel_1.cbm')

        # self.clsmodels[2] = CatBoostClassifier()
        # self.clsmodels[2].load_model(self.MODEL_SAVE_PATH + 'clsmodel_2.cbm')

        # self.models[0] = CatBoostRegressor()
        # self.models[0].load_model(self.MODEL_SAVE_PATH + 'model_0.cbm')

        # self.models[1] = CatBoostRegressor()
        # self.models[1].load_model(self.MODEL_SAVE_PATH + 'model_1.cbm')        
        
        self.clsmodels[0] = joblib.load(self.MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(self.MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(self.MODEL_SAVE_PATH + 'clsmodel_2.sav')        
        self.models[0] = joblib.load(self.MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(self.MODEL_SAVE_PATH + 'model_1.sav')
        
        # 추가학습
        
        self.last_x_train = self.last_x_train[self.last_x_train.columns.difference(y_col_list)]
        last_y_train_binary = (self.last_y_train > -0.02).astype(int)        
        # Classification 모델 추가 학습
        # for i, model in self.clsmodels.items():
        #     logging.info("start fitting classifier")
        #     print(model.tree_count_)  # 트리 개수가 기존보다 증가했는지 확인
        #     print(model.get_params()['learning_rate'])  # 학습률이 줄어들었는지 확인


        #     current_lr = model.get_params()['learning_rate']  # 현재 학습률 가져오기
        #     new_lr = current_lr / 5  # 학습률 감소
        #     cat_features = ['sector']  # 범주형 변수 리스트 추가
        #     # 🔹 새로운 모델을 생성하면서 새로운 학습률 적용
        #     new_model = CatBoostRegressor(
        #         iterations=200,  # 추가 학습을 위한 iteration 설정
        #         learning_rate=new_lr,  # 🔥 새로운 학습률 적용
        #         task_type="CPU",  # 기존 모델이 CPU 기반이므로 동일하게 설정
        #         **{k: model.get_params()[k] for k in ["depth", "loss_function", "eval_metric", "l2_leaf_reg", "colsample_bylevel", \
        #                                               "max_ctr_complexity", "bagging_temperature", "random_strength", "early_stopping_rounds", "allow_writing_files"]}
        #     )
        #     cat_features = ['sector']  # 범주형 변수 리스트 추가
        #     # 🔹 추가 학습 수행 (init_model을 이전 모델로 설정)
        #     new_model.fit(
        #         self.last_x_train, 
        #         last_y_train_binary, 
        #         init_model=model,  # 🔥 기존 모델을 초기 모델로 설정
        #         cat_features=cat_features
        #     )

        #     # 이전 모델을 init_model로 설정하여 추가 학습 진행
        #     # model.fit(self.last_x_train, last_y_train_binary, init_model=model, verbose=100, cat_features=cat_features)

        #     filename = self.MODEL_SAVE_PATH + f'last_clsmodel_{str(i)}.cbm'
        #     new_model.save_model(filename)
        #     logging.info(f"model {i} score : {new_model.score(self.last_x_train, last_y_train_binary)}")

        #     # Feature Importance 저장
        #     ftr_importances_values = new_model.get_feature_importance()
        #     ftr_importances = pd.Series(ftr_importances_values, index=self.last_x_train.columns)
        #     ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
        #     logging.info(ftr_top20)
        #     ftr_importances.to_csv(self.MODEL_SAVE_PATH + f'last_clsmodel_{str(i)}_importances.csv')
        #     print(new_model.tree_count_)  # 트리 개수가 기존보다 증가했는지 확인
        #     print(new_model.get_params()['learning_rate'])  # 학습률이 줄어들었는지 확인



        # # Regression 모델 추가 학습
        # for i, model in self.models.items():
        #     logging.info("start fitting CatBoostRegressor")
        #     print(model.tree_count_)  # 트리 개수가 기존보다 증가했는지 확인
        #     print(model.get_params()['learning_rate'])  # 학습률이 줄어들었는지 확인            
        #     current_lr = model.get_params()['learning_rate']  # 현재 학습률 가져오기
        #     new_lr = current_lr / 5  # 학습률 감소

        #     # 🔹 새로운 모델을 생성하면서 새로운 학습률 적용
        #     new_model = CatBoostRegressor(
        #         iterations=200,  # 추가 학습을 위한 iteration 설정
        #         learning_rate=new_lr,  # 🔥 새로운 학습률 적용
        #         task_type="CPU",  # 기존 모델이 CPU 기반이므로 동일하게 설정
        #         **{k: model.get_params()[k] for k in ["depth", "loss_function", "eval_metric", "l2_leaf_reg", "colsample_bylevel", \
        #                                               "max_ctr_complexity", "bagging_temperature", "random_strength", "early_stopping_rounds", "allow_writing_files"]}
        #     )
        #     cat_features = ['sector']  # 범주형 변수 리스트 추가
        #     # 🔹 추가 학습 수행 (init_model을 이전 모델로 설정)
        #     new_model.fit(
        #         self.last_x_train, 
        #         last_y_train_binary, 
        #         init_model=model,  # 🔥 기존 모델을 초기 모델로 설정
        #         cat_features=cat_features
        #     )

        #     filename = self.MODEL_SAVE_PATH + f'last_model_{str(i)}.cbm'
        #     new_model.save_model(filename)
        #     logging.info(f"model {i} score : {new_model.score(self.last_x_train, self.last_y_train)}")

        #     # Feature Importance 저장
        #     ftr_importances_values = new_model.get_feature_importance()
        #     ftr_importances = pd.Series(ftr_importances_values, index=self.last_x_train.columns)
        #     ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
        #     ftr_importances.to_csv(self.MODEL_SAVE_PATH + f'last_model_{str(i)}_importances.csv')

        #     logging.info(ftr_top20)
        #     print(new_model.tree_count_)  # 트리 개수가 기존보다 증가했는지 확인
        #     print(new_model.get_params()['learning_rate'])  # 학습률이 줄어들었는지 확인            
        
        dtrain = xgboost.DMatrix(self.last_x_train, label=last_y_train_binary, enable_categorical=True)                        
        for i, model in self.clsmodels.items():
            logging.info("start fitting classifier")
            previous_booster = model.get_booster()
            current_lr = model.get_params()['learning_rate']  # 현재 학습률 가져오기
            new_lr = current_lr/5  
            model.set_params(learning_rate=new_lr)
            model.set_params(n_estimators=int(model.get_params()['n_estimators'] / 4))
            model.set_params(early_stopping_rounds=None)
            num_rounds = model.get_params()['n_estimators']
            model.set_params(predictor='gpu_predictor')
            model.fit(self.last_x_train, last_y_train_binary, verbose=100)

            # model.train(dtrain, xgb_model=previous_booster, verbose=100)
            filename = self.MODEL_SAVE_PATH + 'last_clsmodel_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.last_x_train, last_y_train_binary))
            
            ftr_importances_values = model.feature_importances_
            ftr_importances = pd.Series(ftr_importances_values, index = self.last_x_train.columns)
            ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            logging.info(ftr_top20)   
            ftr_importances = ftr_importances.sort_values(ascending=False)  # 정렬 후 저장
            ftr_importances.to_csv(self.MODEL_SAVE_PATH + f'last_clsmodel_{str(i)}_importances.csv')

        dtrain = xgboost.DMatrix(self.last_x_train, label=self.last_y_train, enable_categorical=True)                                    
        for i, model in self.models.items():
            logging.info("start fitting XGBRegressor")
            previous_booster = model.get_booster()
            current_lr = model.get_params()['learning_rate']  # 현재 학습률 가져오기
            new_lr = current_lr/5  
            model.set_params(learning_rate=new_lr)
            model.set_params(n_estimators=int(model.get_params()['n_estimators'] / 4))
            model.set_params(early_stopping_rounds=None)
            num_rounds = model.get_params()['n_estimators']
            model.set_params(predictor='gpu_predictor')
            model.fit(self.last_x_train, self.last_y_train.values.ravel(), verbose=100)
            # model.train(dtrain, xgb_model=previous_booster, verbose=100)
            filename = self.MODEL_SAVE_PATH + 'last_model_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.last_x_train, self.last_y_train))
            
            ftr_importances_values = model.feature_importances_
            ftr_importances = pd.Series(ftr_importances_values, index = self.last_x_train.columns)
            ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            ftr_importances = ftr_importances.sort_values(ascending=False)  # 정렬 후 저장
            ftr_importances.to_csv(self.MODEL_SAVE_PATH + f'last_model_{str(i)}_importances.csv')

            logging.info(ftr_top20)   

        self.clsmodels = dict()
        self.models = dict()

        # self.clsmodels[0] = CatBoostClassifier()
        # self.clsmodels[0].load_model(self.MODEL_SAVE_PATH + 'last_clsmodel_0.cbm')

        # self.clsmodels[1] = CatBoostClassifier()
        # self.clsmodels[1].load_model(self.MODEL_SAVE_PATH + 'last_clsmodel_1.cbm')

        # self.clsmodels[2] = CatBoostClassifier()
        # self.clsmodels[2].load_model(self.MODEL_SAVE_PATH + 'last_clsmodel_2.cbm')

        # self.models[0] = CatBoostRegressor()
        # self.models[0].load_model(self.MODEL_SAVE_PATH + 'last_model_0.cbm')

        # self.models[1] = CatBoostRegressor()
        # self.models[1].load_model(self.MODEL_SAVE_PATH + 'last_model_1.cbm')   
        
        self.clsmodels[0] = joblib.load(self.MODEL_SAVE_PATH + 'last_clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(self.MODEL_SAVE_PATH + 'last_clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(self.MODEL_SAVE_PATH + 'last_clsmodel_2.sav')
        self.models[0] = joblib.load(self.MODEL_SAVE_PATH + 'last_model_0.sav')
        self.models[1] = joblib.load(self.MODEL_SAVE_PATH + 'last_model_1.sav')

        
        aidata_dir = self.conf['ROOT_PATH'] + '/ml_per_year/'
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
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble2'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble3'
            pred_col_list.append(pred_col_name)            

        with open("final_top_40_percent_average.txt", "r") as file:
            read_value = file.read()
            self.final_top_40_percent_average = float(read_value)  

        with open("drop_col_list.json", 'r') as f:
            self.drop_col_list = json.load(f)

        # START : 최근 년도 data 모두 읽고 최신 데이터만 남긴다.
        ldf = pd.DataFrame()
        for i in [1,2,3,4]:
            latest_data_path = aidata_dir + f'rnorm_ml_2024_Q{i}.csv'
            df = pd.read_csv(latest_data_path)
            ldf = pd.concat([ldf, df], axis=0)
        # year_period를 기준으로 내림차순 정렬
            pred_col_name = 'clsmodel_' + str(i) + '_prediction'
        ldf = ldf.sort_values(by='year_period', ascending=False)
        # symbol을 기준으로 중복 제거
        ldf = ldf.drop_duplicates(subset='symbol', keep='first')        

        symbol_means = ldf.groupby('symbol')['volume_mul_price'].mean().reset_index()
        top_symbols = symbol_means[symbol_means['volume_mul_price'] >= self.final_top_40_percent_average]
        ldf = ldf[ldf['symbol'].isin(top_symbols['symbol'])]
        
        ldf = ldf.drop(columns=self.drop_col_list, errors='ignore')
        ldf = ldf[ldf['price_diff_prediction'].isnull()]
        # 현재 rnorm_fs*.csv 파일 첫 column에 index 들어가 있어서 index drop 해주는 코드 넣음
        # print(ldf)
        # ldf = ldf.drop(df.columns[0], axis=1)
        # print(ldf)
        # END  : 최근 년도 data 모두 읽고 최신 데이터만 남긴다.
        self.sector_list = list(ldf['sector'].unique())
        self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
        ldf = ldf.drop('sector', axis=1) 
        print("before dtable len : ", len(ldf))
        ldf = self.drop_many_nan_row(ldf)
        print("after dtable len : ", len(ldf))
                
        latest_input = ldf[ldf.columns.difference(y_col_list)]
        latest_input = self.except_col_df(latest_input)
    
        latest_input['sector'] = latest_input['sector'].fillna("_missing_").astype(str)
        latest_input['sector'] = latest_input['sector'].astype('category')        

        # latest_input = latest_input.fillna(0)

        latest_input.sample(10).to_csv(self.MODEL_SAVE_PATH+"latest_input_tmp.csv")

        # self.x_train의 컬럼 목록과 self.ldf의 컬럼 목록을 set로 변환
        # x_train_columns = set(self.x_train.columns)
        # ldf_columns = set(latest_input)
        # # self.x_train에만 있는 컬럼
        # only_in_x_train = x_train_columns - ldf_columns
        # # self.ldf에만 있는 컬럼
        # only_in_ldf = ldf_columns - x_train_columns
        # # 결과 출력
        # print("self.x_train에만 있는 컬럼:", only_in_x_train)
        # print("self.x_train에만 있는 컬럼:", only_in_x_train)
        # print("self.x_train에만 있는 컬럼:", only_in_x_train)
        # print("latest_input에만 있는 컬럼:", only_in_ldf)
        # print("latest_input에만 있는 컬럼:", only_in_ldf)
        # print("latest_input에만 있는 컬럼:", only_in_ldf)
        
        # input = self.clean_feature_names(input)
        preds = np.empty((0, latest_input.shape[0]))
        for i, model in self.clsmodels.items():    
            logging.info(f"classification model # {i}")
            pred_col_name = 'clsmodel_' + str(i) + '_prediction'
            y_probs = model.predict_proba(latest_input)[:, 1]
            # 상위 10%의 예측 확률값을 threshold로 설정
            threshold = np.percentile(y_probs, THRESHOLD)
            # 새로운 threshold로 예측값 생성
            y_predict_binary = (y_probs > threshold).astype(int)
            logging.info(f"20% positive threshold == {threshold}")
            ldf[pred_col_name] = y_predict_binary
        
        for i, model in self.models.items():    
            pred_bin_col_name_0 = 'clsmodel_0_prediction'
            pred_bin_col_name_1 = 'clsmodel_1_prediction'
            pred_bin_col_name_2 = 'clsmodel_2_prediction'
            pred_col_name = 'model_' + str(i) + '_prediction'
            correct_col_name = 'clsmodel_' + str(i) + '_correct'
            pred_col_name_wbinary_0 = 'model_' + str(i) + '_prediction_wbinary_0'
            pred_col_name_wbinary_1 = 'model_' + str(i) + '_prediction_wbinary_1'
            pred_col_name_wbinary_2 = 'model_' + str(i) + '_prediction_wbinary_2'
            pred_col_name_wbinary_ensemble = 'model_' + str(i) + '_prediction_wbinary_ensemble'
            pred_col_name_wbinary_ensemble2 = 'model_' + str(i) + '_prediction_wbinary_ensemble2'
            pred_col_name_wbinary_ensemble3 = 'model_' + str(i) + '_prediction_wbinary_ensemble3'

            loss_col_name = 'model_' + str(i) + '_prediction_loss'
            loss_bin_col_name_0 = 'model_' + str(i) + '_prediction_wbinary_loss_0'
            loss_bin_col_name_1 = 'model_' + str(i) + '_prediction_wbinary_loss_1'
            loss_bin_col_name_2 = 'model_' + str(i) + '_prediction_wbinary_loss_2'

            y_predict = model.predict(latest_input)
            
            ldf[pred_col_name] = y_predict
            ldf[pred_col_name_wbinary_0] = np.where(ldf[pred_bin_col_name_0] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_1] = np.where(ldf[pred_bin_col_name_1] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_2] = np.where(ldf[pred_bin_col_name_2] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_ensemble] = np.where(((ldf[pred_bin_col_name_0] == 0) | (ldf[pred_bin_col_name_1] == 0)), -1, y_predict)
            ldf[pred_col_name_wbinary_ensemble2] = np.where(((ldf[pred_bin_col_name_1] == 0) | (ldf[pred_bin_col_name_2] == 0)), -1, y_predict)
            
            condition = (
                (ldf[[pred_bin_col_name_0, pred_bin_col_name_1, pred_bin_col_name_2]] == 0).sum(axis=1) >= 2
            )

            ldf[pred_col_name_wbinary_ensemble3] = np.where(condition, -1, y_predict)            
            preds = np.vstack((preds, y_predict[None,:]))

        ldf['ai_pred_avg'] = np.average(preds, axis=0)        
        ldf.to_csv(self.MODEL_SAVE_PATH+"latest_prediction.csv")
        topk_list = [(0,3), (0,7), (0, 15)]
        for s, e in topk_list:
            # logging.info("top" + str(s) + " ~ " + str(e))
            for col in pred_col_list:
                top_k_df = ldf.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                top_k_df.to_csv(self.MODEL_SAVE_PATH+'latest_prediction_{}_top{}-{}.csv'.format(col, s, e))

        # if PER_SECTOR == True:
    
        #     pred_col_list = ['ai_pred_avg'] 
        #     for i in range(2):
        #         pred_col_name = 'model_' + str(i) + '_prediction'
        #         pred_col_list.append(pred_col_name)
        #         pred_col_name = 'model_' + str(i) + '_prediction_wbinary_0'
        #         pred_col_list.append(pred_col_name)
        #         pred_col_name = 'model_' + str(i) + '_prediction_wbinary_1'
        #         pred_col_list.append(pred_col_name)
        #         pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble'
        #         pred_col_list.append(pred_col_name)     

        #     ldf = pd.DataFrame()
        #     for i in [1,2,3,4]:
        #         latest_data_path = aidata_dir + f'rnorm_ml_2024_Q{i}.csv'
        #         df = pd.read_csv(latest_data_path)
        #         ldf = pd.concat([ldf, df], axis=0)
        #     # year_period를 기준으로 내림차순 정렬    
        #     ldf = ldf.sort_values(by='year_period', ascending=False)
        #     # symbol을 기준으로 중복 제거
        #     ldf = ldf.drop_duplicates(subset='symbol', keep='first')        
            
        #     symbol_means = ldf.groupby('symbol')['volume_mul_price'].mean().reset_index()
        #     top_symbols = symbol_means[symbol_means['volume_mul_price'] >= self.final_top_40_percent_average]
        #     ldf = ldf[ldf['symbol'].isin(top_symbols['symbol'])]                
            
        #     ldf = ldf.drop(columns=self.drop_col_list, errors='ignore')
            
        #     self.sector_list = list(ldf['sector'].unique())
        #     self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
        #     ldf = self.drop_many_nan_row(ldf)
        #     print("after dtable len : ", len(ldf))
                        
            
        #     self.sector_models = dict()
        #     for sec in self.sector_list:
        #         for i in range(2):
        #             cur_sec_model_path = self.MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))
        #             k = (sec, i)
        #             print("model path : ", cur_sec_model_path)
        #             self.sector_models[k] = joblib.load(cur_sec_model_path)
            
        #     all_preds = []
        #     for sec in self.sector_list:
        #         sec_df = ldf[ldf['sector']==sec]
        #         sec_df = sec_df.drop('sector', axis=1) 
        #         indata = sec_df[sec_df.columns.difference(['symbol'])]
        #         print(indata)
        #         indata = indata[indata.columns.difference(y_col_list)]
        #         indata = self.except_col_df(indata)
        #         preds = np.empty((0, indata.shape[0]))

        #         for i in range(2):
        #             k = (sec, i)
        #             model = self.sector_models[k]
        #             pred_col_name = 'model_' + str(i) + '_prediction'
        #             y_predict3 = model.predict(indata)
        #             sec_df[pred_col_name] = y_predict3
        #             preds = np.vstack((preds, y_predict3[None,:]))
                    
        #         sec_df['ai_pred_avg'] = np.average(preds, axis=0)        
        #         sec_df.to_csv(self.MODEL_SAVE_PATH+"sec_{}_latest_prediction.csv".format(sec))        
        #         topk_list = [(0,3), (0,7), (0, 15)]
        #         for s, e in topk_list:
        #             logging.info("top" + str(s) + " ~ " + str(e))
        #             for col in pred_col_list:
        #                 top_k_df = sec_df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
        #                 top_k_df.to_csv(self.MODEL_SAVE_PATH+'latest_prediction_{}_{}_top{}-{}.csv'.format(col, sec, s, e))
        #                 symbols = top_k_df['symbol'].to_list()
        #                 preds = top_k_df[col].to_list()
        #                 for i, sym in enumerate(symbols):
        #                     all_preds.append([(e-s), sec, col, i, sym, preds[i]])
                
        #         col_name = ['k', 'sector', 'model', 'i', 'symbol', 'pred']
        #         pred_df = pd.DataFrame(all_preds, columns=col_name)
        #         pred_df.to_csv(self.MODEL_SAVE_PATH+'allsector_latest_pred_df.csv', index=False)
                