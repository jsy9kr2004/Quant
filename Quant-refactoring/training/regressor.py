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

from datasets import Dataset
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

#TODO: PER_SECTOR True일 때 처리 모두 짜야함(sector 별 예측)
PER_SECTOR=False
MODEL_SAVE_PATH=""
THRESHOLD=92

# 입력으로 넣으면 안되는 colume들이라 중간에 drop하기 위한 list
y_col_list= ["symbol",
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
    def __init__(self, conf):
        self.conf = conf
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        print(self.conf)
        aidata_dir = conf['ROOT_PATH'] + '/ml_per_year/'
        print("aidata path : " + aidata_dir)
        if not os.path.exists(aidata_dir):
            print("there is no ai data : " + aidata_dir)
            return 
            
        self.train_files = []
        for year in range(int(conf['TRAIN_START_YEAR']), int(conf['TRAIN_END_YEAR'])+1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.csv"
                self.train_files.append(path)
        
        self.test_files = []
        for year in range(int(conf['TEST_START_YEAR']), int(conf['TEST_END_YEAR'])+1):
            for Q in ['Q1', 'Q2', 'Q3', 'Q4']:
                path = aidata_dir + "rnorm_ml_" + str(year) + f"_{Q}.csv"
                self.test_files.append(path)
                
        print("train file list : ", self.train_files)
        print("test file list : ", self.test_files)
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.test_df_list = []
        self.n_sector = 0
        self.sector_list = []
        self.sector_train_dfs = dict()
        self.sector_test_dfs = dict()
        self.sector_test_df_lists = []
        self.clsmodels = dict()

        self.models = dict()
        self.sector_models=dict()
        self.sector_cls_models=dict()
        self.sector_x_train = dict()
        self.sector_y_train = dict()

        self.drop_col_list = []
        
    def clean_feature_names(self, df):
        # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
        new_n_list = list(new_names.values())
        # [LightGBM] Feature appears more than one time.
        new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
        df = df.rename(columns=new_names)        
        return df

    def dataload(self):
        for fpath in self.train_files:
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['price_diff'])
            self.train_df = pd.concat([self.train_df, df], axis=0)
        # 의미없는 column 날리기
        # 기준 설정
        missing_threshold = 0.8
        same_value_threshold = 0.95
        # 의미 없는 열을 찾기 위한 리스트 초기화
        columns_to_drop = []
        # 각 열에 대해 비어 있는 값의 비율 계산 및 기준 초과하는 열 추가
        for col in self.train_df.columns:
            missing_ratio = self.train_df[col].isna().mean()
            if missing_ratio > missing_threshold:
                columns_to_drop.append(col)
            else:
                # 고유 값의 비율 계산
                top_value_ratio = self.train_df[col].value_counts(normalize=True, dropna=False).iloc[0]
                if top_value_ratio > same_value_threshold:
                    columns_to_drop.append(col)
        columns_to_drop = [col for col in columns_to_drop if col not in y_col_list]
        # 의미 없는 열 제거
        self.train_df = self.train_df.drop(columns=columns_to_drop)
        self.drop_col_list = columns_to_drop
        # for c in columns_to_drop:
        #     print(c)
        print(f'Removed columns # : {len(columns_to_drop)}')
        print(f'Cleaned DataFrame shape: {self.train_df.shape}')

        print("in train set before dtable len : ", len(self.train_df))
        self.train_df['nan_count_per_row'] = self.train_df.isnull().sum(axis=1)        
        # 40%도 값이 없는 row 날리기(60%까지 nan이여도 남김)
        filtered_row = self.train_df['nan_count_per_row'] < int(len(self.train_df.columns)*0.6) 
        self.train_df = self.train_df.loc[filtered_row,:]
        print("in train set after dtable len : ", len(self.train_df))            

        # TODO : 이건 make_mldata에서 처리할 것이기 때문에 지워야 됨 
        self.train_df["sector"] = self.train_df["industry"].map(sector_map)                         
        sector_list = list(self.train_df['sector'].unique())
        sector_list = [x for x in sector_list if str(x) != 'nan']
        for sec in sector_list:
            sec_mask = self.train_df['sector'] == sec
            sec_mean = self.train_df.loc[sec_mask, 'price_dev'].mean()
            self.train_df.loc[sec_mask, 'sec_price_dev_subavg'] = self.train_df.loc[sec_mask, 'price_dev'] - sec_mean
            
        if PER_SECTOR == True:
            print(self.train_df['sector'].value_counts())
            self.sector_list = list(self.train_df['sector'].unique())
            self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
            for sec in self.sector_list:
                self.sector_train_dfs[sec] = self.train_df[self.train_df['sector']==sec].copy()
                print(self.sector_train_dfs[sec])

        self.test_df_list = []
        for fpath in self.test_files:
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['price_diff'])
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            print("in test set before dtable len : ", len(df))
            df['nan_count_per_row'] = df.isnull().sum(axis=1)
            filtered_row = df['nan_count_per_row'] < int(len(df.columns)*0.6)
            df = df.loc[filtered_row,:]
            print("in test set after dtable len : ", len(df))

            # TODO : 이건 make_mldata에서 처리할 것이기 때문에 지워야 됨 
            df["sector"] = df["industry"].map(sector_map)                 
            sector_list = list(df['sector'].unique())
            sector_list = [x for x in sector_list if str(x) != 'nan']
            for sec in sector_list:
                sec_mask = df['sector'] == sec
                sec_mean = df.loc[sec_mask, 'price_dev'].mean()
                df.loc[sec_mask, 'sec_price_dev_subavg'] = df.loc[sec_mask, 'price_dev'] - sec_mean
                
            # df = df.fillna(0)
            self.test_df = pd.concat([self.test_df, df], axis=0)
            self.test_df_list.append([fpath, df])
            if PER_SECTOR == True:
                for sec in self.sector_list:
                    self.sector_test_df_lists.append([fpath, df[df['sector']==sec].copy(), sec])
    
        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)
        logging.debug("test_df shape : ")
        logging.debug(self.test_df.shape)

        # self.train_df.to_csv(self.conf['ROOT_PATH'] + '/train_df.csv', index=False)
        # self.test_df.to_csv(self.conf['ROOT_PATH'] + '/test_df.csv', index=False)

        positive_count = (self.train_df['price_dev'] > 0).sum()
        negative_count = (self.train_df['price_dev'] < 0).sum()
        logging.info("positive # : {}, negative # : {}".format(positive_count, negative_count))
        
        self.x_train = self.train_df[self.train_df.columns.difference(y_col_list)]
        self.y_train = self.train_df[['price_dev_subavg']]
        self.y_train_cls = self.train_df[['price_dev']]

        for sec in self.sector_list:
            print("sector : ", sec)
            self.sector_x_train[sec] = self.sector_train_dfs[sec][self.sector_train_dfs[sec].columns.difference(y_col_list)]
            self.sector_y_train[sec] = self.sector_train_dfs[sec][['sec_price_dev_subavg']]
        
        self.x_test = self.test_df[self.test_df.columns.difference(y_col_list)]
        self.y_test = self.test_df[['price_dev_subavg']]
        self.y_test_cls = self.test_df[['price_dev']]
    
    def def_model(self):
        self.clsmodels[0] = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=500, learning_rate=0.1, gamma=0, subsample=0.8,
                                        colsample_bytree=0.8, max_depth=8, objective='binary:logistic', eval_metric='logloss')
        self.clsmodels[1] = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=500, learning_rate=0.1, gamma=0, subsample=0.8,
                                        colsample_bytree=0.8, max_depth=9, objective='binary:logistic', eval_metric='logloss')
        self.clsmodels[2] = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=500, learning_rate=0.1, gamma=0, subsample=0.8,
                                        colsample_bytree=0.8, max_depth=10, objective='binary:logistic', eval_metric='logloss')
        # LightGBM Classification Model
        self.clsmodels[3] = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', n_estimators=1000, max_depth=8, learning_rate=0.1, 
                                               device='gpu', boost_from_average=False)
        # grid search로 찾은 lgb parameter : {'learning_rate': 0.01, 'max_depth': 6, 'min_child_samples': 30, 'n_estimators': 1000, 'num_leaves': 31}
            
        self.models[0] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.1, gamma=0, subsample=0.8,
                                        colsample_bytree=0.8, max_depth=8, objective='reg:squarederror', eval_metric='rmse')    
        self.models[1] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.1, gamma=0, subsample=0.8,
                                        colsample_bytree=0.8, max_depth=10, objective='reg:squarederror', eval_metric='rmse')
        
        # LightGBM Regression Model -> 정확도 안나와서 사용 안함
        # self.models[1] = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',  max_depth=8, learning_rate=0.1, n_estimators=1000,
        #                                    subsample=0.8, colsample_bytree=0.8, device='gpu')
        
    
        for sec in self.sector_list:
            cur_key = (sec, 0)
            self.sector_models[cur_key] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0.01, subsample=0.8,
                                        colsample_bytree=0.7, max_depth=7) #BEST
            cur_key = (sec, 1)
            self.sector_models[cur_key]  = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0.01, subsample=0.8,
                                        colsample_bytree=0.7, max_depth=8)    
    def train(self):
        
        # 파라미터 그리드 설정 lgb
        # param_grid = {
        #     'n_estimators': [1000],
        #     'max_depth': [6, 8, 10, 12],
        #     'learning_rate': [0.01, 0.05, 0.1],
        #     'num_leaves': [31, 50, 70],
        #     'min_child_samples': [20, 30, 40]
        # }
        # # LGBM 모델 초기화
        # lgbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', device='gpu', boost_from_average=False)
        # # GridSearchCV 초기화
        # grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        # 모델 학습
        # self.x_train = self.clean_feature_names(self.x_train)
        # 레이블을 0과 1로 변환 (학습 직전)
        # y_train_binary = (self.y_train_cls > 0).astype(int)        
        # grid_search.fit(self.x_train, y_train_binary)
        # # 최적 파라미터 출력
        # print("Best parameters found: ", grid_search.best_params_)
        # print("Best accuracy: ", grid_search.best_score_)
        # exit()
        
        # # for parameter tuning XGB
        # params = {
        #     'learning_rate': np.arange(0.05, 0.3, 0.05),
        #     'max_depth': range(3, 10),
        #     'n_estimators': range(50, 500, 50),
        #     'colsample_bytree': np.arange(0.3, 1.0, 0.1),
        #     'subsample': np.arange(0.5, 1.0, 0.1),
        #     'gamma': [0, 1, 5]
        # }
        # # Create an XGBoost regressor
        # xgb = xgboost.XGBRegressor()
        # # Choose a cross-validation strategy
        # cv = KFold(n_splits=5, shuffle=True)
        # # Define the objective function
        # objective = 'reg:squarederror'
        # # Define the optimization algorithm
        # algorithm = 'random'
        # # Define the search method
        # search = RandomizedSearchCV(xgb, params, n_iter=100, cv=cv, scoring='neg_mean_squared_error', random_state=42)
        # # Fit the model
        # search.fit(self.x_train, self.y_train.values.ravel())
        # # Print the best hyperparameters
        # print(search.best_params_)
        # exit()
        
        MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS/'
        self.def_model()
        
        if not os.path.exists(MODEL_SAVE_PATH):
            print("creating MODELS path : " + MODEL_SAVE_PATH)
            os.makedirs(MODEL_SAVE_PATH)

        self.x_train = self.clean_feature_names(self.x_train)
        # 레이블을 0과 1로 변환 (학습 직전)
        y_train_binary = (self.y_train_cls > 0).astype(int)
        for i, model in self.clsmodels.items():
            logging.info("start fitting classifier")
            model.fit(self.x_train, y_train_binary)
            filename = MODEL_SAVE_PATH + 'clsmodel_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, y_train_binary))
            
        for i, model in self.models.items():
            logging.info("start fitting XGBRegressor")
            model.fit(self.x_train, self.y_train.values.ravel())
            filename = MODEL_SAVE_PATH + 'model_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, self.y_train))
            # logging.info("end fitting RandomForestRegressor")
            # ftr_importances_values = model.feature_importances_
            # ftr_importances = pd.Series(ftr_importances_values, index = self.x_train.columns)
            # ftr_importances.to_csv(MODEL_SAVE_PATH+'model_importances.csv')
            # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            # logging.info(ftr_top20)   
        
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
                    # ftr_importances_values = model.feature_importances_
                    # ftr_importances = pd.Series(ftr_importances_values, index = self.sector_x_train[sec].columns)
                    # ftr_importances.to_csv(MODEL_SAVE_PATH + sec + '_model_importances.csv')
                    # ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
                    # logging.info(ftr_top20)

           
    def evaluation(self):

        MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS/'
        self.models = dict()
        self.clsmodels = dict()
        self.clsmodels[0] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_2.sav')
        self.clsmodels[3] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_3.sav')
        
        self.models[0] = joblib.load(MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(MODEL_SAVE_PATH + 'model_1.sav')

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
            y_test = df[['price_dev_subavg']]
            y_test_cls = df[['price_dev']]
            y_test_binary = (y_test_cls > 0).astype(int)

            preds = np.empty((0,x_test.shape[0]))            
            
            df['label'] = y_test
            df['label_binary'] = y_test_binary
            
            x_test = self.clean_feature_names(x_test)
            
            for i, model in self.clsmodels.items():    
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

                y_predict = model.predict(x_test)
                
                df[pred_col_name] = y_predict
                df[pred_col_name_wbinary_0] = np.where(df[pred_bin_col_name_0] == 0, -1, y_predict)
                df[pred_col_name_wbinary_1] = np.where(df[pred_bin_col_name_1] == 0, -1, y_predict)
                df[pred_col_name_wbinary_2] = np.where(df[pred_bin_col_name_2] == 0, -1, y_predict)
                df[pred_col_name_wbinary_3] = np.where(df[pred_bin_col_name_3] == 0, -1, y_predict)
                df[pred_col_name_wbinary_ensemble] = np.where(((df[pred_bin_col_name_1] == 0) | (df[pred_bin_col_name_3] == 0)), -1, y_predict)
                df[pred_col_name_wbinary_ensemble2] = np.where(((df[pred_bin_col_name_1] == 0) | (df[pred_bin_col_name_2] == 0)), -1, y_predict)
                
                condition = (
                    (df[[pred_bin_col_name_1, pred_bin_col_name_2, pred_bin_col_name_3]] == 0).sum(axis=1) >= 2
                )

                df[pred_col_name_wbinary_ensemble3] = np.where(condition, -1, y_predict)
                
                preds = np.vstack((preds, y_predict[None,:]))
                df[loss_col_name] = abs(df['label'] - y_predict)
                df[loss_bin_col_name_0] = abs(df['label'] - df[pred_col_name_wbinary_0])
                df[loss_bin_col_name_1] = abs(df['label'] - df[pred_col_name_wbinary_1])
                df[loss_bin_col_name_2] = abs(df['label'] - df[pred_col_name_wbinary_2])
                df[loss_bin_col_name_3] = abs(df['label'] - df[pred_col_name_wbinary_3])
                logging.info(f"eval : model i : {i} loss : {df[loss_col_name].mean()} loss_wbin_0 {df[loss_bin_col_name_0].mean()} loss_wbin_1 {df[loss_bin_col_name_1].mean()}\
                    loss_wbin_2 {df[loss_bin_col_name_2].mean()} loss_wbin_3 {df[loss_bin_col_name_3].mean()}")
                if test_idx != 0:
                    logging.info(f"accumulated eval : model i : {i} loss : {full_df[loss_col_name].mean()} loss_wbin_0 {full_df[loss_bin_col_name_0].mean()} loss_wbin_1 {full_df[loss_bin_col_name_1].mean()}\
                    loss_wbin_2 {full_df[loss_bin_col_name_2].mean()} loss_wbin_3 {full_df[loss_bin_col_name_3].mean()}")

            
            df['ai_pred_avg'] = np.average(preds, axis=0)
            df['ai_pred_avg_loss'] = abs(df['label']-df['ai_pred_avg'])
            full_df = pd.concat([full_df, df], ignore_index=True)
            df.to_csv(MODEL_SAVE_PATH + "prediction_ai_{}.csv".format(tdate))

            # 각 model의 top_k 종목의 period_price_diff 합을 구해서 model 최종 평가
            topk_period_earning_sums = []
            topk_list = [(0,3), (0,7), (0,15)]
            for s, e in topk_list:
                logging.info("top" + str(s) + " ~ "  + str(e) )
                k = str(s) + '~' + str(e)
                for col in pred_col_list:
                    top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                    logging.info("")
                    logging.info(col)
                    logging.info(("label"))
                    logging.info((top_k_df['price_dev'].sum()/(e-s+1)))
                    logging.info(("pred"))
                    logging.info((top_k_df[col].sum()/(e-s+1)))
                    topk_period_earning_sums.append(top_k_df['price_dev'].sum())
                    
                    top_k_df.to_csv(MODEL_SAVE_PATH+'prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))
                    model_eval_hist.append([tdate, col, k, top_k_df['price_dev'].sum()/(e-s+1), top_k_df[col].sum()/(e-s+1), 
                                            abs(top_k_df[col].sum()/(e-s+1) - top_k_df['price_dev'].sum()/(e-s+1)), int(top_k_df[col].sum()/(e-s+1) > 0), 
                                            top_k_df['ai_pred_avg'].sum()/(e-s+1), top_k_df['model_0_prediction'].sum()/(e-s+1), top_k_df['model_1_prediction'].sum()/(e-s+1),
                                            top_k_df['model_0_prediction_wbinary_0'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_0'].sum()/(e-s+1),
                                            top_k_df['model_0_prediction_wbinary_1'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_1'].sum()/(e-s+1),
                                            top_k_df['model_0_prediction_wbinary_2'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_2'].sum()/(e-s+1),
                                            top_k_df['model_0_prediction_wbinary_3'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_3'].sum()/(e-s+1),
                                            top_k_df['model_0_prediction_wbinary_ensemble'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_ensemble'].sum()/(e-s+1),
                                            top_k_df['model_0_prediction_wbinary_ensemble2'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_ensemble2'].sum()/(e-s+1),
                                            top_k_df['model_0_prediction_wbinary_ensemble3'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_ensemble3'].sum()/(e-s+1)
                                            ])
        
        col_name = ['start_date', 'model', 'krange', 'avg_earning_per_stock', 'cur_model_pred', 'loss_y_and_pred', 
                    'cur_model_pred_ispositive', 'avg_pred', 'model0_pred', 'model1_pred', 
                    'model0_pred_wbinary_0', 'model1_pred_wbinary_0',
                    'model0_pred_wbinary_1', 'model1_pred_wbinary_1', 
                    'model0_pred_wbinary_2', 'model1_pred_wbinary_2',
                    'model0_pred_wbinary_3', 'model1_pred_wbinary_3',
                    'model0_pred_wbinary_ensemble', 'model1_pred_wbinary_ensemble',
                    'model0_pred_wbinary_ensemble2', 'model1_pred_wbinary_ensemble2',
                    'model0_pred_wbinary_ensemble3', 'model1_pred_wbinary_ensemble3'
                    ]
        
        pred_df = pd.DataFrame(model_eval_hist, columns=col_name)
        logging.info(pred_df)
        pred_df.to_csv(MODEL_SAVE_PATH+'pred_df_topk.csv', index=False)
        full_df.to_csv(MODEL_SAVE_PATH+'prediction_ai.csv', index=False)

        if PER_SECTOR == True:
            testdates=set()
            allsector_topk_df=pd.DataFrame()
            self.sector_models = dict()
            for sec in self.sector_list:
                for i in range(2):
                    filename = MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))
                    k = (sec, i)
                    self.sector_models[k] = joblib.load(MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))
            sector_model_eval_hist = []
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
                
                sector_preds = np.empty((0,x_test.shape[0]))
                df['label'] = y_test
                
                
                y_probs = self.clsmodels[2].predict_proba(x_test)[:, 1]
                # 상위 10%의 예측 확률값을 threshold로 설정
                threshold = np.percentile(y_probs, THRESHOLD)
                # 새로운 threshold로 예측값 생성
                y_predict_binary = (y_probs > threshold).astype(int)
                
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
                # 각 model의 top_k 종목의 period_price_diff 합을 구해서 model 최종 평가
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
                        sector_model_eval_hist.append([tdate, sec, col, k, top_k_df['price_dev'].sum()/(e-s+1), top_k_df[col].sum()/(e-s+1), 
                                            abs(top_k_df[col].sum()/(e-s+1) - top_k_df['price_dev'].sum()/(e-s+1)), int(top_k_df[col].sum()/(e-s+1) > 0), 
                                            top_k_df['ai_pred_avg'].sum()/(e-s+1), top_k_df['model_0_prediction'].sum()/(e-s+1), 
                                            top_k_df['model_1_prediction'].sum()/(e-s+1), 
                                            top_k_df['model_0_prediction_wbinary_2'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary_2'].sum()/(e-s+1)])

            
            col_name = ['start_date', 'sector', 'model', 'krange', 'avg_earning_per_stock', 'cur_model_pred', 'loss_y_and_pred', 
                    'cur_model_pred_ispositive', 'avg_pred', 'model0_pred', 'model1_pred',
                    'model0_pred_wbinary_2', 'model1_pred_wbinary_2']
            pred_df = pd.DataFrame(sector_model_eval_hist, columns=col_name)
            print(pred_df)
            pred_df.to_csv(MODEL_SAVE_PATH+'allsector_pred_df.csv'.format(sec), index=False)
        
        
    def latest_prediction(self):
        
        MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS/'
        self.clsmodels = dict()
        self.clsmodels[0] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_2.sav')
        self.clsmodels[3] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_3.sav')
        self.models = dict()
        self.models[0] = joblib.load(MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(MODEL_SAVE_PATH + 'model_1.sav')
        
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
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_3'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble2'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary_ensemble3'
            pred_col_list.append(pred_col_name)            

        # START : 최근 년도 data 모두 읽고 최신 데이터만 남긴다.
        ldf = pd.DataFrame()
        for i in [1,2,3,4]:
            latest_data_path = aidata_dir + f'rnorm_fs_2024_Q{i}.csv'
            df = pd.read_csv(latest_data_path)
            ldf = pd.concat([ldf, df], axis=0)
        # year_period를 기준으로 내림차순 정렬
        ldf = ldf.sort_values(by='year_period', ascending=False)
        # symbol을 기준으로 중복 제거
        ldf = ldf.drop_duplicates(subset='symbol', keep='first')        
        ldf = ldf.drop(columns=self.drop_col_list, errors='ignore')
        # 현재 rnorm_fs*.csv 파일 첫 column에 index 들어가 있어서 index drop 해주는 코드 넣음
        ldf = ldf.drop(df.columns[0], axis=1)
        # END  : 최근 년도 data 모두 읽고 최신 데이터만 남긴다.
        
        self.sector_list = list(ldf['sector'].unique())
        self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
        ldf = ldf.drop('sector', axis=1) 
        print("before dtable len : ", len(ldf))
        ldf['nan_count_per_row'] = ldf.isnull().sum(axis=1)
        filtered_row = ldf['nan_count_per_row'] < int(len(ldf.columns)*0.6)
        ldf = ldf.loc[filtered_row,:]
        print("after dtable len : ", len(ldf))
                
        input = ldf[ldf.columns.difference(y_col_list)]        
        input = self.clean_feature_names(input)
        preds = np.empty((0, input.shape[0]))
        for i, model in self.clsmodels.items():    
            logging.info(f"classification model # {i}")
            pred_col_name = 'clsmodel_' + str(i) + '_prediction'
            y_probs = model.predict_proba(input)[:, 1]
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

            y_predict = model.predict(input)
            
            ldf[pred_col_name] = y_predict
            ldf[pred_col_name_wbinary_0] = np.where(ldf[pred_bin_col_name_0] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_1] = np.where(ldf[pred_bin_col_name_1] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_2] = np.where(ldf[pred_bin_col_name_2] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_3] = np.where(ldf[pred_bin_col_name_3] == 0, -1, y_predict)
            ldf[pred_col_name_wbinary_ensemble] = np.where(((ldf[pred_bin_col_name_1] == 0) | (ldf[pred_bin_col_name_3] == 0)), -1, y_predict)
            ldf[pred_col_name_wbinary_ensemble2] = np.where(((ldf[pred_bin_col_name_1] == 0) | (ldf[pred_bin_col_name_2] == 0)), -1, y_predict)
            
            condition = (
                (ldf[[pred_bin_col_name_1, pred_bin_col_name_2, pred_bin_col_name_3]] == 0).sum(axis=1) >= 2
            )

            ldf[pred_col_name_wbinary_ensemble3] = np.where(condition, -1, y_predict)            
            preds = np.vstack((preds, y_predict[None,:]))

        ldf['ai_pred_avg'] = np.average(preds, axis=0)        
        ldf.to_csv(MODEL_SAVE_PATH+"latest_prediction.csv")
        topk_list = [(0,3), (0,7), (0, 15)]
        for s, e in topk_list:
            logging.info("top" + str(s) + " ~ " + str(e))
            for col in pred_col_list:
                top_k_df = ldf.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                top_k_df.to_csv(MODEL_SAVE_PATH+'latest_prediction_{}_top{}-{}.csv'.format(col, s, e))

        if PER_SECTOR == True:
            self.sector_models = dict()
            ldf = pd.read_csv(latest_data_path)
            for sec in self.sector_list:
                for i in range(2):
                    filename = MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i))
                    k = (sec, i)
                    print("model path : ", MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))
                    self.sector_models[k] = joblib.load(MODEL_SAVE_PATH + '{}_model_{}.sav'.format(sec, str(i)))
            
            all_preds = []
            for sec in self.sector_list:
                sec_df = ldf[ldf['sector']==sec]
                sec_df = sec_df.drop('sector', axis=1) 
                indata = sec_df[sec_df.columns.difference(['symbol'])]
                print(indata)
                preds = np.empty((0, indata.shape[0]))

                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    y_predict3 = model.predict(indata)
                    sec_df[pred_col_name] = y_predict3
                    preds = np.vstack((preds, y_predict3[None,:]))
                    
                sec_df['ai_pred_avg'] = np.average(preds, axis=0)        
                sec_df.to_csv(MODEL_SAVE_PATH+"sec_{}_latest_prediction.csv".format(sec))        
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
                
                col_name = ['k', 'sector', 'model', 'i', 'symbol', 'pred']
                pred_df = pd.DataFrame(all_preds, columns=col_name)
                pred_df.to_csv(MODEL_SAVE_PATH+'allsector_latest_pred_df.csv', index=False)