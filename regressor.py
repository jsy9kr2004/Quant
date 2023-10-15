import glob
import joblib
import logging
import torch
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as nn_f
import torch.optim as optim
from dateutil.relativedelta import relativedelta
import datetime

from datasets import Dataset
from g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map, sparse_col_list
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from torch.utils.data import DataLoader
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# from torch.utils.tensorboard import SummaryWriter
# import torch.utils.data as data_utils
# from xml.dom import XHTML_NAMESPACE
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error

USE_COL='FULL'
PER_SECTOR=False
MODEL_SAVE_PATH=""
y_col_list=["period_price_diff","earning_diff","sector", "symbol"]


if USE_COL == 'FULL':
    ratio_col_list_wdiff = list(map(lambda x: "diff" + x, meaning_col_list))
    ratio_col_list_wewm = list(map(lambda x: "EWM_" + x, meaning_col_list))
    ratio_col_list_woverMC = list(map(lambda x: "OverMC_" + x, meaning_col_list))
    ratio_col_list_wadaMC = list(map(lambda x: 'adaptiveMC_' + x, cal_ev_col_list))
    ratio_col_list_wprev = ratio_col_list + ratio_col_list_wdiff + ratio_col_list_wewm + ratio_col_list_woverMC + ratio_col_list_wadaMC
    
    # ratio_col_list_wprev - sparse_cols
    # ratio_col_list_wprev = [x for x in ratio_col_list_wprev if x not in sparse_col_list]
    
    ratio_col_list_wprev = list(map(lambda x: x+'_normal', ratio_col_list_wprev))
    ratio_col_list_wprev = ratio_col_list_wprev+ y_col_list

if USE_COL == '97':
    print("just use ratio_col_list")
    ratio_col_list_wprev = list(map(lambda x: x+'_normal', ratio_col_list))
    ratio_col_list_wprev = ratio_col_list_wprev + y_col_list

class Regressor:
    def __init__(self, conf):
        self.conf = conf
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        print(self.conf)
        aidata_dir = conf['ROOT_PATH'] + '/regressor_data_p{0:02d}_m{1:02d}/'.format(self.conf['REBALANCE_PERIOD'], self.conf['START_MONTH'])
        print("aidata path : " + aidata_dir)
        if not os.path.exists(aidata_dir):
            print("there is no ai data : " + aidata_dir)
            return 
            
        self.train_files = []
        for year in range(int(conf['TRAIN_START_YEAR']), int(conf['TRAIN_END_YEAR'])):
                path = aidata_dir + str(year) + "*train_norm.csv"
                year_files = [file for file in glob.glob(path)]
                year_files = [file for file in year_files if 'full' not in file]
                self.train_files.extend(year_files)
        
        self.test_files = []
        for year in range(int(conf['TEST_START_YEAR']), int(conf['TEST_END_YEAR'])+1):
                path = aidata_dir + str(year) + "*train_norm.csv"
                year_files = [file for file in glob.glob(path)]
                year_files = [file for file in year_files if 'full' not in file]
                self.test_files.extend(year_files)
                
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
        self.sector_x_train = dict()
        self.sector_y_train = dict()

    def dataload(self):
        
        # TODO: remove after re-creating DTABLE
        # 절대값 column 제거
        # abs_col_list = list(set(meaning_col_list) - set(ratio_col_list))
        # abs_col_list = [col + '_sorted_normal' for col in abs_col_list]
        # print("drop abs_col_list : ")
        # print(abs_col_list)
        #### 
        
        for fpath in self.train_files:
            print(fpath)
            df = pd.read_csv(fpath)
            # df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.dropna(axis=0, subset=['period_price_diff'])
            
            # TODO: remove after re-creating DTABLE
            # df = df.drop(abs_col_list, axis=1) 
            ##### 
            
            print("before drop null > 500")
            logging.debug(df.shape)  
            df = df[df.isnull().sum(axis=1) < 500]
            print("after drop null > 500")
            logging.debug(df.shape)  
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]       
            self.train_df = pd.concat([self.train_df, df], axis=0)

        if PER_SECTOR == True:
            print(self.train_df['sector'].value_counts())
            self.sector_list = list(self.train_df['sector'].unique())
            self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
            print(self.sector_list)
            for sec in self.sector_list:
                print("sector : ", sec)
                self.sector_train_dfs[sec] = self.train_df[self.train_df['sector']==sec].copy()
                self.sector_train_dfs[sec] = self.sector_train_dfs[sec].fillna(0)
                print(self.sector_train_dfs[sec])
            print(self.sector_list)

        self.test_df_list = []
        for fpath in self.test_files:
            print(fpath)
            df = pd.read_csv(fpath)
            
            # df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.dropna(axis=0, subset=['period_price_diff'])
            # df = df.loc[:, ratio_col_list_wprev]
            
            # TODO: remove after re-creating DTABLE
            # df = df.drop(abs_col_list, axis=1) 
            ##
            
            logging.debug(df.shape)  
            df = df[df.isnull().sum(axis=1) < 500]
            logging.debug(df.shape)  
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]       
            df = df.fillna(0)
            self.test_df = pd.concat([self.test_df, df], axis=0)
            self.test_df_list.append([fpath, df])
            if PER_SECTOR == True:
                for sec in self.sector_list:
                    self.sector_test_df_lists.append([fpath, df[df['sector']==sec].copy(), sec])
    
        
        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)    
        logging.debug('NaN occurrences in Columns:')
        logging.debug(self.train_df.isnull().sum(axis=0))
        logging.debug('NaN occurrences in Rows:')
        logging.debug(self.train_df.isnull().sum(axis=1))
        self.train_df = self.train_df.fillna(0)
        self.test_df = self.test_df.fillna(0)
        
        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)
        logging.debug("test_df shape : ")
        logging.debug(self.test_df.shape)
    
        
        self.x_train = self.train_df[self.train_df.columns.difference(y_col_list)]
        # self.y_train = self.train_df[['earning_diff']]
        self.y_train = self.train_df[['period_price_diff']]
        for sec in self.sector_list:
            print("sector : ", sec)
            self.sector_x_train[sec] = self.sector_train_dfs[sec][self.sector_train_dfs[sec].columns.difference(y_col_list)]
            # self.sector_y_train[sec] = self.sector_train_dfs[sec][['earning_diff']]
            self.sector_y_train[sec] = self.sector_train_dfs[sec][['period_price_diff']]
        
        self.x_test = self.test_df[self.test_df.columns.difference(y_col_list)]
        # self.y_test = self.test_df[['earning_diff']]
        self.y_test = self.test_df[['period_price_diff']]
    
    def def_model(self):
        self.clsmodels[0] = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.01, gamma=0.01, subsample=0.5,
                                        colsample_bytree=0.5, max_depth=4, objective='binary:logistic')
        self.clsmodels[1] = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.01, gamma=0.01, subsample=0.5,
                                        colsample_bytree=0.5, max_depth=5, objective='binary:logistic')
        self.clsmodels[2] = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.01, gamma=0.01, subsample=0.5,
                                        colsample_bytree=0.5, max_depth=6, objective='binary:logistic')
        self.clsmodels[3] = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.01, gamma=0.01, subsample=0.5,
                                        colsample_bytree=0.5, max_depth=7, objective='binary:logistic')
        
        self.models[0] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.01, gamma=0.01, subsample=0.5,
                                        colsample_bytree=0.5, max_depth=8)
        # self.models[1] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.03, gamma=0, subsample=0.5,
        #                                 colsample_bytree=0.5, max_depth=10)
        self.models[1] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.01, gamma=0.01, subsample=0.5,
                                        colsample_bytree=0.5, max_depth=10)
        # self.models[3] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.03, gamma=0, subsample=0.5,
        #                                 colsample_bytree=0.5, max_depth=11)
        # self.models[1] = MLPRegressor(hidden_layer_sizes=(800, 400), activation='relu', solver='adam', random_state=42)
        # self.models[2] = RandomForestRegressor(n_jobs=-1, n_estimators=200, min_samples_split=2, min_samples_leaf=2)
        #self.models[1] = RandomForestRegressor(n_jobs=-1, n_estimators=200, min_samples_split=5, min_samples_leaf=3)

        for sec in self.sector_list:
            
            cur_key = (sec, 0)
            # self.sector_models[cur_key] = RandomForestRegressor(n_jobs=-1, n_estimators=200, min_samples_split=5, min_samples_leaf=3)
            self.sector_models[cur_key] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0.01, subsample=0.7,
                                        colsample_bytree=0.7, max_depth=6) #BEST
            # cur_key = (sec, 1)
            # # self.sector_models[cur_key] = RandomForestRegressor(n_jobs=-1, n_estimators=200, min_samples_split=5, min_samples_leaf=3)
            # self.sector_models[cur_key] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0, subsample=0.5,
            #                             colsample_bytree=0.7, max_depth=7)
            cur_key = (sec, 1)
            self.sector_models[cur_key]  = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0.01, subsample=0.7,
                                        colsample_bytree=0.7, max_depth=8)
            # self.sector_models[cur_key] = MLPRegressor(hidden_layer_sizes=(800, 400), activation='relu', solver='adam', random_state=42)
            # cur_key = (sec, 3)
            # self.sector_models[cur_key] = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1000, learning_rate=0.05, gamma=0, subsample=0.5,
            #                             colsample_bytree=0.6, max_depth=9)
            # self.sector_models[cur_key] = RandomForestRegressor(n_jobs=-1, n_estimators=200, min_samples_split=2, min_samples_leaf=2)
    
    def train(self):

        # # for parameter tuning 
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
        
        MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS_p{0:02d}/'.format(self.conf['REBALANCE_PERIOD'])
        self.def_model()
        
        if not os.path.exists(MODEL_SAVE_PATH):
            print("creating MODELS path : " + MODEL_SAVE_PATH)
            os.makedirs(MODEL_SAVE_PATH)
            
        # 레이블을 0과 1로 변환 (학습 직전)
        y_train_binary = (self.y_train > 0).astype(int)
        for i, model in self.clsmodels.items():
            logging.info("start fitting classifier")
            model.fit(self.x_train, y_train_binary)
            # filename = 'rfg' + str(i) + '_model' + str(self.conf['START_MONTH']) + '.sav'
            filename = MODEL_SAVE_PATH + 'clsmodel_{}.sav'.format(str(i))
            joblib.dump(model, filename)
            logging.info("model {} score : ".format(str(i)))
            logging.info(model.score(self.x_train, y_train_binary))
        
        
            
        for i, model in self.models.items():
            logging.info("start fitting XGBRegressor")
            model.fit(self.x_train, self.y_train.values.ravel())
            # filename = 'rfg' + str(i) + '_model' + str(self.conf['START_MONTH']) + '.sav'
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

        MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS_p{0:02d}/'.format(self.conf['REBALANCE_PERIOD'])
        self.models = dict()
        self.clsmodels = dict()
        self.clsmodels[0] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.clsmodels[2] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_2.sav')
        self.clsmodels[3] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_3.sav')
        
        self.models[0] = joblib.load(MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(MODEL_SAVE_PATH + 'model_1.sav')
        # self.models[2] = joblib.load(MODEL_SAVE_PATH + 'model_2.sav')
        # self.models[3] = joblib.load(MODEL_SAVE_PATH + 'model_3.sav')

        pred_col_list = ['ai_pred_avg'] 

        
        for i in range(2):
            pred_col_name = 'model_' + str(i) + '_prediction'
            pred_col_list.append(pred_col_name)
            pred_col_name = 'model_' + str(i) + '_prediction_wbinary'
            pred_col_list.append(pred_col_name)

        model_eval_hist = []
        full_df = pd.DataFrame()
        for test_idx, (testdate, df) in enumerate(self.test_df_list):
            print("evaluation date : ")
            tmp = testdate.split('\\')
            tmp = [v for v in tmp if v.endswith('.csv')]
            tdate = "_".join(tmp[0].split('_')[0:2])
            print(tdate)
            x_test = df[df.columns.difference(y_col_list)]
            # y_test = df[['earning_diff']]
            y_test = df[['period_price_diff']]
            y_test_binary = (y_test > 0).astype(int)

            preds = np.empty((0,x_test.shape[0]))            
            
            df['label'] = y_test
            df['label_binary'] = y_test_binary

            for i, model in self.clsmodels.items():    
                pred_col_name = 'clsmodel_' + str(i) + '_prediction'
                correct_col_name = 'clsmodel_' + str(i) + '_correct'
                y_predict_binary = model.predict(x_test)
                df[pred_col_name] = y_predict_binary
                df[correct_col_name] = (y_test_binary.values.ravel() == y_predict_binary).astype(int)
                acc = accuracy_score(df['label_binary'], df[pred_col_name])
                print(f"Accuracy for {pred_col_name}: {acc:.4f}")
                
            
            for i, model in self.models.items():    
                pred_bin_col_name = 'clsmodel_2_prediction'
                pred_col_name = 'model_' + str(i) + '_prediction'
                correct_col_name = 'clsmodel_' + str(i) + '_correct'
                pred_col_name_wbinary = 'model_' + str(i) + '_prediction_wbinary'
                loss_col_name = 'model_' + str(i) + '_prediction_loss'
                loss_bin_col_name = 'model_' + str(i) + '_prediction__wbinary_loss'

                y_predict = model.predict(x_test)
                
                df[pred_col_name] = y_predict
                df[pred_col_name_wbinary] = y_predict * df[pred_bin_col_name]
                preds = np.vstack((preds, y_predict[None,:]))
                df[loss_col_name] = abs(df['label'] - y_predict)
                df[loss_bin_col_name] = abs(df['label'] - df[pred_col_name_wbinary])
            
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
                    logging.info(col)
                    logging.info((top_k_df['period_price_diff'].sum()/(e-s+1)))
                    topk_period_earning_sums.append(top_k_df['period_price_diff'].sum())
                    
                    top_k_df.to_csv(MODEL_SAVE_PATH+'prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))
                    model_eval_hist.append([tdate, col, k, top_k_df['period_price_diff'].sum()/(e-s+1), top_k_df[col].sum()/(e-s+1), 
                                            abs(top_k_df[col].sum()/(e-s+1) - top_k_df['period_price_diff'].sum()/(e-s+1)), int(top_k_df[col].sum()/(e-s+1) > 0), 
                                            top_k_df['ai_pred_avg'].sum()/(e-s+1), top_k_df['model_0_prediction'].sum()/(e-s+1), top_k_df['model_1_prediction'].sum()/(e-s+1),
                                            top_k_df['model_0_prediction_wbinary'].sum()/(e-s+1), top_k_df['model_1_prediction_wbinary'].sum()/(e-s+1)
                                            ])
        
        col_name = ['start_date', 'model', 'krange', 'avg_earning_per_stock', 'cur_model_pred', 'loss_y_and_pred', 
                    'cur_model_pred_ispositive', 'avg_pred', 'model0_pred', 'model1_pred', 'model0_pred_wbinary', 'model1_pred_wbinary']
        pred_df = pd.DataFrame(model_eval_hist, columns=col_name)
        print(pred_df)
        pred_df.to_csv(MODEL_SAVE_PATH+'pred_df_topk.csv', index=False)
        full_df.to_csv(MODEL_SAVE_PATH+'prediction_ai.csv', index=False)
        # logging.info(model_eval_hist)

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
                # y_test = df[['earning_diff']]
                y_test = df[['period_price_diff']]
                
                if len(x_test) == 0:
                    continue
                
                sector_preds = np.empty((0,x_test.shape[0]))
                df['label'] = y_test
                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    y_predict = model.predict(x_test)
                    df[pred_col_name] = y_predict
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
                        logging.info((top_k_df['period_price_diff'].sum()/(e-s+1)))
                        topk_period_earning_sums.append(top_k_df['period_price_diff'].sum())
                        top_k_df.to_csv(MODEL_SAVE_PATH+'prediction_{}_{}_{}_top{}-{}.csv'.format(tdate, sec, col, s, e))
                        top_k_df['start_date'] = tdate
                        top_k_df['col'] = col
                        allsector_topk_df = pd.concat([allsector_topk_df, top_k_df])
                        sector_model_eval_hist.append([tdate, sec, col, k, top_k_df['period_price_diff'].sum()/(e-s+1), top_k_df[col].sum()/(e-s+1), 
                                            abs(top_k_df[col].sum()/(e-s+1) - top_k_df['period_price_diff'].sum()/(e-s+1)), int(top_k_df[col].sum()/(e-s+1) > 0), 
                                            top_k_df['ai_pred_avg'].sum()/(e-s+1), top_k_df['model_0_prediction'].sum()/(e-s+1), 
                                            top_k_df['model_1_prediction'].sum()/(e-s+1)])

            col_name = ['start_date', 'sector', 'model', 'krange', 'avg_earning_per_stock', 'cur_model_pred', 'loss_y_and_pred', 
                    'cur_model_pred_ispositive', 'avg_pred', 'model0_pred', 'model1_pred']
            pred_df = pd.DataFrame(sector_model_eval_hist, columns=col_name)
            print(pred_df)
            pred_df.to_csv(MODEL_SAVE_PATH+'allsector_pred_df.csv'.format(sec), index=False)
            # allsector_topk_df.to_csv('./reports/allsector_pred_df.csv', index=False)
            
            
            # persector_hist = []                      
            # for tdate in list(testdates):
            #     for model_name in pred_col_list:
            #         tmpdf = allsector_topk_df[allsector_topk_df['start_date']==tdate]
            #         tmpdf = tmpdf[tmpdf['col']==model_name]
            #         tmpdf = tmpdf.sort_values(by=[model_name], ascending=False, na_position="last")[0:10]
            #         persector_hist.append([tdate, model_name, tmpdf['period_price_diff'].sum()/10, tmpdf[model_name].sum()/10])
            # col_name = ['start_date', 'model', 'avg_earning_per_stock', 'model_preds']
            # pred_df = pd.DataFrame(persector_hist, columns=col_name)
            # pred_df.to_csv('./reports/persector_top10_pred.csv', index=False)
                
            
        
        
    def latest_prediction(self):
        
        MODEL_SAVE_PATH = self.conf['ROOT_PATH'] + '/MODELS_p{0:02d}/'.format(self.conf['REBALANCE_PERIOD'])
        self.clsmodels = dict()
        self.clsmodels[0] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_0.sav')
        self.clsmodels[1] = joblib.load(MODEL_SAVE_PATH + 'clsmodel_1.sav')
        self.models = dict()
        self.models[0] = joblib.load(MODEL_SAVE_PATH + 'model_0.sav')
        self.models[1] = joblib.load(MODEL_SAVE_PATH + 'model_1.sav')
        # self.models[2] = joblib.load(MODEL_SAVE_PATH + 'model_2.sav')
        # self.models[3] = joblib.load(MODEL_SAVE_PATH + 'model_3.sav')
        
        aidata_dir = self.conf['ROOT_PATH'] + '/regressor_data_p{0:02d}_m{1:02d}/'.format(self.conf['REBALANCE_PERIOD'], self.conf['START_MONTH'])

        # abs_col_list = list(set(meaning_col_list) - set(ratio_col_list))
        # abs_col_list = [col + '_sorted_normal' for col in abs_col_list]
        # print("drop abs_col_list : ")
        # print(abs_col_list)     

        latest_data_path = aidata_dir + '2023_09_regressor_train_latest_norm.csv'

        pred_col_list = ['ai_pred_avg']
        for i in range(2): 
            pred_col_name = 'model_' + str(i) + '_prediction'
            pred_col_list.append(pred_col_name)  

        ldf = pd.read_csv(latest_data_path)
        
        self.sector_list = list(ldf['sector'].unique())
        self.sector_list = [x for x in self.sector_list if str(x) != 'nan']
        print(self.sector_list)
        # ldf = ldf.drop(abs_col_list, axis=1) 
        ldf = ldf.drop('sector', axis=1) 
     
        ldf = ldf[ldf.isnull().sum(axis=1) < 500]
        ldf = ldf.fillna(0)
        input = ldf[ldf.columns.difference(['symbol'])]        
        preds = np.empty((0, input.shape[0]))
        for i, model in self.models.items():    
            pred_col_name = 'model_' + str(i) + '_prediction'
            y_predict3 = model.predict(input)
            print(y_predict3)
            print(len(y_predict3))
            ldf[pred_col_name] = y_predict3
            preds = np.vstack((preds, y_predict3[None,:]))
        
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
            # ldf = ldf.drop(abs_col_list, axis=1) 
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
                # sec_df = sec_df[sec_df.isnull().sum(axis=1) < 500]
                sec_df = sec_df.fillna(0)
            
                indata = sec_df[sec_df.columns.difference(['symbol'])]
                print(indata)
                preds = np.empty((0, indata.shape[0]))

                for i in range(2):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    y_predict3 = model.predict(indata)
                    print("i : ", i)
                    print(y_predict3)
                    print(len(y_predict3))
                    
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
                print(pred_df)
                pred_df.to_csv(MODEL_SAVE_PATH+'allsector_latest_pred_df.csv', index=False)

class MyDataset(Dataset):
    def __init__(self, conf):
        aidata_dir = conf['ROOT_PATH'] + '/regressor_data/'
        
        self.train_files = []
        for year in range(int(conf['TRAIN_START_YEAR']), int(conf['TRAIN_END_YEAR'])):
            path = aidata_dir + str(year) + "*train.csv"
            year_files = [file for file in glob.glob(path)]
            self.train_files.extend(year_files)
        
        self.test_files = []
        for year in range(int(conf['TEST_START_YEAR']), int(conf['TEST_END_YEAR'])):
            path = aidata_dir + str(year) + "*train.csv"
            year_files = [file for file in glob.glob(path)]
            self.test_files.extend(year_files)
        
        self.train_df = pd.DataFrame()
        for fpath in self.train_files:     
            print(fpath)
            df = pd.read_csv(fpath)
            # df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.dropna(axis=0, subset=['period_price_diff'])
            
            # df = df.loc[:, ratio_col_list_wprev]
            logging.debug(df.shape)  
            df = df[df.isnull().sum(axis=1) < 5]
            logging.debug(df.shape)  
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]       
            self.train_df = pd.concat([self.train_df, df], axis=0)

        self.test_df_list = []
        self.test_df = pd.DataFrame()
        for fpath in self.test_files:
            print(fpath)
            df = pd.read_csv(fpath)
            # df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.dropna(axis=0, subset=['period_price_diff'])
        
            # df = df.loc[:, ratio_col_list_wprev]
            logging.debug(df.shape)  
            df = df[df.isnull().sum(axis=1) < 5]
            logging.debug(df.shape)  
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]       
            df = df.fillna(0)
            self.test_df = pd.concat([self.test_df, df], axis=0)
            self.test_df_list.append([fpath, df])
            
        self.train_df = self.train_df.fillna(0)
        self.test_df = self.test_df.fillna(0)
        
        x = self.train_df[self.train_df.columns.difference(y_col_list)].values
        # y = self.train_df[['earning_diff']].values
        y = self.train_df[['period_price_diff']].values
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]


class RegressionNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.fc1 = nn.Linear(len(ratio_col_list_wprev)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.dropout(nn_f.sigmoid(self.fc1(x)))
        h = self.fc2(h)
        return h
    
    def mtrain(self):
        # for regression network
        net = RegressionNetwork(self.conf)
        net2 = RegressionNetwork(self.conf)
        optimizer = optim.AdamW(net.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()
        # loss_fn = nn.L1Loss(reduction='mean')

        my_ds = MyDataset(self.conf)
        train_loader = DataLoader(my_ds, batch_size=512, shuffle=True)
        min_loss = 9999
        
        writer = SummaryWriter('scalar/')
        val_loss_sum = 0
        loss_sum = 0
        for epoch in range(1, 10000):
            net.train()
            for i, (data, labels) in enumerate(train_loader):
                pred = net(data)
                loss = loss_fn(pred,labels) 
                loss_sum += loss
                writer.add_scalar("Loss/train", loss, epoch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 50 == 0:
                print("epoch : ", epoch)
                print("min_loss : ", min_loss)
                print("train loss : ", loss_sum)
                if (loss_sum < min_loss) and epoch > 100:
                    min_loss = loss_sum
                    torch.save(net.state_dict(), './model_state_dict.pt')  # 모델 객체의 state_dict 저장
                loss_sum = 0

            test_loader = DataLoader(my_ds, batch_size=1024, shuffle=False)
            # net2.load_state_dict(torch.load('./model_state_dict.pt'))
            net.eval()
            preds = np.empty(shape=(0))
            labels = np.empty(shape=(0))
            
            for i, (data, label) in enumerate(test_loader):
                optimizer.zero_grad()
                pred = net(data)
                loss = loss_fn(pred,label)
                val_loss_sum += loss
                preds = np.append(preds, pred.detach().numpy())
                labels = np.append(labels, label)
            if epoch % 50 == 0:    
                print("val loss : ", val_loss_sum)
                val_loss_sum = 0
        plt.scatter(preds, labels, alpha=0.4)
        plt.xlabel("pred")
        plt.ylabel("labels")
        plt.show()
        writer.close()
        # pred = net(tensor_x_test)
        

class RegressionNetwork1(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.fc1 = nn.Linear(len(ratio_col_list_wprev)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        h = nn_f.leaky_relu(self.fc1(x))
        h = self.fc2(h)
        return h
    

class RegressionNetwork2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.fc1 = nn.Linear(len(ratio_col_list_wprev)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.dropout(nn_f.sigmoid(self.fc1(x)))
        h = self.fc2(h)
        return h


class RegressionNetwork4(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.fc1 = nn.Linear(len(ratio_col_list_wprev)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.dropout(nn_f.sigmoid(self.fc1(x)))
        h = self.fc2(h)
        return h
