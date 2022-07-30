import logging
from xml.dom import XHTML_NAMESPACE
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import numpy as np
import joblib

import glob
import logging

from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


use_col_list = [
"interestCoverage_normal",
"dividendYield_normal",
"inventoryTurnover_normal",
"daysPayablesOutstanding_normal",
"stockBasedCompensationToRevenue_normal",
"dcf_normal",
"capexToDepreciation_normal",
"currentRatio_normal",
"daysOfInventoryOnHand_normal",
"payablesTurnover_normal",
"grahamNetNet_normal",
"capexToRevenue_normal",
"netDebtToEBITDA_normal",
"receivablesTurnover_normal",
"capexToOperatingCashFlow_normal",
"evToOperatingCashFlow_normal",
"evToFreeCashFlow_normal",
"debtToAssets_normal",
"tangibleBookValuePerShare_normal",
"stockBasedCompensation_normal",
"capexPerShare_normal",
"peRatio_normal",
"enterpriseValueOverEBITDA_normal",
"bookValuePerShare_normal",
"shareholdersEquityPerShare_normal",
"pfcfRatio_normal",
"pocfratio_normal",
"daysSalesOutstanding_normal",
"incomeQuality_normal",
"interestDebtPerShare_normal",
"revenuePerShare_normal",
"freeCashFlowPerShare_normal",
"evToSales_normal",
"netIncomePerShare_normal",
"grahamNumber_normal",
"operatingCashFlowPerShare_normal",
"cashPerShare_normal",
"priceToSalesRatio_normal",
"pbRatio_normal",
"ptbRatio_normal",
"investedCapital_normal",
"roic_normal",
"freeCashFlowYield_normal",
"roe_normal",
"returnOnTangibleAssets_normal",
"earningsYield_normal",
"debtToEquity_normal",
"payoutRatio_normal",
"salesGeneralAndAdministrativeToRevenue_normal",
"intangiblesToTotalAssets_normal",
"netDebt_normal",
"ebitdaratio_normal",
"ebitda_normal",
"dividendsperShareGrowth_normal",
"freeCashFlow_normal",
"operatingCashFlow_normal",
"netIncomeGrowth_normal",
"grossProfit_normal",
"epsgrowth_normal",
"epsdilutedGrowth_normal",
"revenueGrowth_normal",
"grossProfitRatio_normal",
"epsdiluted_normal",
"eps_normal",
"debtGrowth_normal",
"tenYDividendperShareGrowthPerShare_normal",
"netIncomeRatio_normal",
"incomeBeforeTaxRatio_normal",
"operatingCashFlowGrowth_normal",
"ebitgrowth_normal",
"operatingIncomeGrowth_normal",
"threeYDividendperShareGrowthPerShare_normal",
"assetGrowth_normal",
"freeCashFlowGrowth_normal",
"sgaexpensesGrowth_normal",
"fiveYDividendperShareGrowthPerShare_normal",
"receivablesGrowth_normal",
"fiveYRevenueGrowthPerShare_normal",
"threeYOperatingCFGrowthPerShare_normal",
"grossProfitGrowth_normal",
"operatingIncomeRatio_normal",
"threeYShareholdersEquityGrowthPerShare_normal",
"fiveYShareholdersEquityGrowthPerShare_normal",
"fiveYOperatingCFGrowthPerShare_normal",
"threeYRevenueGrowthPerShare_normal",
"researchAndDdevelopementToRevenue_normal",
"threeYNetIncomeGrowthPerShare_normal",
"tenYOperatingCFGrowthPerShare_normal",
"tenYRevenueGrowthPerShare_normal",
"tenYShareholdersEquityGrowthPerShare_normal",
"tenYNetIncomeGrowthPerShare_normal",
"weightedAverageSharesGrowth_normal",
"weightedAverageSharesDilutedGrowth_normal",
"fiveYNetIncomeGrowthPerShare_normal",
"bookValueperShareGrowth_normal",
"inventoryGrowth_normal",
"rdexpenseGrowth_normal",
"period_price_diff",
"earning_diff",
"symbol"
]
        
class Regressor:
    
    def __init__(self, conf):
        self.conf = conf
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
        
        logging.info("train file list : ", self.train_files)
        logging.info("test file list : ", self.test_files)
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.test_df_list = []
        
        self.nns = dict()
        self.rfgs = dict()
        self.dnns = dict()
        self.mlr = LinearRegression()
        # self.rfg = RandomForestRegressor(n_jobs=-1, n_estimators=10, min_samples_split=5)
        logging.info("use col list length : ")
        
        self.rfgs[0] = RandomForestRegressor(n_jobs=-1, n_estimators=200)
        self.rfgs[1] = RandomForestRegressor(n_jobs=-1, n_estimators=200, min_samples_leaf=2)
        self.rfgs[2] = RandomForestRegressor(n_jobs=-1, n_estimators=200, min_samples_split=4, min_samples_leaf=2)
        
        self.dnns[1] = RegressionNetwork1(self.conf)
        self.dnns[1].load_state_dict(torch.load('./model_state_dict_candi1.pt'))
        
        self.dnns[2] = RegressionNetwork2(self.conf)
        self.dnns[2].load_state_dict(torch.load('./model_state_dict_255_candi2.pt'))
        
        self.dnns[4] = RegressionNetwork4(self.conf)
        self.dnns[4].load_state_dict(torch.load('./model_state_dict_53_candi4.pt'))

        self.nns[0] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97), 
                activation='relu', solver='lbfgs', max_iter = 20000, verbose = True, alpha=0.001)
        
        self.nns[1] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97), 
                activation='tanh', solver='lbfgs', max_iter = 20000, verbose = True, alpha=0.01)
        
        # self.nns[1] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97, 64, 16), 
        #         activation='relu', solver='lbfgs', max_iter = 20000, verbose = True, alpha=0.01)
        
        # self.nns[1] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97, 16), 
        #         activation='relu', solver='lbfgs', max_iter = 10000, verbose = True, alpha=0.001)
        
        # self.nns[2] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97, 16),
        #       activation='relu', solver='lbfgs', max_iter = 10000, verbose = True, alpha=0.1)
        
        #self.nns[1] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97), 
        #      activation='logistic', solver='adam', max_iter = 10000, verbose = True, learning_rate_init=0.01, batch_size=128, early_stopping=True, tol=0.00001, n_iter_no_change=100)     
        
        #self.nns[3] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97), 
        #     activation='logistic', solver='adam', max_iter = 10000, verbose = True, learning_rate_init=0.001, batch_size=32, early_stopping=True, tol=0.00001, n_iter_no_change=100)
        
        # self.nns[4] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97), 
        #      activation='tanh', solver='adam', max_iter = 10000, verbose = True, learning_rate_init=0.001, early_stopping=True, tol=0.00001)
        
        # self.nns[5] = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 97), 
        #      activation='logistic', solver='adam', max_iter = 10000, verbose = True, learning_rate_init=0.001, early_stopping=True, tol=0.00001)
        
        
        
    def dataload(self):
        
        for fpath in self.train_files:     
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.loc[:, use_col_list]
            logging.debug(df.shape)  
            df = df[df.isnull().sum(axis=1) < 5]
            logging.debug(df.shape)  
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]       
            self.train_df = pd.concat([self.train_df, df], axis=0)


        self.test_df_list = []
        for fpath in self.test_files:
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.loc[:, use_col_list]
            logging.debug(df.shape)  
            df = df[df.isnull().sum(axis=1) < 5]
            logging.debug(df.shape)  
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]       
            df = df.fillna(0)
            self.test_df = pd.concat([self.test_df, df], axis=0)
            self.test_df_list.append([fpath, df])
            
            
        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)    
        logging.debug('NaN occurrences in Columns:')
        logging.debug(self.train_df.isnull().sum(axis=0))
        logging.debug('NaN occurrences in Rows:')
        logging.debug(self.train_df.isnull().sum(axis=1))
        self.train_df = self.train_df.fillna(0)
        self.test_df = self.test_df.fillna(0)
        # self.train_df = self.train_df.fillna(self.train_df.mean())   
        
        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)
        logging.debug("test_df shape : ")
        logging.debug(self.test_df.shape)
        
        
                # x = self.train_df.loc[:, self.train_df.columns != 'earning_diff']
        self.x_train = self.train_df[self.train_df.columns.difference(['earning_diff', 'period_price_diff', 'symbol'])]
        self.y_train = self.train_df[['earning_diff']]
        self.x_test = self.test_df[self.test_df.columns.difference(['earning_diff', 'period_price_diff', 'symbol'])]
        self.y_test = self.test_df[['earning_diff']]
        

    def train(self):        
        
        # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
        logging.info("start fitting LinearRegression")
        self.mlr.fit(self.x_train, self,y_train)
        logging.info( "mlr score : ")
        logging.info(self.mlr.score(self.x_train, self.y_train))
        logging.info("end fitting LinearRegression")
        
        
        
        for i, rfg in self.rfgs.items():
            logging.info("start fitting RandomForestRegressor")
            rfg.fit(self.x_train, self.y_train.values.ravel())
            filename = 'rfg' + str(i) + '_model.sav'
            joblib.dump(rfg, filename)
            logging.info("rfg score : ")
            logging.info(rfg.score(self.x_train, self.y_train))
            logging.info("end fitting RandomForestRegressor")

            ftr_importances_values = rfg.feature_importances_
            ftr_importances = pd.Series(ftr_importances_values, index = self.x_train.columns)
            ftr_importances.to_csv('./rfg_importances.csv')
            ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            logging.info(ftr_top20)              
        # params = { 'n_estimators' : [5, 25, 100],
        #    'max_depth' : [2, 4, 8, 16, 32],
        #    'min_samples_leaf' : [8, 12, 18],
        #    'min_samples_split' : [2, 4, 8, 16, 32, 64]
        #     }
        # # RandomForestClassifier 객체 생성 후 GridSearchCV 수행
        # grid_cv = GridSearchCV(self.rfg, param_grid = params, n_jobs = -1)
        # grid_cv.fit(x_train, y_train.values.ravel())
        # print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
        # print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
    
        
        for i, nn in self.nns.items():
            logging.info("start fitting " + str(i) + "-th MLPRegressor")
            nn.fit(self.x_train, self.y_train.values.ravel())
            logging.info("nn score : ")
            logging.info(nn.score(self.x_train, self.y_train))
            logging.info("end fitting " + str(i) + "-th MLPRegressor")
            filename = 'nn' + str(i) + '_model.sav'
            joblib.dump(nn, filename)
        
        self.evaluation(self.x_train, self.x_test, self.y_train, self.y_test)
        # weight 출력
        # logging.debug("result regression. weight : ")
        # logging.debug(x_train.columns)
        # logging.debug(self.mlr.coef_)
        # weight_df = pd.DataFrame(self.mlr.coef_, columns=x_train.columns)
        # # weight_df.to_csv("./weight.csv", index=False)
        # print(weight_df)
           
    def evaluation(self):
        
        
        self.rfgs = dict()
        self.rfgs[0] = joblib.load('rfg0_model.sav')
        self.rfgs[1] = joblib.load('rfg1_model.sav')
        self.rfgs[2] = joblib.load('rfg2_model.sav')
        
        self.nns = dict()
        self.nns[0] = joblib.load('nn0_model.sav')
        self.nns[1] = joblib.load('nn1_model.sav')
        
        for i, rfg in self.rfgs.items():
            y_predict = rfg.predict(self.x_train)
        
        if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
            plt.scatter(self.y_train, y_predict, alpha=0.4)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("rfg REGRESSION")
            plt.show()   

            plt.figure(figsize=(8,6))
            plt.title('Top 20 Feature Importances')
            sns.barplot(x=ftr_top20, y=ftr_top20.index)
            plt.show()
            
        for i, nn in self.nns.items():
            y_predict = nn.predict(self.x_train)
            if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
                plt.scatter(self.y_train, y_predict, alpha=0.4)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("nn REGRESSION")
                plt.show()
        
        model_eval_hist = []
        for test_idx, (testdate, df) in enumerate(self.test_df_list):
            print("evaluation date : ")
            tdate = "_".join(testdate.split("\\")[4].split('_')[0:2])
            print(tdate)
            
            x_test = df[df.columns.difference(['earning_diff', 'period_price_diff', 'symbol'])]
            y_test = df[['earning_diff']]
            
            preds = np.empty((0,x_test.shape[0]))
            
            # preds = np.vstack((preds, y_predict[None,:]))
            if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
                plt.scatter(y_test, y_predict, alpha=0.4)
                # pd.concat([pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_predict).reset_index()],axis=1).to_csv("./prediction_result.csv")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("mlr REGRESSION")
                plt.show()

            df['label'] = y_test

            for i, rfg in self.rfgs.items():    
                rfg_pred_col_name = 'rfg_' + str(i) + '_prediction'
                y_predict = rfg.predict(x_test)
                df[rfg_pred_col_name] = y_predict
                preds = np.vstack((preds, y_predict[None,:]))
                if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
                    plt.scatter(y_test, y_predict, alpha=0.4)
                    # pd.concat([pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_predict).reset_index()],axis=1).to_csv("./prediction_result.csv")    
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.title(rfg_pred_col_name)
                    plt.show()
            
            
            for i, dnn in self.dnns.items():    
                dnn_pred_col_name = 'dnn_' + str(i) + '_prediction'
                y_predict0 = dnn(torch.tensor(x_test.values, dtype=torch.float32))
                y_predict0 = y_predict0.detach().numpy()
                y_predict0 = y_predict0.ravel()
                df[dnn_pred_col_name] = pd.Series(y_predict0)
                preds = np.vstack((preds, y_predict0[None,:]))

                
            for i, nn in self.nns.items():    
                nn_pred_col_name = 'nn_' + str(i) + '_prediction'
                y_predict = nn.predict(x_test)
                preds = np.vstack((preds, y_predict[None,:]))
                df[nn_pred_col_name] = y_predict
                if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
                    plt.scatter(y_test, y_predict, alpha=0.4)
                    # pd.concat([pd.DataFrame(y_train).reset_index(),pd.DataFrame(y_predict).reset_index()],axis=1).to_csv("./prediction_result.csv")
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.title(nn_pred_col_name)
                    plt.show()                
            
            df['ai_pred_avg'] = np.average(preds, axis=0)
            df.to_csv("./reports/prediction_ai_{}.csv".format(tdate))

            # 각 model의 top_k 종목의 period_price_diff 합을 구해서 model 최종 평가
            # model pred col list
            pred_col_list = ['ai_pred_avg'] 
            for i, rfg in self.rfgs.items(): 
                rfg_pred_col_name = 'rfg_' + str(i) + '_prediction'
                pred_col_list.append(rfg_pred_col_name)
            for i, nn in self.nns.items(): 
                nn_pred_col_name = 'nn_' + str(i) + '_prediction'
                pred_col_list.append(nn_pred_col_name)
            for i, dnn in self.dnns.items(): 
                dnn_pred_col_name = 'dnn_' + str(i) + '_prediction'
                pred_col_list.append(dnn_pred_col_name)
            topk_period_earning_sums = []
            topk_list = [(0,10), (0,20), (0,30), (3,20), (5,20), (3,30), (10,30)]
            for s, e in topk_list:
                logging.info("top" + str(s) + " ~ "  + str(e) )
                k = str(s) + '~' + str(e)
                for col in pred_col_list:
                    top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                    logging.info(col)
                    logging.info((top_k_df['period_price_diff'].sum()/(e-s+1)))
                    topk_period_earning_sums.append(top_k_df['period_price_diff'].sum())
                    top_k_df.to_csv('./reports/prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))
                    model_eval_hist.append([tdate, col, k, (top_k_df['period_price_diff'].sum()/(e-s+1))])
        
        col_name = ['start_date', 'model', 'krange', 'avg_earning_per_stock']
        pred_df = pd.DataFrame(model_eval_hist, columns=col_name)
        print(pred_df)
        pred_df.to_csv('./reports/pred_df.csv', index=False)
        # logging.info(model_eval_hist)
        
    def latest_prediction(self, latest_data_path):

        ldf = pd.read_csv(latest_data_path)
        collist = use_col_list.copy()
        collist.remove("earning_diff")
        collist.remove("period_price_diff")

        ldf = ldf.loc[:, collist]
        ldf = ldf[ldf.isnull().sum(axis=1) < 5]
        ldf = ldf.fillna(0)
        
        input = ldf[ldf.columns.difference(['symbol'])]
        
        preds = np.empty((0,input.shape[0]))

        # y_predict1 = self.mlr.predict(input)
        # ldf['mlr_prediction'] = y_predict1.ravel()
        # preds = np.vstack((preds, y_predict1.ravel()[None,:]))
        
        for i, dnn in self.dnns.items():    
            dnn_pred_col_name = 'dnn_' + str(i) + '_prediction'
            y_predict0 = dnn(torch.tensor(input.values, dtype=torch.float32))
            y_predict0 = y_predict0.detach().numpy()
            y_predict0 = y_predict0.ravel()
            ldf[dnn_pred_col_name] = pd.Series(y_predict0)
            preds = np.vstack((preds, y_predict0[None,:]))
        
        
        for i, nn in self.nns.items():    
            nn_pred_col_name = 'nn_' + str(i) + '_prediction'
            y_predict2 = nn.predict(input)
            ldf[nn_pred_col_name] = y_predict2
            preds = np.vstack((preds, y_predict2[None,:]))

        for i, rfg in self.rfgs.items():    
            rfg_pred_col_name = 'rfg_' + str(i) + '_prediction'
            y_predict3 = rfg.predict(input)
            ldf[rfg_pred_col_name] = y_predict3
            preds = np.vstack((preds, y_predict3[None,:]))
        
        ldf['ai_pred_avg'] = np.average(preds, axis=0)        
        ldf.to_csv("./latest_prediction.csv")
        
        
        pred_col_list = ['ai_pred_avg'] 
        for i, rfg in self.rfgs.items(): 
            rfg_pred_col_name = 'rfg_' + str(i) + '_prediction'
            pred_col_list.append(rfg_pred_col_name)
        for i, nn in self.nns.items(): 
            nn_pred_col_name = 'nn_' + str(i) + '_prediction'
            pred_col_list.append(nn_pred_col_name)
        for i, dnn in self.dnns.items(): 
            dnn_pred_col_name = 'dnn_' + str(i) + '_prediction'
            pred_col_list.append(dnn_pred_col_name)
        topk_list = [(0,10), (0,20), (0,30), (3,20), (5,20), (3,30), (10,30)]
        for s, e in topk_list:
            logging.info("top" + str(s) + " ~ "  + str(e) )
            for col in pred_col_list:
                top_k_df = ldf.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                top_k_df.to_csv('./reports/latest_prediction_{}_top{}-{}.csv'.format(col, s, e))
 
        


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
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.loc[:, use_col_list]
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
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.loc[:, use_col_list]
            logging.debug(df.shape)  
            df = df[df.isnull().sum(axis=1) < 5]
            logging.debug(df.shape)  
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]       
            df = df.fillna(0)
            self.test_df = pd.concat([self.test_df, df], axis=0)
            self.test_df_list.append([fpath, df])
            
        
        self.train_df = self.train_df.fillna(0)
        self.test_df = self.test_df.fillna(0)
        
        x = self.train_df[self.train_df.columns.difference(['earning_diff', 'period_price_diff', 'symbol'])].values
        y = self.train_df[['earning_diff']].values
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
        self.fc1 = nn.Linear(len(use_col_list)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        h = self.dropout(F.sigmoid(self.fc1(x)))
        h = self.fc2(h)
        return h
    
    def mtrain(self):

        # for regression network
        net = RegressionNetwork(self.conf)
        net2 = RegressionNetwork(self.conf)
        optimizer = optim.AdamW(net.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()
        # loss_fn = nn.L1Loss(reduction='mean')

        myDs = MyDataset(self.conf)
        train_loader = DataLoader(myDs, batch_size=512, shuffle=True)
        min_loss = 9999
        
        writer = SummaryWriter('scalar/')
        val_loss_sum = 0
        loss_sum = 0
        for epoch in range(1,10000):
            
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

            test_loader = DataLoader(myDs, batch_size=1024, shuffle=False)
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
        self.fc1 = nn.Linear(len(use_col_list)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = self.fc2(h)
        return h
    

class RegressionNetwork2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.fc1 = nn.Linear(len(use_col_list)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        h = self.dropout(F.sigmoid(self.fc1(x)))
        h = self.fc2(h)
        return h
    

class RegressionNetwork4(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.fc1 = nn.Linear(len(use_col_list)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        h = self.dropout(F.sigmoid(self.fc1(x)))
        h = self.fc2(h)
        return h