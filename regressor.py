import logging
from xml.dom import XHTML_NAMESPACE
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import numpy as np

import glob
import logging

from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


use_col_list = [
"interestCoverage_normal",# "interestCoverage_normal_max_diff",
"dividendYield_normal",# "dividendYield_normal_max_diff",
"inventoryTurnover_normal",# "inventoryTurnover_normal_max_diff",
"daysPayablesOutstanding_normal",# "daysPayablesOutstanding_normal_max_diff",
"stockBasedCompensationToRevenue_normal",# "stockBasedCompensationToRevenue_normal_max_diff",
"dcf_normal",# "dcf_normal_max_diff",
"capexToDepreciation_normal",# "capexToDepreciation_normal_max_diff",
"currentRatio_normal",# "currentRatio_normal_max_diff",
"daysOfInventoryOnHand_normal",# "daysOfInventoryOnHand_normal_max_diff",
"payablesTurnover_normal",# "payablesTurnover_normal_max_diff",
"grahamNetNet_normal",# "grahamNetNet_normal_max_diff",
"capexToRevenue_normal",# "capexToRevenue_normal_max_diff",
"netDebtToEBITDA_normal",# "netDebtToEBITDA_normal_max_diff",
"receivablesTurnover_normal",# "receivablesTurnover_normal_max_diff",
"capexToOperatingCashFlow_normal",# "capexToOperatingCashFlow_normal_max_diff",
"evToOperatingCashFlow_normal",# "evToOperatingCashFlow_normal_max_diff",
"evToFreeCashFlow_normal",# "evToFreeCashFlow_normal_max_diff",
"debtToAssets_normal",# "debtToAssets_normal_max_diff",
"tangibleBookValuePerShare_normal",# "tangibleBookValuePerShare_normal_max_diff",
"stockBasedCompensation_normal",# "stockBasedCompensation_normal_max_diff",
"capexPerShare_normal",# "capexPerShare_normal_max_diff",
"peRatio_normal",# "peRatio_normal_max_diff",
"enterpriseValueOverEBITDA_normal",# "enterpriseValueOverEBITDA_normal_max_diff",
"bookValuePerShare_normal",# "bookValuePerShare_normal_max_diff",
"shareholdersEquityPerShare_normal",# "shareholdersEquityPerShare_normal_max_diff",
"pfcfRatio_normal",# "pfcfRatio_normal_max_diff",
"pocfratio_normal",# "pocfratio_normal_max_diff",
"daysSalesOutstanding_normal",# "daysSalesOutstanding_normal_max_diff",
"incomeQuality_normal",# "incomeQuality_normal_max_diff",
"interestDebtPerShare_normal",# "interestDebtPerShare_normal_max_diff",
"revenuePerShare_normal",# "revenuePerShare_normal_max_diff",
"freeCashFlowPerShare_normal",# "freeCashFlowPerShare_normal_max_diff",
"evToSales_normal",# "evToSales_normal_max_diff",
"netIncomePerShare_normal",# "netIncomePerShare_normal_max_diff",
"grahamNumber_normal",# "grahamNumber_normal_max_diff",
"operatingCashFlowPerShare_normal",# "operatingCashFlowPerShare_normal_max_diff",
"cashPerShare_normal",# "cashPerShare_normal_max_diff",
"priceToSalesRatio_normal",# "priceToSalesRatio_normal_max_diff",
"pbRatio_normal",# "pbRatio_normal_max_diff",
"ptbRatio_normal",# "ptbRatio_normal_max_diff",
"investedCapital_normal",# "investedCapital_normal_max_diff",
"roic_normal",# "roic_normal_max_diff",
"freeCashFlowYield_normal",# "freeCashFlowYield_normal_max_diff",
"roe_normal",# "roe_normal_max_diff",
"returnOnTangibleAssets_normal",# "returnOnTangibleAssets_normal_max_diff",
"earningsYield_normal",# "earningsYield_normal_max_diff",
"debtToEquity_normal",# "debtToEquity_normal_max_diff",
"payoutRatio_normal",# "payoutRatio_normal_max_diff",
"salesGeneralAndAdministrativeToRevenue_normal",# "salesGeneralAndAdministrativeToRevenue_normal_max_diff",
"intangiblesToTotalAssets_normal",# "intangiblesToTotalAssets_normal_max_diff",
"netDebt_normal",# "netDebt_normal_max_diff",
"ebitdaratio_normal",# "ebitdaratio_normal_max_diff",
"ebitda_normal",# "ebitda_normal_max_diff",
"dividendsperShareGrowth_normal",# "dividendsperShareGrowth_normal_max_diff",
"freeCashFlow_normal",# "freeCashFlow_normal_max_diff",
"operatingCashFlow_normal",# "operatingCashFlow_normal_max_diff",
"netIncomeGrowth_normal",# "netIncomeGrowth_normal_max_diff",
"grossProfit_normal",# "grossProfit_normal_max_diff",
"epsgrowth_normal",# "epsgrowth_normal_max_diff",
"epsdilutedGrowth_normal",# "epsdilutedGrowth_normal_max_diff",
"revenueGrowth_normal",# "revenueGrowth_normal_max_diff",
"grossProfitRatio_normal",# "grossProfitRatio_normal_max_diff",
"epsdiluted_normal",# "epsdiluted_normal_max_diff",
"eps_normal",# "eps_normal_max_diff",
"debtGrowth_normal",# "debtGrowth_normal_max_diff",
"tenYDividendperShareGrowthPerShare_normal",# "tenYDividendperShareGrowthPerShare_normal_max_diff",
"netIncomeRatio_normal",# "netIncomeRatio_normal_max_diff",
"incomeBeforeTaxRatio_normal",# "incomeBeforeTaxRatio_normal_max_diff",
"operatingCashFlowGrowth_normal",# "operatingCashFlowGrowth_normal_max_diff",
"ebitgrowth_normal",# "ebitgrowth_normal_max_diff",
"operatingIncomeGrowth_normal",# "operatingIncomeGrowth_normal_max_diff",
"threeYDividendperShareGrowthPerShare_normal",# "threeYDividendperShareGrowthPerShare_normal_max_diff",
"assetGrowth_normal",# "assetGrowth_normal_max_diff",
"freeCashFlowGrowth_normal",# "freeCashFlowGrowth_normal_max_diff",
"sgaexpensesGrowth_normal",# "sgaexpensesGrowth_normal_max_diff",
"fiveYDividendperShareGrowthPerShare_normal",# "fiveYDividendperShareGrowthPerShare_normal_max_diff",
"receivablesGrowth_normal",# "receivablesGrowth_normal_max_diff",
"fiveYRevenueGrowthPerShare_normal",# "fiveYRevenueGrowthPerShare_normal_max_diff",
"threeYOperatingCFGrowthPerShare_normal",# "threeYOperatingCFGrowthPerShare_normal_max_diff",
"grossProfitGrowth_normal",# "grossProfitGrowth_normal_max_diff",
"operatingIncomeRatio_normal",# "operatingIncomeRatio_normal_max_diff",
"threeYShareholdersEquityGrowthPerShare_normal",# "threeYShareholdersEquityGrowthPerShare_normal_max_diff",
"fiveYShareholdersEquityGrowthPerShare_normal",# "fiveYShareholdersEquityGrowthPerShare_normal_max_diff",
"fiveYOperatingCFGrowthPerShare_normal",# "fiveYOperatingCFGrowthPerShare_normal_max_diff",
"threeYRevenueGrowthPerShare_normal",# "threeYRevenueGrowthPerShare_normal_max_diff",
"researchAndDdevelopementToRevenue_normal",# "researchAndDdevelopementToRevenue_normal_max_diff",
"threeYNetIncomeGrowthPerShare_normal",# "threeYNetIncomeGrowthPerShare_normal_max_diff",
"tenYOperatingCFGrowthPerShare_normal",# "tenYOperatingCFGrowthPerShare_normal_max_diff",
"tenYRevenueGrowthPerShare_normal",# "tenYRevenueGrowthPerShare_normal_max_diff",
"tenYShareholdersEquityGrowthPerShare_normal",# "tenYShareholdersEquityGrowthPerShare_normal_max_diff",
"tenYNetIncomeGrowthPerShare_normal",# "tenYNetIncomeGrowthPerShare_normal_max_diff",
"weightedAverageSharesGrowth_normal",# "weightedAverageSharesGrowth_normal_max_diff",
"weightedAverageSharesDilutedGrowth_normal",# "weightedAverageSharesDilutedGrowth_normal_max_diff",
"fiveYNetIncomeGrowthPerShare_normal",# "fiveYNetIncomeGrowthPerShare_normal_max_diff",
"bookValueperShareGrowth_normal",# "bookValueperShareGrowth_normal_max_diff",
"inventoryGrowth_normal",# "inventoryGrowth_normal_max_diff",
"rdexpenseGrowth_normal",# "rdexpenseGrowth_normal_max_diff",
"earning_diff"#
]
        
class Regressor:
    
    def __init__(self, conf):
        self.conf = conf
        
        self.train_files = []
        for year in range(int(conf['START_YEAR']), int(conf['END_YEAR'])):
            traindata_dir = conf['ROOT_PATH'] + '/regressor_data/'
            path = traindata_dir + str(year) + "*train.csv"
            year_files = [file for file in glob.glob(path)]
            self.train_files.extend(year_files)
        print(self.train_files)
        
        logging.debug("train file list : ", self.train_files)
        self.train_df = pd.DataFrame()
        # for linear regression()
        self.mlr = LinearRegression()
        self.rfg = RandomForestRegressor()
        self.nn = MLPRegressor(hidden_layer_sizes=(len(use_col_list)-1, 128, 16), 
                  activation='tanh', solver='lbfgs', max_iter = 100, verbose = True)
        
    def dataload(self):

        for fpath in self.train_files:     
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.loc[:, use_col_list]
            df = df[df.isnull().sum(axis=1) < 100]
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]       
            logging.debug(df.shape)  
            self.train_df = pd.concat([self.train_df, df], axis=0)
            
        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)    
        
        # self.train_df = self.train_df.drop(columns=[
        #     'symbol', 'close_normal', 'close_normal_max_diff', 'date_normal', 'date_normal_max_diff',
        #     'date_y_normal', 'date_y_normal_max_diff', 'fillingDate_normal', 'fillingDate_normal_max_diff',
        #     'acceptedDate_normal', 'acceptedDate_normal_max_diff', 'acceptedDate_y_normal', 'acceptedDate_y_normal_max_diff', 
        #     'calendarYear_normal', 'calendarYear_normal_max_diff', 'calendarYear_x_normal', 'calendarYear_x_normal_max_diff',
        #     'calendarYear_y_normal', 'calendarYear_y_normal_max_diff',
        #     'cik_normal', 'cik_normal_max_diff', 'cik_x_normal', 'cik_x_normal_max_diff', 'cik_y_normal',
        #     'cik_y_normal_max_diff', 'close_normal','close_normal_max_diff','commonStockIssued_normal','commonStockIssued_normal_max_diff',
        #     'commonStockRepurchased_normal','commonStockRepurchased_normal_max_diff','commonStock_normal','commonStock_normal_max_diff',
        #     'marketCap_x_normal','marketCap_x_normal_max_diff','marketCap_y_normal','marketCap_y_normal_max_diff',
        #     'rebalance_day_price_normal','rebalance_day_price_normal_max_diff'
        #     ])
        logging.debug('NaN occurrences in Columns:')
        logging.debug(self.train_df.isnull().sum(axis=0))
        logging.debug('NaN occurrences in Rows:')
        logging.debug(self.train_df.isnull().sum(axis=1))
        self.train_df = self.train_df.fillna(0)
        # self.train_df = self.train_df.fillna(self.train_df.mean())

        for col in self.train_df.columns:
            if col == 'earning_diff':
                continue
            max_value = self.train_df[col].max()
            min_value = self.train_df[col].min()
            self.train_df[col] = ((self.train_df[col] - min_value) / (max_value - min_value))-0.5        
            
        Q1 = np.percentile(self.train_df['earning_diff'], 25)
        Q3 = np.percentile(self.train_df['earning_diff'], 75)
        IQR = Q3 - Q1
        # 0.5 is not fixed.   reference :  1.5 => remove 0.7%,  0 =>  remove 50%
        outlier_step = 3*IQR
        outlier_list_col = self.train_df[(self.train_df['earning_diff'] < (Q1 - outlier_step)) 
                                            | (self.train_df['earning_diff'] > (Q3 + outlier_step))].index
        self.train_df = self.train_df.drop(index=outlier_list_col, axis=0)
        
        
        logging.debug("train_df shape : ")
        logging.debug(self.train_df.shape)

    def train(self):        
        

        
        # x = self.train_df.loc[:, self.train_df.columns != 'earning_diff']
        x = self.train_df[self.train_df.columns.difference(['earning_diff'])]
        y = self.train_df[['earning_diff']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
        self.mlr.fit(x_train, y_train)
        print ( "mlr score : ", self.mlr.score(x_train, y_train) )
        self.nn.fit(x_train, y_train)
        print ( "nn score : ", self.mlr.score(x_train, y_train) )
        self.rfg.fit(x_train, y_train.values.ravel())
        print ( "rfg score : ", self.mlr.score(x_train, y_train) )

        
        # weight 출력
        logging.debug("result regression. weight : ")
        logging.debug(x.columns)
        logging.debug(self.mlr.coef_)
        weight_df = pd.DataFrame(self.mlr.coef_, columns=x.columns)
        weight_df.to_csv("./weight.csv", index=False)
        print(weight_df)
        self.prediction(x_train, x_test, y_train, y_test)
           
    def prediction(self,  x_train, x_test, y_train, y_test):
        y_predict = self.mlr.predict(x_test)
        if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
            plt.scatter(y_test, y_predict, alpha=0.4)
            pd.concat([pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_predict).reset_index()],axis=1).to_csv("./prediction_result.csv")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("MULTIPLE LINEAR REGRESSION")
            plt.show()
            # plt.scatter(self.train_df[['pbRatio_normal']], self.train_df[['earning_diff']], alpha=0.4)
            # plt.show()
        y_predict = self.rfg.predict(x_test)
        if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
            plt.scatter(y_test, y_predict, alpha=0.4)
            pd.concat([pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_predict).reset_index()],axis=1).to_csv("./prediction_result.csv")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("MULTIPLE LINEAR REGRESSION")
            plt.show()
            
        y_predict = self.nn.predict(x_test)
        if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
            plt.scatter(y_test, y_predict, alpha=0.4)
            pd.concat([pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_predict).reset_index()],axis=1).to_csv("./prediction_result.csv")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("MULTIPLE LINEAR REGRESSION")
            plt.show()
        
        # input = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
        # my_predict = mlr.predict(input)


class MyDataset(Dataset):
 
    def __init__(self, conf):

        self.train_files = []
        
        for year in range(int(conf['START_YEAR']), int(conf['END_YEAR'])):
            traindata_dir = conf['ROOT_PATH'] + '/regressor_data/'
            path = traindata_dir + str(year) + "*train.csv"
            year_files = [file for file in glob.glob(path)]
            self.train_files.extend(year_files)
        
        self.train_df = pd.DataFrame()
        for fpath in self.train_files:     
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.loc[:, use_col_list]
            
            # df = df[df.isnull().sum(axis=1) < 50]
            # df = df.loc[:, df.isnull().sum(axis=0) < 100]         
            self.train_df = pd.concat([self.train_df, df], axis=0)            
        # self.train_df = self.train_df.fillna(self.train_df.mean())
        for col in self.train_df.columns:
            if col == 'earning_diff':
                continue
            max_value = self.train_df[col].max()
            min_value = self.train_df[col].min()
            self.train_df[col] = ((self.train_df[col] - min_value) / (max_value - min_value))-0.5
        
        logging.debug(self.train_df.shape)
        
        self.train_df = self.train_df.fillna(0)
        
        
        Q1 = np.percentile(self.train_df['earning_diff'], 25)
        Q3 = np.percentile(self.train_df['earning_diff'], 75)
        IQR = Q3 - Q1
        # 0.5 is not fixed.   reference :  1.5 => remove 0.7%,  0 =>  remove 50%
        outlier_step = 3*IQR
        outlier_list_col = self.train_df[(self.train_df['earning_diff'] < (Q1 - outlier_step)) 
                                            | (self.train_df['earning_diff'] > (Q3 + outlier_step))].index
        self.train_df = self.train_df.drop(index=outlier_list_col, axis=0)
    
        logging.debug(self.train_df.shape)
        
        x = self.train_df.loc[:, self.train_df.columns != 'earning_diff'].values
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
        self.fc1 = nn.Linear(len(use_col_list)-1, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        h = self.dropout(F.tanh(self.fc1(x)))
        h = self.dropout(F.tanh(self.fc2(h)))
        h = self.fc3(h)
        return h
    
    def mtrain(self):

        # for regression network
        net = RegressionNetwork(self.conf)
        net2 = RegressionNetwork(self.conf)
        optimizer = optim.AdamW(net.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        myDs = MyDataset(self.conf)
        train_loader = DataLoader(myDs, batch_size=128, shuffle=False)
        min_loss = 9999
        
        writer = SummaryWriter('scalar/')
        
        for epoch in range(1,201):
            print("epoch : ", epoch)
            net.train()
            loss_sum = 0
            for i, (data, labels) in enumerate(train_loader):
                
                pred = net(data)
                loss = loss_fn(pred,labels) 
                loss_sum += loss
                writer.add_scalar("Loss/train", loss, epoch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if loss_sum < min_loss:
                min_loss = loss
                torch.save(net.state_dict(), './model_state_dict.pt')  # 모델 객체의 state_dict 저장
            print("min_loss : ", min_loss)
            print("loss : ", loss_sum)
            test_loader = DataLoader(myDs, batch_size=1024, shuffle=False)
            # net2.load_state_dict(torch.load('./model_state_dict.pt'))
            net.eval()
            preds = np.empty(shape=(0))
            labels = np.empty(shape=(0))
            for i, (data, label) in enumerate(test_loader):
                optimizer.zero_grad()
                pred = net(data)
                preds = np.append(preds, pred.detach().numpy())
                labels = np.append(labels, label)
            if epoch % 10 == 0:    
                plt.scatter(preds, labels, alpha=0.4)
                plt.xlabel("pred")
                plt.ylabel("labels")
                plt.show()
        writer.close()
        # pred = net(tensor_x_test)
        