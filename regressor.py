import glob
import joblib
import logging
import torch

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as nn_f
import torch.optim as optim

from datasets import Dataset
from g_variables import use_col_list, cal_col_list, cal_ev_col_list, sector_map, sparse_col_list
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# from torch.utils.tensorboard import SummaryWriter
# import torch.utils.data as data_utils
# from xml.dom import XHTML_NAMESPACE
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error

USE_COL='FULL'
PER_SECTOR=True

y_col_list=["period_price_diff","earning_diff","sector", "symbol"]

if USE_COL == 'FULL':
    use_col_list_wydiff = list(map(lambda x: "Ydiff_" + x, cal_col_list))
    use_col_list_wqdiff = list(map(lambda x: "Qdiff_" + x, cal_col_list))
    use_col_list_woverMC = list(map(lambda x: "OverMC_" + x, cal_col_list))
    use_col_list_wadaMC = list(map(lambda x: 'adaptiveMC_' + x, cal_ev_col_list))
    use_col_list_wprev = use_col_list + use_col_list_wydiff + use_col_list_wqdiff + use_col_list_woverMC + use_col_list_wadaMC
    # use_col_list_wprev - sparse_cols
    use_col_list_wprev = [x for x in use_col_list_wprev if x not in sparse_col_list]
    use_col_list_wprev = list(map(lambda x: x+'_normal', use_col_list_wprev))
    use_col_list_wprev = use_col_list_wprev+ y_col_list

if USE_COL == '97':
    print("just use use_col_list")
    use_col_list_wprev = list(map(lambda x: x+'_normal', use_col_list))
    use_col_list_wprev = use_col_list_wprev + y_col_list

class Regressor:
    def __init__(self, conf):
        self.conf = conf
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # aidata_dir = conf['ROOT_PATH'] + '/regressor_data_0' + str(conf['START_MONTH']) + '/'
        aidata_dir = conf['ROOT_PATH'] + '/regressor_data_01/'
        
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
        self.models = dict()
        self.sector_models=dict()
        self.sector_x_train = dict()
        self.sector_y_train = dict()

    def dataload(self):
        for fpath in self.train_files:
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.loc[:, use_col_list_wprev]
            logging.debug(df.shape)  
            df = df[df.isnull().sum(axis=1) < 500]
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
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df.loc[:, use_col_list_wprev]
            
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
        self.y_train = self.train_df[['earning_diff']]
        for sec in self.sector_list:
            print("sector : ", sec)
            self.sector_x_train[sec] = self.sector_train_dfs[sec][self.sector_train_dfs[sec].columns.difference(y_col_list)]
            self.sector_y_train[sec] = self.sector_train_dfs[sec][['earning_diff']]
        
        self.x_test = self.test_df[self.test_df.columns.difference(y_col_list)]
        self.y_test = self.test_df[['earning_diff']]
    
    def def_model(self):
        self.models[0] = RandomForestRegressor(n_jobs=-1, n_estimators=400, min_samples_leaf=2)
        self.models[1] = RandomForestRegressor(n_jobs=-1, n_estimators=400, min_samples_split=8, min_samples_leaf=4)
        self.models[2] = xgboost.XGBRegressor(n_estimators=450, learning_rate=0.05, gamma=0, subsample=0.899,
                                        colsample_bytree=0.8, max_depth=9)
        for sec in self.sector_list:
            cur_key = (sec, 0)
            self.sector_models[cur_key] = RandomForestRegressor(n_jobs=-1, n_estimators=400, min_samples_leaf=2)
            cur_key = (sec, 1)
            self.sector_models[cur_key] = RandomForestRegressor(n_jobs=-1, n_estimators=400, min_samples_split=8, min_samples_leaf=4)
            cur_key = (sec, 2)
            self.sector_models[cur_key] = xgboost.XGBRegressor(n_estimators=450, learning_rate=0.05, gamma=0, subsample=0.899,
                                        colsample_bytree=0.8, max_depth=9)
        
    
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

        for i, model in self.models.items():
            logging.info("start fitting RandomForestRegressor")
            model.fit(self.x_train, self.y_train.values.ravel())
            # filename = 'rfg' + str(i) + '_model' + str(self.conf['START_MONTH']) + '.sav'
            filename = self.conf['ROOT_PATH'] + '/models/' + 'model_' + str(i) + '.sav'
            joblib.dump(model, filename)
            logging.info("rfg score : ")
            logging.info(model.score(self.x_train, self.y_train))
            logging.info("end fitting RandomForestRegressor")
            ftr_importances_values = model.feature_importances_
            ftr_importances = pd.Series(ftr_importances_values, index = self.x_train.columns)
            ftr_importances.to_csv('./reports/model_importances.csv')
            ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
            logging.info(ftr_top20)   
        
        
        if PER_SECTOR == True:
            for sec in self.sector_list:
                for i in range(3):
                    k = (sec, i)
                    model = self.sector_models[k]
                    model.fit(self.sector_x_train[sec], self.sector_y_train[sec].values.ravel())
                    filename = self.conf['ROOT_PATH'] + '/models/' + sec + '_model_' + str(i) +  '.sav'
                    joblib.dump(model, filename)
                    logging.info("model score : ")
                    logging.info(model.score(self.sector_x_train[sec], self.sector_y_train[sec]))
                    logging.info("end fitting RandomForestRegressor")
                    ftr_importances_values = model.feature_importances_
                    ftr_importances = pd.Series(ftr_importances_values, index = self.sector_x_train[sec].columns)
                    ftr_importances.to_csv('./reports/' + sec + '_model_importances.csv')
                    ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
                    logging.info(ftr_top20)

           
    def evaluation(self):
        self.models = dict()
        self.models[0] = joblib.load(self.conf['ROOT_PATH'] + '/models/' + 'model_0.sav')
        self.models[1] = joblib.load(self.conf['ROOT_PATH'] + '/models/' + 'model_1.sav')
        self.models[2] = joblib.load(self.conf['ROOT_PATH'] + '/models/' + 'model_2.sav')

        pred_col_list = ['ai_pred_avg'] 
        for i in range(3):
            pred_col_name = 'model_' + str(i) + '_prediction'
            pred_col_list.append(pred_col_name)

        model_eval_hist = []
        for test_idx, (testdate, df) in enumerate(self.test_df_list):
            print("evaluation date : ")
            tdate = "_".join(testdate.split("\\")[4].split('_')[0:2])
            print(tdate)
            x_test = df[df.columns.difference(y_col_list)]
            y_test = df[['earning_diff']]
            preds = np.empty((0,x_test.shape[0]))            
            
            df['label'] = y_test
            for i, model in self.models.items():    
                pred_col_name = 'model_' + str(i) + '_prediction'
                y_predict = model.predict(x_test)
                df[pred_col_name] = y_predict
                preds = np.vstack((preds, y_predict[None,:]))
            
            df['ai_pred_avg'] = np.average(preds, axis=0)
            df.to_csv("./reports/prediction_ai_{}.csv".format(tdate))

            # 각 model의 top_k 종목의 period_price_diff 합을 구해서 model 최종 평가
            topk_period_earning_sums = []
            topk_list = [(0,9), (0,19), (0,29)]
            for s, e in topk_list:
                logging.info("top" + str(s) + " ~ "  + str(e) )
                k = str(s) + '~' + str(e)
                for col in pred_col_list:
                    top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                    logging.info(col)
                    logging.info((top_k_df['period_price_diff'].sum()/(e-s+1)))
                    topk_period_earning_sums.append(top_k_df['period_price_diff'].sum())
                    
                    top_k_df.to_csv('./reports/prediction_{}_{}_top{}-{}.csv'.format(tdate, col, s, e))
                    model_eval_hist.append([tdate, col, k, top_k_df['period_price_diff'].sum()/(e-s+1), top_k_df['model_0_prediction'].sum()/(e-s+1), top_k_df['model_1_prediction'].sum()/(e-s+1), top_k_df['model_2_prediction'].sum()/(e-s+1)])
        
        col_name = ['start_date', 'model', 'krange', 'avg_earning_per_stock', 'avg_model0_pred', 'avg_model1_pred', 'avg_model2_pred']
        pred_df = pd.DataFrame(model_eval_hist, columns=col_name)
        print(pred_df)
        pred_df.to_csv('./reports/pred_df.csv', index=False)
        # logging.info(model_eval_hist)

        if PER_SECTOR == True:
            testdates=set()
            allsector_topk_df=pd.DataFrame()
            self.sector_models = dict()
            for sec in self.sector_list:
                for i in range(3):
                    filename = self.conf['ROOT_PATH'] + '/models/' + sec + '_model_' + str(i) +  '.sav'
                    k = (sec, i)
                    self.sector_models[k] = joblib.load(self.conf['ROOT_PATH'] + '/models/' +sec + '_model_' + str(i) + '.sav')
            sector_model_eval_hist = []
            for test_idx, (testdate, df, sec) in enumerate(self.sector_test_df_lists):
                tdate = "_".join(testdate.split("\\")[4].split('_')[0:2])
                testdates.add(tdate)
                print("evaluation date : ")
                print(tdate)

                x_test = df[df.columns.difference(y_col_list)]
                y_test = df[['earning_diff']]
                sector_preds = np.empty((0,x_test.shape[0]))
                df['label'] = y_test
                for i in range(3):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    y_predict = model.predict(x_test)
                    df[pred_col_name] = y_predict
                    sector_preds = np.vstack((sector_preds, y_predict[None,:]))

                df['ai_pred_avg'] = np.average(sector_preds, axis=0)
                df.to_csv("./reports/sec_{}_prediction_ai_{}.csv".format(sec, tdate))
                # 각 model의 top_k 종목의 period_price_diff 합을 구해서 model 최종 평가
                topk_period_earning_sums = []
                topk_list = [(0,2)]
                for s, e in topk_list:
                    logging.info("top" + str(s) + " ~ "  + str(e) )
                    k = str(s) + '~' + str(e)
                    for col in pred_col_list:
                        top_k_df = df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                        logging.info(col)
                        logging.info((top_k_df['period_price_diff'].sum()/(e-s+1)))
                        topk_period_earning_sums.append(top_k_df['period_price_diff'].sum())
                        top_k_df.to_csv('./reports/prediction_{}_{}_{}_top{}-{}.csv'.format(tdate, sec, col, s, e))
                        top_k_df['start_date'] = tdate
                        top_k_df['col'] = col
                        allsector_topk_df = pd.concat([allsector_topk_df, top_k_df])
                        sector_model_eval_hist.append([tdate, sec, col, k, top_k_df['period_price_diff'].sum()/(e-s+1), top_k_df['model_0_prediction'].sum()/(e-s+1), top_k_df['model_1_prediction'].sum()/(e-s+1), top_k_df['model_2_prediction'].sum()/(e-s+1)])                        

            col_name = ['start_date', 'sector', 'model', 'krange', 'avg_earning_per_stock', 'avg_model0_pred', 'avg_model1_pred', 'avg_model2_pred']
            pred_df = pd.DataFrame(sector_model_eval_hist, columns=col_name)
            print(pred_df)
            pred_df.to_csv('./reports/sector_{}_pred_df.csv'.format(sec), index=False)
            allsector_topk_df.to_csv('./reports/allsector_pred_df.csv', index=False)
            
            
            persector_hist = []                      
            for tdate in list(testdates):
                for model_name in pred_col_list:
                    tmpdf = allsector_topk_df[allsector_topk_df['start_date']==tdate]
                    tmpdf = tmpdf[tmpdf['col']==model_name]
                    tmpdf = tmpdf.sort_values(by=[model_name], ascending=False, na_position="last")[0:10]
                    persector_hist.append([tdate, model_name, tmpdf['period_price_diff'].sum()/10, tmpdf[model_name].sum()/10])
            col_name = ['start_date', 'model', 'avg_earning_per_stock', 'model_preds']
            pred_df = pd.DataFrame(persector_hist, columns=col_name)
            pred_df.to_csv('./reports/persector_top10_pred.csv', index=False)
                
            
        
        
    def latest_prediction(self, latest_data_path):
        pred_col_list = ['ai_pred_avg']
        for i, in range(3): 
            pred_col_name = 'model_' + str(i) + '_prediction'
            pred_col_list.append(pred_col_name)  

        ldf = pd.read_csv(latest_data_path)
        collist = use_col_list_wprev.copy()
        collist.remove("earning_diff")
        collist.remove("period_price_diff")
        collist.remove("sector")

        ldf = ldf.loc[:, collist]
        ldf = ldf[ldf.isnull().sum(axis=1) < 500]
        ldf = ldf.fillna(0)
        
        input = ldf[ldf.columns.difference(['symbol'])]
        
        preds = np.empty((0, input.shape[0]))

        for i, model in self.models.items():    
            pred_col_name = 'model_' + str(i) + '_prediction'
            y_predict3 = model.predict(input)
            ldf[pred_col_name] = y_predict3
            preds = np.vstack((preds, y_predict3[None,:]))
        
        ldf['ai_pred_avg'] = np.average(preds, axis=0)        
        ldf.to_csv("./latest_prediction.csv")
        
        topk_list = [(0, 10), (0, 20), (0, 30), (3, 20), (5, 20), (3, 30), (10, 30)]
        for s, e in topk_list:
            logging.info("top" + str(s) + " ~ " + str(e))
            for col in pred_col_list:
                top_k_df = ldf.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                top_k_df.to_csv('./reports/latest_prediction_{}_top{}-{}.csv'.format(col, s, e))


        if PER_SECTOR == True:
                      
            ldf = pd.read_csv(latest_data_path)
            for sec in self.sector_list:
                sec_df = ldf[ldf['sector']==sec]
                collist = use_col_list_wprev.copy()
                collist.remove("earning_diff")
                collist.remove("period_price_diff")
                collist.remove("sector")

                sec_df = sec_df.loc[:, collist]
                sec_df = sec_df[sec_df.isnull().sum(axis=1) < 5]
                sec_df = sec_df.fillna(0)
            
                indata = sec_df[sec_df.columns.difference(['symbol'])]
                preds = np.empty((0, indata.shape[0]))

                for i in range(3):
                    k = (sec, i)
                    model = self.sector_models[k]
                    pred_col_name = 'model_' + str(i) + '_prediction'
                    y_predict3 = model.predict(indata)
                    sec_df[pred_col_name] = y_predict3
                    preds = np.vstack((preds, y_predict3[None,:]))
                    
                sec_df['ai_pred_avg'] = np.average(preds, axis=0)        
                sec_df.to_csv("./sec_{}_latest_prediction.csv".format(sec))        
                topk_list = [(0, 2), (0, 4)]
                for s, e in topk_list:
                    logging.info("top" + str(s) + " ~ " + str(e))
                    for col in pred_col_list:
                        top_k_df = sec_df.sort_values(by=[col], ascending=False, na_position="last")[s:(e+1)]
                        top_k_df.to_csv('./reports/latest_prediction_{}_{}_top{}-{}.csv'.format(col, sec, s, e))
        


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
            df = df.loc[:, use_col_list_wprev]
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
            df = df.loc[:, use_col_list_wprev]
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
        self.fc1 = nn.Linear(len(use_col_list_wprev)-3, 97)
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
        self.fc1 = nn.Linear(len(use_col_list_wprev)-3, 97)
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
        self.fc1 = nn.Linear(len(use_col_list_wprev)-3, 97)
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
        self.fc1 = nn.Linear(len(use_col_list_wprev)-3, 97)
        self.fc2 = nn.Linear(97, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.dropout(nn_f.sigmoid(self.fc1(x)))
        h = self.fc2(h)
        return h
