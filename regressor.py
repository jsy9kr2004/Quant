import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

import glob
import logging

from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import torch.optim as optim

        
class Regressor:
    def __init__(self, conf):
        self.conf = conf
        path = "./reports/*train.csv"
        self.train_files = [file for file in glob.glob(path)]
        logging.debug("tarin file list : ", self.train_files)
        self.train_df = pd.DataFrame()
        # for linear regression()
        self.mlr = LinearRegression()
        
    def dataload(self):
        for fpath in self.train_files:     
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df[df.isnull().sum(axis=1) < 50]
            df = df.loc[:, df.isnull().sum(axis=0) < 100]         
            self.train_df = pd.concat([self.train_df, df], axis=0)
        
        self.train_df = self.train_df.drop(columns=['symbol'])
        logging.debug('NaN occurrences in Columns:')
        logging.debug(self.train_df.isnull().sum(axis=0))
        logging.debug('NaN occurrences in Rows:')
        logging.debug(self.train_df.isnull().sum(axis=1))
        self.train_df = self.train_df.fillna(0)
        logging.debug("train_df shape : ", self.train_df.shape)

    def train(self):        
        x = self.train_df.loc[:, self.train_df.columns != 'b']
        y = self.train_df[['earning_diff']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
        self.mlr.fit(x_train, y_train)
        # weight 출력
        logging.debug("result regression. weight : ")
        logging.debug(self.mlr.coef_)
        self.prediction(x_train, x_test, y_train, y_test)
           
    def prediction(self,  x_train, x_test, y_train, y_test):
        y_predict = self.mlr.predict(x_test)
        if self.conf['PRINT_PLT_IN_REGRESSOR'] == 'Y':
            plt.scatter(y_test, y_predict, alpha=0.4)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("MULTIPLE LINEAR REGRESSION")
            plt.show()
            plt.scatter(self.train_df[['pbRatio_normal']], self.train_df[['earning_diff']], alpha=0.4)
            plt.show()
        
        # input = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
        # my_predict = mlr.predict(input)


class MyDataset(Dataset):
 
    def __init__(self):
        path = "./reports/*train.csv"
        self.train_files = [file for file in glob.glob(path)]
        logging.debug("tarin file list : ", self.train_files)
        self.train_df = pd.DataFrame() 
        for fpath in self.train_files:     
            print(fpath)
            df = pd.read_csv(fpath)
            df = df.dropna(axis=0, subset=['earning_diff'])
            df = df[df.isnull().sum(axis=1) < 50]
            df = df.loc[:, df.isnull().sum(axis=0) < 100]         
            self.train_df = pd.concat([self.train_df, df], axis=0)
        self.train_df = self.train_df.drop(columns=['symbol'])
        self.train_df = self.train_df.fillna(0)

        x = self.train_df.loc[:, self.train_df.columns != 'b'].values
        y = self.train_df[['earning_diff']].values
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]


class RegressionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(371, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 1)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h
    
    def train(self):

        # for regression network
        net = RegressionNetwork()
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        loss = nn.MSELoss()
        
        myDs = MyDataset()
        train_loader = DataLoader(myDs, batch_size=16, shuffle=False)

        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = net(data)
            loss(pred, labels).backward()
            optimizer.step()
        
        # pred = net(tensor_x_test)