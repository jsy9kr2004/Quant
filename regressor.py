import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import torch.optim as optim


class RegressionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h
    
    @staticmethod
    def train():
        df = pd.read_csv("./sample.csv")

        x = df[df.columns.differnece['target']]
        y = df[['target']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
        # for regression network
        net = RegressionNetwork()
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        loss = nn.MSELoss()

        for _ in range(500):
            optimizer.zero_grad()
            pred = net(x_train)
            #print(pred.shape)
            loss(pred, y_train).backward()
            optimizer.step()
        
        pred = net(x_test)
        


class Regressor:
    def __init__(self, main_ctx, conf):
        path = "./reports/EVAL_REPORT_"
        idx = 0
        while True:
            if not os.path.exists(path + str(idx) + ".csv"):
                path = path + str(idx) + ".csv"
                logging.info('REPORT PATH: "{}" ...'.format(path))
                break
            else:
                idx += 1
        idx -= 1
        # df = pd.read_csv("./reports/EVAL_REPORT_{}.csv".format(idx))
        self.df = pd.read_csv("./sample.csv")
        
        # for linear regression()
        self.mlr = LinearRegression()
    
        
    def train(self):
        
        # for linear regression
        x = self.df[self.df.columns.differnece['target']]
        y = self.df[['target']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
        self.mlr.fit(x_train, y_train)
        self.prediction(x_test, y_test)
    
                
        
    def prediction(self, x_test, y_test):
        # input = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
        # my_predict = mlr.predict(input)
        y_predict = self.mlr.predict(x_test)
        plt.scatter(y_test, y_predict, alpha=0.4)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("MULTIPLE LINEAR REGRESSION")
        plt.show()
        # weight ㅊㅜㄹㄹㅕㄱ 
        print(self.mlr.coef_)
        plt.scatter(self.df[['pbRatio']], self.df[['target']], alpha=0.4)
        plt.show()
        
