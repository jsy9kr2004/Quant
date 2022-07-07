import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


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
        self.mlr = LinearRegression()
        
    def train(self):
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