# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import FinanceDataReader as fdr
import os.path
import matplotlib.pyplot as plt
from pandas_datareader import data
import pandas as pd
import yahoo_fin.stock_info as si


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if not os.path.isfile("./NASDAQ.csv"):
        df_NASDAQ = fdr.StockListing('NASDAQ')
        df_NASDAQ.to_csv("NASDAQ.csv")

    symbol = 'GOOGL'
    web = 'yahoo'
    start_date = '2004-08-19'
    end_date = '2020-04-17'
    google_data = data.DataReader(symbol, web, start_date, end_date)
    print(google_data.head(9))
    google_data['Close'].plot()

    nasdaq_ticker = si.tickers_nasdaq()



    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


#from multiprocessing import Pool
#poool = Pool(8)  # 컴퓨터 자원에 알맞게 주시면 될 것 같습니다.!
# return = pool.map(method, parameters)
#sp500_info = pool.map(get_stockinfo, sp500_tickers)