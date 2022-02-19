# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#import FinanceDataReader as fdr
import os.path
#import matplotlib.pyplot as plt
from pandas_datareader import data
import pandas as pd
import numpy as np
import yfinance as yf
import yahoo_fin.stock_info as si
from pykrx import stock
from pykrx import bond

api_key = '3e87de43bcb95f9ea6ee8710ade908b7'

#url_bs = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/AAPL?datatype=csv&apikey=format(api_key)"
#bs = pd.read_json(url_bs)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
 #   if not os.path.isfile("./NASDAQ.csv"):
 #       df_NASDAQ = fdr.StockListing('NASDAQ')
 #       df_NASDAQ.to_csv("NASDAQ.csv")


    if not os.path.isfile("./stock_list.csv") :
        url_stock_list = "https://financialmodelingprep.com/api/v3/available-traded/list?apikey={}".format(api_key)
        stock_list = pd.read_json(url_stock_list)
        stock_list.loc[:, stock_list.columns != 'price'].to_csv("stock_list.csv", index=False, na_rep='NaN')
    else :
        stock_list = pd.read_csv("./stock_list.csv")

    #df_stock_info = pd.DataFrame()
    # for i in range(len(stock_list["symbol"])) :
    #for i in range(3, 4) :
    #    print("{} ({}/{})".format(stock_list["symbol"][i], i, len(stock_list["symbol"])))
    #    df_stock_info = pd.concat([df_stock_info, pd.DataFrame([yf.Ticker(stock_list["symbol"][i]).info])])

    #df_stock_info.to_csv("test.csv", index=False, na_rep='NaN')

    symbol = stock_list["symbol"][1]
    print(symbol)

    url_is = "https://financialmodelingprep.com/api/v3/income-statement/{}?limit=120&apikey={}".format(symbol, api_key)
    pd_is = pd.read_json(url_is)
    url_bs = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/{}?limit=120&apikey={}".format(symbol, api_key)
    pd_bs = pd.read_json(url_bs)
    url_cf = "https://financialmodelingprep.com/api/v3/cash-flow-statement/{}?limit=120&apikey={}".format(symbol, api_key)
    pd_cf = pd.read_json(url_cf)

    pd_mg = pd_is.merge(pd_bs, left_on=['date', 'symbol'], right_on=['date', 'symbol'], how='inner',
                        suffixes=('', '_delme'))
    pd_mg = pd_mg.merge(pd_cf, left_on=['date', 'symbol'], right_on=['date', 'symbol'], how='inner',
                        suffixes=('', '_delme'))
    pd_mg = pd_mg[[c for c in pd_mg.columns if not c.endswith('_delme')]]

    pd_mg.to_csv("{}.csv".format(symbol), index=False, na_rep='NaN')

    #symbol = 'GOOGL'
#    sp500_ticker = si.tickers_sp500()
    #print(si.get_balance_sheet(symbol))

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#from multiprocessing import Pool
#poool = Pool(8)
#return = pool.map(method, parameters)
#sp500_info = pool.map(get_stockinfo, sp500_tickers)

#    symbol = 'GOOGL'
#    web = 'yahoo'
#    start_date = '2004-08-19'
#    end_date = '2020-04-17'
#    google_data = data.DataReader(symbol, web, start_date, end_date)
#    print(google_data.head(9))
#    google_data['Close'].plot()

#df = stock.get_market_fundamental("20220104", "20220206", "005930", freq="m")