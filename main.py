# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import mariadb
from sqlalchemy import create_engine
import pymysql
import os
import pandas as pd
# import json.decoder
# import FinanceDataReader as fdr
# import matplotlib.pyplot as plt
# from pandas_datareader import data
# import numpy as np
# import yfinance as yf
# import yahoo_fin.stock_info as si
# from pykrx import stock
# from pykrx import bond

api_key = '3e87de43bcb95f9ea6ee8710ade908b7'
fmp_url = "https://financialmodelingprep.com/api/v3/"


def create_folder(path):
    if not os.path.exists(path):
        print('Creating Folder "{}" ...'.format(path))
        try:
            os.makedirs(path)
            return True
        except OSError:
            print('Error: Creating "{}" directory.'.format(path))
            return False


def create_database():
    # maria DB 첫 설치 시 아래의 SQL문으로 db와 user를 만들어줘야함
    # use mysql;
    # create database quantdb
    # create user quant@localhost identified by '0710';
    # grant privileges on quantdb.* to quant@localhost;
    # flush privileges
    # try:
    #    conn = mariadb.connect(
    #        user="quant",
    #        password="0710",
    #        host="127.0.0.1",
    #        port=710,
    #        database="quantdb"
    #    )
    # except mariadb.Error as e:
    #    print("Error connecting to MariaDB Platform: {}".format(e))
    #    sys.exit(1)
    pymysql.install_as_MySQLdb()
    engine = create_engine("mysql+pymysql://{}:{}@localhost/{}".format("quant", "0710", "quantdb"))

    # 읽어왔던 URL의 이름과 동일하게 테이블명을 만듦
    # fmp_url + "available-traded/list?apikey={}".format(api_key)
    listed_stock = pd.read_csv("./data/listed_stock.csv")
    listed_stock.to_sql('available_traded', engine, if_exists='replace', index=False, index_label=None, chunksize=512)


def get_fmp_data_by_stock(stock_list, parent_path, url_1, url_2):
    path = parent_path + url_1
    cre_flag = create_folder(path)
    # TODO FMP의 Free 버젼은 250 req per day
    # for i in range(len(stock_list["symbol"])):
    for i in range(4, 7):
        symbol = stock_list["symbol"][i]
        if not os.path.isfile(path + "/{}.csv".format(symbol)):
            print('Creating File "{}/{}.csv" ...'.format(path, symbol))
            json_profile = pd.read_json(fmp_url + "{}/{}?{}apikey={}".format(url_1, symbol, url_2, api_key))
            json_profile.to_csv(path+"/{}.csv".format(symbol), na_rep='NaN')
        else:
            if cre_flag is True:
                # 새로 만드는 경우, 이미 csv가 있다는 건 stock list와 delisted list에 중복 값이 있는 상황 (Duplicate)
                # 리스트에 중복값이 왜 들어가게 되었는지 반드시 확인이 필요함. (가정이 깨짐)
                print('[ERROR] Already Exist "{}/{}.csv"'.format(path, symbol))


def get_fmp_financial_statement(symbol_list_path, new_fs_):
    symbol_list = pd.read_csv(symbol_list_path)
    # for i in range(len(stock_list["symbol"])):
    for i in range(3, 4):
        symbol = symbol_list["symbol"][i]
        if not os.path.isfile("./data/fs/{}.csv".format(symbol)):
            print("Creating [ {} ] Financial Statements File...".format(symbol))
            # TODO limt 내용을 fmp 유료 결제 후 분기 재무제표로 바꿔야 함
            url_is = fmp_url + "income-statement/{}?limit=120&apikey={}".format(symbol, api_key)
            pd_is = pd.read_json(url_is)
            url_bs = fmp_url + "balance-sheet-statement/{}?limit=120&apikey={}".format(symbol, api_key)
            pd_bs = pd.read_json(url_bs)
            url_cf = fmp_url + "cash-flow-statement/{}?limit=120&apikey={}".format(symbol, api_key)
            pd_cf = pd.read_json(url_cf)
            # 재무재표가 income, balance sheet, cash flow 로 나눠서 제공하고 있어 merge 해주는 작업을 해야함
            pd_mg = pd_is.merge(pd_bs, left_on=['date', 'symbol'], right_on=['date', 'symbol'], how='inner',
                                suffixes=('', '_del'))
            pd_mg = pd_mg.merge(pd_cf, left_on=['date', 'symbol'], right_on=['date', 'symbol'], how='inner',
                                suffixes=('', '_del'))
            pd_mg = pd_mg[[c for c in pd_mg.columns if not c.endswith('_del')]]
            pd_mg.to_csv("./data/fs/{}.csv".format(symbol), index=False, na_rep='NaN')
        else:
            if new_fs_ is True:
                # 새로 만드는 경우, 이미 csv가 있다는 건 stock list와 delisted list에 중복 값이 있는 상황
                # 리스트에 중복값이 왜 들어가게 되었는지 반드시 확인이 필요함. (가정이 깨짐)
                print("[ERROR] Already Exist {}.csv".format(symbol))


def get_fmp_stock_info(symbol_list_path):
    symbol_list = pd.read_csv(symbol_list_path)
    # TODO FMP의 Free 버젼은 250 req per day
    # for i in range(len(stock_list["symbol"])):
    for i in range(4, 5):
        # FIXME yahoo finance를 이용해서 stock info를 가져옴. api 사용량 제한으로 2일 이상 돌려야해서 시간 날 때 해야함
        # df_stock_info = pd.DataFrame()
        # for i in range(len(stock_list["symbol"])) :
        # for i in range(3, 4) :
        #    print("{} ({}/{})".format(stock_list["symbol"][i], i, len(stock_list["symbol"])))
        #    df_stock_info = pd.concat([df_stock_info, pd.DataFrame([yf.Ticker(stock_list["symbol"][i]).info])])
        # df_stock_info.to_csv("test.csv", index=False, na_rep='NaN')

        symbol = symbol_list["symbol"][i]
        # TODO yahoo 대신에 FMP에서 가져와도 될 듯. 필요했던 데이터인 ipoDate와 Sector 정보를 아래에서 얻어올 수 있음
        url_profile = fmp_url + "profile/{}?apikey={}".format(symbol, api_key)
        json_profile = pd.read_json(url_profile)
        json_profile.to_csv("./data/profile", index=False, na_rep='NaN')


def get_fmp():
    create_folder("./data")

    if not os.path.isfile("./data/listed_stock.csv"):
        print("Creating Stock File...")
        url_stock = fmp_url + "available-traded/list?apikey={}".format(api_key)
        listed_stock = pd.read_json(url_stock)
        listed_stock.to_csv("./data/listed_stock.csv", na_rep='NaN')
        # FIXME finance-datareader를 이용해서 NASDAQ stock list 받음
        # if not os.path.isfile("./NASDAQ.csv"):
        #       df_NASDAQ = fdr.StockListing('NASDAQ')
        #       df_NASDAQ.to_csv("NASDAQ.csv")
    else:
        listed_stock = pd.read_csv("./data/listed_stock.csv")

    if not os.path.isfile("./data/delisted_stock.csv"):
        print("Creating Delisted Stock File...")
        i = 1
        delisted_stock = pd.DataFrame()
        while True:
            url_del_stock = fmp_url + "delisted-companies?page={}&apikey={}".format(i, api_key)
            json_del_stock = pd.read_json(url_del_stock)
            print("Get Delisted Stock File - {} page".format(i))
            if json_del_stock.empty is True:
                delisted_stock.to_csv("./data/delisted_stock.csv", na_rep='NaN')
                break
            else:
                # FIXME ignore_index 들어가야하는지 직접 데이터 확인해봐야 함
                delisted_stock = pd.concat([delisted_stock, json_del_stock])
            i += 1
    else:
        delisted_stock = pd.read_csv("./data/delisted_stock.csv")

    stock = pd.concat([listed_stock, delisted_stock], ignore_index=True)

    # Create Stock Info Folder & Files
    get_fmp_data_by_stock(stock, "./data/", "profile", "")

    # Create Financial Statement Folder & Files (income, balance-sheet, cash-flow)
    create_folder("./data/financial-statement/")

    # TODO FMP 유료 결제 후 분기에서 연간 데이터를 분기 데이터로 바꾸기 위해 url_2 값(limit=120&)을 변경해주어야 함
    get_fmp_data_by_stock(stock, "./data/financial-statement/", "income-statement", "limit=120&")
    get_fmp_data_by_stock(stock, "./data/financial-statement/", "balance-sheet-statement", "limit=120&")
    get_fmp_data_by_stock(stock, "./data/financial-statement/", "cash-flow-statement", "limit=120&")


if __name__ == '__main__':

    get_fmp()
    create_database()

    ################################################################################################
    # (1) tickers를 이용한 재무재표 예제
    # symbol = 'GOOGL'
    # sp500_ticker = si.tickers_sp500()
    # print(si.get_balance_sheet(symbol))

    # (2) MultiProcessing 예제
    # from multiprocessing import Pool
    # pool = Pool(8)
    # return = pool.map(method, parameters)
    # sp500_info = pool.map(, sp500_tickers)

    # (3) DataReader 예제
    # symbol = 'GOOGL'
    # web = 'yahoo'
    # start_date = '2004-08-19'
    # end_date = '2020-04-17'
    # google_data = data.DataReader(symbol, web, start_date, end_date)
    # print(google_data.head(9))
    # google_data['Close'].plot()
    # df = stock.get_market_fundamental("20220104", "20220206", "005930", freq="m")
    ################################################################################################
