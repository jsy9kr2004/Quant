# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import mariadb
import re
import os
import pandas as pd
# import json.decoder
# import FinanceDataReader as fdr
# import matplotlib.pyplot as plt
# from pandas_datareader import data
import numpy as np
# import yfinance as yf
# import yahoo_fin.stock_info as si
# from pykrx import stock
# from pykrx import bond

API_KEY = '52a6facc9bba04d228d0babd4c98156c'
FMP_URL = "https://financialmodelingprep.com"
EX_SYMBOL = "APPL"
ROOT_PATH = "./data"


def create_folder(path):
    if not os.path.exists(path):
        print('Creating Folder "{}" ...'.format(path))
        try:
            os.makedirs(path)
            return True
        except OSError:
            print('Error: Creating "{}" directory.'.format(path))
            return False


def get_fmp_data(main_url, data_list, extra_url, need_symbol, is_v4):
    # brief : 순차적으로 넣고자 하는 값을 url에 포함시켜서 돌려줌
    # input : main_url(url의 main 값), data_list (url에 넣고자하는 list), extra_url(뒤에 나머지)
    #            v4_flag(v4 url 형식을 사용할지에 대한 flag)
    # output : none
    # example 1 : /api/v3/discounted-cash-flow/AAPL?apikey=***
    #               get_fmp_data_by_list("./data". "dicounted-cash-flow", symbol_list, "")
    #               ./data 폴더 아래에 discounted-cash-flow 폴더를 만들고, list element별 csv를 만듦
    # example 2 : /api/v3/etf-holder/SPY?apikey=***
    #               get_fmp_data_by_list("./data", "etf-holder", etf_list, "")
    # example 3 : /api/v3/income-statement-as-reported/AAPL?period=quarter&limit=50&apikey=***
    #               get_fmp_data_by_list("./data/financial-statement", "income-statement-as-reported",
    #                                                                     symbol_list, "period=quarter&limit=50&")
    # [주의] extra_url을 확인해서 필요하다면 &를 넣어주어야 함
    if is_v4 is True:
        api_url = FMP_URL + "/api/v4/"
    else:
        api_url = FMP_URL + "/api/v3/"

    # file 한개만 만들고 나가면 되는 애들부터 처리
    if len(data_list) == 0:
        if main_url.rfind("/") != -1:
            create_folder(ROOT_PATH + "/" + main_url[:main_url.rfind("/")])
        if not os.path.isfile("{}/{}.csv".format(ROOT_PATH, main_url)):
            api_url += "{}?{}apikey={}".format(main_url, extra_url, API_KEY)
            print('Creating File "{}/{}.csv" <- "{}"'.format(ROOT_PATH, main_url, api_url))
            try:
                json_data = pd.read_json(api_url)
            except ValueError:
                print("[Warning] No Data. Or Different Data Type")
                return
            json_data.to_csv("{}/{}.csv".format(ROOT_PATH, main_url), na_rep='NaN')
        return

    path = ROOT_PATH + "/" + main_url
    cre_flag = create_folder(path)

    # TODO FMP의 Free 버젼은 250 req per day / 일부만 돌리기 위해 i로 range를 살짝 줘서 돌림 (for test)
    # for elem in data_list:
    for i in range(0, 1):
        elem = data_list[i]
        if not os.path.isfile(path + "/{}.csv".format(elem)):
            if is_v4 is True:
                if need_symbol is True:
                    api_url += "{}?symbol={}&{}apikey={}".format(main_url, elem, extra_url, API_KEY)
                else:
                    api_url += "{}/{}?{}apikey={}".format(main_url, elem, extra_url, API_KEY)
            else:
                api_url += "{}/{}?{}apikey={}".format(main_url, elem, extra_url, API_KEY)
            print('Creating File "{}/{}.csv" <- "{}"'.format(path, elem, api_url))
            try:
                json_data = pd.read_json(api_url)
            except ValueError:
                print("[Warning] No Data. Or Different Data Type")
                continue
            json_data.to_csv(path+"/{}.csv".format(elem), na_rep='NaN')
        else:
            if cre_flag is True:
                # 새로 만드는 경우, 이미 csv가 있다는 건 stock list와 delisted list에 중복 값이 있는 상황 (Duplicate)
                # 리스트에 중복값이 왜 들어가게 되었는지 반드시 확인이 필요함. (가정이 깨짐)
                print('[ERROR] Already Exist "{}/{}.csv"'.format(path, elem))


def get_fmp_data_only_one(path, main_url, v4_flag=False):
    # brief : 하나의 값만 읽어오고자 할 때 사용됨
    # input : path(저장하고자 하는 폴더 폴더), main_url(url의 main 값), extra_url(뒤에 나머지 url)
    #            v4_flag(v4 url 형식을 사용할지에 대한 flag)
    # output : none
    # example 1 :  /api/v3/financial-statement-symbol-lists
    #                   get_fmp_data_only_one("./data", "financial-statement-symbol-lists", "")
    #                   ./data 폴더 아래에 financial-statement-symbol-lists.csv 파일을 만듦
    if not os.path.isfile("{}/{}.csv".format(path, main_url)):
        print('Creating File "{}/{}.csv" ...'.format(path, main_url))
        json_data = pd.read_json(fmp_url + "{}?apikey={}".format(main_url, api_key))
        json_data.to_csv("{}/{}.csv".format(path, main_url), na_rep='NaN')


def get_fmp(api_list):
    create_folder("./data")

    if not os.path.isfile("./data/available-traded.csv"):
        print("Creating Stock File...")
        url_symbol = FMP_URL + "available-traded/list?apikey={}".format(API_KEY)
        available_traded = pd.read_json(url_symbol)
        available_traded.to_csv("./data/available-traded.csv", na_rep='NaN')
        # FIXME finance-datareader를 이용해서 NASDAQ stock list 받음 (필요할까?)
        # if not os.path.isfile("./NASDAQ.csv"):
        #       df_NASDAQ = fdr.StockListing('NASDAQ')
        #       df_NASDAQ.to_csv("NASDAQ.csv")
    else:
        available_traded = pd.read_csv("./data/available-traded.csv")

    if not os.path.isfile("./data/delisted_stock.csv"):
        print("Creating Delisted Stock File...")
        i = 1
        delisted_stock = pd.DataFrame()
        while True:
            url_del_stock = FMP_URL + "delisted-companies?page={}&apikey={}".format(i, API_KEY)
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

    stock = pd.concat([available_traded, delisted_stock], ignore_index=True)

    for i in range(len(api_list)):
        need_symbol = True if api_list[i].find("AAPL") != -1 else False
        is_v4 = True if api_list[i].split('/')[2] == "v4" else False
        # Code에 박아 넣은 값인 8은 url의 앞부분인 /api/v4/ 의 길이. v3와 v4 코드 통합을 위해 박아넣음
        main_url = api_list[i].split('?')[0][8:]
        extra_url = "" if api_list[i].find("?") == -1 else api_list[i].split('?')[1]
        if need_symbol is True:
            if is_v4 is True:
                extra_url = "" if len(extra_url) == 10 else extra_url[12:]
            else:
                main_url = main_url[:-5]
        if extra_url != "":
            extra_url = extra_url + "&"
        #    print("{}\nextra_url : {}".format(api_list[i], extra_url))
        # FIXME 디버깅용 로그라 120칸 넘김 나중에 수정 예정
        print("{}\nmain_url : {} / extra_url : {} / need_symbol : {}".format(api_list[i], main_url, extra_url, need_symbol))
        get_fmp_data(main_url, "" if need_symbol is False else stock["symbol"], extra_url, need_symbol, is_v4)


def get_api_list():
    api_df = pd.read_csv("./api_list.csv", header=0, usecols=["URL"])
    api_df = api_df.dropna()
    api_list = api_df.values.tolist()
    for i in range(len(api_list)):
        if str(api_list[i]).find(FMP_URL) != -1:
            api_list[i] = str(api_list[i])[2: str(api_list[i]).find("apikey") - 1].replace(FMP_URL, "").strip()
        elif str(api_list[i]).find('\\') != -1:
            api_list[i] = str(api_list[i])[2: str(api_list[i]).find('\\')]
        else:
            api_list[i] = str(api_list[i])[2:-2]
    return api_list


if __name__ == '__main__':
    api_list = get_api_list()
    get_fmp(api_list)
    # get_fmp_es()
    # create_mariadb()
    # create_database()

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

    # (4) dataframe을 이용한 merge 예시
    # url_is = fmp_url + "income-statement/{}?limit=120&apikey={}".format(symbol, api_key)
    # pd_is = pd.read_json(url_is)
    # url_bs = fmp_url + "balance-sheet-statement/{}?limit=120&apikey={}".format(symbol, api_key)
    # pd_bs = pd.read_json(url_bs)
    # url_cf = fmp_url + "cash-flow-statement/{}?limit=120&apikey={}".format(symbol, api_key)
    # pd_cf = pd.read_json(url_cf)
    # 재무재표가 income, balance sheet, cash flow 로 나눠서 제공하고 있어 merge 해주는 작업을 해야함
    # pd_mg = pd_is.merge(pd_bs, left_on=['date', 'symbol'], right_on=['date', 'symbol'], how='inner',
    #                    suffixes=('', '_del'))
    # pd_mg = pd_mg.merge(pd_cf, left_on=['date', 'symbol'], right_on=['date', 'symbol'], how='inner',
    #                    suffixes=('', '_del'))
    # pd_mg = pd_mg[[c for c in pd_mg.columns if not c.endswith('_del')]]
    # pd_mg.to_csv("./data/fs/{}.csv".format(symbol), index=False, na_rep='NaN')

    # def create_database():
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
    # pymysql.install_as_MySQLdb()
    # engine = create_engine("mysql+pymysql://{}:{}@localhost/{}".format("quant", "0710", "quantdb"))

    # 읽어왔던 URL의 이름과 동일하게 테이블명을 만듦
    # fmp_url + "available-traded/list?apikey={}".format(api_key)
    # listed_stock = pd.read_csv("./data/listed_stock.csv")
    # listed_stock.to_sql('available_traded', engine, if_exists='replace', index=False, index_label=None, chunksize=512)

    ################################################################################################
