import sys
import mariadb
import re
import os
import pandas as pd
import numpy as np
# import json.decoder
# import FinanceDataReader as fdr
# import matplotlib.pyplot as plt
# from pandas_datareader import data
# import yfinance as yf
# import yahoo_fin.stock_info as si
# from pykrx import stock
# from pykrx import bond

API_KEY = '52a6facc9bba04d228d0babd4c98156c'
FMP_URL = "https://financialmodelingprep.com"
EX_SYMBOL = "AAPL"
ROOT_PATH = "./data"
SYMBOL = pd.DataFrame()
START_YEAR = 2021
END_YEAR = 2022


def create_folder(path):
    if not os.path.exists(path):
        print('Creating Folder "{}" ...'.format(path))
        try:
            os.makedirs(path)
            return True
        except OSError:
            print('Error: Creating "{}" directory.'.format(path))
            return False


def get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix=""):
    # brief : 순차적으로 넣고자 하는 값을 url에 포함시켜서 돌려줌
    # input : main_url(url의 main 값), extra_url(뒤에 나머지),
    #         need_symbol(url에 symbol 값이 들어가지는에 대한 boolean. v3는 불필요. v4 주소 처리때문에 필요)
    #           TODO data_list를 완전히 없앤 이유는 나중에는 need_symbol을 data_type flag로 바꿔서 필요한 리스트를
    #                이 함수 내에서 알아서 넣는 것이 더 효율적이기에 list 자체는 완전히 받지 않을 예정  예) SYMBOL이 글로벌로 바뀜
    #         v4_flag(v4 url 형식을 사용할지에 대한 flag)
    #         file_postfix(csv 파일 뒤에 붙는 다른 구분자를 넣고 싶은 경우 사용 예) AAPL_2022_1.csv)
    # output : none
    # example 1 : /api/v3/discounted-cash-flow/AAPL?apikey=***
    #               get_fmp_data_by_list("dicounted-cash-flow", "", True, False)
    #               ./data 폴더 아래에 discounted-cash-flow 폴더를 만들고, list element별 csv를 만듦
    # example 2 : /api/v3/income-statement-as-reported/AAPL?period=quarter&limit=50&apikey=***
    #               get_fmp_data_by_list("income-statement-as-reported",
    #                                    "period=quarter&limit=50&", symbol_list, True, False)

    if need_symbol is False:
        return get_fmp_data_only_one(main_url, extra_url, is_v4, file_postfix)

    path = ROOT_PATH + "/" + main_url
    cre_flag = create_folder(path)

    # TODO 일부만 돌리기 위해 i로 range를 살짝 줘서 돌림 (for test)
    # for elem in SYMBOL:
    for i in range(0, 2):
        elem = SYMBOL[i]
        if not os.path.isfile(path + "/{}.csv".format(elem + file_postfix)):
            if is_v4 is True:
                # TODO symbol 이 외에 list가 올 것이기에 need_symbol flag를 두고 있으나, symbol 이외에는 아직 당장 필요한 것이
                #       없어서 이대로 두었으나 이 loop는 symbol 이외의 list에 대한 대비가 아래 if 문 이외에는 되어 있지 않음
                if need_symbol is True:
                    api_url = FMP_URL + "/api/v4/{}?symbol={}&{}apikey={}".format(main_url, elem, extra_url, API_KEY)
                else:
                    api_url = FMP_URL + "/api/v4/{}?{}apikey={}".format(main_url, extra_url, API_KEY)
            else:
                api_url = FMP_URL + "/api/v3/{}/{}?{}apikey={}".format(main_url, elem, extra_url, API_KEY)
            print('Creating File "{}/{}.csv" <- "{}"'.format(path, elem + file_postfix, api_url))
            try:
                json_data = pd.read_json(api_url)
            except ValueError:
                print("[Warning] No Data. Or Different Data Type")
                continue
            # 읽어왔는데 비어 있을 수 있음. ValueError는 Format이 안맞는 경우고 이 경우는 page=50 과 같은 extra_url 처리 때문
            json_data.to_csv(path+"/{}.csv".format(elem + file_postfix), na_rep='NaN')
            if json_data.empty is True:
                return False
        else:
            if cre_flag is True:
                # 새로 만드는 경우, 이미 csv가 있다는 건 stock list와 delisted list에 중복 값이 있는 상황 (Duplicate)
                # 리스트에 중복값이 왜 들어가게 되었는지 반드시 확인이 필요함. (가정이 깨짐)
                print('[ERROR] Already Exist "{}/{}.csv"'.format(path, elem + file_postfix))
    return True


def get_fmp_data_only_one(main_url, extra_url, is_v4, file_postfix):
    # brief : 하나의 값만 읽어오고자 할 때 사용됨
    # input : main_url(url의 main 값), extra_url(뒤에 나머지 url), v4_flag(v5 url 형식을 사용할지에 대한 flag),
    #        file_postfix(csv 파일 뒤에 붙는 다른 구분자를 넣고 싶은 경우 사용 예) AAPL_2022_1.csv)
    # output : none
    # example 1 :  /api/v3/financial-statement-symbol-lists
    #                   get_fmp_data_only_one("financial-statement-symbol-lists", "", False, "")
    #                   ./data 폴더 아래에 financial-statement-symbol-lists.csv 파일을 만듦
    # FIXME get_fmp_data 함수랑 이쁘게 잘 합쳐보고 싶은데 결국 한번 합쳤다 다시 때어냄
    if is_v4 is True:
        api_url = FMP_URL + "/api/v4/"
    else:
        api_url = FMP_URL + "/api/v3/"

    if main_url.rfind("/") != -1:
        create_folder(ROOT_PATH + "/" + main_url[:main_url.rfind("/")])
    if not os.path.isfile("{}/{}.csv".format(ROOT_PATH, main_url + file_postfix)):
        api_url += "{}?{}apikey={}".format(main_url, extra_url, API_KEY)
        print('Creating File "{}/{}.csv" <- "{}"'.format(ROOT_PATH, main_url + file_postfix, api_url))
        try:
            json_data = pd.read_json(api_url)
        except ValueError:
            print("[Warning] No Data. Or Different Data Type")
            return False
        # 읽어왔는데 비어 있을 수 있음. ValueError는 Format이 안맞는 경우고 이 경우는 page=50 과 같은 extra_url 처리 때문
        json_data.to_csv("{}/{}.csv".format(ROOT_PATH, main_url + file_postfix), na_rep='NaN')
        if json_data.empty is True:
            return False
    return True


def get_fmp_data_preprocessing(main_url, extra_url, need_symbol, is_v4):
    if extra_url.find("year") != -1:
        for year in range(START_YEAR, END_YEAR + 1):
            if re.match("year=[0-9]{4}&period=Q[0-9]{1}", extra_url):
                for quater in range(1, 5):
                    extra_url = re.sub('year=[0-9]{4}&period=Q[0-9]{1}', "[Y_P]", extra_url)
                    file_postfix = "_" + str(year) + "_P" + str(quater)
                    extra_url = extra_url.replace("[Y_P]", "year={}&period=Q{}".format(year, quater))
                    get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)
            elif re.match("quarter=[0-9]{1}&year=[0-9]{4}", extra_url):
                for quater in range(1, 5):
                    extra_url = re.sub('quarter=[0-9]{1}&year=[0-9]{4}', "[Y_Q]", extra_url)
                    file_postfix = "_" + str(year) + "_Q" + str(quater)
                    extra_url = extra_url.replace("[Y_Q]", "quarter={}&year={}".format(quater, year))
                    get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)
            else:
                extra_url = re.sub('year=[0-9]{4}', "[YEAR]", extra_url)
                file_postfix = "_" + str(year)
                extra_url = extra_url.replace("[YEAR]", "year={}".format(year))
                get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)

    elif extra_url.find("from") != -1:
        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                extra_url = re.sub('from=[0-9]{4}-[0-9]{2}-[0-9]{2}&to=[0-9]{4}-[0-9]{2}-[0-9]{2}', "[FT]", extra_url)
                file_postfix = "_" + str(year) + "_" + str(month)
                extra_url = extra_url.replace("[FT]", "from={0}-{1:02d}-01&to={0}-{1:02d}-31".format(year, month))
                get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)

    elif extra_url.find("page") != -1:
        i = 0
        while True:
            extra_url = re.sub('page=[0-9]{1,2}', "[PAGE]", extra_url)
            file_postfix = "_" + str(i)
            extra_url = extra_url.replace("[PAGE]", "page={}".format(i))
            if get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix) is False:
                break;
            # FIXME ignore_index 들어가야하는지 직접 데이터 확인해봐야 함
            #       성용아 주석 좀 정확히 쓰자 얼마나 됐다고 이게 먼말인지 이해를 못하게 썼니..
            #       page 처리와 delisted_stock 처리의 문제로 보이는데 이해해서 필요 없으면 지우자
            # delisted_stock = pd.concat([delisted_stock, json_del_stock])
            i += 1

    elif extra_url.find("date") != -1:
        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                for day in range (1, 32):
                    extra_url = re.sub('date=[0-9]{4}-[0-9]{2}-[0-9]{2}', "[DATE]", extra_url)
                    file_postfix = "_" + str(year) + "_" + str(month) + "_"+ str(day)
                    extra_url = extra_url.replace("[DATE]", "date={0}-{1:02d}-{2:02d}".format(year, month, day))
                    get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)

    else:
        get_fmp_data(main_url, extra_url, need_symbol, is_v4)


def set_symbol():
    if os.path.isfile(ROOT_PATH + "/available-traded/list.csv") is True \
            and os.path.isfile(ROOT_PATH + "/delisted-companies.csv") is True:
        available_traded = pd.read_csv(ROOT_PATH + "/available-traded/list.csv")
        delisted_stock = pd.read_csv(ROOT_PATH + "/delisted-companies.csv")
        stock = pd.concat([available_traded, delisted_stock], ignore_index=True)
        global SYMBOL
        SYMBOL = stock["symbol"]


def get_fmp(api_list):
    create_folder(ROOT_PATH)
    set_symbol()

    for i in range(len(api_list)):
        need_symbol = True if api_list[i].find(EX_SYMBOL) != -1 else False
        # SYMBOL 이 없는 건, SYMBOL을 만들기 위한 file도 만들어지기 전이기 때문에 두번 돌려서 SYMBOL 안쓰는 것부터 만듦
        if need_symbol is True and SYMBOL.empty is True:
            continue
        is_v4 = True if api_list[i].split('/')[2] == "v4" else False
        # Code에 박아 넣은 값인 8은 url의 앞부분인 /api/v4/ 의 길이. v3와 v4 코드 통합을 위해 박아넣음
        main_url = api_list[i].split('?')[0][8:]
        extra_url = "" if api_list[i].find("?") == -1 else api_list[i].split('?')[1]
        if need_symbol is True:
            if is_v4 is True:
                extra_url = "" if len(extra_url) == 10 else extra_url[12:]
            else:
                main_url = main_url[:-5]
        # limit 제거
        extra_url = re.sub('[&]{0,1}limit=[0-9]{2,3}[&]{0,1}', "", extra_url)
        if extra_url != "":
            extra_url = extra_url + "&"
            print("{}\nextra_url : {}".format(api_list[i], extra_url))
        # FIXME 디버깅용 로그라 120칸 넘김 나중에 수정 예정. 두 줄이면 뺐더 넣다 하기 귀찮
        # print("\n{}\nmain_url : {} / extra_url : {} / need_symbol : {} / is_v4 : {}".format(api_list[i], main_url, extra_url, need_symbol, is_v4))
        get_fmp_data_preprocessing(main_url, extra_url, need_symbol, is_v4)


def get_api_list():
    api_df = pd.read_csv("./api_list.csv", header=0, usecols=["URL"])
    api_df = api_df.dropna()
    api_list = api_df.values.tolist()
    for i in range(len(api_list)):
        api_list[i] = str(api_list[i]).replace(" ","")
        # https 부터 적은 경우에 대한 처리
        if str(api_list[i]).find(FMP_URL) != -1:
            api_list[i] = str(api_list[i])[2: str(api_list[i]).find("apikey") - 1].replace(FMP_URL, "")
        # 여러줄이 들어간 경우, 가장 앞에 써진 url 만을 돌린다. \n을 찾아서 뒤를 전부 지워주는 작업
        elif str(api_list[i]).find('\\') != -1:
            api_list[i] = str(api_list[i])[2: str(api_list[i]).find('\\')]
        else:
            api_list[i] = str(api_list[i])[2:-2]
    return api_list


if __name__ == '__main__':
    api_list = get_api_list()
    # 굳이 symbol을 채우기 위해 별도의 작업을 하는 것보다 2번 돌리는게 효율적
    get_fmp(api_list)
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
