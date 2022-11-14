import datetime
import calendar
import json
import logging
import multiprocessing
import os
import re
from time import sleep
import time
import urllib
import urllib.error
import urllib.request

from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler

import dateutil.utils
from dateutil.relativedelta import relativedelta
import requests
import pandas as pd


class FMP:
    def __init__(self, config, main_ctx):
        self.fmp_url = config['FMP_URL']
        self.api_key = config['API_KEY']
        self.ex_symbol = config['EX_SYMBOL']
        self.target_stock_path = config['TARGET_STOCK_LIST']
        self.target_api_path = config['TARGET_API_LIST']
        self.symbol_list = pd.DataFrame()
        self.current_list = pd.DataFrame()
        self.main_ctx = main_ctx

    def flatten_json(self, js, expand_all=False):
        df = pd.json_normalize(json.loads(js) if type(js) == str else js)
        # get first column that contains lists
        ex = df.applymap(type).astype(str).eq("<class 'list'>").all().to_frame(name='bool')
        isin_classlist = (ex['bool'] == True).any()
        if isin_classlist == True:
            col = df.applymap(type).astype(str).eq("<class 'list'>").all().idxmax()
            # explode list and expand embedded dictionaries
            df = df.explode(col).reset_index(drop=True)
            df = df.drop(columns=[col]).join(df[col].apply(pd.Series), rsuffix=f".{col}")
            # any lists left?
            if expand_all and df.applymap(type).astype(str).eq("<class 'list'>").any(axis=1).all():
                df = self.flatten_json(df.to_dict("records"))
            return df
        else:
            return df

    def get_fmp_data(self, main_url, extra_url, need_symbol, is_v4, file_postfix=""):
        """
        brief : 순차적으로 넣고자 하는 값을 url에 포함시켜서 돌려줌
        input : main_url(url의 main 값), extra_url(뒤에 나머지),
                need_symbol(url에 symbol 값이 들어가지는에 대한 boolean. v3는 불필요. v4 주소 처리때문에 필요)
                v4_flag(v4 url 형식을 사용할지에 대한 flag)
                file_postfix(csv 파일 뒤에 붙는 다른 구분자를 넣고 싶은 경우 사용 예) AAPL_2022_1.csv)
        TODO data_list를 완전히 없앤 이유는 나중에는 need_symbol을 data_type flag로 바꿔서 필요한 리스트를
             이 함수 내에서 알아서 넣는 것이 더 효율적이기에 list 자체는 완전히 받지 않을 예정  예) SYMBOL이 글로벌로 바뀜
        output : none
        example 1 : /api/v3/discounted-cash-flow/AAPL?apikey=***
                    get_fmp_data_by_list("dicounted-cash-flow", "", True, False)
                    ./data 폴더 아래에 discounted-cash-flow 폴더를 만들고, list element별 csv를 만듦
        example 2 : /api/v3/income-statement-as-reported/AAPL?period=quarter&limit=50&apikey=***
                   get_fmp_data_by_list("income-statement-as-reported",
                                        "period=quarter&limit=50&", symbol_list, True, False)
        """
        # [RULE] 모든 File들은 Path가 ./data/* 로 시작해야 함. ./data나 ./data/*/* 와 같은 path는 가질 수 없음
        path = self.main_ctx.root_path + "/" + main_url.replace("/", "-").replace("-", "_")
        cre_flag = self.main_ctx.create_dir(path)

        if need_symbol == False:
            data_list = [path[path.rfind("/") + 1:]]
        else:
            # 일부만 돌리기 위해 앞에 5개만 가져옴 (for test) / 나중에 else만 없애면 됨.
            # data_list = self.symbol_list.head(5)
            data_list = self.symbol_list
            if main_url == "historical-price-full":
                logging.info("add ETF symbols to get historical-price-full data")
                data_list = data_list.append(pd.Series(["SPY", "IVV", "VTI", "VOO", "QQQ", "VEA", "IEFA"]))
            
        # for elem in SYMBOL:
        start = time.time()
        if need_symbol == True:
            data_list = data_list.dropna()
        for elem in data_list:
            # TODO url_data = "" 와 같은 줄이 필요할 듯? except 후 continue로 들어갈 때 이전 값이 들어있음. 초기화 필요?
            # json_data = ""
            if (not os.path.isfile(path + "/{}.parquet".format(str(elem) + file_postfix))) \
                    and (not os.path.isfile(path + "/{}.parquetx".format(str(elem) + file_postfix))):
                if is_v4 == True:
                    # TODO symbol 이 외에 list가 올 것이기에 need_symbol flag를 두고 있으나, symbol 이외에는 아직 당장 필요한 것이
                    #       없어서 이대로 두었으나 이 loop는 symbol 이외의 list에 대한 대비가 아래 if 문 이외에는 되어 있지 않음
                    if need_symbol == True:
                        api_url = self.fmp_url + "/api/v4/{}?symbol={}&{}apikey={}".format(main_url, elem, extra_url,
                                                                                           self.api_key)
                    else:
                        api_url = self.fmp_url + "/api/v4/{}?{}apikey={}".format(main_url, extra_url, self.api_key)
                else:
                    if need_symbol == True:
                        api_url = self.fmp_url + "/api/v3/{}/{}?{}apikey={}".format(main_url, elem, extra_url,
                                                                                    self.api_key)
                    else:
                        api_url = self.fmp_url + "/api/v3/{}?{}apikey={}".format(main_url, extra_url, self.api_key)
                try:
                    # TODO 결제 PLAN 더 비싼거 쓰면 sleep 지워도 됨
                    # logging.info("sleep 0.2s")
                    # sleep(0.2)
                    logging.info('Creating File "{}/{}.parquet" <- "{}"'.format(path, elem + file_postfix, api_url))
                    # json_data = pd.read_json(api_url)
                    end = time.time()
                    remain_sec = (0.2 - (end - start))
                    if remain_sec > 0:
                        sleep(remain_sec)
                    start = time.time()
                    url_data = requests.get(api_url)
                    
                except ValueError:
                    # logging.warning("No Data. Or Different Data Type")
                    continue
                except urllib.error.HTTPError:
                    logging.warning("HTTP Error 400, API_URL : ", api_url)
                    continue
                # 읽어왔는데 비어 있을 수 있음. ValueError는 Format이 안맞는 경우고 이 경우는 page=50 과 같은 extra_url 처리 때문
                json_text = url_data.text
                try:
                    json_data = json.loads(json_text)
                except json.decoder.JSONDecodeError:
                    logging.error("json.decoder.JSONDecodeError")
                    if need_symbol == True:
                        continue
                    else:
                        return False
                if json_data == [] or json_data == {}:
                    logging.info("No Data in URL")
                    f = open(path+"/{}.parquetx".format(elem + file_postfix), 'w')
                    f.close()
                    if need_symbol == True:
                        continue
                    else:
                        return False
                json_data = self.flatten_json(json_data, expand_all=True)
                # json_data.to_parquet(path+"/{}.csv".format(elem + file_postfix), na_rep='NaN')
                json_data.to_parquet(path+"/{}.parquet".format(elem + file_postfix))
                if json_data.empty == True:
                    logging.info("No Data in CSV")
                    if need_symbol == True:
                        continue
                    else:
                        return False
            else:
                if cre_flag == True:
                    # 새로 만드는 경우, 이미 csv가 있다는 건 stock list와 delisted list에 중복 값이 있는 상황 (Duplicate)
                    # 리스트에 중복값이 왜 들어가게 되었는지 반드시 확인이 필요함. (가정이 깨짐)
                    logging.error('Already Exist "{}/{}.parquet"'.format(path, elem + file_postfix))
                else:
                    logging.info('Alread Exist File "{}/{}.parquet"'.format(path, elem + file_postfix))
        return True

    def get_fmp_data_preprocessing(self, main_url, extra_url, need_symbol, is_v4):
        if extra_url.find("year") != -1:
            for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
                if re.match("year=[0-9]{4}&period=Q[0-9]{1}", extra_url):
                    for quater in range(1, 5):
                        extra_url = re.sub('year=[0-9]{4}&period=Q[0-9]{1}', "[Y_P]", extra_url)
                        file_postfix = "_" + str(year) + "_P" + str(quater)
                        extra_url = extra_url.replace("[Y_P]", "year={}&period=Q{}".format(year, quater))
                        self.get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)
                elif re.match("quarter=[0-9]{1}&year=[0-9]{4}", extra_url):
                    for quater in range(1, 5):
                        extra_url = re.sub('quarter=[0-9]{1}&year=[0-9]{4}', "[Y_Q]", extra_url)
                        file_postfix = "_" + str(year) + "_Q" + str(quater)
                        extra_url = extra_url.replace("[Y_Q]", "quarter={}&year={}".format(quater, year))
                        self.get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)
                else:
                    extra_url = re.sub('year=[0-9]{4}', "[YEAR]", extra_url)
                    file_postfix = "_" + str(year)
                    extra_url = extra_url.replace("[YEAR]", "year={}".format(year))
                    self.get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)
        elif extra_url.find("from") != -1:
            for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
                # for month in range(1, 13):
                # if dateutil.utils.today() < datetime.datetime(year, month, 1):
                #    break
                extra_url = re.sub('from=[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}&to=[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}', "[FT]",
                                   extra_url)
                file_postfix = "_" + str(year)
                # file_postfix = "_" + str(year) + "_" + str(month)
                # day = calendar.monthrange(year, month)[1]
                # extra_url = extra_url.replace("[FT]", "from={0}-{1}-01&to={0}-{1}-{2}".format(year, month, day))
                extra_url = extra_url.replace("[FT]", "from={0}-01-01&to={0}-12-31".format(year))
                self.get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)
        elif extra_url.find("page") != -1:
            i = 0
            while True:
                extra_url = re.sub('page=[0-9]{1,4}', "[PAGE]", extra_url)
                file_postfix = "_" + str(i)
                extra_url = extra_url.replace("[PAGE]", "page={}".format(i))
                if self.get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix) == False:
                    break
                i += 1
        elif extra_url.find("date") != -1:
            for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
                for month in range(1, 13):
                    for day in range(1, 32):
                        extra_url = re.sub('date=[0-9]{4}-[0-9]{2}-[0-9]{2}', "[DATE]", extra_url)
                        file_postfix = "_" + str(year) + "_" + str(month) + "_" + str(day)
                        extra_url = extra_url.replace("[DATE]", "date={0}-{1:02d}-{2:02d}".format(year, month, day))
                        self.get_fmp_data(main_url, extra_url, need_symbol, is_v4, file_postfix)
        else:
            self.get_fmp_data(main_url, extra_url, need_symbol, is_v4)

    def set_symbol(self):
        """fmp api 로 얻어온 stock_list 와 delisted companies 에서 exchange 가 NASDAQ, NYSE인 symbol들의 list 를 만드는 함수"""
        # fmp api로 얻어온 stock_list 불러오기 
        path = self.main_ctx.root_path + "/stock_list/stock_list.parquet"
        if os.path.isfile(path) == True:
            symbol_list = pd.read_parquet(path)
        else:
            return

        # stock_list에서 type "stock", exchange "NASDQA", "NYSE" 만 가져오기 이를 filtered_symbol에 저장
        # symbol_list = symbol_list.drop(symbol_list.columns[0], axis=1)
        filtered_symbol = symbol_list[(symbol_list['type'] == "stock")
                                      & ((symbol_list['exchangeShortName'] == 'NASDAQ')
                                         | (symbol_list['exchangeShortName'] == 'NYSE'))]
        filtered_symbol = filtered_symbol.reset_index(drop=True)
        filtered_symbol = filtered_symbol.drop(['price', 'exchange', 'name'], axis=1)        
        all_symbol = filtered_symbol
        # target_stock_symbol 과 delisted stock symbol 합쳐 필요한 symbol list 완성
        file_list = os.listdir(self.main_ctx.root_path + "/delisted_companies/")
        for file in file_list:
            if os.path.splitext(file)[1] == ".parquet":
                # delisted = pd.read_parquet(self.main_ctx.root_path + "/delisted_companies/" + file, index_col=None)
                delisted = pd.read_parquet(self.main_ctx.root_path + "/delisted_companies/" + file)
                if delisted.empty == True:
                    continue
                # drop index column
                # delisted = delisted.drop(delisted.columns[0], axis=1)
                delisted = delisted.reset_index(drop=True)
                delisted = delisted[((delisted['exchange'] == 'NASDAQ') | (delisted['exchange'] == 'NYSE'))]
                delisted.rename(columns={'exchange':'exchangeShortName'}, inplace=True)
                delisted = delisted.drop(['companyName'], axis=1)
                all_symbol = pd.concat([all_symbol, delisted])
        all_symbol.to_parquet('./allsymbol.parquet')
        all_symbol = all_symbol.drop_duplicates('symbol', keep='first')
        all_symbol = all_symbol.reset_index(drop=True)
        self.symbol_list = all_symbol["symbol"]

        all_symbol["delistedDate"] = pd.to_datetime(all_symbol["delistedDate"])
        recent_date = all_symbol["delistedDate"].max()
        recent_date -= relativedelta(months=1)
        query = '(delistedDate >= "{}") or (delistedDate == "NaT") or (delistedDate == "None")'.format(recent_date)
        current_symbol = all_symbol.query(query)
        current_symbol.to_parquet('./current_list.parquet')
        current_symbol = current_symbol.drop_duplicates('symbol', keep='first')
        current_symbol = current_symbol.reset_index(drop=True)
        self.current_list = current_symbol["symbol"]

        # logging.info("in set_symbol() list=")
        logging.info(self.symbol_list)

    def get_fmp(self, api_url):
        # multiprocessing 할 예정이기에 로거를 다시 세팅해야 함
        self.main_ctx.set_multi_logger()

        self.main_ctx.create_dir(self.main_ctx.root_path)
        # for i in range(len(api_list)):
        need_symbol = True if api_url.find(self.ex_symbol) != -1 else False
        # SYMBOL 이 없는 건, SYMBOL을 만들기 위한 file도 만들어지기 전이기 때문에 두번 돌려서 SYMBOL 안쓰는 것부터 만듦
        if (need_symbol == True) and (self.symbol_list.empty == True):
            return
        is_v4 = True if api_url.split('/')[2] == "v4" else False
        # Code에 박아 넣은 값인 8은 url의 앞부분인 /api/v4/ 의 길이. v3와 v4 코드 통합을 위해 박아넣음
        main_url = api_url.split('?')[0][8:]
        extra_url = "" if api_url.find("?") == -1 else api_url.split('?')[1]
        if need_symbol == True:
            if is_v4 == True:
                extra_url = "" if len(extra_url) == 10 else extra_url[12:]
            else:
                main_url = main_url[:-5]
        # limit 제거
        extra_url = re.sub('[&]{0,1}limit=[0-9]{2,3}[&]{0,1}', "", extra_url)
        if extra_url != "":
            extra_url = extra_url + "&"
            logging.info("{}\n\textra_url : {}".format(api_url, extra_url))
        logging.info("\n\t{}\n\tmain_url : {} / extra_url : {} "
                     "/ need_symbol : {} / is_v4 : {}".format(api_url, main_url, extra_url, need_symbol, is_v4))
        self.get_fmp_data_preprocessing(main_url, extra_url, need_symbol, is_v4)

    def get_api_list(self):
        api_df = pd.read_csv(self.target_api_path, header=0, usecols=["URL"])
        api_df = api_df.dropna()
        api_list = api_df.values.tolist()
        for i in range(len(api_list)):
            api_list[i] = str(api_list[i]).replace(" ", "")
            # https 부터 적은 경우에 대한 처리
            if str(api_list[i]).find(self.fmp_url) != -1:
                api_list[i] = str(api_list[i])[2: str(api_list[i]).find("apikey") - 1].replace(self.fmp_url, "")
            # 여러줄이 들어간 경우, 가장 앞에 써진 url 만을 돌린다. \n을 찾아서 뒤를 전부 지워주는 작업
            elif str(api_list[i]).find('\\') != -1:
                api_list[i] = str(api_list[i])[2: str(api_list[i]).find('\\')]
            else:
                api_list[i] = str(api_list[i])[2:-2]
        return api_list

    @staticmethod
    def remove_files(path, only_parquet=False):
        for file in os.listdir(path):
            if only_parquet is True and not file.endswith(".parquet"):
                continue
            else:
                os.remove(os.path.join(path, file))

    def remove_current_list_files(self, base_path, check_target=True):
        logging.info("[Check Remove Files] Path : " + str(base_path))
        today = dateutil.utils.today()
        for symbol in self.current_list:
            path = base_path + "/" + str(symbol) + ".parquet"
            if os.path.isfile(path):
                if check_target is True:
                    # row = pd.read_parquet(path, nrows=1)
                    row = pd.read_parquet(path)
                    if row["date"].empty is True:
                        os.remove(path)
                        continue
                    update_date = datetime.datetime.strptime(row["date"].max(), "%Y-%m-%d")
                    if (today - update_date) < datetime.timedelta(days=75):
                        continue
                os.remove(path)

    @staticmethod
    def remove_current_year(base_path):
        # 우선은 사용되지 않는 함수
        today = dateutil.utils.today()
        year = today.strftime("%Y")
        if os.path.isfile(base_path + str(year) + ".parquet"):
            os.remove(base_path + str(year) + ".parquet")

    @staticmethod
    def xxx_remove_current_month(base_path):
        # 우선은 사용되지 않는 함수
        today = dateutil.utils.today()
        year = today.strftime("%Y")
        month = today.strftime("%m")
        if os.path.isfile(base_path + str(year) + "_" + str(month) + ".parquet"):
            os.remove(base_path + str(year) + "_" + str(month) + ".parquet")
        else:
            if not os.path.isfile(base_path + str(year) + "_" + str(month) + ".parquetx"):
                if os.path.isfile(base_path + str(year) + "_" + str(int(month)-1) + ".parquet"):
                    os.remove(base_path + str(year) + "_" + str(int(month)-1) + ".parquet")

    @staticmethod
    def skip_remove_check():
        today = datetime.datetime.today()
        if os.path.isfile("./config/update_date.txt"):
            fd = open("./config/update_date.txt", "r")
            update_date = fd.readline()
            fd.close()
            update_date = datetime.datetime.strptime(update_date, "%Y-%m-%d")
            # today = datetime.datetime.strptime(today, "%Y-%m-%d")
            if (today - update_date) < datetime.timedelta(days=1):
                logging.info("Skip Remove Files")
                return True
        return False

    def remove_first_loop(self):
        if os.path.isfile("./allsymbol.parquet"):
            os.remove("./allsymbol.parquet")
        if os.path.isfile("./current_list.parquet"):
            os.remove("./current_list.parquet")
        self.remove_files("./data/delisted_companies")
        self.remove_files("./data/stock_list")
        self.remove_files("./data/symbol_available_indexes")

        self.remove_current_year("./data/earning_calendar/earning_calendar_")

    def remove_second_loop(self):
        self.remove_current_list_files("./data/income_statement")
        self.remove_current_list_files("./data/balance_sheet_statement")
        self.remove_current_list_files("./data/cash_flow_statement")

        self.remove_current_list_files("./data/key_metrics")
        self.remove_current_list_files("./data/financial_growth")

        for symbol in self.current_list:
            self.remove_current_year("./data/historical_price_full/" + str(symbol) + "_")

        # 얘가 문제
        self.remove_current_list_files("./data/historical_daily_discounted_cash_flow")

        # 얘는 매번 불러 올 필요가 있는가?
        # self.remove_current_list_files("./data/profile", False)
        self.remove_current_list_files("./data/historical_market_capitalization", False)

        write_fd = open("./config/update_date.txt", "w")
        today = datetime.date.today()
        write_fd.write(str(today))
        write_fd.close()

    def get_new(self):
        api_list = self.get_api_list()
        if self.skip_remove_check() is False:
            self.remove_first_loop()
        # self.get_fmp(api_list)
        install_mp_handler()
        with Pool(processes=multiprocessing.cpu_count(), initializer=install_mp_handler()) as pool:
            pool.map(self.get_fmp, api_list)
        self.set_symbol()
        if self.skip_remove_check() is False:
            self.remove_second_loop()
        install_mp_handler()
        with Pool(processes=multiprocessing.cpu_count(), initializer=install_mp_handler()) as pool:
            pool.map(self.get_fmp, api_list)
        # self.get_fmp(api_list)
