import datetime
import dateutil.utils
import json
import logging
import multiprocessing
import os
import re
import requests
import time
import urllib
import urllib.error
import urllib.request

import pandas as pd

from dateutil.relativedelta import relativedelta
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler
from time import sleep


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

    def return_fmp(self, logger, run_multi, res):
        if run_multi is True:
            logger.handlers.clear()
            return res
        else:
            self.main_ctx.set_default_logger()
            return res

    def get_fmp_data_loop(self, fmp_info, run_multi=True):
        """get_fmp_data를 parallel하게 처리하기 위해 만든 함수"""
        # multiprocessing 으로 처리되므로 로거를 다시 세팅해야 함
        if run_multi is True:
            logger = self.main_ctx.get_multi_logger()
        else:
            logger = logging.getLogger()
        # 결제 PLAN에 관계없이 parallel로 수행하기 때문에 control 할 수 없음 (필요시 다른 방안 모색)
        # start = time.time()
        # end = time.time()
        # remain_sec = (0.2 - (end - start))
        # if remain_sec > 0:
        #     sleep(remain_sec)
        # start = time.time()
        # TODO pool map 하기 위해서 fmp_info에 list로 넣고 처리한 부분인데 코드가 예쁘지가 않다..
        while True:
            path, elem, file_postfix, api_url = fmp_info
            try:
                logger.info('Creating File "{}/{}.parquet" <- "{}"'.format(path, elem + file_postfix, api_url))
                # json_data = pd.read_json(api_url)
                url_data = requests.get(api_url)
            except ValueError:
                logger.debug("No Data. Or Different Data Type")
                return self.return_fmp(logger, run_multi, False)
            except urllib.error.HTTPError:
                logger.warning("HTTP Error 400, API_URL : ", api_url)
                return self.return_fmp(logger, run_multi, False)
            # 읽어왔는데 비어 있을 수 있음. ValueError와 다름.
            # ValueError는 Format이 안맞는 경우고, 이 경우는 page=50 과 같은 extra_url 처리 때문
            json_text = url_data.text
            if "Limit Reach" in json_text:
                logger.error("Limit Reach. Please upgrade your plan or visit our documentation")
                sleep(1)
                continue
            try:
                json_data = json.loads(json_text)
            except json.decoder.JSONDecodeError:
                logger.error("json.decoder.JSONDecodeError")
                return self.return_fmp(logger, run_multi, False)
            if json_data == [] or json_data == {}:
                logger.info("No Data in URL")
                # 비어있는 표시를 해주기 위해 parquet 뒤에 x를 붙인 file만 만들고 fd close
                f = open(path + "/{}.parquetx".format(elem + file_postfix), 'w')
                f.close()
                logger.handlers.clear()
                return self.return_fmp(logger, run_multi, False)
            json_data = self.flatten_json(json_data, expand_all=True)
            # dcf 값에 대한 별도 예외처리 로직
            if 'dcf' in json_data.columns:
                json_data['dcf'] = json_data['dcf'].astype(float)
            # json_data.to_csv(path + "/{}.csv".format(elem + file_postfix), na_rep='NaN')
            json_data.to_parquet(path + "/{}.parquet".format(elem + file_postfix))
            if json_data.empty == True:
                logger.info("No Data in CSV")
                logger.handlers.clear()
                return self.return_fmp(logger, run_multi, False)
            return self.return_fmp(logger, run_multi, True)

    def get_fmp_data(self, main_url, extra_url, need_symbol, is_v4, file_postfix=""):
        """
        순차적으로 넣고자 하는 값을 url에 포함시켜서 돌려줌
        :param : main_url(url의 main 값), extra_url(뒤에 나머지),
                need_symbol(url에 symbol 값이 들어가지는에 대한 boolean. v3는 불필요. v4 주소 처리때문에 필요)
                v4_flag(v4 url 형식을 사용할지에 대한 flag)
                file_postfix(parquet 파일 뒤에 붙는 다른 구분자를 넣고 싶은 경우 사용 예) AAPL_2022_1.csv)
        TODO data_list를 완전히 없앤 이유는 나중에는 need_symbol을 data_type flag로 바꿔서 필요한 리스트를
             이 함수 내에서 알아서 넣는 것이 더 효율적이기에 list 자체는 완전히 받지 않을 예정  예) SYMBOL이 글로벌로 바뀜
        :return : True / False : api Call 한 곳에서 더 불러야 할지 경정하기 위한 용도
        example 1 : /api/v3/discounted-cash-flow/AAPL?apikey=*** 인 경우,
                    get_fmp_data_by_list("dicounted-cash-flow", "", True, False)
                    ./data 폴더 아래에 discounted-cash-flow 폴더를 만들고, list element별 parquet을 만듦
        example 2 : /api/v3/income-statement-as-reported/AAPL?period=quarter&limit=50&apikey=*** 인 경우,
                   get_fmp_data_by_list("income-statement-as-reported",
                                        "period=quarter&limit=50&", symbol_list, True, False)
        """
        # [RULE] 모든 Folder들은 Path가 ./data/* 로 시작해야 함. ./data나 ./data/*/* 와 같은 path는 가질 수 없음
        path = self.main_ctx.root_path + "/" + main_url.replace("/", "-").replace("-", "_")
        cre_flag = self.main_ctx.create_dir(path)

        if need_symbol == False:
            data_list = [path[path.rfind("/") + 1:]]
        else:
            # data_list = self.symbol_list.head(5) # Only For Debug!
            data_list = self.symbol_list
            if main_url == "historical-price-full":
                logging.info("add ETF symbols to get historical-price-full data")
                data_list = data_list.append(pd.Series(["SPY", "IVV", "VTI", "VOO", "QQQ", "VEA", "IEFA"]))
            data_list = data_list.dropna()

        fmp_info_list = []

        for elem in data_list:
            # TODO url_data = "" 와 같은 줄이 필요할 듯? except 후 continue로 들어갈 때 이전 값이 들어있음. 초기화 필요?
            api_url = None
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
                fmp_info_list.append([path, elem, file_postfix, api_url])
            else:
                if cre_flag == True:
                    # 새로 만드는 경우, 이미 csv가 있다는 건 stock list와 delisted list에 중복 값이 있는 상황 (Duplicate)
                    # 리스트에 중복값이 왜 들어가게 되었는지 반드시 확인이 필요함. (가정이 깨짐)
                    logging.error('Already Exist "{}/{}.parquet"'.format(path, elem + file_postfix))
                else:
                    logging.info('Already Exist File "{}/{}.parquet"'.format(path, elem + file_postfix))

        if need_symbol is False:
            # symbol list가 들어가지 않는 경우 data list는 단 하나만 존재함을 가정하고 있기에 return 해주는 거라 assert 체크 필요
            # (정정) symbol list가 들어가더라도 모두 다 존재해서 fmp_info_list가 1개 혹은 그 이하 일 수 있음
            # assert만 체크하자
            assert len(fmp_info_list) <= 1, "DATA List 0 or 1 != len(fmp_info_list):{}".format(len(fmp_info_list))

        if len(fmp_info_list) == 1:
            return self.get_fmp_data_loop(fmp_info_list[0], False)
        elif len(fmp_info_list) > 1:
            with Pool(processes=multiprocessing.cpu_count(), initializer=install_mp_handler()) as pool:
                pool.map(self.get_fmp_data_loop, fmp_info_list)

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
        """fmp api 로 얻어온 stock_list 와 delisted companies 에서 exchange가 NASDAQ, NYSE인 symbol들의 list 를 만드는 함수"""
        path = self.main_ctx.root_path + "/stock_list/stock_list.parquet"
        if os.path.isfile(path) == True:
            symbol_list = pd.read_parquet(path)
        else:
            return

        # stock_list에서 type "stock", exchange "NASDQA", "NYSE" 만 가져오기 이를 filtered_symbol에 저장
        # read_parquet를 index를 가져오지 않으므로, symbol_list.drop(symbol_list.columns[0], axis=1) 를 하면 안 됨
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

        logging.info("in set_symbol() lit = " + str(self.symbol_list))

    def get_fmp(self, api_url):
        """preprocesing 함수를 call하기 위해 url 형태를 확인하고, argument를 나누는 작업"""
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
        if os.path.isdir(path) is False:
            return
        for file in os.listdir(path):
            if only_parquet is True and not file.endswith(".parquet"):
                continue
            else:
                os.remove(os.path.join(path, file))

    def remove_current_list_files(self, base_path, check_target=True):
        """
        현재 기준으로 지워야할 파일들을 찾아서 지움
        :param : base_path(지워야할 Folder 경로), check_target(True인 경우 직접 row를 읽고 지울지말지 판단)
        """
        logging.info("[Check Remove Files] Path : " + str(base_path))
        if os.path.isdir(base_path) is False:
            return
        today = dateutil.utils.today()
        for symbol in self.current_list:
            path = base_path + "/" + str(symbol) + ".parquet"
            if os.path.isfile(path):
                if check_target is True:
                    # 현재는 한 줄만 읽어오는 함수를 찾지 못해 전체를 읽고 있음.
                    # pd.read_parquet(path, nrows=1)와 같은 옵션은 없는 것으로 보임
                    row = pd.read_parquet(path)
                    if "date" in row.columns:
                        if row["date"].empty is True:
                            os.remove(path)
                            continue
                    else:
                        os.remove(path)
                        continue
                    update_date = datetime.datetime.strptime(row["date"].max(), "%Y-%m-%d")
                    if (today - update_date) < datetime.timedelta(days=75):
                        continue
                os.remove(path)

    @staticmethod
    def remove_current_year(base_path):
        today = dateutil.utils.today()
        year = today.strftime("%Y")
        if os.path.isfile(base_path + str(year) + ".parquet"):
            os.remove(base_path + str(year) + ".parquet")

    @staticmethod
    def xxx_remove_current_month(base_path):
        """우선은 사용되지 않는 함수. 월별 처리로 다시 바뀌는 경우, xxx를 지우고 사용하면 됨"""
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

    def validation_check(self):
        """
        아래와 같은 에러 메시지가 발생하는 경우가 존재
        1) Limit Reach . Please upgrade your plan or visit our documentation for more details
           at https://site.financialmodelingprep.com/
        2) Error Message
        파일의 내용을 확인해보고 위와 같은 메시지를 파일에 적어놓은 경우 해당 파일을 삭제
        :return: bool 삭제해야 할 파일이 존재하는지 여부
        """

    def remove_first_loop(self):
        """
        symbol list를 현재 기준으로 다시 만들고, 지워야하기 때문에 먼저 리스트에 영향을 끼칠만한 내용부터 지움
        remove_first (symbol list 관련 삭제) -> get_fmp (symbol list 관련 내용들을 다시 받아옴
        -> remove_second (새로 받은 symbol list 기준으로 삭제) -> get_fmp (second로 인해 지워진 data 다시 받아옴)
        """
        if os.path.isfile("./allsymbol.parquet"):
            os.remove("./allsymbol.parquet")
        if os.path.isfile("./current_list.parquet"):
            os.remove("./current_list.parquet")
        self.remove_files("./data/delisted_companies")
        self.remove_files("./data/stock_list")
        self.remove_files("./data/symbol_available_indexes")

        self.remove_current_year("./data/earning_calendar/earning_calendar_")

    def remove_second_loop(self):
        """remove_first_loop 설명 참조"""
        self.remove_current_list_files("./data/income_statement")
        self.remove_current_list_files("./data/balance_sheet_statement")
        self.remove_current_list_files("./data/cash_flow_statement")
        self.remove_current_list_files("./data/key_metrics")
        self.remove_current_list_files("./data/financial_growth")

        for symbol in self.current_list:
            self.remove_current_year("./data/historical_price_full/" + str(symbol) + "_")

        # FIXME 가장 오래 걸리는 함수 Top2 이기에 다른 방식을 고민해보기
        self.remove_current_list_files("./data/historical_daily_discounted_cash_flow")
        self.remove_current_list_files("./data/historical_market_capitalization", False)

        # profile은 굳이 update 하지 않기로 결정함
        # self.remove_current_list_files("./data/profile", False)

    def get_new(self):
        """
        symbol list가 없거나(전체 새로 download) 갱신이 필요한 경우, symbol list 관련 api들을 먼저 call 해주어야 하는데,
        이렇게 되면 api가 추가되고 삭제 될 때마다 매번 코드를 수정해주어야 하며, 분류 작업이 귀찮고 get_fmp 함수 내부가 지저분해짐
        새로 만든 symbol list가 없는 경우, get_fmp에서 api를 call하지 않고 return만 해주도록 코드 작성
        여기서는 get_fmp만 두 번 돌려주면 됨.
        update가 모두 완료되고 나면 "./config/update_date.txt"에 현재 날짜를 기록해 하루 이내에 다시 update한 경우 skip (실수 방지)
        """
        api_list = self.get_api_list()

        if self.skip_remove_check() is False:
            self.remove_first_loop()

        for url in api_list:
            self.get_fmp(url)

        self.set_symbol()

        if self.skip_remove_check() is False:
            self.remove_second_loop()

        for url in api_list:
            self.get_fmp(url)

        write_fd = open("./config/update_date.txt", "w")
        today = datetime.date.today()
        write_fd.write(str(today))
        write_fd.close()

        if self.validation_check() is False:
            logging.critical("Validation Check False!! Please run the program again after a few minutes!!")
            exit()

