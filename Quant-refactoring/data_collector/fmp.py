from data_collector.fmp_api import FMPAPI
from data_collector.fmp_fetch_worker import fetch_fmp

import datetime
import dateutil.utils
import logging
import os
import pandas as pd
from dateutil.relativedelta import relativedelta

class FMP:
    def __init__(self, main_ctx):
        self.main_ctx = main_ctx
        self.symbol_list = pd.DataFrame()
        self.current_list = pd.DataFrame()

        self.logger = self.main_ctx.get_logger('fmp')
    
    def __get_api_list(self):
        """
        FMP Api 리스트를 만들어주는는 함수
        """
        url_df = pd.read_csv(self.main_ctx.target_api_list, header=0, usecols=["URL"])
        url_df = url_df.dropna()
        url_list = url_df['URL'].tolist()
        api_list = [FMPAPI(self.main_ctx, url) for url in url_list]

        return api_list

    def __fetch_ticker_list(self, api_list):
        """
        ticker list와 관련된 api로부터 데이터를 받아와주는 함수
        """
        self.logger.info('fetching ticker list start (stock_list, delisted_companies)')

        stock_list_api = [api for api in api_list if api.converted_category == 'stock_list']
        if len(stock_list_api) == 0:
            self.logger.error('stock list는 받아온다는 전제')
            raise Exception('stock list는 받아온다는 전제')
        stock_list_api = stock_list_api[0]

        delisted_companies_api = [api for api in api_list if api.converted_category == 'delisted_companies']
        if len(delisted_companies_api) == 0:
            self.logger.error('delisted companies는 받아온다는 전제')
            raise Exception('delisted companies는 받아온다는 전제')
        delisted_companies_api = delisted_companies_api[0]

        self.logger.info('fetching ticker list done')

        return fetch_fmp(self.main_ctx, [stock_list_api, delisted_companies_api])

    def __set_symbol(self):
        """
        fmp api 로 얻어온 stock_list 와 delisted companies 에서 exchange가 NASDAQ, NYSE인 symbol들의 list 를 만드는 함수
        """
        self.logger.info('set symbol list start')
        path = self.main_ctx.root_path + "/stock_list/stock_list.csv"
        if os.path.isfile(path):
            symbol_list = pd.read_csv(path)
        else:
            self.logge.error(f'file({path}) is not existed')
            return

        # stock_list에서 type "stock", exchange "NASDQA", "NYSE" 만 가져오기 이를 filtered_symbol에 저장
        # read_parquet를 index를 가져오지 않으므로, symbol_list.drop(symbol_list.columns[0], axis=1) 를 하면 안 됨
        filtered_symbol = symbol_list[(symbol_list['type'] == "stock")
                                      & ((symbol_list['exchangeShortName'] == 'NASDAQ')
                                         | (symbol_list['exchangeShortName'] == 'NYSE'))]
        filtered_symbol = filtered_symbol.dropna(subset=['symbol'])
        filtered_symbol = filtered_symbol.reset_index(drop=True)
        filtered_symbol = filtered_symbol.drop(['price', 'exchange', 'name'], axis=1)        
        all_symbol = filtered_symbol
        # target_stock_symbol 과 delisted stock symbol 합쳐 필요한 symbol list 완성
        file_list = os.listdir(self.main_ctx.root_path + "/delisted_companies/")
        for file in file_list:
            if os.path.splitext(file)[1] == ".csv":
                # delisted = pd.read_parquet(self.main_ctx.root_path + "/delisted_companies/" + file, index_col=None)
                delisted = pd.read_csv(self.main_ctx.root_path + "/delisted_companies/" + file)
                if delisted.empty == True:
                    continue
                delisted = delisted[((delisted['exchange'] == 'NASDAQ') | (delisted['exchange'] == 'NYSE'))]
                delisted = delisted.dropna(subset=['symbol'])
                delisted = delisted.reset_index(drop=True)
                delisted.rename(columns={'exchange':'exchangeShortName'}, inplace=True)
                delisted = delisted.drop(['companyName'], axis=1)
                all_symbol = pd.concat([all_symbol, delisted])
        # all_symbol.to_parquet('./allsymbol.parquet')
        all_symbol.to_csv('./allsymbol.csv', index=False)
        all_symbol = all_symbol.drop_duplicates('symbol', keep='first')
        all_symbol = all_symbol.reset_index(drop=True)
        self.symbol_list = all_symbol["symbol"].to_list()

        all_symbol["delistedDate"] = pd.to_datetime(all_symbol["delistedDate"])
        recent_date = all_symbol["delistedDate"].max()
        recent_date -= relativedelta(months=1)  ############################### 의미 잘 모르겠음
        query = '(delistedDate >= "{}") or (delistedDate == "NaT") or (delistedDate == "None")'.format(recent_date)
        current_symbol = all_symbol.query(query)
        # current_symbol.to_parquet('./current_list.parquet')
        current_symbol.to_csv('./current_list.csv', index=False)
        current_symbol = current_symbol.drop_duplicates('symbol', keep='first')
        current_symbol = current_symbol.reset_index(drop=True)
        self.current_list = current_symbol["symbol"].to_list()

        self.logger.info("in set_symbol() lit = " + str(self.symbol_list))
        self.logger.info('set symbol list done')

    def __fetch_data(self, api_list):        
        """
        나머지 api로부터 데이터를 받아와주는 함수
        """
        self.logger.info('fetching the rest start')

        rest_api_list = [api for api in api_list if api.converted_category not in ['stock_list', 'delisted_companies']]
        for api in rest_api_list:
            if api.need_symbol:
                api.symbol_list = self.symbol_list

        self.logger.info('fetching the rest done')

        return fetch_fmp(self.main_ctx, rest_api_list)

    @staticmethod
    def remove_files(path, only_csv=True):
        if os.path.isdir(path) is False:
            return
        for file in os.listdir(path):
            if only_csv is True and not file.endswith(".csv"):
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
            path = base_path + "/" + str(symbol) + ".csv"
            if os.path.isfile(path):
                if check_target is True:
                    # 현재는 한 줄만 읽어오는 함수를 찾지 못해 전체를 읽고 있음.
                    # pd.read_parquet(path, nrows=1)와 같은 옵션은 없는 것으로 보임
                    row = pd.read_csv(path)
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
        if os.path.isfile(base_path + str(year) + ".csv"):
            os.remove(base_path + str(year) + ".csv")
        if os.path.isfile(base_path + str(year) + ".csvx"):
            os.remove(base_path + str(year) + ".csvx")

    def skip_remove_check(self):
        today = datetime.datetime.today()
        if os.path.isfile("./config/update_date.txt"):
            fd = open("./config/update_date.txt", "r")
            update_date = fd.readline()
            fd.close()
            update_date = datetime.datetime.strptime(update_date, "%Y-%m-%d")
            # today = datetime.datetime.strptime(today, "%Y-%m-%d")
            if (today - update_date) < datetime.timedelta(days=1):
                self.logger.info('Skip Remove Files')
                return True
        return False

    @staticmethod
    def validation_check():
        """
        아래와 같은 에러 메시지가 발생하는 경우가 존재
        1) Limit Reach . Please upgrade your plan or visit our documentation for more details
           at https://site.financialmodelingprep.com/
        2) Error Message
        파일의 내용을 확인해보고 위와 같은 메시지를 파일에 적어놓은 경우 해당 파일을 삭제
        :return: bool 삭제해야 할 파일이 존재하는지 여부
        """
        basepath = 'E:\qt\data'
        flag = True
        del_count = 0
        pass_count = 0
        for dir_name in os.listdir(basepath):
            if os.path.isdir(os.path.join(basepath, dir_name)):
                cur_path = os.path.join(basepath, dir_name)
                par_list = [file for file in os.listdir(cur_path) if file.endswith('csv')]
                for p in par_list:
                    df = pd.read_csv(os.path.join(cur_path, p))
                    # df.str.contains('Limit').any().any()
                    if df.filter(regex='Limit').empty is False or df.filter(regex='Error').empty is False:
                        logging.debug(os.path.join(cur_path, p))
                        os.remove(os.path.join(cur_path, p))
                        del_count += 1
                        flag = False
                    else:
                        pass_count += 1
                logging.info("[ {} ] Delete file count : {} / Total file count {} ".format(cur_path, del_count,
                                                                                           del_count + pass_count))
                del_count = 0
                pass_count = 0
        return flag

    def remove_first_loop(self):
        """
        symbol list를 현재 기준으로 다시 만들고, 지워야하기 때문에 먼저 리스트에 영향을 끼칠만한 내용부터 지움
        remove_first (symbol list 관련 삭제) -> get_fmp (symbol list 관련 내용들을 다시 받아옴
        -> remove_second (새로 받은 symbol list 기준으로 삭제) -> get_fmp (second로 인해 지워진 data 다시 받아옴)
        """
        if os.path.isfile("./allsymbol.csv"):
            os.remove("./allsymbol.csv")
        if os.path.isfile("./current_list.csv"):
            os.remove("./current_list.csv")
        self.remove_files(self.main_ctx.root_path+"/delisted_companies")
        self.remove_files(self.main_ctx.root_path+"/stock_list")

    def remove_second_loop(self):
        """remove_first_loop 설명 참조"""
        self.remove_files(self.main_ctx.root_path+"/symbol_available_indexes")
        self.remove_current_year(self.main_ctx.root_path+"/earning_calendar/earning_calendar_")

        self.remove_current_list_files(self.main_ctx.root_path+"/income_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/balance_sheet_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/cash_flow_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/key_metrics")
        self.remove_current_list_files(self.main_ctx.root_path+"/financial_growth")

        for symbol in self.current_list:
            self.remove_current_year(self.main_ctx.root_path+"/historical_price_full/" + str(symbol) + "_")

        # FIXME 가장 오래 걸리는 함수 Top2 이기에 다른 방식을 고민해보기
        self.remove_current_list_files(self.main_ctx.root_path+"/historical_daily_discounted_cash_flow")
        self.remove_current_list_files(self.main_ctx.root_path+"/historical_market_capitalization", False)

        # profile은 굳이 update 하지 않기로 결정함
        self.remove_current_list_files(self.main_ctx.root_path+"/profile", False)

    def collect(self):
        """
        symbol list가 없거나(전체 새로 download) 갱신이 필요한 경우, symbol list 관련 api들을 먼저 call 해주어야 하는데,
        이렇게 되면 api가 추가되고 삭제 될 때마다 매번 코드를 수정해주어야 하며, 분류 작업이 귀찮고 get_fmp 함수 내부가 지저분해짐
        새로 만든 symbol list가 없는 경우, get_fmp에서 api를 call하지 않고 return만 해주도록 코드 작성
        여기서는 get_fmp만 두 번 돌려주면 됨.
        update가 모두 완료되고 나면 "./config/update_date.txt"에 현재 날짜를 기록해 하루 이내에 다시 update한 경우 skip (실수 방지)
        """
        
        api_list = self.__get_api_list() # query_parmas_str -> condition_str

        # if self.skip_remove_check() is False:
        #     self.remove_first_loop()

        self.__fetch_ticker_list(api_list)
        self.__set_symbol() # symbol에 etf symbol 추가하는거 빼먹음
        print("after set_symbol : {}".format(self.symbol_list))

        # if self.skip_remove_check() is False:
        #     self.remove_second_loop()

        self.__fetch_data(api_list)

        write_fd = open("./config/update_date.txt", "w")
        today = datetime.date.today()
        write_fd.write(str(today))
        write_fd.close()

        if self.validation_check() is False:
            logging.critical("Validation Check False!! Please run the program again after a few minutes!!")
            exit()

