import copy
import csv
import datetime
import logging
import multiprocessing
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from collections import defaultdict
from dateutil.relativedelta import relativedelta
from functools import reduce
from g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from functools import partial
from warnings import simplefilter

pd.options.display.width = 30
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
CHUNK_SIZE = 20480


class Backtest:
    def __init__(self, main_ctx, conf, plan_handler, rebalance_period):
        """
        Back test 실행 순서
        Backtest.run() -> start_year ~ recent_date 까지 rebalancing period 만큼 지나가면서,
        date_handler(date table) load -> plan_handler.run() -> eval_handler.set_best_k()
        모든 기간에 대해서 plan과 set_best_k() 마치고나서, self.eval_handler.run(self.price_table) 실행
        eval_handler.run()은  self.cal_price() -> self.cal_earning() -> self.print_report() 순 실행
        cal_price() : 각 period에서 best_k 주식들의 시작 price, rebalance date price를 채워줌.
        cal_earning() : 두 price의 차이를 가지고 best_k 주식들이 얼마나 올랐는지 채워줌(y값)
        print_report() : earning, price를 포함한 RANK, EVAL report 출력.
        """
        self.main_ctx = main_ctx
        self.conf = conf
        self.plan_handler = plan_handler
        self.rebalance_period = rebalance_period

        self.symbol_table = pd.DataFrame()
        self.price_table = pd.DataFrame()
        self.fs_table = pd.DataFrame()
        self.metrics_table = pd.DataFrame()
        self.table_year = main_ctx.start_year
        
        self.eval_report_path = self.create_report("EVAL")
        self.rank_report_path = self.create_report("RANK")
        self.ai_report_path = self.create_report("AI")
        self.avg_report_path = self.create_report("AVG")

        # backtest 시 이용될 디렉토리를 만들고 시작
        self.main_ctx.create_dir(self.conf['ROOT_PATH'] + "/DATE_TABLE")
        self.main_ctx.create_dir(self.conf['ROOT_PATH'] + "/PLANED_DATE_TABLE")

        self.load_bt_table(main_ctx.start_year)
        self.eval_handler = EvaluationHandler(self)
        self.run() 

    def create_report(self, report_type):
        if report_type in self.conf['REPORT_LIST']:
            path = "./reports/" + report_type + "_REPORT_"
            idx = 0
            while True:
                if not os.path.exists(path + str(idx) + ".csv"):
                    path = path + str(idx) + ".csv"
                    logging.info('REPORT PATH: "{}" ...'.format(path))
                    break
                else:
                    idx += 1

            if report_type == "EVAL" or report_type == "RANK":
                with open(path, 'w', newline='') as file:
                    writer = csv.writer(file, delimiter=",")
                    writer.writerow(["COMMON"])
                    writer.writerow(["Report Date", datetime.datetime.now().strftime('%m-%d %H:%M')])
                    writer.writerow(["Rebalance Period", str(self.rebalance_period) + " Month",
                                     "Start Year", self.main_ctx.start_year,
                                     "End Year", self.main_ctx.end_year])
                    writer.writerow(["K", self.plan_handler.k_num])
                    writer.writerow("")

                    if report_type == "EVAL":
                        writer.writerow(["PLAN HANDLER"])
                        writer.writerow(["key", "key_dir", "weight", "diff", "base", "base_dir"])
                        for plan in self.plan_handler.plan_list:
                            dict_writer = csv.DictWriter(file, fieldnames=plan["params"])
                            dict_writer.writerow(plan["params"])
            return path
        else:
            return None

    def data_from_database(self, query):
        """
        데이터베이스로부터 chunk 단위로 테이블을 읽어오고 반환함
        :param query: 데이터베이스에 전송할 쿼리
        :return: 데이터베이스로부터 읽어온 테이블
        """
        logging.info("Query : " + query)
        chunks = pd.read_sql_query(sql=query, con=self.main_ctx.conn, chunksize=CHUNK_SIZE)
        table = pd.DataFrame()
        for df in chunks:
            table = pd.concat([table, df])
        return table

    def load_bt_table(self, year):
        """
        추후에 database에서 가져올 데이터가 많을 걸 대비해서 __init__ 함수에서 세팅하지 않고, 해당 함수에서 세팅토록 함
        일부 필요한 내용한 init하거나 분할해서 가져오려고 한다면 쿼리가 더 복잡해질 수 있기에 따로 빼놓음
        init에서 세팅하지 않은 이유를 코드에 도입. 해당하는 year에 값을 가져오도록 변경
        """
        query = "SELECT * FROM PRICE WHERE date BETWEEN '" \
                + str(datetime.datetime(self.main_ctx.start_year, 1, 1)) + "'" \
                + " AND '" + str(datetime.datetime(self.main_ctx.end_year, 12, 31)) + "'"
        
        if self.conf['STORAGE_TYPE'] == "DB":
            self.symbol_table = self.data_from_database("SELECT * FROM symbol_list")
            self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
            self.price_table = self.data_from_database(query)
            self.fs_table = self.data_from_database("SELECT * FROM financial_statement")
            self.metrics_table = self.data_from_database("SELECT * FROM METRICS")

        if self.conf['STORAGE_TYPE'] == "PARQUET":
            # self.symbol_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/symbol_list.parquet")
            self.symbol_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/symbol_list.csv")
            self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
            # self.price_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/price.parquet")
            self.price_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/price.csv")
            self.price_table['date'] = pd.to_datetime(self.price_table['date'])

            self.fs_table = pd.DataFrame()
            # self.fs_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/financial_statement.parquet")    
            for year in range(self.main_ctx.start_year-3, self.main_ctx.start_year+1):
                # tmp_fs = pd.read_parquet(self.main_ctx.root_path + "/VIEW/financial_statement_"
                #                          + str(year) + ".parquet")
                tmp_fs = pd.read_csv(self.main_ctx.root_path + "/VIEW/financial_statement_" + str(year) + ".csv",
                                     parse_dates=['fillingDate_x', 'acceptedDate_x'],
                                     dtype={'reportedCurrency_x': str, 'period_x': str,
                                            'link_x': str, 'finalLink_x': str})
                self.fs_table = pd.concat([tmp_fs, self.fs_table])
            del tmp_fs
            self.fs_table['date'] = pd.to_datetime(self.fs_table['date'])
            self.fs_table['fillingDate'] = pd.to_datetime(self.fs_table['fillingDate'])

            self.metrics_table = pd.DataFrame()
            # self.metrics_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/metrics.parquet")
            for year in range(self.main_ctx.start_year-3, self.main_ctx.start_year+1):
                # tmp_metrics = pd.read_parquet(self.main_ctx.root_path + "/VIEW/metrics_" + str(year) + ".parquet")
                tmp_metrics = pd.read_csv(self.main_ctx.root_path + "/VIEW/metrics_" + str(year) + ".csv",
                                          dtype={'period_x': str, 'period_y': str})
                self.metrics_table = pd.concat([tmp_metrics, self.metrics_table])
            del tmp_metrics
            self.metrics_table['date'] = pd.to_datetime(self.metrics_table['date'])

    def reload_bt_table(self, year):
        logging.info("reload_bt_table, year : {}".format(year))
        self.fs_table = pd.DataFrame()
        # self.fs_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/financial_statement.parquet")
        for y in range(year-3, year+1):
            # tmp_fs = pd.read_parquet(self.main_ctx.root_path + "/VIEW/financial_statement_"
            #                          + str(y) + ".parquet")
            tmp_fs = pd.read_csv(self.main_ctx.root_path + "/VIEW/financial_statement_" + str(y) + ".csv",
                                 parse_dates = ['fillingDate_x', 'acceptedDate_x'],
                                 dtype = {'reportedCurrency_x': str, 'period_x': str,
                                          'link_x': str, 'finalLink_x': str})
            self.fs_table = pd.concat([tmp_fs, self.fs_table])
            del tmp_fs
        self.fs_table['date'] = pd.to_datetime(self.fs_table['date'])
        self.fs_table['fillingDate'] = pd.to_datetime(self.fs_table['fillingDate'])

        self.metrics_table = pd.DataFrame()
        # self.metrics_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/metrics.parquet")
        for y in range(year-3, year+1):
            # tmp_metrics = pd.read_parquet(self.main_ctx.root_path + "/VIEW/metrics_" + str(y) + ".parquet")
            tmp_metrics = pd.read_csv(self.main_ctx.root_path + "/VIEW/metrics_" + str(y) + ".csv")
            self.metrics_table = pd.concat([tmp_metrics, self.metrics_table])
        del tmp_metrics
        self.metrics_table['date'] = pd.to_datetime(self.metrics_table['date'])

    def get_trade_date(self, pdate):
        """개장일이 아닐 수도 있기에 보정해주는 함수"""
        # pdate =  pdate.date()
        post_date = pdate - relativedelta(days=10)
        res = self.price_table.query("date >= @post_date and date <= @pdate")
        if res.empty:
            return None
        else:
            return res.iloc[0].date

    def run(self):
        """
        main에서 전달받은 plan list에 따른 backtest를 진행하는 함수로 크게 2개의 파트로 구분되어 있음
        [1] date 별로 plan list에 따라 plan_handler.date_handler.score를 계산해넣고,
            상위권 symbol을 self.best_k에 추가
        [2] best_k 에서 가져와서 MDD나 샤프지수와 같은 전략 전체에 필요한 계산값들을 계산해서 채워 넣음
        전략 별로 하나의 backtest class, plan_handler (1:1 mapping)가 만들어지며 생성의 주체는 main
        date_handler는 다수 만들어지며 생성 주체는 backtest이며 생성 후
        backtest에서 본인에게 mapping되어 있는 plan_handler에게 달아줌.
        """
        date = datetime.datetime(self.main_ctx.start_year, self.conf['START_MONTH'], self.conf['START_DATE'])
        recent_date = self.price_table["date"].max()
        end_date = datetime.datetime(self.main_ctx.end_year, 12, 31)
        if end_date > recent_date:
            end_date = recent_date
        cur_table_year = self.main_ctx.start_year
        # START OF WHILE #
        while True:
            tdate = self.get_trade_date(date)
            if tdate is None:
                logging.info("tradable date is None. break")
                break            
            logging.info("Backtest Run : " + str(tdate.strftime("%Y-%m-%d")))
            # Date에 맞게 DateHandler ( symbol, price, fs, metrics table ) 만들고 plan run
            self.plan_handler.date_handler = DateHandler(self, tdate)
            logging.info("complete set date_handler date : {}".format(tdate.strftime("%Y-%m-%d")))
            self.plan_handler.run(self.conf)

            if date != recent_date:
                self.eval_handler.set_best_k(tdate, date+relativedelta(months=self.rebalance_period),
                                            self.plan_handler.date_handler)
            else: # recent_date이면 current_best 뽑기
                self.eval_handler.print_current_best(self.plan_handler.date_handler)
                break
            
            if (date + relativedelta(months=self.rebalance_period)) <= end_date:
                date += relativedelta(months=self.rebalance_period)
                if date.year != cur_table_year:
                    cur_table_year = date.year
                    self.reload_bt_table(cur_table_year)
            else:
                # 마지막 loop 에 도달하면 최근 date 로 한번 돌아서 print 해준 후에 루프를 빠져 나가도록 함
                if self.eval_report_path is not None:
                    date = recent_date
                    if date.year != cur_table_year:
                        cur_table_year = date.year
                        self.reload_bt_table(cur_table_year)
                else:
                    # evaluation report 를 뽑지 않으면 current 추천을 스킵하고 나감
                    break
        # END OF WHILE # 

        logging.info("DateHandler.global_sparse_col : ")        
        for k, v in DateHandler.global_sparse_col.items():
            logging.debug(f"{k} - {v}")
            
        logging.info("START Evaluation")
        self.eval_handler.run(self.price_table)


class PlanHandler:
    def __init__(self, k_num, absolute_score, main_ctx):
        self.plan_list = None
        self.date_handler = None
        self.k_num = k_num
        self.absolute_score = absolute_score
        self.main_ctx = main_ctx

    def run(self, conf):
        """
        main에서 짜여진 전략(function pointer)을 순서대로 호출
        plan_list와 date_handler를 채워주고 불러워줘야 함.
        현재는 plan_list는 main, date_handler는 Backtest 에서 채워줌.
        """
        assert self.plan_list is not None, "Empty Plan List"
        assert self.date_handler is not None, "Empty Date Handler"
        # START : save / read planed dtable from csv
        pdate = self.date_handler.date
        planed_dtable_path = conf['ROOT_PATH']\
                             + "/PLANED_DATE_TABLE/planed_dtable_{}.csv".format(pdate.strftime('%Y-%m-%d'))

        if not os.path.exists(planed_dtable_path):
            logging.info("there is no planed date_table : " + planed_dtable_path)
            logging.info("start to create planed_date_table : " + planed_dtable_path)
            # with Pool(processes=4, initializer=install_mp_handler()) as pool:
            # with Pool(processes=multiprocessing.cpu_count() - 4, initializer=install_mp_handler()) as pool:
            #    df_list = pool.map(self.plan_run, self.plan_list)
            # full_df = pd.concat(df_list, ignore_index=True)
            # full_df = full_df.groupby('symbol', as_index=False).last()
            # TODO multiprocessing 처리 시에 table의 크기가 비이상적으로 커지는 현상이 있어 우선 serial 하게 처리하는 것으로 변경
            #     첫 시도 이후에는 planed_dtalbe_path의 파일을 읽을 것이기 때문에 첫 loop만 느려질 것으로 예상됨
            i = 0
            for plan in self.plan_list:
                logging.debug("[{}/{}] {} processing....".format(i, len(self.plan_list), str(plan["params"]["key"])))
                self.plan_run(plan)
                i += 1
            # full_df = reduce(lambda df1, df2: pd.merge(df1, df2, on='symbol'), df_list)
            # replace reduce + merge -> concat + groupby (for memory usage optimization)

            # self.date_handler.dtable = pd.merge(self.date_handler.dtable, full_df, how='left', on=['symbol'])
            score_col_list = self.date_handler.dtable.columns.str.contains("_score")
            self.date_handler.dtable['score'] = self.date_handler.dtable.loc[:, score_col_list].sum(axis=1)
            self.date_handler.dtable.to_csv(conf['ROOT_PATH'] + "/PLANED_DATE_TABLE/"
                                            + 'planed_dtable_{}.csv'.format(pdate.strftime('%Y-%m-%d')), index=False)
        else:
            logging.info("there is csv file for this date. read planed date table from csv")
            self.date_handler.dtable = pd.read_csv(planed_dtable_path)
        # END :save / read planed dtable from csv
        # logging.debug(self.date_handler.dtable.sort_values(by=['score'], ascending=False)[['symbol', 'score']])

    @staticmethod
    def plan_run(plan):
        return plan["f_name"](plan["params"])

    def single_metric_plan_no_parallel(self, params):
        """
        아래의 single metric_plan과 하는 일은 동일하나 parallel 하게 동작하지 않도록 변경한 것
        """
        if self.absolute_score - params["diff"] * self.k_num < 0:
            logging.warning("Wrong params['diff'] : TOO BIG! SET UNDER " + str(self.absolute_score/self.k_num))
        key = str(params["key"])
        # all feature was preprocessed ( high is good ) in Datehandler
        rank_name = key + '_rank'
        self.date_handler.dtable[rank_name] = self.date_handler.dtable[key+"_sorted"].rank(ascending=False,
                                                                                           method='min',
                                                                                           na_option='bottom')

        score_name = key + '_score'
        self.date_handler.dtable.loc[self.date_handler.dtable[rank_name] <= self.k_num, score_name]\
            = self.absolute_score - ((self.date_handler.dtable.loc[self.date_handler.dtable[rank_name] <= self.k_num,
                                                                   rank_name] - 1) * params["diff"])
        # 나머지 행의 결측값을 0으로 채우기
        self.date_handler.dtable[score_name] = self.date_handler.dtable[score_name].fillna(0)

    def single_metric_plan(self, params):
        """
        single metric(PBR, PER ... )에 따라 plan_handler.date_handler.symbol_list의 score column에 값을 갱신해주는 함수.
        params에 plan의 parameter들이 아래와 같이 들어옴
        params["key"]        : plan에서 사용할 종목의 지표(ex: PER, PBR, 영업이익 ...)
        params["key_dir"]    : 지표가 낮을수록 좋은지(low) 높을수록 좋은지(high)
        params["weight"]     : score update시 weight
        params["diff"]       : 이 지표로 각 종목간 score 차이
        params["base"]       : 특정 threshold 이상/이하의 종목은 score 주지 않음
        params["base_dir"]   : "base"로 준 threshold 이상/이하(</>) 선택
        """
        logger = self.main_ctx.get_multi_logger()
        # logger.debug("[plan] key : {}, key_dir : {}, weight : {}, "
        #             "diff : {}, base : {}, base_dir : {}".format(params["key"], params["key_dir"], params["weight"],
        #                                                          params["diff"], params["base"], params["base_dir"]))
        if self.absolute_score - params["diff"] * self.k_num < 0:
            logger.warning("Wrong params['diff'] : TOO BIG! SET UNDER " + str(self.absolute_score/self.k_num))

        key = str(params["key"])
        # all feature was preprocessed ( high is good ) in Datehandler
        top_k_df = self.date_handler.dtable.sort_values(by=[key+"_sorted"], ascending=False,
                                                        na_position="last")[:self.k_num]
        # logger.debug(top_k_df[['symbol', params["key"]]])
        symbols = top_k_df['symbol']
        del top_k_df
        return_df = self.date_handler.dtable[['symbol']]
        delta = self.absolute_score
        # 경고처리 무시
        pd.set_option('mode.chained_assignment', None)
        local_score_name = key + '_score'
        for sym in symbols:
            return_df.loc[(self.date_handler.dtable.symbol == sym), local_score_name] = params["weight"] * delta
            delta = delta - params["diff"]
        local_rank_name = key+'_rank'
        return_df[local_rank_name] = return_df[local_score_name].rank(method='min', ascending=False)
        return_df[local_rank_name] = return_df[local_rank_name].fillna(-1)
        # .astype(int)
        return_df[local_score_name] = return_df[local_score_name].fillna(0)
        # .astype(int)
        # logger.debug(return_df[[local_score_name, local_rank_name]])
        return return_df

time_periods = [3, 6, 9, 12, 15, 18, 21, 24]

def compute_ewm(feature_values):
    ewm6 = pd.Series(feature_values).ewm(span=6).mean()
    ewm2 = pd.Series(feature_values).ewm(span=2).mean()
    return ewm6[0], ewm2[0]

def compute_column_ewm(fs_metrics, col):
    columns_to_select = [f"prev{t}_{col}" for t in time_periods]
    feature_values_matrix = fs_metrics[columns_to_select].values

    results = [compute_ewm(feature_values) for feature_values in feature_values_matrix]
    ewm6s, ewm2s = zip(*results)

    return col, ewm6s, ewm2s

class DateHandler:
    global_sparse_col = defaultdict(int)

    def __init__(self, backtest, date):
        pd.set_option('mode.chained_assignment', None)
        logging.info("in datehandler date : " + date.strftime("%Y-%m-%d"))
        self.date = date
        # date handler 안에서 table은 하나 (symbol + price + fs + metric )
        self.dtable = None
        self.init_data(backtest)

    def init_data(self, backtest):
        logging.info("START init_data in date handler ")
        
        # 미리 parquet로 저장해둔 DATE handler table을 읽어들임.
        trade_date = backtest.get_trade_date(self.date)
        dtable_path = backtest.conf['ROOT_PATH'] + "/DATE_TABLE/dtable_"\
                                                 + str(trade_date.year) + '_' + str(trade_date.month) + '_'\
                                                 + str(trade_date.day) + '.parquet'
        if not os.path.exists(dtable_path):
            logging.info("there is no date_table : ")
            self.create_dtable(backtest)
        else:
            logging.info("there is parquet file for this date. read date table from parquet.")
            self.dtable = pd.read_parquet(dtable_path)
            # industry to sector.
            self.dtable["sector"] = self.dtable["industry"].map(sector_map)

    # for create_dtable()
    def get_price_for_dtable(self, backtest):
        # db에서 delistedDate null 이  df에서는 NaT로 들어옴.
        query = '(delistedDate >= "{}") or (delistedDate == "NaT") or (delistedDate == "None")'.format(self.date)
        self.dtable = backtest.symbol_table.query(query)
        self.dtable = self.dtable.assign(score=0)
        trade_date = backtest.get_trade_date(self.date)
        price = backtest.price_table.query("date == @trade_date")
        # logging.info("Price Table : \n" + str(price))
        price = price[['symbol', 'date', 'close', 'volume', 'marketCap']]
        price = price.drop_duplicates('symbol', keep='first')

        self.dtable = pd.merge(self.dtable, price, how='left', on='symbol')
        self.dtable['volume_mul_price'] =  self.dtable['close'] * self.dtable['volume']
        
        # 총거래액 적은 주식 제외 (얼마나 버려야할까 ? -> )
        # self.dtable = self.dtable[~self.dtable.isin([np.inf, -np.inf]).any(axis=1)]
        # self.dtable = self.dtable[self.dtable['volume_mul_price'] > 1000000] # TODO threshold
        self.dtable = self.dtable.nlargest(int(len(self.dtable)*0.50), 'volume_mul_price', keep='first')

    # for create_dtable()
    def get_fs_metrics(self, backtest):
        prev = self.date - relativedelta(months=6)
        # self.fs = self.get_date_latest_per_symbol(backtest.fs_table, self.date)
        fs = backtest.fs_table.copy()
        fs = fs[fs.fillingDate <= self.date]
        fs = fs[prev <= fs.fillingDate]
        fs = fs.drop_duplicates('symbol', keep='first')
        metrics = backtest.metrics_table.copy()
        fs_metrics = pd.merge(fs, metrics, how='left', on=['symbol', 'date'])
        
        # marketCap 과의 직접 계산이 필요한 column들을 추가 해 줌
        cap = self.dtable.copy()
        cap = cap[['symbol', 'marketCap']]
        cap.rename(columns={'marketCap': 'cal_marketCap'}, inplace=True)
        fs_metrics = pd.merge(fs_metrics, cap, how='left', on=['symbol'])

        for col in meaning_col_list:
            new_col_name = 'OverMC_' + col
            fs_metrics[new_col_name] = np.where(fs_metrics['cal_marketCap'] > 0,
                                                fs_metrics[col]/fs_metrics['cal_marketCap'], np.nan)
            # fs_metrics[new_col_name] = fs_metrics.apply(\
            #     lambda x : x[col]/x['cal_marketCap'] \
            #     if  (x['cal_marketCap']!=0 and (not pd.isnull(x['cal_marketCap']))) \
            #     else np.nan, axis='columns')
        return fs_metrics

    def create_dtable(self, backtest):
        # 미리 parquet로 저장해둔 DATE handler table이 없어서 새로 만듬.

        # 1) dtable에 최신 price 가져와서 merge. + 총거래액 하위 50% 버리기
        self.get_price_for_dtable(backtest)

        # 2) fs, metrics 불러오기 + marketcap으로 나누어서 모두 상대값으로 변환
        fs_metrics = self.get_fs_metrics(backtest)

        # for prev_n in range(3, 17):
        for prev_n in time_periods:
            prefix_col_name = "prev" + str(prev_n) + "_"
            # prev_m month 전 fs, metrics 와 모든 column 빼서 diff column 만들기
            prev_q = self.date - relativedelta(months=prev_n)
            prev = self.date - relativedelta(months=prev_n+4)
            prev_fs = backtest.fs_table.copy()
            prev_fs = prev_fs[prev_fs.fillingDate <= prev_q]
            prev_fs = prev_fs[prev <= prev_fs.fillingDate]

            prev_fs = prev_fs.drop_duplicates('symbol', keep='first')
            metrics = backtest.metrics_table.copy()
            prev_fs_metrics = pd.merge(prev_fs, metrics, how='left', on=['symbol', 'date'])
            symbols = prev_fs_metrics['symbol']
            prev_fs_metrics = prev_fs_metrics[meaning_col_list]
            prev_fs_metrics = prev_fs_metrics.rename(columns=lambda x: prefix_col_name + x)
            prev_fs_metrics['symbol'] = symbols
            fs_metrics = pd.merge(fs_metrics, prev_fs_metrics, how='left', on=['symbol'])
            # fs_metrics = fs_metrics[~fs_metrics.isin([np.inf, -np.inf]).any(axis=1)]

        # 4,8,12,16개월 변화량
        for prev_n in time_periods:
            prefix_col_name = "prev" + str(prev_n) + "_"            
            # diff column
            for col in meaning_col_list:
                # 변화량
                new_col_name = "diff" + str(prev_n) + "_" + col
                fs_metrics[new_col_name] = np.where(fs_metrics[prefix_col_name+col] > 0,
                                                    (fs_metrics[col] - fs_metrics[prefix_col_name+col])
                                                    / fs_metrics[prefix_col_name+col], np.nan)
                # fs_metrics[new_col_name] = fs_metrics.apply(lambda x : \
                #     (x[col] - x[prefix_col_name+col]) / x[prefix_col_name+col] \
                #     if (x[prefix_col_name +col]!=0 and (not pd.isnull(x[prefix_col_name +col]))) \
                #     else np.nan, \
                #     axis='columns')
        # 이동 평균 -> 지수 이동 평균이 최근 데이터에 가중치 부여해서 추세 변화 빠르게 감지하므로 지수 이동평균으로 대채
        # for col in meaning_col_list:        
        #     new_col_name = "mavg_" + col
        #     for prev_n in [4, 8, 12, 16]:
        #         time_columns = 'prev_' + str(prev_n) +'_' + col
        #     fs_metrics[new_col_name] = fs_metrics[time_columns].mean(axis=1)
        
        # 지수 이동평균 계산
        # for col in meaning_col_list:
        #     new_col_name = "emavg_" + col
        #     alpha = 2 / ( (prev_n/4) + 1)  # n은 이동평균 기간
        #     for prev_n in [4, 8, 12, 16]:
        #         time_columns = 'prev' + str(prev_n) +'_' + col
        #     fs_metrics[new_col_name] = fs_metrics[time_columns].ewm(alpha=alpha, adjust=False).mean()

        func = partial(compute_column_ewm, fs_metrics)
        with Pool(multiprocessing.cpu_count()) as pool:
            results = list(tqdm(pool.imap(func, meaning_col_list), total=len(meaning_col_list)))

        for col, ewm6s, ewm2s in results:
            fs_metrics[f"EWM6_{col}"] = ewm6s
            fs_metrics[f"EWM2_{col}"] = ewm2s
            
        # ARIMA 계산
        # for col in meaning_col_list:
        #     for i, row in fs_metrics.iterrows():
        #         feature_values = np.array([row[f"prev{t}_{col}"] for t in [3, 6, 9, 12, 15, 18]])

        #         # ARIMA 모델 적합
        #         model = ARIMA(feature_values, order=(1,1,1))
        #         results = model.fit()

        #         # 예측값
        #         forecasted_values = results.forecast(steps=2)
        #         print("forecast : {}".format(forecasted_values))
        #         # 잔차
        #         residuals = results.resid
        #         # ARIMA 모델의 계수
        #         ar_coefficients = results.arparams
        #         ma_coefficients = results.maparams
        #         fs_metrics.loc[i, f"ARIMA_ar_{col}"] = ar_coefficients
        #         fs_metrics.loc[i, f"ARIMA_ma_{col}"] = ma_coefficients
        #         # 잔차의 통계적 특성
        #         mean_residual = np.mean(residuals)
        #         std_residual = np.std(residuals)
        #         skewness_residual = pd.Series(residuals).skew()
        #         kurtosis_residual = pd.Series(residuals).kurt()
        #         fs_metrics.loc[i, f"ARIMA_meanresidual_{col}"] = mean_residual
        #         fs_metrics.loc[i, f"ARIMA_stdresidual_{col}"] = std_residual
        #         fs_metrics.loc[i, f"ARIMA_skewnessresidual_{col}"] = skewness_residual
        #         fs_metrics.loc[i, f"ARIMA_kurtosisresidual_{col}"] = kurtosis_residual

        #         # ACF 및 PACF 값
        #         acf_values = acf(residuals, nlags=4)
        #         pacf_values = pacf(residuals, nlags=4, method='ols')
        #         fs_metrics.loc[i, f"ARIMA_acf_{col}"] = acf_values
        #         fs_metrics.loc[i, f"ARIMA_pacf_{col}"] = pacf_values

        #         # 모델의 적합도 통계치
        #         aic = results.aic
        #         bic = results.bic
        #         fs_metrics.loc[i, f"ARIMA_aic_{col}"] = aic
        #         fs_metrics.loc[i, f"ARIMA_bic_{col}"] = bic

        # prev column 제거
        for prev_n in time_periods:
            prefix_col_name = "prev" + str(prev_n) + "_"   
            for col in meaning_col_list:
                fs_metrics = fs_metrics.drop([prefix_col_name+col], axis=1)
        # 절대값 column 제거
        abs_col_list = list(set(meaning_col_list) - set(ratio_col_list))
        for col in abs_col_list:
            fs_metrics = fs_metrics.drop([col], axis=1)                
            
        # 50% 넘게 비어있는 row drop
        logging.info("before fs_metric len : {}".format(len(fs_metrics)))
        fs_metrics['nan_count_per_row'] = fs_metrics.isnull().sum(axis=1)
        filtered_row = fs_metrics['nan_count_per_row'] < int(len(fs_metrics.columns)*0.5)
        fs_metrics = fs_metrics.loc[filtered_row,:]
        logging.info("after fs_metric len : {}".format(len(fs_metrics)))
        
        # "bookValuePerShare", # 시총 / bookValuePerShare = PBR
        # "eps", # (시총/eps = 유사 PER)
        # "netdebt", # (netdebt + 시총) = EV
        # "operatingCashflow", # ev / operatingCashflow = evToOperatingCashFlow
        # "FreeCashflow", # ev / FreeCashflow = evToFreeCashflow
        # "ebitda", #  ev / ebitda = enterpriseValueOverEBITDA
        # "revenues" # ev/revenues =  evToSales

        fs_metrics["adaptiveMC_ev"] = fs_metrics['cal_marketCap'] + fs_metrics["netDebt"]
        for col in cal_ev_col_list:
            new_col_name = 'adaptiveMC_' + col
            fs_metrics[new_col_name] = np.where(fs_metrics[col] > 0,
                                                fs_metrics['adaptiveMC_ev']/fs_metrics[col], np.nan)
            # fs_metrics[new_col_name] = fs_metrics.apply(lambda x : \
            #         x['adaptiveMC_ev']/x[col] \
            #         if (x[col]!=0 and (not pd.isnull(x[col]))) \
            #         else np.nan, \
            #         axis='columns')
        # logging.debug("COL_NAME in fs_table : ")
        # for col in backtest.fs_table.columns:
        #     logging.debug(col)
        # logging.debug("COL_NAME in fs_metrics : ")
        # for col in fs_metrics.columns:
        #     logging.debug(col)

        highlow = pd.read_csv('./sort.csv', header=0)
        for feature in fs_metrics.columns:
            # if ( feature not in ratio_col_list) and (not feature.startswith('Ydiff')) and (not feature.startswith('Qdiff'))\
            #             and (not feature.startswith('OverMC')) and (not feature.startswith('adaptiveMC')):
            #     continue
            feature_sortedvalue_col_name = feature + "_sorted"                
            if str(feature).startswith("diff") or str(feature).startswith("EWM6_") or str(feature).startswith("EWM2_")\
                    or str(feature).startswith("OverMC_") or str(feature).startswith("adaptiveMC_"):
                feature_name = str(feature).split('_')[1]
            else:
                feature_name = feature
            if (feature_name not in meaning_col_list) and (feature_name not in ratio_col_list):
                continue
            # 음수 처리
            f = highlow.query("name == @feature_name")
            if f.empty:
                continue
            else:
                fs_metrics[feature_sortedvalue_col_name] = fs_metrics[feature].copy() # .astype('float32')
                if f.iloc[0].sort == "low":
                    try:
                        feat_max = fs_metrics[feature].max()
                        # 양수는 낮을 수록 좋아지도록 만들고, 음수는 양수의 제일 낮은 값보다 더 안좋게만들고
                        fs_metrics[feature_sortedvalue_col_name] = \
                            [s*(-1) if s >= 0 else (s - feat_max) for s in fs_metrics[feature]]
                        # fs_metrics[feature_sortedvalue_col_name] = fs_metrics[feature_sortedvalue_col_name].astype('float32')
                    except Exception as e:
                        logging.info(str(e))
                        continue
                        
            # # normalization ( 0~20000 ). range is not fixed
            # feature_normal_col_name = feature + "_normal"
            # try:
            #     max_value = fs_metrics[feature_sortedvalue_col_name].max()
            #     min_value = fs_metrics[feature_sortedvalue_col_name].min()
            #     fs_metrics[feature_normal_col_name] \
            #         = (((fs_metrics[feature_sortedvalue_col_name] - min_value) * 20000) / (max_value - min_value))
            #     fs_metrics[feature_normal_col_name] = fs_metrics[feature_normal_col_name].astype(int)
            #     # fs_metrics = fs_metrics.astype({feature_normal_col_name: 'float32'})
            #     # fs_metrics = fs_metrics.astype({feature_sortedvalue_col_name: 'float32'})
            # except Exception as e:
            #     logging.info(str(e))
            #     continue        
            # # fs_metrics = fs_metrics.astype({feature: 'float32'})


        self.dtable = pd.merge(self.dtable, fs_metrics, how='left', on='symbol')
        
        # # 40% 넘게 비어있는 row drop
        # print("before dtable len : ", len(self.dtable))
        # self.dtable['nan_count_per_row'] = self.dtable.isnull().sum(axis=1)
        # filtered_row = self.dtable['nan_count_per_row'] < int(len(self.dtable.columns)*0.4)
        # self.dtable = self.dtable.loc[filtered_row,:]
        # print("after dtable len : ", len(self.dtable))

        # 50% 넘게 비어있는 column 누적
        columns_with_nan_above_threshold = self.dtable.columns[self.dtable.isnull().sum(axis=0)
                                                               >= int(len(self.dtable)*0.5)].tolist()
        for c in columns_with_nan_above_threshold:
            DateHandler.global_sparse_col[c] += 1
        logging.info("DateHandler.global_sparse_col : ")
        for k, v in DateHandler.global_sparse_col.items():
            logging.debug(f"{k} - {v}")
    
        self.dtable["sector"] = self.dtable["industry"].map(sector_map)
        self.dtable = self.dtable.drop_duplicates('symbol', keep='first')
        self.dtable.to_csv(backtest.conf['ROOT_PATH'] + "/DATE_TABLE/"
                + 'dtable_' + str(self.date.year) + '_'
                + str(self.date.month) + '_' + str(self.date.day) + '.csv', index=False)
        self.dtable.to_parquet(backtest.conf['ROOT_PATH'] + "/DATE_TABLE/"
                               + 'dtable_' + str(self.date.year) + '_' + str(self.date.month) + '_' + str(self.date.day)
                               + '.parquet', engine="pyarrow")
        
        logging.info("END create dtable in date handler ")


class EvaluationHandler:
    def __init__(self, backtest):
        self.best_k = []
        self.historical_earning_per_rebalanceday = []
        self.backtest = backtest
        self.member_cnt = self.cal_member_cnt()
        self.accumulated_earning = 0
        self.MDD = 0
        self.sharp = 0
        self.total_asset = backtest.conf['TOTAL_ASSET']

    def cal_member_cnt(self):
        """상위 몇 종목을 구매할 것인가에 대한 계산. 현재는 상위 4개의 주식을 매 period 마다 구매하는 것으로 되어 있음"""
        return self.backtest.conf['MEMBER_CNT']

    def print_current_best(self, scored_dh):
        best_symbol = scored_dh.dtable[scored_dh.dtable.volume > 10000]
        best_symbol = best_symbol.sort_values(by=["score"], axis=0, ascending=False).head(self.member_cnt)
        best_symbol = best_symbol.assign(count=0)
        best_symbol.to_csv('./result.csv')

    def set_best_k(self, date, rebalance_date, scored_dh):
        """plan_handler.date_handler.symbol_list에 score를 보고 best_k에 append 해주는 함수."""
        if self.backtest.eval_report_path is not None:
            best_symbol = scored_dh.dtable[scored_dh.dtable.volume > 10000]
            best_symbol = best_symbol.sort_values(by=["score"], axis=0, ascending=False).head(self.member_cnt * 2)
            best_symbol = best_symbol.assign(count=0)
        else:
            best_symbol = pd.DataFrame()
        reference_group = pd.DataFrame()
        period_earning_rate = 0
        self.best_k.append([date, rebalance_date, best_symbol, reference_group, period_earning_rate])

    def print_ai_data(self, df_for_reg, date, latest):
        
        if date.day > 15 & date.month == 12:
            mmonth = 1;
        elif date.day > 15 & date.month < 12:
            mmonth = date.month+1;
        elif date.day < 15:
            mmonth = date.month
        
        df_for_reg = df_for_reg.drop_duplicates('symbol', keep='first')
        symbols_tmp = df_for_reg['symbol']
        

        period_price_diff_tmp = pd.DataFrame()
        if latest == False:
            period_price_diff_tmp = df_for_reg['period_price_diff']

        sorted_col_list = []
        for col in df_for_reg.columns:
            sorted_col_list.append(col)
        sorted_col_list = list(filter(lambda x: x.endswith('_sorted'), sorted_col_list))

        # ratio_col_list_wydiff = list(map(lambda x: "Ydiff_" + x, meaning_col_list))
        # ratio_col_list_wqdiff = list(map(lambda x: "Qdiff_" + x, meaning_col_list))
        # ratio_col_list_woverMC = list(map(lambda x: "OverMC_" + x, meaning_col_list))
        # ratio_col_list_wadaMC = list(filter(lambda x: x.startswith('adaptiveMC_'), df_for_reg.columns))

        # ratio_col_list_wprev = ratio_col_list + ratio_col_list_wydiff + ratio_col_list_wqdiff + ratio_col_list_woverMC + ratio_col_list_wadaMC

        # df_for_reg = df_for_reg[[ratio_col_list_wprev]]
        # df_for_reg['symbol'] = symbols_tmp
        sector_list = list(df_for_reg['sector'].unique())
        sector_list = [x for x in sector_list if str(x) != 'nan']
        if latest == False:


            df_for_reg['period_price_diff'] = period_price_diff_tmp
            df_for_reg['earning_diff'] \
                = df_for_reg['period_price_diff'] - df_for_reg['period_price_diff'].mean()

            for sec in sector_list:
                print("sec earning _diff")
                print(sec)
                # 'sector' 열이 'it'인 경우에 대해서만 'price_diff' 열 생성
                sec_mask = df_for_reg['sector'] == sec
                sec_mean = df_for_reg.loc[sec_mask, 'period_price_diff'].mean()
                # df_for_reg[sec_mask, 'sec_earning_diff'] = df_for_reg.loc[sec_mask, 'period_price_diff'] - sec_mean
                df_for_reg.loc[sec_mask, 'sec_earning_diff'] = df_for_reg.loc[sec_mask, 'period_price_diff'] - sec_mean
                print(df_for_reg.loc[sec_mask])

            df_for_reg = df_for_reg.dropna(subset=['period_price_diff'])

        # remove outlier
        logging.info("before removing outlier # rows : " + str(df_for_reg.shape[0]))
        logging.info("before removing outlier # columns : " + str(df_for_reg.shape[1]))
        df_for_reg.to_csv('./beforeremoving.csv', index=False)

        # # remove sparse cols
        # df_for_reg = df_for_reg.drop(sparse_col_list, axis=1)
        # # ratio_col_list_wprev - sparse_cols
        # ratio_col_list2 = [x for x in ratio_col_list if x not in sparse_col_list]
        # ratio_col_list_wprev = [x for x in ratio_col_list_wprev if x not in sparse_col_list]
        
        # df_for_reg.replace(-np.inf, np.nan, inplace=True)
        # df_for_reg = df_for_reg.dropna(thresh=int(len(df_for_reg.columns)*0.6))
        # fill nan with min value per col
        def fill_nan_with_min(df):    
            # Select only numeric columns
            numeric_columns = df.select_dtypes(include=['number'])
            # Compute the minimum values for numeric columns
            min_values = numeric_columns.min().to_dict()
            # Fill NaN values in the original dataframe using the computed minimum values
            return df.fillna(min_values)                    
                    
        # filled_df = fill_nan_with_min(df_for_reg)
        # df_for_reg = filled_df
        # Define a function to remove the top 1%(var : high=0.99) and bottom 1%(var : low = 0.01) from a column
        def get_extreme_percentile_indices(column, low, high):
            lower_bound = column.quantile(low)
            upper_bound = column.quantile(high)
            return column[(column < lower_bound) | (column > upper_bound)].index
                
        # remove outliar rows   
        # outlier_rows = set()
        # for col in sorted_col_list:
        #     try:
        #         column_indices = get_extreme_percentile_indices(df_for_reg[col], 0.002, 0.998)
        #         if len(column_indices) < 10:
        #             outlier_rows.update(column_indices)
        #     except Exception as e:
        #         logging.info(str(e))
        #         continue

        # if latest == False:
        #     column_indices = get_extreme_percentile_indices(df_for_reg['period_price_diff'], 0.002, 0.998)
        #     outlier_rows.update(column_indices)

        # df_for_reg = df_for_reg.drop(index=outlier_rows, axis=0)
        logging.info("after removing outlier # rows : " + str(df_for_reg.shape[0]))
        logging.info("after removing outlier # columns : " + str(df_for_reg.shape[1]))

        
        #표준화
        # new_std_columns_names = [cname+"_std" for cname in sorted_col_list] 
        # scaler = StandardScaler()
        # scaled_columns = scaler.fit_transform(df_for_reg[sorted_col_list])
        # df_for_reg[new_std_columns_names] = scaled_columns

        #정규화
        sorted_col_list.append("volume_mul_price")
        for col in sorted_col_list:
            feature_normal_col_name = col + "_normal"
            try:
                max_value = df_for_reg[col].max()
                min_value = df_for_reg[col].min()
                df_for_reg[feature_normal_col_name] = (df_for_reg[col] - min_value) / (max_value - min_value)
                            
            except Exception as e:
                logging.info(str(e))
                continue


        normal_col_list = df_for_reg.columns.str.contains("_normal")
        df_for_reg_norm = df_for_reg.loc[:, normal_col_list]
        
        # std_col_list = df_for_reg.columns.str.contains("_std")
        # df_for_reg_std = df_for_reg.loc[:, normal_col_list]
        
        if latest == False:
            df_for_reg_norm['period_price_diff'] = df_for_reg['period_price_diff']
            df_for_reg_norm['earning_diff'] = df_for_reg['earning_diff']
            df_for_reg_norm['sec_earning_diff'] = df_for_reg['sec_earning_diff']
            # df_for_reg_std['period_price_diff'] = df_for_reg['period_price_diff']
            # df_for_reg_std['earning_diff'] = df_for_reg['earning_diff']
        df_for_reg_norm['symbol'] = df_for_reg['symbol']
        df_for_reg_norm['sector'] = df_for_reg['sector']
        # df_for_reg_std['symbol'] = df_for_reg['symbol']
        # df_for_reg_std['sector'] = df_for_reg['sector']
        
        traindata_path = self.backtest.conf['ROOT_PATH'] + '/regressor_data_p{0:02d}_m{1:02d}/'.format(
                                self.backtest.conf['REBALANCE_PERIOD'], self.backtest.conf['START_MONTH'])
        
        if not os.path.exists(traindata_path):
            print("creating traindate_path : " + traindata_path)
            os.makedirs(traindata_path)

        if latest == False:
            # df_for_reg_std.to_csv(traindata_path + '{0}_{1:02d}_regressor_train_std.csv'.format(date.year, mmonth),
            #                         index=False)
            df_for_reg_norm.to_csv(traindata_path + '{0}_{1:02d}_regressor_train_norm.csv'.format(date.year, mmonth),
                                    index=False)
            print("print ai data to " + traindata_path + '{0}_{1:02d}_regressor_train_norm.csv'.format(date.year, mmonth))
            # df_for_reg.to_csv(traindata_path + '{0}_{1:02d}_full_regressor_train.csv'.format(date.year, mmonth),
            #                         index=False)
        else:
            df_for_reg_norm.to_csv(
                traindata_path + '{}_{:02d}_regressor_train_latest_norm.csv'.format(date.year, mmonth), index=False)
            # df_for_reg_std.to_csv(
            #     traindata_path + '{0}_{1:02d}_regressor_train_latest_std.csv'.format(date.year, mmonth), index=False)

    def cal_price(self):
        pd.set_option('mode.chained_assignment', None)
        """best_k 의 ['price', 'rebalance_day_price'] column을 채워주는 함수"""
        logging.info("best k length : %d", len(self.best_k))
        for idx, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            if date.year != self.backtest.table_year:
                logging.info("Reload BackTest Table. year : {} -> {}".format(self.backtest.table_year, date.year))
                self.backtest.load_bt_table(date.year)
                self.backtest.table_year = date.year

            # lastest date
            if (idx == len(self.best_k) - 1) and (len(self.best_k) != 1):
                logging.info("print latest data : " + date.strftime("%Y-%m-%d"))
                self.best_k[idx][3] = start_dh.dtable
                self.best_k[idx][3] = self.best_k[idx][3][self.best_k[idx][3].close > 0.000001]
                # self.best_k[idx][3] = self.best_k[idx][3][self.best_k[idx][3].volume_mul_price > 1000000]
                self.best_k[idx][3].rename(columns={'close': 'price'}, inplace=True)
                if self.backtest.ai_report_path is not None:
                    df_for_reg = self.best_k[idx][3].copy()
                    self.print_ai_data(df_for_reg, date, True)
                break

            if idx == 0:
                start_dh = DateHandler(self.backtest, date)
            end_dh = DateHandler(self.backtest, rebalance_date)

            if self.backtest.rank_report_path is not None or self.backtest.ai_report_path is not None:
                logging.info("rank/ai report cur date : {}".format(date))
                self.best_k[idx][3] = start_dh.dtable
                rebalance_date_price_df = end_dh.dtable[['symbol', 'close']]
                rebalance_date_price_df.rename(columns={'close': 'rebalance_day_price'}, inplace=True)
                self.best_k[idx][3] = pd.merge(self.best_k[idx][3], rebalance_date_price_df, how='outer', on='symbol')
                self.best_k[idx][3] = self.best_k[idx][3][self.best_k[idx][3].close > 0.000001]
                # self.best_k[idx][3] = self.best_k[idx][3][self.best_k[idx][3].volume_mul_price > 1000000]
                diff = self.best_k[idx][3]['rebalance_day_price'] - self.best_k[idx][3]['close']
                self.best_k[idx][3]['period_price_diff'] = diff / self.best_k[idx][3]['close']
                # self.best_k[idx][3] = pd.merge(self.best_k[idx][3], start_dh.fs_metrics, how='left', on='symbol')

                if self.backtest.ai_report_path is not None:
                    df_for_reg = self.best_k[idx][3].copy()
                    self.print_ai_data(df_for_reg, date, False)

                if self.backtest.rank_report_path is not None:
                    for feature in self.best_k[idx][3].columns:
                        if '_sorted' in feature:
                            feature_rank_col_name = feature + "_rank"
                            self.best_k[idx][3][feature_rank_col_name] \
                                = self.best_k[idx][3][feature].rank(method='max', ascending=False)

                self.best_k[idx][3] \
                    = self.best_k[idx][3].sort_values(by=["period_price_diff"],
                                                      axis=0, ascending=False)[:self.backtest.conf['TOP_K_NUM']]
            # evaluation report 만 뽑는 경우
            else:
                self.best_k[idx][3] = pd.DataFrame()

            if self.backtest.eval_report_path is not None:
                syms = best_group['symbol']
                for sym in syms:
                    if start_dh.dtable.loc[(start_dh.dtable['symbol'] == sym), 'close'].empty:
                        logging.debug("there is no price in start_dh FMP API  symbol : {}".format(sym))
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'price'] = 0
                    else:
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'price'] \
                            = start_dh.dtable.loc[(start_dh.dtable['symbol'] == sym), 'close'].values[0]

                    if end_dh.dtable.loc[(end_dh.dtable['symbol'] == sym), 'close'].empty:
                        logging.debug("there is no price in end_dh FMP API  symbol : {}".format(sym))
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'rebalance_day_price'] = 0
                    else:
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'rebalance_day_price'] \
                            = end_dh.dtable.loc[(end_dh.dtable['symbol'] == sym), 'close'].values[0]
                        if end_dh.dtable.loc[(end_dh.dtable['symbol'] == sym), 'close'].values[0] < 0.01:
                            logging.debug("close price already 0 : {}".format(sym))

                logging.debug(str(self.best_k[idx][2]))
                self.best_k[idx][2] = self.best_k[idx][2][self.best_k[idx][2].rebalance_day_price > 0.000001]
            self.best_k[idx][2] = self.best_k[idx][2].head(self.member_cnt)
            start_dh = copy.deepcopy(end_dh)
            logging.info(str(idx) + " " + str(date))

    def cal_earning_no_parallel(self):
        """backtest로 계산한 plan의 수익률을 serial하게 계산하는 함수"""
        logging.info("START cal_earning")
        params = copy.deepcopy(self.best_k)
        logging.info("in cal_earning : params : ")
        for best in self.best_k:
            (date, rebalance_date, best_group, reference_group, period_earning_rate) = best
            if 'price' not in best_group.columns:
                logging.warning("No Price Column!!")
                continue
            total_asset = 100000000
            stock_cnt = (total_asset / len(best_group)) / best_group['price']
            stock_cnt = stock_cnt.replace([np.inf, -np.inf], 0)
            stock_cnt = stock_cnt.fillna(0)
            # stock_cnt = stock_cnt.astype(int)
            price_mul_stock_cnt = best_group['price'] * stock_cnt
            my_asset_period = price_mul_stock_cnt.sum()
            remain_asset = total_asset - price_mul_stock_cnt.sum()
            if my_asset_period == 0:
                return
            # MDD 계산을 위해 이 구간에서 각 종목별 구매 개수 저장
            best[2]['count'] = stock_cnt
            # rebalance date의 가격으로 구매한 종목들 판매했을 때 자산 계산
            rebalance_day_price_mul_stock_cnt = best_group['rebalance_day_price'] * stock_cnt
            best[2]['period_earning'] = rebalance_day_price_mul_stock_cnt - price_mul_stock_cnt
            period_earning = rebalance_day_price_mul_stock_cnt.sum() - price_mul_stock_cnt.sum()
            best[4] = period_earning


    @staticmethod
    def cal_earning_func(best_k):
        # logger = self.backtest.main_ctx.get_multi_logger()

        (date, rebalance_date, best_group, reference_group, period_earning_rate) = best_k
        # logger.debug("in cal_earning_func  " + date.strftime("%Y-%m-%d"))
        if 'price' not in best_group.columns:
            # logger.warning("No Price Column!!")
            return
        # logger.debug(best_group)

        total_asset = 100000000
        stock_cnt = (total_asset / len(best_group)) / best_group['price']
        stock_cnt = stock_cnt.replace([np.inf, -np.inf], 0)
        stock_cnt = stock_cnt.fillna(0)
        # stock_cnt = stock_cnt.astype(int)
        price_mul_stock_cnt = best_group['price'] * stock_cnt
        my_asset_period = price_mul_stock_cnt.sum()
        remain_asset = total_asset - price_mul_stock_cnt.sum()
        if my_asset_period == 0:
            return
        
        # MDD 계산을 위해 이 구간에서 각 종목별 구매 개수 저장
        best_k[2]['count'] = stock_cnt

        # rebalance date의 가격으로 구매한 종목들 판매했을 때 자산 계산
        rebalance_day_price_mul_stock_cnt = best_group['rebalance_day_price'] * stock_cnt
        best_k[2]['period_earning'] = rebalance_day_price_mul_stock_cnt - price_mul_stock_cnt
        period_earning = rebalance_day_price_mul_stock_cnt.sum() - price_mul_stock_cnt.sum()
        best_k[4] = period_earning
        # logger.debug("in mp cal_earning_func : best k : ")
        # logger.debug(best_k)
        return best_k

    def cal_earning(self):
        """backtest로 계산한 plan의 수익률을 계산하는 함수"""
        logging.info("START cal_earning")
        params = copy.deepcopy(self.best_k)
        # with Pool(processes=multiprocessing.cpu_count()) as pool:
        # install_mp_handler()
        logging.info("in cal_earning : params : ")
        # print(params)
        with Pool(processes=multiprocessing.cpu_count()-4, initializer=install_mp_handler()) as pool:
        #with Pool(processes=2, initializer=install_mp_handler()) as pool:
            df_list = pool.map(self.cal_earning_func, params)
            # print("return df_list : ", df_list)
        df_list = list(filter(None.__ne__, df_list))
        logging.info("in cal_earning : df_list : ")
        # logging.info(df_list)
        # full_df = reduce(lambda df1, df2: pd.concat(df1, df2), df_list)
        # 잘들어가는지 check
        self.best_k = df_list
        for elem in df_list:
            if elem == None:
                continue
            (date, rebalance_date, best_group, reference_group, period_earning) = elem
            logging.debug(date)
            logging.debug(period_earning)
            # DEBUG
            if self.backtest.main_ctx.log_lvl == 10:
                best_group.to_csv("./earning_test.csv")

    def cal_mdd(self, price_table):
        """MDD를 계산해서 채워주는 함수"""
        best_asset = -1
        worst_asset = self.total_asset * 100000
        for i, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            if i == 0:
                prev_date = date
                continue
            # if i == len(self.best_k)-1:
            #     break
            else:
                # prev_date ~ date 까지 모든 date에 대해 자산 총액 계산
                allday_price_allsymbol = []
                syms = best_group['symbol']
                # symbol 별로 rebalancing day 기준으로 prev_date ~ date 의 price 정보 가져오고,
                # rebalancing day에 계산한 symbol 당 구매 수 column인 'count' 와 'close' 가격 곱해서 종목별 일별 자산 구함
                for sym in syms:
                    allday_price_per_symbol = price_table.query("(symbol == @sym) and "
                                                                "(date <= @date and date >= @prev_date)")
                    if allday_price_per_symbol.empty:
                        continue
                    else:
                        # FIXME SettingWithCopyWarning
                        count_per_sym = best_group.loc[(best_group.symbol == sym), 'count'].values
                        # allday_price_per_symbol['my_asset'] = allday_price_per_symbol['close'] * count_per_sym
                        allday_price_per_symbol \
                            = allday_price_per_symbol.assign(my_asset=lambda x: x.close * count_per_sym)
                        allday_price_allsymbol.append(allday_price_per_symbol)

                if allday_price_allsymbol == "":
                    logging.warning("allday_price_allsymbol is empty. can't calc MDD.")
                    return
                # 각 종목별 일별 자산을 모두 더하여 일별 총자산 구함
                accum_df = pd.DataFrame()
                for j, df in enumerate(allday_price_allsymbol):
                    df = df.reset_index(drop=True)
                    if j == 0:
                        accum_df = df[['date', 'my_asset']]
                    else:
                        accum_df = accum_df[['my_asset']] + df[['my_asset']]

                # concat 'date' column
                accum_df['date'] = df['date']

                # memory
                if accum_df['my_asset'].max(axis=0) > best_asset:
                    best_asset = accum_df['my_asset'].max(axis=0)
                    best_date = accum_df.loc[accum_df['my_asset'].idxmax(), 'date']
                if accum_df['my_asset'].min(axis=0) < worst_asset:
                    worst_asset = accum_df['my_asset'].min(axis=0)
                    worst_date = accum_df.loc[accum_df['my_asset'].idxmin(), 'date']
                prev_date = date    # update prev_date
        mdd = ((worst_asset / best_asset) - 1) * 100
        logging.info("MDD : {:.2f}%, best date : {}, worst date : {}".format(mdd, best_date, worst_date))
        self.MDD = mdd

    def cal_sharp(self):
        """sharp를 계산해서 채워주는 함수"""
        sharp = 0
        self.sharp = sharp

    @staticmethod
    def write_csv(path, date, rebalance_date, elem):
        fd = open(path, 'a')
        writer = csv.writer(fd, delimiter=",")
        writer.writerow("")
        writer.writerow(["start", date, "end", rebalance_date])
        fd.close()
        elem.to_csv(path, mode="a")

    def print_report(self):
        plan_earning = 1
        total_asset = 100000000
        accumulated_earning = 100
        max_local_plan_earning = -9999999999999
        min_local_plan_earning = 9999999999999
        for idx, (date, rebalance_date, eval_elem, rank_elem, period_earning_rate) in enumerate(self.best_k):
            local_plan_earning = period_earning_rate / total_asset
            accumulated_earning = accumulated_earning * (1.0 + local_plan_earning)
            if self.backtest.eval_report_path is not None:
                if max_local_plan_earning < local_plan_earning:
                    max_local_plan_earning = local_plan_earning
                if min_local_plan_earning > local_plan_earning:
                    min_local_plan_earning = local_plan_earning
                self.write_csv(self.backtest.eval_report_path, date, rebalance_date, eval_elem)
                fd = open(self.backtest.eval_report_path, 'a', newline='')
                writer = csv.writer(fd, delimiter=",")
                writer.writerow([str(period_earning_rate)])
                writer.writerow([str(accumulated_earning)])
                fd.close()
            if self.backtest.rank_report_path is not None:
                if idx <= self.backtest.conf['RANK_PERIOD']:
                    rank_partial_path = self.backtest.rank_report_path[:-4]+'_' + str(date.year)+'_' + str(date.month)\
                                        + '_' + str(date.day) + '.csv'
                    rank_elem.to_csv(rank_partial_path, index=False)
                    self.write_csv(self.backtest.rank_report_path, date, rebalance_date, rank_elem)
            if self.backtest.avg_report_path is not None:
                rank_elem.to_csv(self.backtest.avg_report_path, mode="a", index=False, header=False)

        # period.to_csv(self.backtest.eval_report_path, mode="a", column=columns)
        if self.backtest.eval_report_path is not None:
            ref_total_earning_rates = dict()
            for ref_sym in self.backtest.conf['REFERENCE_SYMBOL']:
                start_date = self.backtest.get_trade_date(datetime.datetime(self.backtest.main_ctx.start_year, 1, 1))
                end_date = self.backtest.get_trade_date(datetime.datetime(self.backtest.main_ctx.end_year, 12, 31))
                if end_date is None:
                    # get_trade_date는 기준일로부터 4일 이내 거래일을 가져오는 함수
                    # 가장 마지막 년도에는 12월 31일을 기준으로 가져오면 None이 return 됨
                    end_date = self.backtest.price_table["date"].max()
                    logging.debug(ref_sym)
                    logging.debug("start_date : " + start_date.strftime("%Y-%m-%d")
                                  + "    end_date : " + end_date.strftime("%Y-%m-%d"))
                reference_earning_df = self.backtest.price_table.query(
                    "(symbol == @ref_sym) and ((date == @start_date) or (date == @end_date))")
                logging.debug(reference_earning_df)
                if len(reference_earning_df) == 2:
                    reference_earning = reference_earning_df.iloc[1]['close'] - reference_earning_df.iloc[0]['close']
                    ref_total_earning_rate = (reference_earning / reference_earning_df.iloc[0]['close']) * 100
                    ref_total_earning_rates[ref_sym] = ref_total_earning_rate
                else:
                    logging.info("REFERENCE_SYMBOL [ " + str(ref_sym) + " ] ( "
                                 + start_date.strftime("%Y-%m-%d") + " ~ "
                                 + end_date.strftime("%Y-%m-%d") + " ) is Strange Value!!! NEED CHECK!!!")
                    ref_total_earning_rates[ref_sym] = 0

            # plan_earning = self.historical_earning_per_rebalanceday \
            #                                                   [len(self.historical_earning_per_rebalanceday)-1][3]\
            #               - self.historical_earning_per_rebalanceday[0][2]
            # plan_total_earning_rate = (plan_earning / self.historical_earning_per_rebalanceday[0][2]) * 100
            plan_total_earning_rate = (accumulated_earning - 100) / 100

            logging.warning("TOP_K_NUM : " + str(self.backtest.conf['TOP_K_NUM'])
                            + ", MEMBER_CNT : " + str(self.backtest.conf['MEMBER_CNT'])
                            + ", ABSOLUTE_SCORE : " + str(self.backtest.conf['ABSOLUTE_SCORE'])
                            + ", Our_Earning : " + str(plan_total_earning_rate)
                            + ", MAX_LOCAL_PLAN_EARNING : " + str(max_local_plan_earning)
                            + ", MIN_LOCAL_PLAN_EARNING : " + str(min_local_plan_earning)
                            )
            fd = open(self.backtest.eval_report_path, 'a')
            writer = csv.writer(fd, delimiter=",")
            writer.writerow("")
            writer.writerow(["ours", plan_total_earning_rate])
            for ref_sym, total_earning_rate in ref_total_earning_rates.items():
                writer.writerow([ref_sym, total_earning_rate])
            fd.close()

    def run(self, price_table):
        self.cal_price()
        if self.backtest.eval_report_path is not None:
            # self.cal_earning()
            self.cal_earning_no_parallel()
            # self.cal_mdd(price_table)
            # self.cal_sharp()
        self.print_report()
