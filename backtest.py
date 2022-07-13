import copy
import csv
import datetime
import logging
import multiprocessing
import os

import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from multiprocessing import Pool
from functools import reduce

CHUNK_SIZE = 20480


class Backtest:
    def __init__(self, main_ctx, conf, plan_handler, rebalance_period):
        self.main_ctx = main_ctx
        self.conf = conf
        self.plan_handler = plan_handler
        self.rebalance_period = rebalance_period
        self.eval_handler = EvaluationHandler(self)
        # 아래 값들은 init_bt_from_db 에서 세팅해주나, 가려지는 값이 없도록(init만 봤을 때 calss value가 모두 보이도록) 나열함
        self.symbol_table = ""
        self.price_table = ""
        self.fs_table = ""
        self.metrics_table = ""
        if conf['PRINT_EVAL_REPORT'] == 'Y':
            self.eval_report_path = self.create_report("EVAL")
        if conf['PRINT_RANK_REPORT'] == 'Y':
            self.rank_report_path = self.create_report("RANK")
        self.table_year = 0

        self.run()

    def create_report(self, report_type):
        path = "./reports/" + report_type + "_REPORT_"
        idx = 0
        while True:
            if not os.path.exists(path + str(idx) + ".csv"):
                path = path + str(idx) + ".csv"
                logging.info('REPORT PATH: "{}" ...'.format(path))
                break
            else:
                idx += 1

        with open(path, 'w') as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(["COMMON"])
            writer.writerow(["Report Date", datetime.datetime.now().strftime('%m-%d %H:%M')])
            writer.writerow(["Rebalance Period", str(self.rebalance_period) + " Month",
                             "Start Year", self.main_ctx.start_year,
                             "End Year", self.main_ctx.end_year])
            writer.writerow(["K", self.plan_handler.k_num])
            writer.writerow("")

            writer.writerow(["PLAN HANDLER"])
            for plan in self.plan_handler.plan_list:
                writer.writerow(plan["params"])
                dict_writer = csv.DictWriter(file, fieldnames=plan["params"])
                dict_writer.writerow(plan["params"])
                writer.writerow("")
        return path

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

    def reload_bt_table(self, year):
        """
        추후에 database에서 가져올 데이터가 많을 걸 대비해서 __init__ 함수에서 세팅하지 않고, 해당 함수에서 세팅토록 함
        일부 필요한 내용한 init하거나 분할해서 가져오려고 한다면 쿼리가 더 복잡해질 수 있기에 따로 빼놓음
        init에서 세팅하지 않은 이유를 코드에 도입. 해당하는 year에 값을 가져오도록 변경
        """
        query = "SELECT * FROM PRICE WHERE date BETWEEN '" \
                + str(datetime.datetime(self.main_ctx.start_year, 1, 1)) + "'" \
                + " AND '" + str(datetime.datetime(self.main_ctx.end_year, 12, 31)) + "'"
        self.symbol_table = pd.DataFrame()
        self.price_table = pd.DataFrame()
        if self.conf['USE_DB'] == "Y":
            self.symbol_table = self.data_from_database("SELECT * FROM symbol_list")
            self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
            self.price_table = self.data_from_database(query)
            self.fs_table = self.data_from_database("SELECT * FROM financial_statement")
            self.metrics_table = self.data_from_database("SELECT * FROM METRICS")

        if self.conf['USE_DATAFRAME'] == "Y":
            if self.symbol_table.empty:
                self.symbol_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/symbol_list.parquet")
                self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
            if self.price_table.empty:
                self.price_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/price.parquet")

            self.fs_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/financial_statement_"
                                            + str(year) + ".parquet")
            self.metrics_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/metrics_" + str(year) + ".parquet")

            if year != self.main_ctx.start_year:
                prev_fs = pd.read_parquet(self.main_ctx.root_path + "/VIEW/financial_statement_"
                                          + str(year-1) + ".parquet")
                self.fs_table = pd.concat([prev_fs, self.fs_table])
                prev_metrics = pd.read_parquet(self.main_ctx.root_path + "/VIEW/metrics_" + str(year-1) + ".parquet")
                self.metrics_table = pd.concat([prev_metrics, self.metrics_table])

    def get_trade_date(self, pdate):
        """개장일이 아닐 수도 있기에 보정해주는 함수"""
        # pdate =  pdate.date()
        post_date = pdate + relativedelta(days=4)

        res = self.price_table.query("date <= @post_date and date >=@pdate")
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
        date = datetime.datetime(self.main_ctx.start_year, 4, 1)
        while date <= datetime.datetime(self.main_ctx.end_year, 12 - self.rebalance_period, 30):
            if date.year != self.table_year:
                logging.info("Reload BackTest table. year : {} -> {}".format(self.table_year, date.year))
                self.reload_bt_table(date.year)
                self.table_year = date.year
            date = self.get_trade_date(date)
            # get_trade_date 에서 price table 을 이용해야 하기에 reload_bt_table을 먼저 해주어야 함
            if date is None:
                break
            logging.info("Backtest Run : " + str(date.strftime("%Y-%m-%d")))
            self.plan_handler.date_handler = DateHandler(self, date)
            logging.debug("complete set date_handler date : {}".format(date.strftime("%Y-%m-%d")))
            
            if self.conf['NEED_EVALUATION'] == 'Y':
                self.plan_handler.run()
            self.eval_handler.set_best_k(date, date+relativedelta(months=self.rebalance_period),
                                         self.plan_handler.date_handler)
            # day를 기준으로 하려면 아래를 사용하면 됨. 31일 기준으로 하면 우리가 원한 한달이 아님
            # date += relativedelta(days=self.rebalance_period)
            date += relativedelta(months=self.rebalance_period)

        date = self.price_table["date"].max()
        logging.debug("Recent date : " + str(date))
        self.reload_bt_table(date.year)
        self.plan_handler.date_handler = DateHandler(self, date)
        self.plan_handler.run()
        self.eval_handler.print_current_best(self.plan_handler.date_handler)

        if (self.conf['PRINT_RANK_REPORT'] == 'Y') | (self.conf['PRINT_EVAL_REPORT'] == 'Y') |\
                (self.conf['PRINT_AI'] == 'Y'):
            logging.info("START Evaluation")
            self.eval_handler.run(self.price_table)


class PlanHandler:
    def __init__(self, k_num, absolute_score):
        self.plan_list = None
        self.date_handler = None
        self.k_num = k_num
        self.absolute_score = absolute_score

    def run(self):
        """
        main에서 짜여진 전략(function pointer)을 순서대로 호출
        plan_list와 date_handler를 채워주고 불러워줘야 함.
        현재는 plan_list는 main, date_handler는 Backtest 에서 채워줌.
        """
        assert self.plan_list is not None, "Empty Plan List"
        assert self.date_handler is not None, "Empty Date Handler"

        with Pool(processes=multiprocessing.cpu_count()*2) as pool:
            df_list = pool.map(self.plan_run, self.plan_list)

        full_df = reduce(lambda df1, df2: pd.merge(df1, df2, on='symbol'), df_list)
        self.date_handler.symbol_list = pd.merge(self.date_handler.symbol_list, full_df, how='left', on=['symbol'])
        score_col_list = self.date_handler.symbol_list.columns.str.contains("_score")
        self.date_handler.symbol_list['score'] = self.date_handler.symbol_list.loc[:,score_col_list].sum(axis=1)
        logging.debug(self.date_handler.symbol_list.sort_values(by=['score'], ascending=False)[['symbol', 'score']])

    @staticmethod
    def plan_run(plan):
        return plan["f_name"](plan["params"])

    def single_metric_plan(self, params):
        """single metric(PBR, PER ... )에 따라 plan_handler.date_handler.symbol_list의 score column에 값을 갱신해주는 함수.
           params에 plan의 parameter들이 아래와 같이 들어옴
           params["key"]        : plan에서 사용할 종목의 지표(ex: PER, PBR, 영업이익 ...)
           params["key_dir"]    : 지표가 낮을수록 좋은지(low) 높을수록 좋은지(high)
           params["weight"]     : score update시 weight
           params["diff"]       : 이 지표로 각 종목간 score 차이
           params["base"]       : 특정 threshold 이상/이하의 종목은 score 주지 않음
           params["base_dir"]   : "base"로 준 threshold 이상/이하(</>) 선택
        """
        logging.info("[plan] key : {}, key_dir : {}, weight : {}, "
                     "diff : {}, base : {}, base_dir : {}".format(params["key"], params["key_dir"], params["weight"],
                                                                  params["diff"], params["base"], params["base_dir"]))

        if self.absolute_score - params["diff"] * self.k_num < 0:
            logging.warning("Wrong params['diff'] : TOO BIG! SET UNDER " + str(self.absolute_score/self.k_num))

        key = str(params["key"])
        if params["key_dir"] == "low":
            top_k_df = self.date_handler.fs_metircs[self.date_handler.fs_metircs[key] > 0]
            top_k_df = top_k_df.sort_values(by=[key], ascending=True, na_position="last")[:self.k_num]
        elif params["key_dir"] == "high":
            top_k_df = self.date_handler.fs_metircs.sort_values(by=[key], ascending=False,
                                                                na_position="last")[:self.k_num]
        else:
            logging.error("Wrong params['key_dir'] : ", params["key_dir"], "params['key_dir'] must be 'low' or 'high'")
            return

        if params["base_dir"] == ">":
            top_k_df = top_k_df[top_k_df[key] > params["base"]]
        elif params["base_dir"] == "<":
            top_k_df = top_k_df[top_k_df[key] < params["base"]]
        else:
            logging.error("Wrong params['base_dir'] : ", params["base_dir"], " params['base_dir'] must be '>' or '<'")
            return

        logging.debug(top_k_df[['symbol', params["key"]]])

        symbols = top_k_df['symbol']
        return_df = self.date_handler.symbol_list[['symbol']]
        delta = self.absolute_score
        # 경고처리 무시
        pd.set_option('mode.chained_assignment', None)
        for sym in symbols:
            local_score_name = key + '_score'
            return_df.loc[(self.date_handler.symbol_list.symbol == sym), local_score_name]\
                = params["weight"] * delta
            delta = delta - params["diff"]
        local_rank_name = key+'_rank'
        return_df[local_rank_name] = return_df[local_score_name].rank(method='min', ascending=False)
        logging.debug(return_df[[local_score_name, local_rank_name]])
        return return_df


class DateHandler:
    def __init__(self, backtest, date):
        self.date = date
        # date = datetime.datetime.combine(date, datetime.datetime.min.time())
        # query = '(date == "{}")'.format(self.date)
        # db에서 delistedDate null 이  df에서는 NaT로 들어옴.
        query = '(delistedDate >= "{}") or (delistedDate == "NaT") or (delistedDate == "None")'.format(date)
        self.symbol_list = backtest.symbol_table.query(query)
        self.symbol_list = self.symbol_list.assign(score=0)

        trade_date = backtest.get_trade_date(date)
        self.price = backtest.price_table.query("date == @trade_date")
        self.price = self.price.drop_duplicates('symbol', keep='first')
        # self.price = self.get_date_latest_per_symbol(backtest.price_table, self.date)
        self.symbol_list = pd.merge(self.symbol_list, self.price, how='left', on='symbol')

        prev = self.date - relativedelta(months=4)
        # self.fs = self.get_date_latest_per_symbol(backtest.fs_table, self.date)
        self.fs = backtest.fs_table.copy()
        self.fs = self.fs[self.fs.date <= self.date]
        self.fs = self.fs[prev <= self.fs.date]
        self.fs = self.fs.drop_duplicates('symbol', keep='first')

        # self.metrics = self.get_date_latest_per_symbol(backtest.metrics_table, self.date)
        self.metrics = backtest.metrics_table.copy()
        self.metrics = self.metrics[self.metrics.date <= self.date]
        self.metrics = self.metrics[prev <= self.metrics.date]
        self.metrics = self.metrics.drop_duplicates('symbol', keep='first')

        self.fs_metircs = pd.merge(self.fs, self.metrics, how='outer', on='symbol')


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
        best_symbol_info = pd.merge(scored_dh.symbol_list, scored_dh.metrics, how='outer', on='symbol')
        best_symbol_info = pd.merge(best_symbol_info, scored_dh.fs, how='outer', on='symbol')
        best_symbol = best_symbol_info.sort_values(by=["score"], axis=0, ascending=False).head(self.member_cnt)
        best_symbol = best_symbol.assign(count=0)
        best_symbol.to_csv('./result.csv')

    def set_best_k(self, date, rebalance_date, scored_dh):
        """plan_handler.date_handler.symbol_list에 score를 보고 best_k에 append 해주는 함수."""
        best_symbol_info = pd.merge(scored_dh.symbol_list, scored_dh.metrics, how='outer', on='symbol')
        best_symbol_info = pd.merge(best_symbol_info, scored_dh.fs, how='outer', on='symbol')
        best_symbol = best_symbol_info.sort_values(by=["score"], axis=0, ascending=False).head(self.member_cnt)
        # best_symbol = best_symbol.assign(price=0)
        best_symbol = best_symbol.assign(count=0)
        reference_group = pd.DataFrame()
        period_earning_rate = 0
        self.best_k.append([date, rebalance_date, best_symbol, reference_group, period_earning_rate])

    def cal_price(self):
        """best_k 의 ['price', 'rebalance_day_price'] column을 채워주는 함수"""
        for idx, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            if date.year != self.backtest.table_year:
                logging.info("Reload BackTest Table. year : {} -> {}".format(self.backtest.table_year, date.year))
                self.backtest.reload_bt_table(date.year)
                self.backtest.table_year = date.year
                            
            if idx == 0:
                start_dh = DateHandler(self.backtest, date)
            end_dh = DateHandler(self.backtest, rebalance_date)

            if (self.backtest.conf['PRINT_RANK_REPORT'] == 'Y') or (self.backtest.conf['PRINT_AI'] == 'Y'):
                self.best_k[idx][3] = start_dh.price
                rebalance_date_price_df = end_dh.price[['symbol', 'close']]
                rebalance_date_price_df.rename(columns={'close':'rebalance_day_price'}, inplace=True)
                self.best_k[idx][3] = pd.merge(self.best_k[idx][3], rebalance_date_price_df, how='outer', on='symbol')
                self.best_k[idx][3] = self.best_k[idx][3][self.best_k[idx][3].close > 0.000001]
                diff = self.best_k[idx][3]['rebalance_day_price'] - self.best_k[idx][3]['close']
                self.best_k[idx][3]['period_price_diff'] = diff / self.best_k[idx][3]['close']
                self.best_k[idx][3] = pd.merge(self.best_k[idx][3], start_dh.metrics, how='outer', on='symbol')
                self.best_k[idx][3] = pd.merge(self.best_k[idx][3], start_dh.fs, how='outer', on='symbol')
                if self.backtest.conf['PRINT_AI'] == 'Y':
                    for feature in self.best_k[idx][3].columns:
                        feature_normal_col_name = feature + "_normal"
                        try:
                            max_value = self.best_k[idx][3][feature].max()
                            min_value = self.best_k[idx][3][feature].min()
                            self.best_k[idx][3][feature_normal_col_name] \
                                = (self.best_k[idx][3][feature] - min_value)/(max_value - min_value)
                        except Exception as e:
                            logging.info(str(e))
                            continue
                        self.best_k[idx][3]['earning_diff'] \
                            = self.best_k[idx][3]['period_price_diff'] - self.best_k[idx][3]['period_price_diff'].mean()
                    normal_col_list = self.best_k[idx][3].columns.str.contains("_normal")
                    df_for_reg = self.best_k[idx][3].loc[:,normal_col_list]
                        
                    for col in df_for_reg.columns:
                        new_col_name = col + "_max_diff"
                        max_v = df_for_reg[col].max()
                        df_for_reg[new_col_name] = max_v - df_for_reg[col]
                
                    df_for_reg['earning_diff'] = self.best_k[idx][3]['earning_diff']
                    df_for_reg['symbol'] = self.best_k[idx][3]['symbol']                                                
                    df_for_reg.to_csv('./reports/{}_{}_regressor_train.csv'.format(date.year, date.month), index=False)
                    
                else:
                    for feature in self.best_k[idx][3].columns:
                        feature_rank_col_name = feature + "_rank"
                        self.best_k[idx][3][feature_rank_col_name] = self.best_k[idx][3][feature].rank(method='min')
                # self.best_k[idx][3] = self.best_k[idx][3].sort_values(by=["period_price_diff"], axis=0, ascending=False)
                self.best_k[idx][3] = self.best_k[idx][3].sort_values(by=["period_price_diff"], axis=0, ascending=False)[:self.backtest.conf['TOP_K_NUM']]
            else:
                self.best_k[idx][3] = pd.DataFrame()

            if self.backtest.conf['NEED_EVALUATION'] == 'Y':
                syms = best_group['symbol']
                for sym in syms:
                    if start_dh.price.loc[(start_dh.price['symbol'] == sym), 'close'].empty:
                        logging.debug("there is no price in FMP API  symbol : {}".format(sym))
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'price'] = 0
                    else:
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'price']\
                            = start_dh.price.loc[(start_dh.price['symbol'] == sym), 'close'].values[0]

                    if end_dh.price.loc[(end_dh.price['symbol'] == sym), 'close'].empty:
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'rebalance_day_price'] = 0
                    else:
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'rebalance_day_price']\
                            = end_dh.price.loc[(end_dh.price['symbol'] == sym), 'close'].values[0]
                
                self.best_k[idx][2] = self.best_k[idx][2][self.best_k[idx][2].price > 0.000001]

            start_dh = copy.deepcopy(end_dh)
            logging.debug(str(idx) + " " + str(date))
            logging.debug(str(self.best_k[idx][2]))
    
    def cal_earning(self):
        """backtest로 계산한 plan의 수익률을 계산하는 함수"""
        base_asset = self.total_asset
        prev = 0
        best_asset = -1
        worst_asset = self.total_asset * 1000
        for idx, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            # TODO best_k 맞게 사고 남은 짜투리 금액 처리
            stock_cnt = (self.total_asset / len(best_group)) / best_group['price']
            stock_cnt = stock_cnt.replace([np.inf, -np.inf], 0)
            stock_cnt = stock_cnt.fillna(0)
            stock_cnt = stock_cnt.astype(int)
            price_mul_stock_cnt = best_group['price'] * stock_cnt
            my_asset_period = price_mul_stock_cnt.sum()
            remain_asset = self.total_asset - price_mul_stock_cnt.sum()
            
            if my_asset_period == 0:
                continue

            # MDD 계산을 위해 이 구간에서 각 종목별 구매 개수 저장
            self.best_k[idx][2]['count'] = stock_cnt
            
            # rebalance date의 가격으로 구매한 종목들 판매했을 때 자산 계산
            rebalance_day_price_mul_stock_cnt = best_group['rebalance_day_price'] * stock_cnt
            self.best_k[idx][2]['period_earning'] = rebalance_day_price_mul_stock_cnt - price_mul_stock_cnt
            period_earning = rebalance_day_price_mul_stock_cnt.sum() - price_mul_stock_cnt.sum()
            
            prev = self.total_asset
            self.total_asset = remain_asset + rebalance_day_price_mul_stock_cnt.sum()

            logging.debug("cur idx : {}, prev : {}, earning : {:.2f},"
                          "earning_rate : {}, asset : {}".format(idx, idx-1, period_earning,
                                                                 period_earning / (self.total_asset - period_earning),
                                                                 self.total_asset))
            self.best_k[idx][4] = period_earning / (self.total_asset-period_earning)

            self.historical_earning_per_rebalanceday.append([date, period_earning, prev, self.total_asset, best_group])

            # for rebalanced day price based MDD
            if self.total_asset > best_asset:
                best_asset = my_asset_period
            if self.total_asset < worst_asset:
                worst_asset = my_asset_period
        logging.info("Rebalanced day price based MDD : {:.2f} %".format(((worst_asset / best_asset) - 1) * 100))

    def cal_mdd(self, price_table):
        """MDD를 계산해서 채워주는 함수"""
        best_asset = -1
        worst_asset = self.total_asset * 100000
        for i, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            if i == 0:
                prev_date = date
                continue
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
                        allday_price_per_symbol\
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

                # update prev_date
                prev_date = date
        mdd = ((worst_asset / best_asset) - 1) * 100
        logging.info("MDD : {:.2f}%, best date : {}, worst date : {}".format(mdd, best_date, worst_date))
        self.MDD = mdd

    def cal_sharp(self):
        """sharp를 계산해서 채워주는 함수"""
        sharp = 0
        self.sharp = sharp

    @staticmethod
    def write_csv(path, date, rebalance_date, elem, columns):
        fd = open(path, 'a')
        writer = csv.writer(fd, delimiter=",")
        writer.writerow("")
        writer.writerow(["start", date, "end", rebalance_date])
        fd.close()
        elem.to_csv(path, columns=columns, mode="a")

    def print_report(self):
        # eval_columns = ["symbol", "score", "price", "rebalance_day_price", "count", "period_earning",
        #                "pbRatio", "pbRatio_rank", "pbRatio_score", "peRatio", "peRatio_rank", "peRatio_score",
        #                "ipoDate", "delistedDate"]
        # "period_earning" 우선 삭제
        for idx, (date, rebalance_date, eval_elem, rank_elem, period_earning_rate) in enumerate(self.best_k):
            if self.backtest.conf['PRINT_EVAL_REPORT'] == 'Y' and self.backtest.conf['NEED_EVALUATION'] == 'Y':
                self.write_csv(self.backtest.eval_report_path, date, rebalance_date, eval_elem, eval_elem.columns.tolist())
                fd = open(self.backtest.eval_report_path, 'a')
                writer = csv.writer(fd, delimiter=",")
                writer.writerow(str(period_earning_rate))
                fd.close()
            if self.backtest.conf['PRINT_RANK_REPORT'] == 'Y':
                if idx <= self.backtest.conf['RANK_PERIOD']:
                    self.write_csv(self.backtest.rank_report_path, date, rebalance_date, rank_elem, rank_elem.columns.tolist())
            # period.to_csv(self.backtest.eval_report_path, mode="a", column=columns)

        if self.backtest.conf['PRINT_EVAL_REPORT'] == 'Y' and self.backtest.conf['NEED_EVALUATION'] == 'Y':
            ref_total_earning_rates = dict()
            for ref_sym in self.backtest.conf['REFERENCE_SYMBOL']:
                start_date = self.backtest.get_trade_date(datetime.datetime(self.backtest.main_ctx.start_year, 1, 1))
                end_date = self.backtest.get_trade_date(datetime.datetime(self.backtest.main_ctx.end_year, 12, 31))
                reference_earning_df= self.backtest.price_table.query("(symbol == @ref_sym) and ((date == @start_date) or (date == @end_date))")
                logging.debug(ref_sym)
                logging.debug(reference_earning_df)
                reference_earning = reference_earning_df.iloc[1]['close'] - reference_earning_df.iloc[0]['close']
                ref_total_earning_rate = (reference_earning / reference_earning_df.iloc[0]['close']) * 100
                ref_total_earning_rates[ref_sym] = ref_total_earning_rate

            plan_earning = self.historical_earning_per_rebalanceday[len(self.historical_earning_per_rebalanceday)-1][3]\
                           - self.historical_earning_per_rebalanceday[0][2]
            plan_total_earning_rate = (plan_earning / self.historical_earning_per_rebalanceday[0][2]) * 100

            logging.info("Our Earning : " + str(plan_total_earning_rate))
            fd = open(self.backtest.eval_report_path, 'a')
            writer = csv.writer(fd, delimiter=",")
            writer.writerow("")
            writer.writerow(["ours", plan_total_earning_rate])
            for ref_sym, total_earning_rate in ref_total_earning_rates.items():
                writer.writerow([ref_sym, total_earning_rate])
            fd.close()

    def run(self, price_table):
        self.cal_price()
        if self.backtest.conf['NEED_EVALUATION'] == 'Y':
            self.cal_earning()
            self.cal_mdd(price_table)
            # self.cal_sharp()
        self.print_report()


class SymbolHandler:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.price = self.get_symbol_price()
        self.financial_statement = self.get_symbol_fs()

    def get_symbol_price(self):
        return self.symbol

    def get_symbol_fs(self):
        return self.symbol
