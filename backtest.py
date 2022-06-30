import csv
import datetime
from dateutil.relativedelta import relativedelta
import logging
import os

import pandas as pd
import numpy as np

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
        self.backtest_table_year = 0

        self.run()

    def create_report(self, type):
        path = "./reports/" + type + "_REPORT_"
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
        if self.conf['USE_DB'] == "Y":
            self.symbol_table = self.data_from_database("SELECT * FROM symbol_list")
            self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
            self.price_table = self.data_from_database(query)
            self.fs_table = self.data_from_database("SELECT * FROM financial_statement")
            self.metrics_table = self.data_from_database("SELECT * FROM METRICS")

        if self.conf['USE_DATAFRAME'] == "Y":
            if self.symbol_table == "":
                self.symbol_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/symbol_list.parquet")
                self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
            self.price_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/price_" + str(year) + ".parquet")
            self.fs_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/financial_statement_"
                                            + str(year) + ".parquet")
            self.metrics_table = pd.read_parquet(self.main_ctx.root_path + "/VIEW/metrics_" + str(year) + ".parquet")

    def get_trade_date(self, pdate):
        """개장일이 아닐 수도 있기에 보정해주는 함수"""
        post_date = pdate + relativedelta(days=4)
        post_date = post_date.date()
        pdate = pdate.date()
        res = self.price_table.query("date <= @post_date and date >=@pdate")
        if res.empty:
            return None
        else:
            return res.iloc[0].date

    def run(self):
        """
        main에서 전달받은 plan list에 따른 backtest를 진행하는 함수로 크게 2개의 파트로 구분되어 있음
        [1] date 별로 plan list에 따라 plan_handler.date_handler.score를 계산해넣고,
            상위권 symbol을 self.best_symbol_group에 추가
        [2] best_symbol_group 에서 가져와서 MDD나 샤프지수와 같은 전략 전체에 필요한 계산값들을 계산해서 채워 넣음
        전략 별로 하나의 backtest class, plan_handler (1:1 mapping)가 만들어지며 생성의 주체는 main
        date_handler는 다수 만들어지며 생성 주체는 backtest이며 생성 후
        backtest에서 본인에게 mapping되어 있는 plan_handler에게 달아줌.
        """
        date = datetime.datetime(self.main_ctx.start_year, 1, 1)
        while date <= datetime.datetime(self.main_ctx.end_year, 12 - self.rebalance_period, 30):
            if date.year != self.backtest_table_year:
                self.reload_bt_table(date.year)
                self.backtest_table_year = date.year
            date = self.get_trade_date(date)
            # get_trade_date 에서 price table 을 이용해야 하기에 reload_bt_table을 먼저 해주어야 함
            if date is None:
                break
            logging.info("in Backtest run() date : " + str(date))
            self.plan_handler.date_handler = DateHandler(self, date)
            self.plan_handler.run()
            self.eval_handler.set_best_symbol_group(date, date+relativedelta(
                months=self.rebalance_period), self.plan_handler.date_handler)
            # day를 기준으로 하려면 아래를 사용하면 됨. 31일 기준으로 하면 우리가 원한 한달이 아님
            # date += relativedelta(days=self.rebalance_period)
            date += relativedelta(months=self.rebalance_period)
        if (self.conf['PRINT_RANK_REPORT'] == 'Y') | (self.conf['PRINT_EVAL_REPORT'] == 'Y'):
            self.eval_handler.run(self.price_table)


class PlanHandler:
    def __init__(self, k_num):
        self.plan_list = None
        self.date_handler = None
        self.k_num = k_num

    def run(self):
        """
        main에서 짜여진 전략(function pointer)을 순서대로 호출
        plan_list와 date_handler를 채워주고 불러워줘야 함.
        현재는 plan_list는 main, date_handler는 Backtest 에서 채워줌.
        """
        assert self.plan_list is not None, "Empty Plan List"
        assert self.date_handler is not None, "Empty Date Handler"
        for plan in self.plan_list:
            plan["f_name"](plan["params"])

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
        logging.info("[pbr] key : {}, key_dir : {}, weight : {}, "
                     "diff : {}, base : {}, base_dir : {}".format(params["key"], params["key_dir"], params["weight"],
                                                                  params["diff"], params["base"], params["base_dir"]))
        key = str(params["key"])
        if params["key_dir"] == "low":
            top_k_df = self.date_handler.metrics.sort_values(by=[key], ascending=True)[:self.k_num]
        elif params["key_dir"] == "high":
            top_k_df = self.date_handler.metrics.sort_values(by=[key], ascending=False)[:self.k_num]
        else:
            logging.error("Wrong params['key_dir'] : ", params["key_dir"], " params['key_dir'] must be 'low' or 'high'")
            return

        if params["base_dir"] == ">":
            top_k_df = top_k_df[top_k_df[key] > params["base"]]
        elif params["base_dir"] == "<":
            top_k_df = top_k_df[top_k_df[key] < params["base"]]
        else:
            logging.error("Wrong params['base_dir'] : ", params["base_dir"], " params['base_dir'] must be '>' or '<'")
            return

        print(top_k_df[['symbol', params["key"]]])
        symbols = top_k_df['symbol']
        delta = 100
        for sym in symbols:
            prev_score = self.date_handler.symbol_list[self.date_handler.symbol_list['symbol'] == sym]['score']
            self.date_handler.symbol_list.loc[(self.date_handler.symbol_list.symbol == sym), 'score']\
                = prev_score + params["weight"] * delta
            local_score_name = key + '_score'
            self.date_handler.symbol_list.loc[(self.date_handler.symbol_list.symbol == sym), local_score_name]\
                = params["weight"] * delta
            delta = delta - params["diff"]
        local_rank_name = key + '_rank'
        self.date_handler.symbol_list[local_rank_name] = \
            self.date_handler.symbol_list[local_score_name].rank(method='min', ascending=False)
        # print(self.date_handler.symbol_list[[local_score_name, local_rank_name]])
        # print(self.date_handler.symbol_list.sort_values(by=['score'], ascending=False)[['symbol', 'score']])


class DateHandler:
    def __init__(self, backtest, date):
        self.date = date
        # query = '(date == "{}")'.format(self.date)
        # db에서 delistedDate null 이  df에서는 NaT로 들어옴.
        query = '(delistedDate >= "{}") or (delistedDate == "NaT")'.format(self.date)
        self.symbol_list = backtest.symbol_table.query(query)
        self.symbol_list = self.symbol_list.assign(score=0)
        self.price = self.get_date_latest_per_symbol(backtest.price_table, self.date)
        self.symbol_list = pd.merge(self.symbol_list, self.price, how='left', on='symbol')
        self.fs = self.get_date_latest_per_symbol(backtest.fs_table, self.date)
        self.metrics = self.get_date_latest_per_symbol(backtest.metrics_table, self.date)

    def get_date_latest_per_symbol(self, table, date):
        date_latest = pd.DataFrame()
        syms = self.symbol_list['symbol']
        tmp = [pd.DataFrame()]
        # TODO 모든 symbol 다 돌면 오래걸려서 10개로 줄임. 나중에 삭제
        for sym in syms[:10]:
            # TODO date 기준에 date - 3달 ~ date로 넣기
            prev_q_date = date - relativedelta(months=3)
            past = table.query("(symbol == @sym) and (date <= @date and date >= @prev_q_date)")
            if past.empty:
                continue
            else:
                # past 는 date 이전 모든 fs들, 이 중 첫번째 row가 가장 최신 fs. iloc[0]로 첫 row 가져옴.
                date_latest = date_latest.append(past.iloc[0])
        # FIXME 왜 symbol 당 row가 2개씩 들어가있나 ?
        date_latest = date_latest.drop_duplicates('symbol', keep='first')
        return date_latest


class EvaluationHandler:
    def __init__(self, backtest):
        self.best_symbol_group = []
        self.historical_earning_per_rebalanceday = []
        self.backtest = backtest
        self.member_cnt = self.cal_member_cnt()
        self.accumulated_earning = 0
        self.MDD = 0
        self.sharp = 0
        self.total_asset = 100000000

    @staticmethod
    def cal_member_cnt():
       # TODO 상위 몇 종목을 구매할 것인가에 대한 계산. 현재는 상위 4개의 주식을 매 period 마다 구매하는 것으로 되어 있음
       return 4

    def set_best_symbol_group(self, date, rebalance_date, scored_datehandler):
        """plan_handler.date_handler.symbol_list에 score를 보고 best_symbol_group에 append 해주는 함수."""
        best_symbol_info = pd.merge( scored_datehandler.symbol_list, scored_datehandler.metrics, how='outer', on='symbol')
        best_symbol_info = pd.merge( best_symbol_info, scored_datehandler.fs, how='outer', on='symbol')
        best_symbol = best_symbol_info.sort_values(by=["score"], axis=0, ascending=False).head(self.member_cnt)
        # print("set_best_symbol_group()")
        # print(best_symbol)
        # best_symbol = best_symbol.assign(price=0)
        best_symbol = best_symbol.assign(count=0)
        reference_group = pd.DataFrame()
        self.best_symbol_group.append([date, rebalance_date, best_symbol, reference_group])

    def cal_price(self):
        """best_symbol_group 의 ['price', 'rebalance_day_price'] column을 채워주는 함수"""
        for idx, (date, rebalance_date, best_group, reference_group) in enumerate(self.best_symbol_group):
            
            if idx == 0:
                start_datehandler = DateHandler(self.backtest, date)
            end_datehandler = DateHandler(self.backtest, rebalance_date)

            if self.backtest.conf['NEED_EVALUATION'] == 'Y':
                self.best_symbol_group[idx][3] = start_datehandler.price
                rebalance_date_price_df = end_datehandler.price[['symbol', 'close']]
                rebalance_date_price_df.rename(columns={'close':'rebalance_day_price'}, inplace=True)
                self.best_symbol_group[idx][3] = pd.merge(
                    self.best_symbol_group[idx][3], rebalance_date_price_df, 
                    how='outer', on='symbol')
                self.best_symbol_group[idx][3]['period_price_diff'] = \
                    self.best_symbol_group[idx][3]['rebalance_day_price'] - self.best_symbol_group[idx][3]['close']
                self.best_symbol_group[idx][3] = pd.merge(
                    self.best_symbol_group[idx][3], start_datehandler.metrics, how='outer', on='symbol')
                self.best_symbol_group[idx][3] = pd.merge(
                    self.best_symbol_group[idx][3], start_datehandler.fs, how='outer', on='symbol' )
                for feature in self.best_symbol_group[idx][3].columns:
                    feature_rank_col_name = feature + "_rank"
                    self.best_symbol_group[idx][3][feature_rank_col_name] = \
                        self.best_symbol_group[idx][3][feature].rank(method='min')
                self.best_symbol_group[idx][3] = self.best_symbol_group[idx][3].sort_values(
                    by=["period_price_diff"], axis=0, ascending=False)

                syms = best_group['symbol']
                for sym in syms:
                    if start_datehandler.price.loc[(start_datehandler.price['symbol']==sym), 'close'].empty:
                        self.best_symbol_group[idx][2].loc[(self.best_symbol_group[idx][2].symbol == sym), 'price'] = 0
                    else:
                        self.best_symbol_group[idx][2].loc[(self.best_symbol_group[idx][2].symbol == sym), 'price']\
                        = start_datehandler.price.loc[(start_datehandler.price['symbol']==sym), 'close'].values[0]

                    if end_datehandler.price.loc[(end_datehandler.price['symbol']==sym), 'close'].empty:
                        self.best_symbol_group[idx][2].loc[(self.best_symbol_group[idx][2].symbol == sym), 'rebalance_day_price'] = 0
                    else:
                        self.best_symbol_group[idx][2].loc[(self.best_symbol_group[idx][2].symbol == sym), 'rebalance_day_price']\
                            = end_datehandler.price.loc[(end_datehandler.price['symbol']==sym), 'close'].values[0]
                
            start_datehanler = end_datehandler
            # print(idx, " ", date, "\n", self.best_symbol_group[idx][2])

    def cal_earning(self): 
        """backtest로 계산한 plan의 수익률을 계산하는 함수"""
        base_asset = self.total_asset
        prev = 0
        best_asset = -1
        worst_asset = self.total_asset * 1000
        for idx, (date, rebalance_date, best_group, reference_group) in enumerate(self.best_symbol_group):
            # TODO best_symbol_group 맞게 사고 남은 짜투리 금액 처리
            stock_cnt = (self.total_asset / self.member_cnt) / best_group['price']
            stock_cnt = stock_cnt.replace([np.inf, -np.inf], 0)
            stock_cnt = stock_cnt.fillna(0)
            stock_cnt = stock_cnt.astype(int)
            price_mul_stock_cnt = best_group['price'] * stock_cnt
            my_asset_period = price_mul_stock_cnt.sum()
            remain_asset = self.total_asset - price_mul_stock_cnt.sum()
            
            if my_asset_period == 0:
                continue

            # MDD 계산을 위해 이 구간에서 각 종목별 구매 개수 저장
            self.best_symbol_group[idx][2]['count'] = stock_cnt
            
            # rebalance date의 가격으로 구매한 종목들 판매했을 때 자산 계산
            rebalance_day_price_mul_stock_cnt = best_group['rebalance_day_price'] * stock_cnt
            self.best_symbol_group[idx][2]['period_earning'] = rebalance_day_price_mul_stock_cnt - price_mul_stock_cnt
            period_earning = rebalance_day_price_mul_stock_cnt.sum() - price_mul_stock_cnt.sum()
            
            prev = self.total_asset
            self.total_asset = remain_asset + rebalance_day_price_mul_stock_cnt.sum()

            # print("date : ", date, "\nbest group : \n")
            # print(best_group[['symbol', 'price', 'rebalance_day_price', 'count']])
            print("cur idx : {} prev : {} earning : {:.2f} asset : {}".format(idx, idx-1, period_earning,
                                                                              self.total_asset))
            # print(best_group['symbol', 'price', 'rebalance_day_price'])

            self.historical_earning_per_rebalanceday.append([date, period_earning, prev, self.total_asset, best_group])

            # for rebalanced day price based MDD
            if self.total_asset > best_asset:
                best_asset = my_asset_period
            if self.total_asset < worst_asset:
                worst_asset = my_asset_period
        print("rebalanced day price based MDD : {:.2f} %".format(((worst_asset / best_asset) - 1) * 100))

    def cal_mdd(self, price_table):
        """MDD를 계산해서 채워주는 함수"""
        best_asset = -1
        worst_asset = self.total_asset * 100000
        for i, (date, rebalance_date, best_group, reference_group) in enumerate(self.best_symbol_group):
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
        print("MDD : {:.2f}%, best date : {}, worst date : {}".format(mdd, best_date, worst_date))
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
        eval_columns = ["symbol", "score", "price", "rebalance_day_price", "count", "period_earning",
                        "pbRatio", "pbRatio_rank", "pbRatio_score", "peRatio", "peRatio_rank", "peRatio_score",
                        "ipoDate", "delistedDate"]
        for idx, (date, rebalance_date, eval_elem, rank_elem) in enumerate(self.best_symbol_group):
            if self.backtest.conf['PRINT_EVAL_REPORT'] == 'Y' and self.backtest.conf['NEED_EVALUATION'] == 'Y':
                self.write_csv(self.backtest.eval_report_path, date, rebalance_date, eval_elem, eval_columns)
            if self.backtest.conf['PRINT_RANK_REPORT'] == 'Y':
                self.write_csv(self.backtest.rank_report_path, date, rebalance_date, rank_elem, rank_elem.columns.tolist())
            # period.to_csv(self.backtest.eval_report_path, mode="a", column=columns)

        ref_total_earning_rates = dict()
        for ref_sym in self.backtest.conf['REFERENCE_SYMBOL']:
            start_date = self.backtest.get_trade_date(datetime.datetime(self.backtest.main_ctx.start_year, 1, 1))
            end_date = self.backtest.get_trade_date(datetime.datetime(self.backtest.main_ctx.end_year, 12, 31))
            reference_earning_df = self.backtest.price_table.query("(symbol == @ref_sym) and ((date == @start_date) or (date == @end_date))")
            reference_earning = reference_earning_df.iloc[0]['close'] - reference_earning_df.iloc[1]['close']
            ref_total_earning_rate = (reference_earning / reference_earning_df.iloc[1]['close']) * 100
            ref_total_earning_rates[ref_sym] = ref_total_earning_rate

        plan_earning = self.historical_earning_per_rebalanceday [ len(self.historical_earning_per_rebalanceday)-1 ][3]\
                        - self.historical_earning_per_rebalanceday[0][2]
        plan_total_earning_rate = (plan_earning / self.historical_earning_per_rebalanceday[0][2]) * 100

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
            self.cal_sharp()
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
