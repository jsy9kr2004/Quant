from cmath import nan
import copy
import csv
import datetime
import logging
import multiprocessing
import os

import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from functools import reduce
from multiprocessing import Pool

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

CHUNK_SIZE = 20480

use_col_list = [
"interestCoverage",
"dividendYield",
"inventoryTurnover",
"daysPayablesOutstanding",
"stockBasedCompensationToRevenue",
"dcf",
"capexToDepreciation",
"currentRatio",
"daysOfInventoryOnHand",
"payablesTurnover",
"grahamNetNet",
"capexToRevenue",
"netDebtToEBITDA",
"receivablesTurnover",
"capexToOperatingCashFlow",
"evToOperatingCashFlow",
"evToFreeCashFlow",
"debtToAssets",
"tangibleBookValuePerShare",
"stockBasedCompensation",
"capexPerShare",
"peRatio",
"enterpriseValueOverEBITDA",
"bookValuePerShare",
"shareholdersEquityPerShare",
"pfcfRatio",
"pocfratio",
"daysSalesOutstanding",
"incomeQuality",
"interestDebtPerShare",
"revenuePerShare",
"freeCashFlowPerShare",
"evToSales",
"netIncomePerShare",
"grahamNumber",
"operatingCashFlowPerShare",
"cashPerShare",
"priceToSalesRatio",
"pbRatio",
"ptbRatio",
"investedCapital",
"roic",
"freeCashFlowYield",
"roe",
"returnOnTangibleAssets",
"earningsYield",
"debtToEquity",
"payoutRatio",
"salesGeneralAndAdministrativeToRevenue",
"intangiblesToTotalAssets",
"netDebt",
"ebitdaratio",
"ebitda",
"dividendsperShareGrowth",
"freeCashFlow",
"operatingCashFlow",
"netIncomeGrowth",
"grossProfit",
"epsgrowth",
"epsdilutedGrowth",
"revenueGrowth",
"grossProfitRatio",
"epsdiluted",
"eps",
"debtGrowth",
"tenYDividendperShareGrowthPerShare",
"netIncomeRatio",
"incomeBeforeTaxRatio",
"operatingCashFlowGrowth",
"ebitgrowth",
"operatingIncomeGrowth",
"threeYDividendperShareGrowthPerShare",
"assetGrowth",
"freeCashFlowGrowth",
"sgaexpensesGrowth",
"fiveYDividendperShareGrowthPerShare",
"receivablesGrowth",
"fiveYRevenueGrowthPerShare",
"threeYOperatingCFGrowthPerShare",
"grossProfitGrowth",
"operatingIncomeRatio",
"threeYShareholdersEquityGrowthPerShare",
"fiveYShareholdersEquityGrowthPerShare",
"fiveYOperatingCFGrowthPerShare",
"threeYRevenueGrowthPerShare",
"researchAndDdevelopementToRevenue",
"threeYNetIncomeGrowthPerShare",
"tenYOperatingCFGrowthPerShare",
"tenYRevenueGrowthPerShare",
"tenYShareholdersEquityGrowthPerShare",
"tenYNetIncomeGrowthPerShare",
"weightedAverageSharesGrowth",
"weightedAverageSharesDilutedGrowth",
"fiveYNetIncomeGrowthPerShare",
"bookValueperShareGrowth",
"inventoryGrowth",
"rdexpenseGrowth",
]

class Backtest:
    def __init__(self, main_ctx, conf, plan_handler, rebalance_period):
        self.main_ctx = main_ctx
        self.conf = conf
        self.plan_handler = plan_handler
        self.rebalance_period = rebalance_period
        self.eval_handler = EvaluationHandler(self)
        # ?????? ????????? init_bt_from_db ?????? ???????????????, ???????????? ?????? ?????????(init??? ?????? ??? calss value??? ?????? ????????????) ?????????
        self.symbol_table = ""
        self.price_table = ""
        self.fs_table = ""
        self.metrics_table = ""
        self.reload_bt_table(main_ctx.start_year)

        self.table_year = main_ctx.start_year

        self.eval_report_path = self.create_report("EVAL")
        self.rank_report_path = self.create_report("RANK")
        self.ai_report_path = self.create_report("AI")
        self.avg_report_path = self.create_report("AVG")

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
        else:
            return None

    def data_from_database(self, query):
        """
        ??????????????????????????? chunk ????????? ???????????? ???????????? ?????????
        :param query: ????????????????????? ????????? ??????
        :return: ??????????????????????????? ????????? ?????????
        """
        logging.info("Query : " + query)
        chunks = pd.read_sql_query(sql=query, con=self.main_ctx.conn, chunksize=CHUNK_SIZE)
        table = pd.DataFrame()
        for df in chunks:
            table = pd.concat([table, df])
        return table

    def reload_bt_table(self, year):
        """
        ????????? database?????? ????????? ???????????? ?????? ??? ???????????? __init__ ???????????? ???????????? ??????, ?????? ???????????? ???????????? ???
        ?????? ????????? ????????? init????????? ???????????? ??????????????? ????????? ????????? ??? ???????????? ??? ????????? ?????? ?????????
        init?????? ???????????? ?????? ????????? ????????? ??????. ???????????? year??? ?????? ??????????????? ??????
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
        """???????????? ?????? ?????? ????????? ??????????????? ??????"""
        # pdate =  pdate.date()
        post_date = pdate + relativedelta(days=4)

        res = self.price_table.query("date <= @post_date and date >=@pdate")
        if res.empty:
            return None
        else:
            return res.iloc[-1].date

    def run(self):
        """
        main?????? ???????????? plan list??? ?????? backtest??? ???????????? ????????? ?????? 2?????? ????????? ???????????? ??????
        [1] date ?????? plan list??? ?????? plan_handler.date_handler.score??? ???????????????,
            ????????? symbol??? self.best_k??? ??????
        [2] best_k ?????? ???????????? MDD??? ??????????????? ?????? ?????? ????????? ????????? ??????????????? ???????????? ?????? ??????
        ?????? ?????? ????????? backtest class, plan_handler (1:1 mapping)??? ??????????????? ????????? ????????? main
        date_handler??? ?????? ??????????????? ?????? ????????? backtest?????? ?????? ???
        backtest?????? ???????????? mapping?????? ?????? plan_handler?????? ?????????.
        """
        date = datetime.datetime(self.main_ctx.start_year, self.conf['START_MONTH'], self.conf['START_DATE'])
        recent_date = self.price_table["date"].max()
        while True:
            if date.year != self.table_year:
                logging.info("Reload BackTest table. year : {} -> {}".format(self.table_year, date.year))
                self.reload_bt_table(date.year)
                self.table_year = date.year
            tdate = self.get_trade_date(date)
            # get_trade_date ?????? price table ??? ???????????? ????????? reload_bt_table??? ?????? ???????????? ???
            if tdate is None:
                break
            logging.info("Backtest Run : " + str(tdate.strftime("%Y-%m-%d")))

            if self.eval_report_path is not None:
                self.plan_handler.date_handler = DateHandler(self, tdate)
                logging.debug("complete set date_handler date : {}".format(tdate.strftime("%Y-%m-%d")))
                self.plan_handler.run()

            if date != recent_date:
                self.eval_handler.set_best_k(tdate, date+relativedelta(months=self.rebalance_period),
                                             self.plan_handler.date_handler)
            else:
                self.eval_handler.print_current_best(self.plan_handler.date_handler)
                break
            # day??? ???????????? ????????? ????????? ???????????? ???. 31??? ???????????? ?????? ????????? ?????? ????????? ??????
            # date += relativedelta(days=self.rebalance_period)
            if date <= datetime.datetime(self.main_ctx.end_year, 12 - self.rebalance_period, 30):
                date += relativedelta(months=self.rebalance_period)
            else:
                # ????????? loop ??? ???????????? ?????? date ??? ?????? ????????? print ?????? ?????? ????????? ?????? ???????????? ???
                if self.eval_report_path is not None:
                    date = recent_date
                # evaluation report ??? ?????? ????????? current ????????? ???????????? ????????? ???
                else:
                    break

        if self.rank_report_path is not None or self.eval_report_path is not None or self.ai_report_path is not None:
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
        main?????? ????????? ??????(function pointer)??? ???????????? ??????
        plan_list??? date_handler??? ???????????? ??????????????? ???.
        ????????? plan_list??? main, date_handler??? Backtest ?????? ?????????.
        """
        assert self.plan_list is not None, "Empty Plan List"
        assert self.date_handler is not None, "Empty Date Handler"

        with Pool(processes=multiprocessing.cpu_count()-2) as pool:
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
        """single metric(PBR, PER ... )??? ?????? plan_handler.date_handler.symbol_list??? score column??? ?????? ??????????????? ??????.
           params??? plan??? parameter?????? ????????? ?????? ?????????
           params["key"]        : plan?????? ????????? ????????? ??????(ex: PER, PBR, ???????????? ...)
           params["key_dir"]    : ????????? ???????????? ?????????(low) ???????????? ?????????(high)
           params["weight"]     : score update??? weight
           params["diff"]       : ??? ????????? ??? ????????? score ??????
           params["base"]       : ?????? threshold ??????/????????? ????????? score ?????? ??????
           params["base_dir"]   : "base"??? ??? threshold ??????/??????(</>) ??????
        """
        logging.info("[plan] key : {}, key_dir : {}, weight : {}, "
                     "diff : {}, base : {}, base_dir : {}".format(params["key"], params["key_dir"], params["weight"],
                                                                  params["diff"], params["base"], params["base_dir"]))

        if self.absolute_score - params["diff"] * self.k_num < 0:
            logging.warning("Wrong params['diff'] : TOO BIG! SET UNDER " + str(self.absolute_score/self.k_num))

        key = str(params["key"])

        # all feature was preprocessed ( high is good ) in Datehandler
        top_k_df = self.date_handler.fs_metrics.sort_values(by=[key], ascending=False, na_position="last")[:self.k_num]
        top_k_df = self.date_handler.fs_metrics.sort_values(by=[key+"_sorted"], ascending=False,
                                                                na_position="last")[:self.k_num]

        # if params["base_dir"] == ">":
        #     top_k_df = top_k_df[top_k_df[key] > params["base"]]
        # elif params["base_dir"] == "<":
        #     top_k_df = top_k_df[top_k_df[key] < params["base"]]
        # else:
        #     logging.error("Wrong params['base_dir'] : ", params["base_dir"], " params['base_dir'] must be '>' or '<'")
        #     return

        logging.debug(top_k_df[['symbol', params["key"]]])

        symbols = top_k_df['symbol']
        return_df = self.date_handler.symbol_list[['symbol']]
        delta = self.absolute_score
        # ???????????? ??????
        pd.set_option('mode.chained_assignment', None)
        local_score_name = key + '_score'
        for sym in symbols:
            return_df.loc[(self.date_handler.symbol_list.symbol == sym), local_score_name]\
                = params["weight"] * delta
            delta = delta - params["diff"]
        local_rank_name = key+'_rank'
        return_df[local_rank_name] = return_df[local_score_name].rank(method='min', ascending=False)
        logging.debug(return_df[[local_score_name, local_rank_name]])
        return return_df


class DateHandler:
    def __init__(self, backtest, date):
        pd.set_option('mode.chained_assignment', None)

        logging.info("in datehandler date : ")
        logging.info(date)
        
        self.date = date
        # date = datetime.datetime.combine(date, datetime.datetime.min.time())
        # query = '(date == "{}")'.format(self.date)
        # db?????? delistedDate null ???  df????????? NaT??? ?????????.
        query = '(delistedDate >= "{}") or (delistedDate == "NaT") or (delistedDate == "None")'.format(date)
        self.symbol_list = backtest.symbol_table.query(query)
        self.symbol_list = self.symbol_list.assign(score=0)

        trade_date = backtest.get_trade_date(date)
        self.price = backtest.price_table.query("date == @trade_date")
        self.price = self.price.drop_duplicates('symbol', keep='first')

        # self.price = self.get_date_latest_per_symbol(backtest.price_table, self.date)
        self.symbol_list = pd.merge(self.symbol_list, self.price, how='left', on='symbol')

        # filter volume
        # self.symbol_list = self.symbol_list.drop(index=self.symbol_list[self.symbol_list['volume'] < 10000].index)
        self.symbol_list = self.symbol_list[self.symbol_list.volume > 10000]
        del self.price

        prev = self.date - relativedelta(months=3)
        # self.fs = self.get_date_latest_per_symbol(backtest.fs_table, self.date)
        self.fs = backtest.fs_table.copy()
        self.fs = self.fs[self.fs.fillingDate <= self.date]
        self.fs = self.fs[prev <= self.fs.fillingDate]
        self.fs = self.fs.drop_duplicates('symbol', keep='first')

        self.metrics = backtest.metrics_table.copy()
        
        self.fs_metrics = pd.merge(self.fs, self.metrics, how='outer', on=['symbol', 'date'])
        
        del self.metrics
        del self.fs

        highlow = pd.read_csv('./sort.csv', header=0)
        for feature in self.fs_metrics.columns:
            feature_sortedvalue_col_name = feature + "_sorted"
            self.fs_metrics[feature_sortedvalue_col_name] = self.fs_metrics[feature]

            # ?????? ??????
            f = highlow.query("name == @feature")
            if f.empty:
                continue
            else:
                if f.iloc[0].sort == "low":
                    try:
                        feat_max = self.fs_metrics[feature].max()
                        self.fs_metrics[feature_sortedvalue_col_name] = \
                            [s*(-1) if s >= 0 else (s - feat_max) for s in self.fs_metrics[feature]]
                    except Exception as e:
                        logging.info(str(e))
                        continue

            # normalization ( 0~20000 ). range is not fixed
            feature_normal_col_name = feature + "_normal"
            try:
                max_value = self.fs_metrics[feature_sortedvalue_col_name].max()
                min_value = self.fs_metrics[feature_sortedvalue_col_name].min()
                self.fs_metrics[feature_normal_col_name] \
                    = (((self.fs_metrics[feature_sortedvalue_col_name] - min_value) * 20000) / (max_value - min_value))
            except Exception as e:
                logging.info(str(e))
                continue


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
        """?????? ??? ????????? ????????? ???????????? ?????? ??????. ????????? ?????? 4?????? ????????? ??? period ?????? ???????????? ????????? ?????? ??????"""
        return self.backtest.conf['MEMBER_CNT']

    def print_current_best(self, scored_dh):
        best_symbol_info = pd.merge(scored_dh.symbol_list, scored_dh.fs_metrics, how='left', on='symbol')
        # best_symbol_info = pd.merge(best_symbol_info, scored_dh.fs, how='left', on='symbol')
        best_symbol = best_symbol_info.sort_values(by=["score"], axis=0, ascending=False).head(self.member_cnt)
        best_symbol = best_symbol.assign(count=0)
        best_symbol.to_csv('./result.csv')

    def set_best_k(self, date, rebalance_date, scored_dh):
        """plan_handler.date_handler.symbol_list??? score??? ?????? best_k??? append ????????? ??????."""
        if self.backtest.eval_report_path is not None:
            best_symbol_info = pd.merge(scored_dh.symbol_list, scored_dh.fs_metrics, how='left', on='symbol')
            # best_symbol_info = pd.merge(best_symbol_info, scored_dh.fs, how='left', on='symbol')
            best_symbol = best_symbol_info.sort_values(by=["score"], axis=0, ascending=False).head(self.member_cnt)
            # best_symbol = best_symbol.assign(price=0)
            best_symbol = best_symbol.assign(count=0)
        else:
            best_symbol = pd.DataFrame()

        reference_group = pd.DataFrame()
        period_earning_rate = 0
        self.best_k.append([date, rebalance_date, best_symbol, reference_group, period_earning_rate])

    def cal_price(self):
        pd.set_option('mode.chained_assignment', None)

        """best_k ??? ['price', 'rebalance_day_price'] column??? ???????????? ??????"""
        for idx, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            if date.year != self.backtest.table_year:
                logging.info("Reload BackTest Table. year : {} -> {}".format(self.backtest.table_year, date.year))
                self.backtest.reload_bt_table(date.year)
                self.backtest.table_year = date.year
                            
            if idx == 0:
                start_dh = DateHandler(self.backtest, date)
            end_dh = DateHandler(self.backtest, rebalance_date)

            if self.backtest.rank_report_path is not None or self.backtest.ai_report_path is not None:
                self.best_k[idx][3] = start_dh.symbol_list
                rebalance_date_price_df = end_dh.symbol_list[['symbol', 'close']]
                rebalance_date_price_df.rename(columns={'close':'rebalance_day_price'}, inplace=True)
                self.best_k[idx][3] = pd.merge(self.best_k[idx][3], rebalance_date_price_df, how='outer', on='symbol')
                self.best_k[idx][3] = self.best_k[idx][3][self.best_k[idx][3].close > 0.000001]
                diff = self.best_k[idx][3]['rebalance_day_price'] - self.best_k[idx][3]['close']
                self.best_k[idx][3]['period_price_diff'] = diff / self.best_k[idx][3]['close']
                self.best_k[idx][3] = pd.merge(self.best_k[idx][3], start_dh.fs_metrics, how='left', on='symbol')
                # self.best_k[idx][3] = pd.merge(self.best_k[idx][3], start_dh.fs, how='left', on='symbol')

                if self.backtest.conf['PRINT_AI'] == 'Y':
                    df_for_reg = self.best_k[idx][3].copy()
                    df_for_reg = df_for_reg[use_col_list]
                    print("in print_AI")
                    print(df_for_reg)
                    df_for_reg['period_price_diff']  = self.best_k[idx][3]['period_price_diff']
                    df_for_reg['symbol']  = self.best_k[idx][3]['symbol']


                    df_for_reg['earning_diff'] \
                        = df_for_reg['period_price_diff'] - df_for_reg['period_price_diff'].mean()
                    print(df_for_reg)
                    # normal_col_list = self.best_k[idx][3].columns.str.contains("_normal")
                    # df_for_reg = self.best_k[idx][3].loc[:,normal_col_list]
                    # df_for_reg['earning_diff'] = self.best_k[idx][3]['earning_diff']
                    # df_for_reg['symbol'] = self.best_k[idx][3]['symbol']

                    # remove outlier
                    logging.info("before removing outlier # rows : " + str(df_for_reg.shape[0]))
                    logging.info("before removing outlier # columns : " + str(df_for_reg.shape[1]))
                    for col in use_col_list:
                        try:
                            # removing outlier with IQR
                            # candi 1
                            # Q1 = np.nanpercentile(df_for_reg[col], 10)
                            # Q3 = np.nanpercentile(df_for_reg[col], 90)
                            # print("col : ", col , "Q1: ", Q1, "Q3 :", Q3)
                            # IQR = Q3 - Q1
                            # # 0.5 is not fixed.   reference :  1.5 => remove 0.7%,  0 =>  remove 50%
                            # outlier_step = 1.5*IQR
                            # outlier_list_col = df_for_reg[(df_for_reg[col] < (Q1 - outlier_step))
                            #                                     | (df_for_reg[col] > (Q3 + outlier_step))].index
                            # if outlier_list_col.shape[0] < 200:
                            #     df_for_reg = df_for_reg.drop(index=outlier_list_col, axis=0)

                            # candi 2
                            Q1 = np.nanpercentile(df_for_reg[col], 0.5)
                            Q3 = np.nanpercentile(df_for_reg[col], 99.5)
                            print("col : ", col , "Q1: ", Q1, "Q3 :", Q3)
                            IQR = Q3 - Q1
                            outlier_step = 0*IQR
                            outlier_list_col = df_for_reg[(df_for_reg[col] < (Q1 - outlier_step))
                                                                | (df_for_reg[col] > (Q3 + outlier_step))].index
                            if outlier_list_col.shape[0] < 200:
                                df_for_reg = df_for_reg.drop(index=outlier_list_col, axis=0)


                            # removing by count
                            # MID = self.fs_metrics[col].median()
                            # if (MID == nan) or (MID.isnan()):
                            #     print("MID is nan col : ", col)
                            #     continue
                            # logging.debug("start removing outlier col : " + col +  " mid : " + str(MID))

                            # threshold = 200
                            # while True:
                            #     outlier_list_col = self.fs_metrics[(self.fs_metrics[col] < (MID - threshold))
                            #                                     | (self.fs_metrics[col] > (MID + threshold))].index

                                # if outlier_list_col.shape[0] <= 200:

                                #     # logging.debug("outlier undercut col cnt : " +  str(outlier_list_col.shape[0]) + " threshold : "+ str(threshold))
                                #     # print(outlier_list_col.shape)
                                #     threshold -= 5
                                # elif outlier_list_col.shape[0] > 1000:
                                #     # logging.debug("outlier overcut col cnt : " + str(outlier_list_col.shape[0]) + " threshold : " + str(threshold))
                                #     # print(outlier_list_col.shape)
                                #     threshold += 5
                                # if (outlier_list_col.shape[0] <= 1000) & (outlier_list_col.shape[0] > 200) :
                                #     logging.debug("@@@@@@@@ proper outlier col cnt : " +  str(outlier_list_col.shape[0])+ " threshold : " + str(threshold))
                                #     print(outlier_list_col.shape)
                                #     break
                                # if threshold <= 0:
                                #     print(outlier_list_col.shape)
                                #     break
                                # if threshold > 1000:
                                #     logging.debug("there is no proper threshold in col : " + col)
                                #     print(outlier_list_col.shape)
                                #     outlier_list_col = []
                                #     break

                            # self.fs_metrics = self.fs_metrics.drop(index=outlier_list_col, axis=0)

                        except Exception as e:
                            logging.info(str(e))
                            continue
                    logging.info("after removing outlier # rows : " + str(df_for_reg.shape[0]))
                    logging.info("after removing outlier # columns : " + str(df_for_reg.shape[1]))

                    # ?????? ??????
                    for col in use_col_list:
                        highlow = pd.read_csv('./sort.csv', header=0)
                        f = highlow.query("name == @col")
                        if f.empty:
                            continue
                        else:
                            if f.iloc[0].sort == "low":
                                try:
                                    feat_max = df_for_reg[col].max()
                                    df_for_reg[col] = \
                                        [s*(-1) if s >= 0 else (s - feat_max) for s in df_for_reg[col]]
                                except Exception as e:
                                    logging.info(str(e))
                                    continue

                    for col in use_col_list:
                        feature_normal_col_name = col + "_normal"
                        try:
                            max_value = df_for_reg[col].max()
                            min_value = df_for_reg[col].min()
                            df_for_reg[feature_normal_col_name] \
                                = ((df_for_reg[col] - min_value)) / (max_value - min_value)
                        except Exception as e:
                            logging.info(str(e))
                            continue

                    normal_col_list = df_for_reg.columns.str.contains("_normal")
                    df_for_reg_print = df_for_reg.loc[:,normal_col_list]
                    df_for_reg_print['period_price_diff'] = df_for_reg['period_price_diff']
                    df_for_reg_print['earning_diff'] = df_for_reg['earning_diff']
                    df_for_reg_print['symbol'] = df_for_reg['symbol']

                    traindata_path = self.backtest.conf['ROOT_PATH'] + '/regressor_data/'
                    df_for_reg_print.to_csv(traindata_path + '{}_{}_regressor_train.csv'.format(date.year, date.month), index=False)

                if self.backtest.rank_report_path is not None:
                    for feature in self.best_k[idx][3].columns:
                        feature_rank_col_name = feature + "_rank"
                        self.best_k[idx][3][feature_rank_col_name] \
                            = self.best_k[idx][3][feature].rank(method='min', ascending=False)
                self.best_k[idx][3] \
                    = self.best_k[idx][3].sort_values(by=["period_price_diff"],
                                                      axis=0, ascending=False)[:self.backtest.conf['TOP_K_NUM']]
            else:
                self.best_k[idx][3] = pd.DataFrame()

            if self.backtest.eval_report_path is not None:
                syms = best_group['symbol']
                for sym in syms:
                    if start_dh.symbol_list.loc[(start_dh.symbol_list['symbol'] == sym), 'close'].empty:
                        logging.debug("there is no price in FMP API  symbol : {}".format(sym))
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'price'] = 0
                    else:
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'price']\
                            = start_dh.symbol_list.loc[(start_dh.symbol_list['symbol'] == sym), 'close'].values[0]

                    if end_dh.symbol_list.loc[(end_dh.symbol_list['symbol'] == sym), 'close'].empty:
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'rebalance_day_price'] = 0
                    else:
                        self.best_k[idx][2].loc[(self.best_k[idx][2].symbol == sym), 'rebalance_day_price']\
                            = end_dh.symbol_list.loc[(end_dh.symbol_list['symbol'] == sym), 'close'].values[0]
                logging.debug(str(self.best_k[idx][2]))

                # self.best_k[idx][2] = self.best_k[idx][2][self.best_k[idx][2].symbol_list > 0.000001]

            start_dh = copy.deepcopy(end_dh)
            logging.debug(str(idx) + " " + str(date))

    def cal_earning(self):
        """backtest??? ????????? plan??? ???????????? ???????????? ??????"""
        base_asset = self.total_asset
        prev = 0
        best_asset = -1
        worst_asset = self.total_asset * 1000
        for idx, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            # TODO best_k ?????? ?????? ?????? ????????? ?????? ??????
            stock_cnt = (self.total_asset / len(best_group)) / best_group['price']
            stock_cnt = stock_cnt.replace([np.inf, -np.inf], 0)
            stock_cnt = stock_cnt.fillna(0)
            stock_cnt = stock_cnt.astype(int)
            price_mul_stock_cnt = best_group['price'] * stock_cnt
            my_asset_period = price_mul_stock_cnt.sum()
            remain_asset = self.total_asset - price_mul_stock_cnt.sum()
            
            if my_asset_period == 0:
                continue

            # MDD ????????? ?????? ??? ???????????? ??? ????????? ?????? ?????? ??????
            self.best_k[idx][2]['count'] = stock_cnt
            
            # rebalance date??? ???????????? ????????? ????????? ???????????? ??? ?????? ??????
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
        """MDD??? ???????????? ???????????? ??????"""
        best_asset = -1
        worst_asset = self.total_asset * 100000
        for i, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            if i == 0:
                prev_date = date
                continue
            else:
                # prev_date ~ date ?????? ?????? date??? ?????? ?????? ?????? ??????
                allday_price_allsymbol = []
                syms = best_group['symbol']

                # symbol ?????? rebalancing day ???????????? prev_date ~ date ??? price ?????? ????????????,
                # rebalancing day??? ????????? symbol ??? ?????? ??? column??? 'count' ??? 'close' ?????? ????????? ????????? ?????? ?????? ??????
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
                # ??? ????????? ?????? ????????? ?????? ????????? ?????? ????????? ??????
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
        """sharp??? ???????????? ???????????? ??????"""
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
        for idx, (date, rebalance_date, eval_elem, rank_elem, period_earning_rate) in enumerate(self.best_k):
            if self.backtest.eval_report_path is not None:
                self.write_csv(self.backtest.eval_report_path, date, rebalance_date, eval_elem)
                fd = open(self.backtest.eval_report_path, 'a')
                writer = csv.writer(fd, delimiter=",")
                writer.writerow(str(period_earning_rate))
                fd.close()
            if self.backtest.rank_report_path is not None:
                if idx <= self.backtest.conf['RANK_PERIOD']:
                    self.write_csv(self.backtest.rank_report_path, date, rebalance_date, rank_elem)
            # period.to_csv(self.backtest.eval_report_path, mode="a", column=columns)

        if self.backtest.eval_report_path is not None:
            ref_total_earning_rates = dict()
            for ref_sym in self.backtest.conf['REFERENCE_SYMBOL']:
                start_date = self.backtest.get_trade_date(datetime.datetime(self.backtest.main_ctx.start_year, 1, 1))
                end_date = self.backtest.get_trade_date(datetime.datetime(self.backtest.main_ctx.end_year, 12, 31))
                reference_earning_df = self.backtest.price_table.query(
                    "(symbol == @ref_sym) and ((date == @start_date) or (date == @end_date))")
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
        if self.backtest.eval_report_path is not None:
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
