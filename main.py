import logging
import os
import pandas as pd
import sqlalchemy
import yaml

from backtest import Backtest, PlanHandler
from database import Database
from parquet import Parquet
from fmp import FMP
from regressor import Regressor, RegressionNetwork


class MainCtx:
    def __init__(self, config):
        self.start_year = int(config['START_YEAR'])
        self.end_year = int(config['END_YEAR'])
        self.root_path = config['ROOT_PATH']
        self.need_pq_new_year = config['NEED_NEWYEAR_CSV_TO_PQ']
        # 다른 Class와 함수에서 connection이 자주 필요하기에 Databse Class 로 관리하지 않고 main_context로 관리
        # aws_mariadb_url = 'mysql+pymysql://' + config['MARIA_DB_USER'] + ":" + config['MARIA_DB_PASSWD'] + "@" \
        #                   + config['MARIA_DB_ADDR'] + ":" + config['MARIA_DB_PORT'] + "/" + config['MARIA_DB_NAME']
        # self.conn = sqlalchemy.create_engine(aws_mariadb_url)

    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            logging.info('Creating Folder "{}" ...'.format(path))
            try:
                os.makedirs(path)
                return True
            except OSError:
                logging.error('Cannot Creating "{}" directory.'.format(path))
                return False


def get_config():
    with open('config/conf.yaml') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def set_logger(config):
    log_path = "log.txt"
    # if os.path.exists(log_path):
    #    os.remove(log_path)
    logging.basicConfig(level=config['LOG_LVL'],
                        format='[%(asctime)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d) ',
                        handlers=[logging.FileHandler(log_path, mode='a+'), logging.StreamHandler()])


def conf_check(config):
    # REPORT 종류는 EVAL, RANK, AI, AVG 뿐
    for rep_type in config['REPORT_LIST']:
        if rep_type != "EVAL" and rep_type != "RANK" and rep_type != "AI" and rep_type != "AVG":
            logging.critical("Only REPORT_LIST : EVAL, RANK, AI, AVG")
            exit()


if __name__ == '__main__':
    conf = get_config()
    set_logger(conf)
    conf_check(conf)
    main_ctx = MainCtx(conf)

    if conf['RUN_REGRESSION'] == "Y":
        regor = Regressor(conf)
        regor.dataload()
        regor.train()
        latest_data_path = conf['ROOT_PATH'] + '/regressor_data/' + '2022_2_regressor_train.csv'
        regor.latest_prediction(latest_data_path)
        # MLP = RegressionNetwork(conf)
        # MLP.mtrain()
        exit()

    main_ctx.create_dir("./reports")
    if conf['NEED_NEW_GET_FMP'] == "Y":
        fmp = FMP(conf, main_ctx)
        fmp.get_new()

    if conf['USE_DB'] == "Y":
        db = Database(main_ctx)
        if conf['NEED_INSERT_CSV_TO_DB'] == "Y":
            db.insert_csv()
        if conf['NEED_NEW_VIEW_DB'] == "Y":
            db.rebuild_table_view()
    elif conf['USE_DATAFRAME'] == 'Y':
        df_engine = Parquet(main_ctx)
        if conf['NEED_INSERT_CSV_TO_PQ'] == "Y" or conf['NEED_NEWYEAR_CSV_TO_PQ'] == "Y":
            df_engine.insert_csv()
        if conf['NEED_NEW_VIEW_PQ'] == "Y":
            df_engine.rebuild_table_view()
    else:
        logging.error("Check conf.yaml. don't choose db and parquet both")

    plan_handler = PlanHandler(conf['TOP_K_NUM'], conf['ABSOLUTE_SCORE'])
    plan = []
    plan_df = pd.read_csv("./plan.csv")
    plan_info = plan_df.values.tolist()
    for i in range(len(plan_info)):
        plan.append(
            {"f_name": plan_handler.single_metric_plan,
             "params": {"key": plan_info[i][0],
                        "key_dir": plan_info[i][1], "weight": plan_info[i][2],
                        "diff": plan_info[i][3], "base": plan_info[i][4], "base_dir": plan_info[i][5]}}
        )
    plan_handler.plan_list = plan

    for mem_cnt in range(10, 51, 10):
        for top_k_num in range(400, 3201, 400):
            for score_ratio in range(0, 201, 20):
                conf['TOP_K_NUM'] = top_k_num
                conf['MEMBER_CNT'] = mem_cnt
                conf['ABSOLUTE_SCORE'] = int(top_k_num * 10 * (1 + score_ratio / 100))
                logging.warning("TOP_K_NUM : " + str(conf['TOP_K_NUM']) + ", MEMBER_CNT : " + str(conf['MEMBER_CNT']) +
                                ", ABSOLUTE_SCORE : " + str(conf['ABSOLUTE_SCORE']))
    # TOP_K_NUM: 1500
    # MEMBER_CNT: 20
    # ABSOLUTE_SCORE: 30000
    bt = Backtest(main_ctx, conf, plan_handler, rebalance_period=conf['REBALANCE_PERIOD'])

    ################################################################################################
    # (1) tickers를 이용한 재무재표 예제
    # import yfinance as yf
    # import yahoo_fin.stock_info as si
    # from pykrx import stock
    # import pymysql
    # symbol = 'GOOGL'
    # sp500_ticker = si.tickers_sp500()
    # print(si.get_balance_sheet(symbol))

    # (3) DataReader 예제
    # import FinanceDataReader as fdr
    # symbol = 'GOOGL'
    # web = 'yahoo'
    # start_date = '2004-08-19'
    # end_date = '2020-04-17'
    # google_data = data.DataReader(symbol, web, start_date, end_date)
    # print(google_data.head(9))
    # google_data['Close'].plot()
    # df = stock.get_market_fundamental("20220104", "20220206", "005930", freq="m")

    # (4) dataframe을 이용한 merge 예시
    # import numpy as np
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

    # (5) chart 그리기
    # import mariadb
    # import matplotlib.pyplot as plt
    ################################################################################################
