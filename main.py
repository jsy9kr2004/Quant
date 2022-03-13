import sqlalchemy
import yaml

from backtest import Backtest, PlanHandler
from database import Database
from fmp import FMP


class MainCtx:
    def __init__(self, config):
        self.start_year = int(config['START_YEAR'])
        self.end_year = int(config['END_YEAR'])
        self.root_path = config['ROOT_PATH']
        # 다른 Class와 함수에서 connection이 자주 필요하기에 Databse Class 로 관리하지 않고 main_context로 관리
        aws_mariadb_url = 'mysql+pymysql://' + config['MARIA_DB_USER'] + ":" + config['MARIA_DB_PASSWD'] + "@" \
                          + config['MARIA_DB_ADDR'] + ":" + config['MARIA_DB_PORT'] + "/" + config['MARIA_DB_NAME']
        self.conn = sqlalchemy.create_engine(aws_mariadb_url)
        self.cre_tbl_list = ["FULL_LIST", "TMP_FULL_LIST", "PRICE", "FINANCIAL_STATEMENT", "METRICS", "INDEXES"]


def get_config():
    with open('config/conf.yaml') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == '__main__':
    conf = get_config()
    main_ctx = MainCtx(conf)

    if conf['NEED_NEW_GET_FMP'] == "Y":
        fmp = FMP(conf, main_ctx)
        fmp.get_new()

    if conf['NEED_NEW_CREATE_DB'] == "N":
        db = Database(main_ctx)
        db.insert_csv()
        db.rebuild_table_view()

    plan_handler = PlanHandler()
    plan = [
        {"f_name": plan_handler.pbr, "params": {"weight": 1, "diff": 2, "base": 3}},
        {"f_name": plan_handler.per, "params": {"w": 4, "d": 5, "b": 6}}
    ]
    plan_handler.plan_list = plan
    bt = Backtest(main_ctx, plan_handler, rebalance_period=3)

    ################################################################################################
    # (1) tickers를 이용한 재무재표 예제
    # import yfinance as yf
    # import yahoo_fin.stock_info as si
    # from pykrx import stock
    # import pymysql
    # symbol = 'GOOGL'
    # sp500_ticker = si.tickers_sp500()
    # print(si.get_balance_sheet(symbol))

    # (2) MultiProcessing 예제
    # from multiprocessing import Pool
    # pool = Pool(8)
    # return = pool.map(method, parameters)
    # sp500_info = pool.map(, sp500_tickers)

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
