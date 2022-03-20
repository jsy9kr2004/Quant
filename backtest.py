import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta

CHUNK_SIZE = 20480

class Backtest:
    def __init__(self, main_ctx, plan_handler, rebalance_period):
        self.main_ctx = main_ctx
        self.best_symbol_group = []
        self.plan_handler = plan_handler
        self.rebalance_period = rebalance_period
        self.MDD = 0
        self.sharp = 0
        # 아래 값들은 init_bt_from_db 에서 세팅해주나, 가려지는 값이 없도록(init만 봤을 때 calss value가 모두 보이도록) 나열함
        self.price_table = ""
        self.symbol_table = ""
        self.fs_table = ""
        self.metrics_table = ""
        self.init_bt_from_db()
        self.run()

    def data_from_database(self, query):
        """
        데이터베이스로부터 chunk 단위로 테이블을 읽어오고 반환함
        :param query: 데이터베이스에 전송할 쿼리
        :return: 데이터베이스로부터 읽어온 테이블
        """
        chunks = pd.read_sql_query(sql=query, con=self.main_ctx.conn, chunksize=CHUNK_SIZE)
        table = pd.DataFrame()
        for df in chunks:
            table = pd.concat([table, df])
        return table

    def init_bt_from_db(self):
        """
        추후에 database에서 가져올 데이터가 많을 걸 대비해서 __init__ 함수에서 세팅하지 않고, 해당 함수에서 세팅토록 함
        일부 필요한 내용한 init하거나 분할해서 가져오려고 한다면 쿼리가 더 복잡해질 수 있기에 따로 빼놓음
        """
        # TODO 추후에 database에서 가져올 테이블이 많다면, set_bt_from_db 와 같은 함수 안에서 처리 예정
        query = "SELECT * FROM PRICE WHERE date BETWEEN '" \
                + str(datetime(self.main_ctx.start_year, 1, 1)) + "'" \
                + " AND '" + str(datetime(self.main_ctx.end_year, 12, 31)) + "'"
        self.price_table = self.data_from_database(query)
        self.symbol_table = self.data_from_database("SELECT * FROM symbol_list")
        self.symbol_table["score"] = 0
        self.fs_table = self.data_from_database("SELECT * FROM financial_statement")
        self.metrics_table = self.data_from_database("SELECT * FROM METRICS")

    def run(self):
        """
        main에서 전달받은 plan list에 따른 backtest를 진행하는 함수로 크게 2개의 파트로 구분되어 있음
        [1] date 별로 plan list에 따라 plan_handler.date_handler.score를 계산해넣고, 상위권 symbol을 self.best_symbol_group에 추가
        [2] best_symbol_group 에서 가져와서 MDD나 샤프지수와 같은 전략 전체에 필요한 계산값들을 계산해서 채워 넣음
        전략 별로 하나의 backtest class, plan_handler (1:1 mapping)가 만들어지며 생성의 주체는 main
        date_handler는 다수 만들어지며 생성 주체는 backtest이며 생성 후 backtest에서 본인에게 mapping되어 있는 plan_handler에게 달아줌.
        """
        # table "PRICE" 을 table_price dataframe 으로 땡겨옴
        query = "SELECT * FROM PRICE WHERE date BETWEEN '" \
                + str(datetime(self.main_ctx.start_year, 1, 1)) + "'" \
                + " AND '" + str(datetime(self.main_ctx.end_year, 12, 31)) + "'"
        table_price = self.data_from_database(query)
        query = "SELECT * FROM symbol_list"
        table_symbol = self.data_from_database(query)
        query = "SELECT * FROM financial_statement"
        table_fs = self.data_from_database(query)
        query = "SELECT * FROM METRICS"
        table_metrics = self.data_from_database(query)

        date = datetime(self.main_ctx.start_year, 1, 1)
        while date <= datetime(self.main_ctx.end_year, 12, 31):
            self.plan_handler.date_handler = DateHandler(self, date)
            print(self.plan_handler.date_handler.date)
            self.plan_handler.run()
            self.set_best_symbol_group()
            # day를 기준으로 하려면 아래를 사용하면 됨. 31일 기준으로 하면 우리가 원한 한달이 아님
            # date += relativedelta(days=self.rebalance_period)
            date += relativedelta(months=self.rebalance_period)
        print(self.best_symbol_group)
        self.calculate_metrics()

    def set_best_symbol_group(self):
        """plan_handler.date_handler.symbol_list에 score를 보고 best_symbol_group에 append 해주는 함수."""
        self.best_symbol_group.append([self.plan_handler.date_handler.date, ["AAPL", "TESL"]])

    def calculate_metrics(self):
        """MDD와 샤프지수와 같은 전략 평가를 위한 지표 계산을 위한 함수"""
        mdd = 0
        sharp = 0
        for group in self.best_symbol_group:
            for symbol in group[1]:
                symbol_handler = SymbolHandler(symbol, self.main_ctx.start_year, self.main_ctx.end_year)
                # mdd = self.calcuate_mdd(symbol_handler.price)
        self.MDD = mdd
        self.sharp = sharp


class PlanHandler:
    def __init__(self):
        self.plan_list = None
        self.date_handler = None

    def run(self):
        """
        main에서 짜여진 전략(function pointer)을 순서대로 호출
        plan_list와 date_handler를 채워주고 불러워줘야 함.
        현재는 plan_list는 main, date_handler는 Backtest 에서 채워줌.
        """
        # TODO plan_list와 date_handler가 차있는지 확인하는 assert 함수가 있어도 좋을 듯
        for plan in self.plan_list:
            plan["f_name"](plan["params"])

    def pbr(self, params):
        """PBR에 따라 plan_handler.date_handler.symbol_list의 score column에 값을 갱신해주는 함수."""
        print("[pbr] weight : {}, diff : {}, base : {}".format(params["weight"], params["diff"], params["base"]))
        print("pbr : ", self.date_handler.metrics['pbRatio'])
        print("metrics : ", self.date_handler.metrics)
        prev_score = self.date_handler.symbol_list["score"]
        self.date_handler.symbol_list["score"] = prev_score + params["weight"] * self.date_handler.metrics['pbRatio']

    def per(self, params):
        """PER에 따라 plan_handler.date_handler.symbol_list의 score column에 값을 갱신해주는 함수."""
        print("[per] weight : {}, diff : {}, base : {}".format(params['w'], params['d'], params['b']))
        print("per : ", self.date_handler.metrics['peRatio'])


class DateHandler:
    def __init__(self, backtest, date):
        self.date = date
        query = "date == @self.date"
        self.price = self.init_by_query(backtest.price_table, query)
        query = "(date <= @self.date)"
        self.fs = self.init_by_query(backtest.fs_table, query, True)
        self.metrics = self.init_by_query(backtest.metrics_table, query, True)
        # db에서 delistedDate null 이  df에서는 NaT로 들어옴.
        query_str = "(delistedDate >= @self.date) or (delistedDate == 'NaT')"
        self.symbol_list = self.init_by_query(backtest.symbol_table, query)

    def add_score_column(self):
        """get_date_symbol_list 함수에서 dataframe 으로 가져온 symbol_list에 score column을 추가해 주는 함수"""
        self.symbol_list["score"]=0

    def get_date_fs(self, table):
        date_fs = pd.DataFrame()
        syms = self.symbol_list['symbol']
        print(syms)
        for sym in syms:
            query_str = "(symbol == @sym) and (date <= @self.date)"
            past_fs = table.query(query_str)
            if past_fs.empty:
                continue
            else:
                # past_fs 는 date 이전 모든 fs들, 이 중 첫번째 row가 가장 최신 fs. iloc[0]로 첫 row 가져옴.
                date_fs = pd.concat([date_fs, past_fs.iloc[0]])
        return date_fs

    def get_date_metrics(self, table):
        date_metrics = pd.DataFrame()
        syms = self.symbol_list['symbol']
        for sym in syms:
            query_str = "(symbol == @sym) and (date <= @self.date)"
            past_metrics = table.query(query_str)
            if past_metrics.empty:
                continue
            else:
                # past_fs 는 date 이전 모든 fs들, 이 중 첫번째 row가 가장 최신 fs. iloc[0]로 첫 row 가져옴.
                date_metrics = pd.concat([date_metrics, past_metrics.iloc[0]])
        return date_metrics

    @staticmethod
    def init_by_query(table, query, need_iloc = False):
        result = table.query(query)
        if need_iloc is True:
            # 첫번째 row가 가장 최신 fs. iloc[0]로 첫 row 가져옴.
            result = result.iloc[0]
        return result


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