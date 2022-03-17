import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta


class Backtest:
    def __init__(self, main_ctx, plan_handler, rebalance_period):
        self.main_ctx = main_ctx
        self.best_symbol_group = []
        self.plan_handler = plan_handler
        self.rebalance_period = rebalance_period
        self.MDD = 0
        self.sharp = 0
        self.run()

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
        chunks = pd.read_sql_query(sql=query,
                                   con=self.main_ctx.conn, chunksize=20480)
        table_price = pd.DataFrame()
        for df in chunks:
            table_price = pd.concat([table_price, df])    

        # table "symbol_list" 을 table_symbol dataframe 으로 땡겨옴
        query = "SELECT * FROM symbol_list"
        chunks = pd.read_sql_query(sql=query,
                                   con=self.main_ctx.conn, chunksize=20480)
        table_symbol = pd.DataFrame()
        for df in chunks:
            table_symbol = pd.concat([table_symbol, df])    

        # table "financial_statment" 을 table_fs dataframe 으로 땡겨옴
        query = "SELECT * FROM financial_statement"
        chunks = pd.read_sql_query(sql=query,
                                   con=self.main_ctx.conn, chunksize=20480)
        table_fs = pd.DataFrame()
        for df in chunks:
            table_fs = pd.concat([table_fs, df])  


        date = datetime(self.main_ctx.start_year, 1, 1)
        while date <= datetime(self.main_ctx.end_year, 12, 31):
            self.plan_handler.date_handler = DateHandler(table_price, table_symbol, table_fs, date)
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

    def per(self, params):
        """PER에 따라 plan_handler.date_handler.symbol_list의 score column에 값을 갱신해주는 함수."""
        print("[per] weight : {}, diff : {}, base : {}".format(params['w'], params['d'], params['b']))


class DateHandler:
    def __init__(self, table_price, table_symbol, table_fs, date):
        self.table_price = table_price
        self.table_symbol = table_symbol
        self.table_fs = table_fs
        self.date = date
        self.price = self.get_date_price(self.table_price)
        self.symbol_list = self.get_date_symbol_list(self.table_symbol)
        self.add_score_column()
        self.financial_statement = self.get_date_fs(self.table_fs)

    def get_date_price(self, table):
        query_str = "date == @self.date"
        date_price = table.query(query_str)
        return date_price

    def get_date_symbol_list(self, table):
        # db에서 delistedDate null 이  df에서는 NaT로 들어옴.
        query_str = "(delistedDate >= @self.date) or (delistedDate == 'NaT')"
        date_symbol = table.query(query_str)
        return date_symbol

    def add_score_column(self):
        """get_date_symbol_list 함수에서 dataframe 으로 가져온 symbol_list에 score column을 추가해 주는 함수"""
        return self.date

    def get_date_fs(self, table):
        query_str = "(date <= @self.date)"
        past_fs = table.query(query_str)
        # past_fs 는 date 이전 모든 fs들, 이 중 첫번째 row가 가장 최신 fs. iloc[0]로 첫 row 가져옴.
        date_fs = past_fs.iloc[0]
        return date_fs


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
