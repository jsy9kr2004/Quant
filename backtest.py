import pandas as pd

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

CHUNK_SIZE = 20480


class Backtest:
    def __init__(self, main_ctx, plan_handler, rebalance_period):
        self.main_ctx = main_ctx
        self.plan_handler = plan_handler
        self.rebalance_period = rebalance_period
        self.eval_handler = EvaluationHandler()
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
        date = datetime(self.main_ctx.start_year, 1, 1)
        while date <= datetime(self.main_ctx.end_year, 12, 31):
            self.plan_handler.date_handler = DateHandler(self, date)
            self.plan_handler.run()
            self.eval_handler.set_best_symbol_group(date, self.plan_handler.date_handler.symbol_list)
            # day를 기준으로 하려면 아래를 사용하면 됨. 31일 기준으로 하면 우리가 원한 한달이 아님
            # date += relativedelta(days=self.rebalance_period)
            date += relativedelta(months=self.rebalance_period)
        self.eval_handler.run()


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
        assert self.plan_list is not None, "Empty Plan List"
        assert self.date_handler is not None, "Empty Date Handler"
        for plan in self.plan_list:
            plan["f_name"](plan["params"])

    def single_metric_plan(self, params):
        """single metric(PBR, PER ... )에 따라 plan_handler.date_handler.symbol_list의 score column에 값을 갱신해주는 함수."""
        print("[pbr] key : {}, weight : {}, diff : {}, base : {}, base_dir : {}".\
              format(params["key"], params["weight"], params["diff"], params["base"], params["base_dir"]))
        key = str(params["key"])
        topK_df = self.date_handler.metrics.sort_values(by=[key], ascending=True)[:20]
        # TODO: 아래 if 문에서 loc[조건, column명] 으로 조건에 맞는 row의 column 값을 갱신하고자 할 때, 변수인 key로 접근하면 오류 ('pbRatio'로 적으면 정상 작동)
        # TOOD: 위 오류를 잡던지, 우회하는 방법으로 base활용은 for문 밖에서 df 자르는 것으로 바꾼다
        if params["base_dir"] == ">":
            topK_df[topK_df[key] > params["base"]]
        elif params["base_dir"] == "<":
            topK_df[topK_df[key] < params["base"]]
            #if self.date_handler.symbol_list.loc[(self.date_handler.symbol_list.symbol == sym), key] < params["base"]:
        else:
            print("Wrong params['base_dir'] : ", params["base_dir"], " params['base_dir'] must be '>' or '<' ")
            return    
        print(topK_df[['symbol', params["key"]]])
        symbols = topK_df['symbol']
        delta = 100
        for sym in symbols:
            prev_score = self.date_handler.symbol_list[self.date_handler.symbol_list['symbol'] == sym]['score']
            self.date_handler.symbol_list.loc[(self.date_handler.symbol_list.symbol == sym), 'score'] = prev_score + params["weight"] * delta
            delta = delta - params["diff"]
        print(self.date_handler.symbol_list[['symbol', 'score']])


    def per(self, params):
        """PER에 따라 plan_handler.date_handler.symbol_list의 score column에 값을 갱신해주는 함수."""
        print("[per] weight : {}, diff : {}, base : {}".format(params['w'], params['d'], params['b']))


class DateHandler:
    def __init__(self, backtest, date):
        self.date = date
        query = '(date == "{}")'.format(self.date)
        self.price = self.init_by_query(backtest.price_table, query)
        # db에서 delistedDate null 이  df에서는 NaT로 들어옴.
        query = '(delistedDate >= "{}") or (delistedDate == "NaT")'.format(self.date)
        self.symbol_list = self.init_by_query(backtest.symbol_table, query)
        self.symbol_list["score"] = 0
        print(self.symbol_list)

        self.fs = self.get_date_latest_per_symbol(backtest.fs_table, self.date)
        self.metrics = self.get_date_latest_per_symbol(backtest.metrics_table, self.date)

    def get_date_latest_per_symbol(self, table, date):
        date_latest = pd.DataFrame()
        syms = self.symbol_list['symbol']
        # TODO: 모든 symbol 다 돌면 오래걸려서 10개로 줄임. 나중에 삭제
        for sym in syms[:10]:
            # TODO: date 기준에 date-3달~date로 넣기
            prev_Q_date = date - relativedelta(months=3)
            past = table.query("(symbol == @sym) and (date <= @date and date >= @prev_Q_date)")
            if past.empty:
                continue
            else:
                # past 는 date 이전 모든 fs들, 이 중 첫번째 row가 가장 최신 fs. iloc[0]로 첫 row 가져옴.
                date_latest = date_latest.append(past.iloc[0])
        # TODO: 왜 symbol 당 row가 2개씩 들어가있나 ?
        date_latest = date_latest.drop_duplicates('symbol', keep='first')
        return date_latest

    @staticmethod
    def init_by_query(table, query, need_iloc=False):
        result = table.query(query)
        if need_iloc is True:
            # 첫번째 row가 가장 최신 fs. iloc[0]로 첫 row 가져옴.
            result = result.iloc[0]
        return result


class EvaluationHandler:
    def __init__(self):
        self.best_symbol_group = []
        self.member_cnt = self.cal_member_cnt()
        self.MDD = 0
        self.sharp = 0

    @staticmethod
    def cal_member_cnt():
        """
        당시 시장 상황을 고려해서 1 period에 상위 몇 개의 주식을 살지(= 몇 달러 투자할지)를 결정하는 함수
        :return: member cnt
        """
        # TODO 현재는 상위 20개의 주식을 매 period 마다 구매하는 것으로 되어 있음
        return 20

    def set_best_symbol_group(self, date, symbol):
        """plan_handler.date_handler.symbol_list에 score를 보고 best_symbol_group에 append 해주는 함수."""
        symbol = symbol.sort_values(by=["score"], axis=0, ascending=False).head(self.member_cnt)
        symbol['price'] = 0
        self.best_symbol_group.append([date, symbol])

    def cal_price(self):
        """best_symbol_group 의 ['price'] column을 채워주는 함수"""
        pass

    def cal_mdd(self):
        """MDD를 계산해서 채워주는 함수"""
        mdd = 0
        for group in self.best_symbol_group:
            for symbol in group[1]:
                # symbol_handler = SymbolHandler(symbol, self.main_ctx.start_year, self.main_ctx.end_year)
                pass
        self.MDD = mdd

    def cal_sharp(self):
        """sharp를 계산해서 채워주는 함수"""
        sharp = 0
        self.sharp = sharp

    def run(self):
        self.cal_price()
        self.cal_mdd()
        self.cal_sharp()


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