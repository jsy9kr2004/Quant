import os

import pandas as pd
import sqlalchemy

class Database:
    # maria DB 첫 설치 시 아래의 SQL문으로 db와 user를 만들어줘야함
    # use mysql;
    # create database quantdb
    # create user 'quant'@'%' identified by '0710';
    # grant privileges on quantdb.* to 'quant'@'localhost';
    # flush privileges
    def __init__(self, main_ctx):
        self.main_ctx = main_ctx

    def rebuild_table_view(self):
        # 1번 Table
        dfs_symbol_list = pd.read_sql_query(sql="SELECT symbol, exchangeShortName, type FROM stock_list",
                                            con=self.main_ctx.conn, chunksize=20480)
        symbol_list = pd.DataFrame()
        for df in dfs_symbol_list:
            symbol_list = pd.concat([symbol_list, df])
        dfs_delist = pd.read_sql_query(sql="SELECT symbol, exchange, ipoDate, delistedDate FROM delisted_companies",
                                       con=self.main_ctx.conn, chunksize=20480)
        delisted = pd.DataFrame()
        for df in dfs_delist:
            delisted = pd.concat([delisted, df])
        dfs_profile = pd.read_sql_query(sql="SELECT symbol, ipoDate, industry, exchangeShortName FROM profile",
                                        con=self.main_ctx.conn, chunksize=20480)
        profile = pd.DataFrame()
        for df in dfs_profile:
            profile = pd.concat([profile, df])

        delisted.rename(columns={'exchange':'exchangeShortName'}, inplace=True)
        # concat (symbol_list, delisted companies)
        all_symbol = pd.concat([symbol_list, delisted])
        all_symbol = all_symbol.drop_duplicates('symbol', keep='last')

        # merge ((symbol_list, delisted companies), profile)
        all_symbol = all_symbol.merge(profile, how='left', on=['symbol', 'exchangeShortName'])
        all_symbol['ipoDate'] = all_symbol['ipoDate_x'].combine_first(all_symbol['ipoDate_y'])
        all_symbol = all_symbol.drop(['ipoDate_x', 'ipoDate_y'], axis=1)
        all_symbol = all_symbol.drop_duplicates('symbol', keep='last')
        all_symbol = all_symbol[(all_symbol['exchangeShortName'] == 'NASDAQ')
                                | (all_symbol['exchangeShortName'] == 'NYSE')]
        all_symbol = all_symbol.reset_index(drop=True)
        all_symbol.to_sql("symbol_list", self.main_ctx.conn,
                          if_exists='append', index=False, index_label=None, chunksize=512)

        # query = "ALTER TABLE full_list ADD PRIMARY KEY (symbol);"
        # self.main_ctx.conn.execute(query)
        # query = "SELECT c.symbol, c.exchangeShortName, c.type, d.delistedDate," \
        #        " CASE c.ipoDate" \
        #        " IS NULL THEN d.ipoDate" \
        #        " ELSE c.ipoDate" \
        #        " END as ipoDate" \
        #        " FROM (SELECT a.symbol, a.exchangeShortName, a.type, b.industry, b.ipoDate" \
        #        " FROM stock a LEFT OUTER JOIN profile b on a.symbol = b.symbol) c" \
        #        " LEFT OUTER JOIN delisted_companies d on c.symbol = d.symbol;"
        #        # CASE WHEN table3.col3 IS NULL THEN table2.col3 ELSE table3.col3 END as col4
        # result = pd.read_sql_query(sql=query, con=self.main_ctx.conn)
        # print(result)

        # 2번 Table
        query = "CREATE TABLE PRICE " \
                " AS SELECT a.date, a.symbol, a.open, a.high, a.low, a.close, a.volume, " \
                " b.marketCap" \
                " FROM historical_price_full a, historical_market_capitalization b" \
                " WHERE a.symbol = b.symbol" \
                " AND a.date = b.date"
        self.main_ctx.conn.execute(query)

        # 3번 Table
        query = "CREATE VIEW FINANCIAL_STATEMENT" \
                " AS SELECT a.date, a.symbol, a.reportedCurrency, a.fillingDate, a.acceptedDate, a.calendarYear," \
                " a.period, a.revenue, a.grossProfit, a.ebitda, a.operatingIncome, a.netIncome, a.eps, a.epsdiluted," \
                " a.weightedAverageShsOut, a.weightedAverageShsOutDil," \
                " b.inventory AS bs_inventory, b.totalCurrentAssets, b.totalNonCurrentAssets, b.totalAssets," \
                " b.accountPayables as bs_accountPayables, b.totalCurrentLiabilities, b.totalNonCurrentLiabilities," \
                " b.totalLiabilities, b.totalEquity, b.totalDebt, b.netDebt," \
                " c.inventory as cf_inventory, c.accountsPayables as cf_accountsPayables," \
                " c.commonStockIssued, c.commonStockRepurchased, c.dividendsPaid, c.netChangeInCash," \
                " c.cashAtEndOfPeriod, c.operatingCashFlow, c.capitalExpenditure, c.freeCashFlow, a.link, a.finalLink" \
                " FROM income_statement a, balance_sheet_statement b, cash_flow_statement c " \
                " WHERE a.symbol = b.symbol AND b.symbol = c.symbol" \
                " AND a.date = b.date AND b.date = c.date;"
        self.main_ctx.conn.execute(query)

        # 4번 Table
        query = "CREATE VIEW METRICS" \
                " AS SELECT a.date, a.symbol, a.period, a.netIncomePerShare, a.marketCap, a.enterpriseValue," \
                " a.peRatio, a.priceToSalesRatio, a.pbRatio, a.enterpriseValueOverEBITDA, a.debtToEquity," \
                " a.dividendYield, a.payoutRatio, a.netCurrentAssetValue, a.roe, a.capexPerShare," \
                " b.revenueGrowth, b.grossProfitGrowth, b.ebitgrowth, b.operatingIncomeGrowth, b.netIncomeGrowth," \
                " b.epsgrowth, b.epsdilutedGrowth, b.dividendsperShareGrowth, b.operatingCashFlowGrowth," \
                " b.freeCashFlowGrowth, b.assetGrowth, b.bookValueperShareGrowth, b.debtGrowth, c.dcf" \
                " FROM key_metrics a, financial_growth b, historical_daily_discounted_cash_flow c" \
                " WHERE a.symbol = b.symbol AND b.symbol = c.symbol" \
                " AND a.date = b.date AND b.date = c.date;"
        self.main_ctx.conn.execute(query)

        # 5번 Table
        query = "ALTER TABLE symbol_available_indexes RENAME INDEXES;"
        self.main_ctx.conn.execute(query)

    def insert_csv(self):
        # Drop All Tables
        dir_list = os.listdir(self.main_ctx.root_path)
        for table in dir_list + self.main_ctx.cre_tbl_list:
            query = "DROP TABLE IF EXISTS {} RESTRICT;".format(table)
            self.main_ctx.conn.execute(query)
            # TODO 안예쁘다 바꾸자. VIEW 인지 TABLE인지 몰라서 이렇게 두번 날림
            query = "DROP VIEW IF EXISTS {} RESTRICT;".format(table)
            self.main_ctx.conn.execute(query)

        # Inster All CSV file
        for directory in dir_list:
            file_list = os.listdir(self.main_ctx.root_path + "/" + directory)
            for file in file_list:
                target = pd.read_csv(self.main_ctx.root_path + "/" + directory + "/" + file, index_col=None)
                # drop index column
                target = target.drop(target.columns[0], axis=1)
                target = target.reset_index(drop=True)
                try:
                    target.to_sql(directory, self.main_ctx.conn,
                              if_exists='append', index=False, index_label=None, chunksize=512)
                except sqlalchemy.exc.DataError:
                    print("error {} table".format(directory)) 
                print("Complete creation of {} table".format(directory))

        params = [
            ['income_statement', 'date'], ['income_statement', 'fillingDate'], ['income_statement', 'acceptedDate'],
            ['balance_sheet_statement', 'date'], ['balance_sheet_statement', 'fillingDate'],
            ['balance_sheet_statement', 'acceptedDate'],
            ['cash_flow_statement', 'date'], ['cash_flow_statement', 'fillingDate'],
            ['cash_flow_statement', 'acceptedDate'],
            ['key_metrics', 'date'], ['financial_growth', 'date'], ['historical_daily_discounted_cash_flow', 'date'],
            ['earning_calendar', 'date'], ['profile', 'ipoDate'], ['historical_market_capitalization', 'date'],
            ['delisted_companies', 'ipoDate'], ['delisted_companies', 'delistedDate'],
            ['historical_price_full', 'date']
        ]
        for param in params:
            query = "ALTER TABLE {} MODIFY {} DATETIME;".format(str(param[0]), str(param[1]))
            print(query)
            self.main_ctx.conn.execute(query)
