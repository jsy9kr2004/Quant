import logging
import os
import pandas as pd


from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv

class Parquet:
    def __init__(self, main_ctx):
        self.main_ctx = main_ctx
        self.tables = dict()

    def csv2pq(self, df, path, tablename):
        csv_save_path = path + tablename + ".csv"
        pq_save_path = path + tablename + ".parquet"
        df.to_csv(csv_save_path)
        pq.write_table(csv.read_csv(csv_save_path), pq_save_path)
        

    def read_from_pq(self):
        dir_list = os.listdir(self.main_ctx.root_path)
        
        for directory in dir_list:
            
            csv_path = self.main_ctx.root_path + "/" + directory + "/" + directory + ".csv"
            pq_path = self.main_ctx.root_path + "/" + directory + "/" + directory + ".parquet"
            if directory == 'stock_list' or directory == 'symbol_available_indexes':
                pq.write_table(csv.read_csv(csv_path), pq_path)
            self.tables[directory] = pq.read_table(source=pq_path).to_pandas()
            logging.info("read from pq : {}".format(directory))
            # self.tables[directory] = pd.read_parquet(pq_path, engine='pyarrow')

    def rebuild_table_view(self):
        # 1번 Table
        symbol_list = self.tables['stock_list'][['symbol', 'exchangeShortName', 'type']]      
        delisted = self.tables['delisted_companies'][['symbol', 'exchange', 'ipoDate', 'delistedDate']]
        profile = self.tables['profile'][['symbol', 'ipoDate', 'industry', 'exchangeShortName']]
        
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
        self.tables['symbol_list'] = all_symbol
        logging.info("create symbol_list df")
        
        self.csv2pq(all_symbol, self.main_ctx.root_path + "/", "symbol_list")


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

        # 2번 Table
        price = self.tables['historical_price_full'][['date', 'symbol', 'close', 'volume']]
        marketcap = self.tables['historical_market_capitalization'][['date', 'symbol', 'marketCap']]
        price_marketcap = pd.merge(price, marketcap, how='left', on=['symbol', 'date'])
        self.tables['price'] = price_marketcap
        del price
        del marketcap
        self.csv2pq(price_marketcap, self.main_ctx.root_path + "/", "price")
        logging.info("create price df")

        
        # query = "CREATE TABLE PRICE " \
        #         " AS SELECT a.date, a.symbol, a.open, a.high, a.low, a.close, a.volume, " \
        #         " b.marketCap" \
        #         " FROM historical_price_full a, historical_market_capitalization b" \
        #         " WHERE a.symbol = b.symbol" \
        #         " AND a.date = b.date"
        # logging.info(query)
        # self.main_ctx.conn.execute(query)

        # 3번 Table

        self.tables['financial_statement'] = self.tables['income_statement'].merge(
                                                self.tables['balance_sheet_statement'],
                                                how='outer', on=['date', 'symbol']).merge(
                                                    self.tables['cash_flow_statement'], 
                                                    how='outer', on=['date', 'symbol']
                                                )
        self.csv2pq(self.tables['financial_statement'], self.main_ctx.root_path + "/", "financial_statement")

        # query = "CREATE VIEW FINANCIAL_STATEMENT" \
        #         " AS SELECT a.date, a.symbol, a.reportedCurrency, a.fillingDate, a.acceptedDate, a.calendarYear," \
        #         " a.period, a.revenue, a.grossProfit, a.ebitda, a.operatingIncome, a.netIncome, a.eps, a.epsdiluted," \
        #         " a.weightedAverageShsOut, a.weightedAverageShsOutDil," \
        #         " b.inventory AS bs_inventory, b.totalCurrentAssets, b.totalNonCurrentAssets, b.totalAssets," \
        #         " b.accountPayables as bs_accountPayables, b.totalCurrentLiabilities, b.totalNonCurrentLiabilities," \
        #         " b.totalLiabilities, b.totalEquity, b.totalDebt, b.netDebt," \
        #         " c.inventory as cf_inventory, c.accountsPayables as cf_accountsPayables," \
        #         " c.commonStockIssued, c.commonStockRepurchased, c.dividendsPaid, c.netChangeInCash," \
        #         " c.cashAtEndOfPeriod, c.operatingCashFlow, c.capitalExpenditure, c.freeCashFlow, a.link, a.finalLink" \
        #         " FROM income_statement a, balance_sheet_statement b, cash_flow_statement c " \
        #         " WHERE a.symbol = b.symbol AND b.symbol = c.symbol" \
        #         " AND a.date = b.date AND b.date = c.date;"
        # self.main_ctx.conn.execute(query)
        logging.info("create financial_statement df")
        

        # 4번 Table
        # query = "CREATE VIEW METRICS" \
        #         " AS SELECT a.date, a.symbol, a.period, a.netIncomePerShare, a.marketCap, a.enterpriseValue," \
        #         " a.peRatio, a.priceToSalesRatio, a.pbRatio, a.enterpriseValueOverEBITDA, a.debtToEquity," \
        #         " a.dividendYield, a.payoutRatio, a.netCurrentAssetValue, a.roe, a.capexPerShare," \
        #         " b.revenueGrowth, b.grossProfitGrowth, b.ebitgrowth, b.operatingIncomeGrowth, b.netIncomeGrowth," \
        #         " b.epsgrowth, b.epsdilutedGrowth, b.dividendsperShareGrowth, b.operatingCashFlowGrowth," \
        #         " b.freeCashFlowGrowth, b.assetGrowth, b.bookValueperShareGrowth, b.debtGrowth, c.dcf" \
        #         " FROM key_metrics a, financial_growth b, historical_daily_discounted_cash_flow c" \
        #         " WHERE a.symbol = b.symbol AND b.symbol = c.symbol" \
        #         " AND a.date = b.date AND b.date = c.date;"
        # self.main_ctx.conn.execute(query)
        self.tables['key_metrics']
        self.tables['financial_growth']
        self.tables['historical_daily_discounted_cash_flow']
        
        self.tables['metrics'] = self.tables['key_metrics'].merge(
                                        self.tables['financial_growth'],
                                        how='outer', on=['date', 'symbol']).merge(
                                            self.tables['historical_daily_discounted_cash_flow'], 
                                            how='outer', on=['date', 'symbol']
                                        )
        
        self.csv2pq(self.tables['metrics'], self.main_ctx.root_path + "/", "metrics")

        logging.info("create metrics df")

        # 5번 Table
        self.tables['indexes'] = self.tables['symbol_available_indexes']
        # query = "ALTER TABLE symbol_available_indexes RENAME INDEXES;"
        # self.main_ctx.conn.execute(query)
        self.csv2pq(self.tables['indexes'], self.main_ctx.root_path + "/", "indexes")

        logging.info("create indexes df")
        

    @staticmethod
    def read_csv_mp(filename):
        'converts a filename to a pandas dataframe'
        return csv.read_csv(filename).to_pandas()
    
    def insert_csv(self):
        # wrap your csv importer in a function that can be mapped
        # merge all csvs per directoy
        dir_list = os.listdir(self.main_ctx.root_path)
        logging.info("directory list : {}".format(dir_list))
        for directory in dir_list:    
            csv_save_path = self.main_ctx.root_path + "/" + directory + "/" + directory + ".csv"
            pq_save_path = self.main_ctx.root_path + "/" + directory + "/" + directory + ".parquet"
            if (directory != 'stock_list') and (directory != 'symbol_available_indexes'):
                if os.path.exists(csv_save_path):
                    os.remove(csv_save_path)
                if os.path.exists(pq_save_path):
                    os.remove(pq_save_path)
                
            file_list = [self.main_ctx.root_path + "/" + directory + "/" +file for file in os.listdir(self.main_ctx.root_path + "/" + directory) if file.endswith(".csv")] 
            full_df = pd.DataFrame()
            
            # using multi-processing 
            with Pool(processes=8) as pool: # or whatever your hardware can support
                # have your pool map the file names to dataframes
                df_list = pool.map(self.read_csv_mp, file_list)
                # reduce the list of dataframes to a single dataframe
                full_df = pd.concat(df_list, ignore_index=True)
            
            # using single processing
            # for file in file_list:
            #     # tmp_df = pd.read_csv(self.main_ctx.root_path + "/" + directory + "/" + file, index_col=None)
            #         tmp_df = csv.read_csv(self.main_ctx.root_path + "/" + directory + "/" + file).to_pandas()
            #         # drop index column
            #         tmp_df = tmp_df.drop(tmp_df.columns[0], axis=1)
            #         full_df = pd.concat([full_df, tmp_df])
            
            full_df = full_df.drop(full_df.columns[0], axis=1)
            self.tables[directory] = full_df
            full_df.to_csv(csv_save_path, index=False)
            pq.write_table(csv.read_csv(csv_save_path), pq_save_path)
            logging.info("create df in tables dict : {}".format(directory))
            # check_df = pd.read_parquet(save_path, engine='pyarrow')
    
                        
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

