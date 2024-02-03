import datetime
import logging
import os

import multiprocessing as mp
import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
# import numpy as np
# from dateutil.relativedelta import relativedelta
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler
# from pyarrow import csv
from tqdm import tqdm


class Parquet:
    def __init__(self, main_ctx):
        global PQPATH
        self.main_ctx = main_ctx
        self.tables = dict()
        self.view_path = self.main_ctx.root_path + "/VIEW/"
        self.rawpq_path = self.main_ctx.root_path + "/parquet/"
        PQPATH = self.main_ctx.root_path + "/parquet/"

        self.main_ctx.create_dir(self.view_path)
        self.main_ctx.create_dir(self.rawpq_path)
        self.main_ctx.create_dir(PQPATH)

    def rebuild_table_view(self):
        # 1번 Table
        # symbol_list = pd.read_parquet(self.rawpq_path + "stock_list.parquet",
        #                               columns=['symbol', 'exchangeShortName', 'type'])
        # delisted = pd.read_parquet(self.rawpq_path + "delisted_companies.parquet",
        #                            columns=['symbol', 'exchange', 'ipoDate', 'delistedDate'])
        # profile = pd.read_parquet(self.rawpq_path + "profile.parquet",
        #                           columns=['symbol', 'ipoDate', 'industry', 'exchangeShortName'])
        symbol_list = pd.read_csv(self.rawpq_path + "stock_list.csv", usecols=['symbol', 'exchangeShortName', 'type'])
        delisted = pd.read_csv(self.rawpq_path + "delisted_companies.csv",
                               usecols=['symbol', 'exchange', 'ipoDate', 'delistedDate'])
        profile = pd.read_csv(self.rawpq_path + "profile.csv",
                              usecols=['symbol', 'ipoDate', 'industry', 'exchangeShortName'])
        delisted.rename(columns={'exchange':'exchangeShortName'}, inplace=True)

        # concat (symbol_list, delisted companies)
        all_symbol = pd.concat([symbol_list, delisted])
        all_symbol = all_symbol.drop_duplicates('symbol', keep='first')
        # merge ((symbol_list, delisted companies), profile)
        all_symbol = all_symbol.merge(profile, how='left', on=['symbol', 'exchangeShortName'])
        all_symbol['ipoDate'] = all_symbol['ipoDate_x'].combine_first(all_symbol['ipoDate_y'])
        all_symbol = all_symbol.drop(['ipoDate_x', 'ipoDate_y'], axis=1)
        all_symbol = all_symbol.drop_duplicates('symbol', keep='first')
        all_symbol = all_symbol[(all_symbol['exchangeShortName'] == 'NASDAQ')
                                | (all_symbol['exchangeShortName'] == 'NYSE')]

        # ['symbol_list', 'ipoDate'], ['symbol_list', 'delistedDate']
        all_symbol['ipoDate'] = all_symbol['ipoDate'].astype('datetime64[ns]')
        all_symbol['delistedDate'] = all_symbol['delistedDate'].astype('datetime64[ns]')
        
        all_symbol = all_symbol.reset_index(drop=True)
        # all_symbol.to_parquet(self.view_path + "symbol_list.parquet", engine="pyarrow", compression="gzip")
        all_symbol.to_csv(self.view_path + "symbol_list.csv", index=False)
        logging.info("create symbol_list df")
        del all_symbol

        # 2번 Table
        # price = pd.read_parquet(self.rawpq_path + "historical_price_full.parquet",
        #                         columns=['date', 'symbol', 'close', 'volume'])
        # marketcap = pd.read_parquet(self.rawpq_path + "historical_market_capitalization.parquet",
        #                             columns=['date', 'symbol', 'marketCap'])
        price = pd.read_csv(self.rawpq_path + "historical_price_full.csv",
                            usecols=['date', 'symbol', 'close', 'volume'])
        marketcap = pd.read_csv(self.rawpq_path + "historical_market_capitalization.csv",
                                usecols=['date', 'symbol', 'marketCap'])
        price_marketcap = pd.merge(price, marketcap, how='left', on=['symbol', 'date'])
        del price
        del marketcap
        invalid_dates = price_marketcap[~price_marketcap['date'].str.match(r'^\d{4}-\d{2}-\d{2}$')]
        # print(invalid_dates)
        # ['price', 'date']
        price_marketcap['date'] = price_marketcap['date'].astype('datetime64[ns]')
        # price_marketcap.to_parquet(self.view_path + "price.parquet", engine="pyarrow", compression="gzip")
        price_marketcap.to_csv(self.view_path + "price.csv", index=False)
        
        logging.info("create price df")
        # for year in range(self.main_ctx.start_year - 1, self.main_ctx.end_year + 1):
        #     price_peryear = price_marketcap[price_marketcap['date'].between(datetime.datetime(year, 1, 1),
        #                                                                     datetime.datetime(year, 12, 31))]
        #     price_peryear.to_parquet(self.view_path + "price_" + str(year) + ".parquet",
        #                              engine="pyarrow", compression="gzip")
        # logging.info("create price parquet per year")
        # del price_peryear
        del price_marketcap

        # 3번 Table
        # income_statement = pd.read_parquet(self.rawpq_path + "income_statement.parquet")
        # balance_sheet_statement = pd.read_parquet(self.rawpq_path + "balance_sheet_statement.parquet")
        # cash_flow_statement = pd.read_parquet(self.rawpq_path + "cash_flow_statement.parquet")
        income_statement = pd.read_csv(self.rawpq_path + "income_statement.csv")
        balance_sheet_statement = pd.read_csv(self.rawpq_path + "balance_sheet_statement.csv")
        cash_flow_statement = pd.read_csv(self.rawpq_path + "cash_flow_statement.csv")
    
        financial_statement = income_statement.merge(balance_sheet_statement,
                                                     how='outer', on=['date', 'symbol']).merge(cash_flow_statement,
                                                                                               how='outer',
                                                                                               on=['date', 'symbol'])
        financial_statement['date'] = pd.to_datetime(financial_statement['date'], errors='coerce')
        # financial_statement['date'] = financial_statement['date'].astype('datetime64[ns]')
        # financial_statement['acceptedDate'] = financial_statement['acceptedDate'].astype('datetime64[ns]')
        financial_statement['fillingDate'] = financial_statement['fillingDate'].astype('datetime64[ns]')
                                                
        # financial_statement.to_parquet(self.view_path + "financial_statement.parquet",
        #                                engine="pyarrow", compression="gzip")
        financial_statement.to_csv(self.view_path + "financial_statement.csv", index=False)
        logging.info("create financial_statement df")
        for year in range(self.main_ctx.start_year - 1, self.main_ctx.end_year + 1):
            fs_peryear = financial_statement[financial_statement['date'].between(datetime.datetime(year, 1, 1),
                                                                                 datetime.datetime(year, 12, 31))]
            # fs_peryear.to_parquet(self.view_path + "financial_statement_" + str(year) + ".parquet",
            #                       engine="pyarrow", compression="gzip")
            fs_peryear.to_csv(self.view_path + "financial_statement_" + str(year) + ".csv", index=False)
        logging.info("create financial_statement parquet per year")
        
        del income_statement
        del balance_sheet_statement
        del cash_flow_statement
        del financial_statement
        # del fs_peryear

        # 4번 Table
        # key_metrics = pd.read_parquet(self.rawpq_path + "key_metrics.parquet")
        # financial_growth = pd.read_parquet(self.rawpq_path + "financial_growth.parquet")
        # historical_daily_discounted_cash_flow = pd.read_parquet(self.rawpq_path
        #                                                         + "historical_daily_discounted_cash_flow.parquet")
        key_metrics = pd.read_csv(self.rawpq_path + "key_metrics.csv")
        financial_growth = pd.read_csv(self.rawpq_path + "financial_growth.csv")
        historical_daily_discounted_cash_flow = pd.read_csv(self.rawpq_path+"historical_daily_discounted_cash_flow.csv")
        metrics = key_metrics.merge(financial_growth,
                                    how='outer', on=['date', 'symbol']).merge(historical_daily_discounted_cash_flow,
                                                                              how='left', on=['date', 'symbol'])
        metrics['date'] = metrics['date'].astype('datetime64[ns]')
        # metrics.to_parquet(self.view_path + "metrics.parquet", engine="pyarrow", compression="gzip")
        metrics.to_csv(self.view_path + "metrics.csv", index=False)
        logging.info("create metrics df")

        for year in range(self.main_ctx.start_year - 1, self.main_ctx.end_year + 1):
            metrics_peryear = metrics[metrics['date'].between(datetime.datetime(year, 1, 1),
                                                              datetime.datetime(year, 12, 31))]
            # metrics_peryear.to_parquet(self.view_path + "metrics_" + str(year) + ".parquet",
            #                         engine="pyarrow", compression="gzip")
            metrics_peryear.to_csv(self.view_path + "metrics_" + str(year) + ".csv", index=False)
                   
        logging.info("create metrics parquet per year")
        
        del financial_growth
        del key_metrics
        del metrics
        # del metrics_peryear

        # 5번 Table
        # indexes = pd.read_parquet(self.rawpq_path + "symbol_available_indexes.parquet")
        # indexes.to_parquet(self.view_path + "indexes.parquet", engine="pyarrow", compression="gzip")
        indexes = pd.read_csv(self.rawpq_path + "symbol_available_indexes.csv")
        indexes.to_csv(self.view_path + "indexes.csv", index=False)
        logging.info("create indexes df")

    # @staticmethod
    # def read_pq_mp(self, filename):
    @staticmethod
    def read_pq_mp(filename):
        c_proc = mp.current_process()
        # PQPATH = self.main_ctx.root_path        
        csv_save_path = './' + str(c_proc.pid)+"_mp.csv"
        try:
            if filename.split('/')[1] == 'historical_price_full':
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename, usecols=['date', 'symbol', 'close', 'volume'])
                elif filename.endswith('.parquet'):
                    df = pd.read_parquet(filename, columns=['date', 'symbol', 'close', 'volume'])
            else:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename)
                elif filename.endswith('.parquet'):
                    df = pd.read_parquet(filename)
                
        except Exception as e:
            logging.info(str(e))
            return
        if not os.path.exists(csv_save_path):
            df.to_csv(csv_save_path, index=False, mode='w', encoding='utf-8-sig')
        else:
            df.to_csv(csv_save_path, index=False, mode='a', encoding='utf-8-sig', header=False)
        return
            
    def insert_csv(self):
        # wrap your csv importer in a function that can be mapped
        # merge all csvs per directoy
        # dir_list = os.listdir(self.main_ctx.root_path)
        dir_list = ["key_metrics", "stock_list", "symbol_available_indexes",
                    "balance_sheet_statement", "cash_flow_statement",
                    "delisted_companies", "earning_calendar",
                    "financial_growth", "historical_daily_discounted_cash_flow", "historical_market_capitalization",
                    "historical_price_full", "income_statement", "profile"]
        # dir_list = ["key_metrics"]
        
        logging.info("directory list : {}".format(dir_list))
        for directory in tqdm(dir_list):
            csv_save_path = self.rawpq_path + directory + ".csv"
            # pq_save_path = self.rawpq_path + directory + ".parquet"
            if (directory != 'stock_list') and (directory != 'symbol_available_indexes'):
                if os.path.exists(csv_save_path):
                    os.remove(csv_save_path)
                
            # TODO: historical price 는 year 별로 나눠서 csv 만들어 놓기 
            file_list = [self.main_ctx.root_path + "/" + directory + "/" + file
                         for file in os.listdir(self.main_ctx.root_path + "/" + directory)
                         if (file.endswith(".parquet") or file.endswith(".csv"))]

            full_df = pd.DataFrame()
            with Pool(processes=7, initializer=install_mp_handler()) as pool:
                pool.map(self.read_pq_mp, file_list)
                
            df_all_years = pd.DataFrame()
            # mp_file_list = [self.rawpq_path + file for file in os.listdir(self.rawpq_path) if file.endswith("mp.csv")]
            mp_file_list = ['./' + file for file in os.listdir('./') if file.endswith("mp.csv")]
            for files in mp_file_list:
                df = pd.read_csv(files, low_memory=False)
                df_all_years = pd.concat([df_all_years, df])
            # df_all_years = df_all_years.drop(df_all_years.columns[0], axis=1)
            # df_all_years.to_parquet(pq_save_path, index=False)
            df_all_years.to_csv(csv_save_path, index=False)
            # TODO 삭제 잠시 임시로 주석처리
            for files in mp_file_list:
                os.remove(files)
            # pq.write_table(csv.read_csv(pq_save_path), pq_save_path)
            logging.info("create df in tables dict : {}".format(directory))
 