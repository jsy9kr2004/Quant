import datetime
import logging
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dateutil.relativedelta import relativedelta
from multiprocessing import Pool
from pyarrow import csv


class Parquet:
    def __init__(self, main_ctx):
        self.main_ctx = main_ctx
        self.tables = dict()
        self.view_path = self.main_ctx.root_path + "/VIEW/"
        self.rawpq_path = self.main_ctx.root_path + "/parquet/"

    def rebuild_table_view(self):
        # 1번 Table
        symbol_list = pd.read_parquet(self.rawpq_path + "stock_list.parquet", columns=['symbol', 'exchangeShortName', 'type'])
        delisted = pd.read_parquet(self.rawpq_path + "delisted_companies.parquet",
                                   columns=['symbol', 'exchange', 'ipoDate', 'delistedDate'])
        profile = pd.read_parquet(self.rawpq_path + "profile.parquet",
                                  columns=['symbol', 'ipoDate', 'industry', 'exchangeShortName'])
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

        # ['symbol_list', 'ipoDate'], ['symbol_list', 'delistedDate']
        all_symbol['ipoDate'] = all_symbol['ipoDate'].astype('datetime64[ns]')
        all_symbol['delistedDate'] = all_symbol['delistedDate'].astype('datetime64[ns]')
        
        all_symbol = all_symbol.reset_index(drop=True)
        all_symbol.to_parquet(self.view_path + "symbol_list.parquet", engine="pyarrow", compression="gzip")
        logging.info("create symbol_list df")
        del all_symbol

        # 2번 Table
        price = pd.read_parquet(self.rawpq_path + "historical_price_full.parquet",
                                columns=['date', 'symbol', 'close', 'volume'])
        marketcap = pd.read_parquet(self.rawpq_path + "historical_market_capitalization.parquet",
                                    columns=['date', 'symbol', 'marketCap'])
        price_marketcap = pd.merge(price, marketcap, how='left', on=['symbol', 'date'])
        del price
        del marketcap
        
        # ['price', 'date']
        price_marketcap['date'] = price_marketcap['date'].astype('datetime64[ns]')
        price_marketcap.to_parquet(self.view_path + "price.parquet", engine="pyarrow", compression="gzip")
        
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
        income_statement = pd.read_parquet(self.rawpq_path + "income_statement.parquet")
        balance_sheet_statement = pd.read_parquet(self.rawpq_path + "balance_sheet_statement.parquet")
        cash_flow_statement = pd.read_parquet(self.rawpq_path + "cash_flow_statement.parquet")
    
        financial_statement = income_statement.merge(balance_sheet_statement,
                                                     how='outer', on=['date', 'symbol']).merge(cash_flow_statement,
                                                                                               how='outer',
                                                                                               on=['date', 'symbol'])
        financial_statement['date'] = financial_statement['date'].astype('datetime64[ns]')
        financial_statement['acceptedDate'] = financial_statement['acceptedDate'].astype('datetime64[ns]')
        financial_statement['fillingDate'] = financial_statement['fillingDate'].astype('datetime64[ns]')
                                                
        financial_statement.to_parquet(self.view_path + "financial_statement.parquet", engine="pyarrow", compression="gzip")
        
        logging.info("create financial_statement df")
        for year in range(self.main_ctx.start_year - 1, self.main_ctx.end_year + 1):
            fs_peryear = financial_statement[financial_statement['date'].between(datetime.datetime(year, 1, 1),
                                                                                 datetime.datetime(year, 12, 31))]
            fs_peryear.to_parquet(self.view_path + "financial_statement_" + str(year) + ".parquet",
                                  engine="pyarrow", compression="gzip")
        logging.info("create price parquet per year")
        
        del income_statement
        del balance_sheet_statement
        del cash_flow_statement
        del financial_statement
        del fs_peryear

        # 4번 Table
        key_metrics = pd.read_parquet(self.rawpq_path + "key_metrics.parquet")
        financial_growth = pd.read_parquet(self.rawpq_path + "financial_growth.parquet")
        historical_daily_discounted_cash_flow = pd.read_parquet(self.rawpq_path
                                                                + "historical_daily_discounted_cash_flow.parquet")
        
        metrics = key_metrics.merge(financial_growth,
                                    how='outer', on=['date', 'symbol']).merge(historical_daily_discounted_cash_flow,
                                                                              how='outer', on=['date', 'symbol'])
        metrics['date'] = metrics['date'].astype('datetime64[ns]')
        metrics.to_parquet(self.path + "metrics.parquet", engine="pyarrow", compression="gzip")
        logging.info("create metrics df")

        for year in range(self.main_ctx.start_year - 1, self.main_ctx.end_year + 1):
            metrics_peryear = metrics[metrics['date'].between(datetime.datetime(year, 1, 1),
                                                              datetime.datetime(year, 12, 31))]
            metrics_peryear.to_parquet(self.path + "metrics_" + str(year) + ".parquet",
                                       engine="pyarrow", compression="gzip")
                   
        # logging.info("create price parquet per year")
        
        del financial_growth
        del key_metrics
        del metrics
        del metrics_peryear

        # 5번 Table
        indexes = pd.read_parquet(self.rawpq_path + "symbol_available_indexes.parquet")
        indexes.to_parquet(self.view_path + "indexes.parquet", engine="pyarrow", compression="gzip")
        logging.info("create indexes df")

    @staticmethod
    def read_csv_mp(filename):
        """converts a filename to a pandas dataframe"""
        return csv.read_csv(filename).to_pandas()
    
    def insert_csv(self):
        # wrap your csv importer in a function that can be mapped
        # merge all csvs per directoy
        dir_list = os.listdir(self.main_ctx.root_path)
        if self.main_ctx.need_pq_new_year == 'Y':
            dir_list = ['earning_calendar', 'historical_price_full']
        logging.info("directory list : {}".format(dir_list))
        for directory in dir_list:
            if not os.path.isdir(directory):
                continue
            csv_save_path = self.path + directory + ".csv"
            pq_save_path = self.path + directory + ".parquet"
            if (directory != 'stock_list') and (directory != 'symbol_available_indexes'):
                if os.path.exists(csv_save_path):
                    os.remove(csv_save_path)
                if os.path.exists(pq_save_path):
                    os.remove(pq_save_path)
                
            file_list = [self.main_ctx.root_path + "/" + directory + "/"
                         + file for file in os.listdir(self.main_ctx.root_path
                                                       + "/" + directory) if file.endswith(".csv")]
            full_df = pd.DataFrame()
            with Pool(processes=8) as pool:
                # or whatever your hardware can support
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
