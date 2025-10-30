"""Financial Modeling Prep (FMP) Data Collector Module.

This module provides the main FMP data collection interface for fetching financial data
from the Financial Modeling Prep API. It handles symbol list management, data collection,
file cleanup, and validation of downloaded data.

The FMP class orchestrates the entire data collection workflow:
1. Fetches stock and delisted company lists
2. Builds symbol lists for NASDAQ and NYSE exchanges
3. Collects financial data for all symbols
4. Validates and cleans up outdated files

Typical usage example:
    from context import MainContext

    main_ctx = MainContext()
    fmp = FMP(main_ctx)
    fmp.collect()  # Start the data collection process
"""

from data_collector.fmp_api import FMPAPI
from data_collector.fmp_fetch_worker import fetch_fmp

import datetime
import dateutil.utils
import logging
import os
import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import List, Optional


class FMP:
    """Main class for collecting financial data from Financial Modeling Prep API.

    This class manages the entire data collection workflow including fetching ticker lists,
    building symbol lists for NASDAQ and NYSE exchanges, collecting financial data,
    and cleaning up outdated files.

    Attributes:
        main_ctx: Main context object containing configuration and shared resources.
        symbol_list (list): Complete list of stock symbols including delisted companies.
        current_list (list): List of currently active stock symbols.
        logger: Logger instance for this class.

    Example:
        >>> from context import MainContext
        >>> main_ctx = MainContext()
        >>> fmp = FMP(main_ctx)
        >>> fmp.collect()
    """

    def __init__(self, main_ctx) -> None:
        """Initialize the FMP data collector.

        Args:
            main_ctx: Main context object containing configuration like API keys,
                root paths, and logging setup.
        """
        self.main_ctx = main_ctx
        self.symbol_list: List[str] = pd.DataFrame()
        self.current_list: List[str] = pd.DataFrame()

        self.logger = self.main_ctx.get_logger('fmp')

    def __get_api_list(self) -> List[FMPAPI]:
        """Create a list of FMPAPI objects from the target API list CSV.

        Reads the API list from the configured CSV file and creates FMPAPI objects
        for each URL entry. Empty rows are dropped.

        Returns:
            List[FMPAPI]: List of initialized FMPAPI objects for data collection.

        Raises:
            FileNotFoundError: If the target API list file doesn't exist.
            KeyError: If the CSV doesn't have a 'URL' column.
        """
        url_df = pd.read_csv(self.main_ctx.target_api_list, header=0, usecols=["URL"])
        url_df = url_df.dropna()
        url_list = url_df['URL'].tolist()
        api_list = [FMPAPI(self.main_ctx, url) for url in url_list]

        return api_list

    def __fetch_ticker_list(self, api_list: List[FMPAPI]) -> None:
        """Fetch stock list and delisted companies data from FMP API.

        Filters the API list to find stock_list and delisted_companies endpoints,
        validates that both exist, and fetches data from both.

        Args:
            api_list (List[FMPAPI]): Complete list of FMPAPI objects to filter.

        Raises:
            Exception: If stock_list or delisted_companies API is not found in the list.
        """
        self.logger.info('fetching ticker list start (stock_list, delisted_companies)')

        # Find and validate stock_list API
        stock_list_api = [api for api in api_list if api.converted_category == 'stock_list']
        if len(stock_list_api) == 0:
            self.logger.error('stock list는 받아온다는 전제')
            raise Exception('stock list는 받아온다는 전제')
        stock_list_api = stock_list_api[0]

        # Find and validate delisted_companies API
        delisted_companies_api = [api for api in api_list if api.converted_category == 'delisted_companies']
        if len(delisted_companies_api) == 0:
            self.logger.error('delisted companies는 받아온다는 전제')
            raise Exception('delisted companies는 받아온다는 전제')
        delisted_companies_api = delisted_companies_api[0]

        self.logger.info('fetching ticker list done')

        return fetch_fmp(self.main_ctx, [stock_list_api, delisted_companies_api])

    def __set_symbol(self) -> None:
        """Build symbol lists from stock_list and delisted_companies data.

        This method:
        1. Loads the stock_list CSV file
        2. Filters for stocks on NASDAQ and NYSE exchanges
        3. Merges with delisted companies from the same exchanges
        4. Creates two lists:
           - symbol_list: All symbols (including delisted)
           - current_list: Recently active symbols (delisted within 1 month or currently active)

        The symbol lists are saved to CSV files (allsymbol.csv and current_list.csv)
        for debugging and validation purposes.

        Raises:
            FileNotFoundError: If stock_list.csv doesn't exist.
        """
        self.logger.info('set symbol list start')
        path = self.main_ctx.root_path + "/stock_list/stock_list.csv"
        if os.path.isfile(path):
            symbol_list = pd.read_csv(path)
        else:
            self.logger.error(f'file({path}) is not existed')
            return

        # Filter stock_list for stocks on NASDAQ and NYSE exchanges
        # Note: read_csv doesn't include index, so don't drop the first column
        filtered_symbol = symbol_list[(symbol_list['type'] == "stock")
                                      & ((symbol_list['exchangeShortName'] == 'NASDAQ')
                                         | (symbol_list['exchangeShortName'] == 'NYSE'))]
        filtered_symbol = filtered_symbol.dropna(subset=['symbol'])
        filtered_symbol = filtered_symbol.reset_index(drop=True)
        filtered_symbol = filtered_symbol.drop(['price', 'exchange', 'name'], axis=1)
        all_symbol = filtered_symbol

        # Merge with delisted companies from NASDAQ and NYSE
        file_list = os.listdir(self.main_ctx.root_path + "/delisted_companies/")
        for file in file_list:
            if os.path.splitext(file)[1] == ".csv":
                delisted = pd.read_csv(self.main_ctx.root_path + "/delisted_companies/" + file)
                if delisted.empty == True:
                    continue
                # Filter for NASDAQ and NYSE exchanges
                delisted = delisted[((delisted['exchange'] == 'NASDAQ') | (delisted['exchange'] == 'NYSE'))]
                delisted = delisted.dropna(subset=['symbol'])
                delisted = delisted.reset_index(drop=True)
                delisted.rename(columns={'exchange':'exchangeShortName'}, inplace=True)
                delisted = delisted.drop(['companyName'], axis=1)
                all_symbol = pd.concat([all_symbol, delisted])

        # Save all symbols to CSV and create complete symbol list
        all_symbol.to_csv('./allsymbol.csv', index=False)
        all_symbol = all_symbol.drop_duplicates('symbol', keep='first')
        all_symbol = all_symbol.reset_index(drop=True)
        self.symbol_list = all_symbol["symbol"].to_list()

        # Create current symbol list (recently active stocks)
        # TODO: Clarify the purpose of subtracting 1 month from the most recent delisted date
        all_symbol["delistedDate"] = pd.to_datetime(all_symbol["delistedDate"])
        recent_date = all_symbol["delistedDate"].max()
        recent_date -= relativedelta(months=1)  # Subtract 1 month from most recent delisted date

        # Query for symbols that are either:
        # 1. Delisted recently (within 1 month)
        # 2. Not delisted (NaT or None)
        query = '(delistedDate >= "{}") or (delistedDate == "NaT") or (delistedDate == "None")'.format(recent_date)
        current_symbol = all_symbol.query(query)
        current_symbol.to_csv('./current_list.csv', index=False)
        current_symbol = current_symbol.drop_duplicates('symbol', keep='first')
        current_symbol = current_symbol.reset_index(drop=True)
        self.current_list = current_symbol["symbol"].to_list()

        self.logger.info("in set_symbol() lit = " + str(self.symbol_list))
        self.logger.info('set symbol list done')

    def __fetch_data(self, api_list: List[FMPAPI]) -> None:
        """Fetch data from all APIs except stock_list and delisted_companies.

        Filters out the ticker list APIs and fetches data from remaining endpoints.
        For APIs that require symbols, the symbol_list is assigned before fetching.

        Args:
            api_list (List[FMPAPI]): Complete list of FMPAPI objects.
        """
        self.logger.info('fetching the rest start')

        # Filter for non-ticker-list APIs
        rest_api_list = [api for api in api_list if api.converted_category not in ['stock_list', 'delisted_companies']]

        # Assign symbol list to APIs that need it
        for api in rest_api_list:
            if api.need_symbol:
                api.symbol_list = self.symbol_list

        self.logger.info('fetching the rest done')

        return fetch_fmp(self.main_ctx, rest_api_list)

    @staticmethod
    def remove_files(path: str, only_csv: bool = True) -> None:
        """Remove files from the specified directory.

        Args:
            path (str): Directory path to clean up.
            only_csv (bool, optional): If True, only removes CSV files.
                If False, removes all files. Defaults to True.
        """
        if os.path.isdir(path) is False:
            return
        for file in os.listdir(path):
            if only_csv is True and not (file.endswith(".csv") or file.endswith(".csvx")):
                continue
            os.remove(os.path.join(path, file))

    def remove_current_list_files(self, base_path: str, check_target: bool = True) -> None:
        """Remove outdated files for symbols in the current list.

        Iterates through current_list symbols and removes their data files if:
        1. check_target is False: Removes all files unconditionally
        2. check_target is True: Removes files only if they're older than 75 days

        Args:
            base_path (str): Base directory path containing symbol data files.
            check_target (bool, optional): If True, checks file age before removal.
                Defaults to True.

        Note:
            Files must have a 'date' column to check age. Files without a date
            column or with empty date values are removed immediately.
        """
        logging.info("[Check Remove Files] Path : " + str(base_path))
        if os.path.isdir(base_path) is False:
            return
        today = dateutil.utils.today()

        for symbol in self.current_list:
            path = base_path + "/" + str(symbol) + ".csv"
            if os.path.isfile(path):
                if check_target is True:
                    # Read the entire file to check the date
                    # TODO: Find a more efficient way to read just the first/last row
                    row = pd.read_csv(path)

                    # Check if file has a date column
                    if "date" in row.columns:
                        if row["date"].empty is True:
                            os.remove(path)
                            continue
                    else:
                        # No date column, remove the file
                        os.remove(path)
                        continue

                    # Check if file is older than 75 days
                    update_date = datetime.datetime.strptime(row["date"].max(), "%Y-%m-%d")
                    if (today - update_date) < datetime.timedelta(days=75):
                        continue

                os.remove(path)

    @staticmethod
    def remove_current_year(base_path: str) -> None:
        """Remove files for the current year from the specified base path.

        Removes both .csv and .csvx files with the current year suffix.

        Args:
            base_path (str): Base path pattern (e.g., 'path/to/data_').
                Current year will be appended to form 'path/to/data_2025.csv'.

        Example:
            >>> FMP.remove_current_year('/data/historical_price_full/AAPL_')
            # Removes: /data/historical_price_full/AAPL_2025.csv
            #          /data/historical_price_full/AAPL_2025.csvx
        """
        today = dateutil.utils.today()
        year = today.strftime("%Y")
        if os.path.isfile(base_path + str(year) + ".csv"):
            os.remove(base_path + str(year) + ".csv")
        if os.path.isfile(base_path + str(year) + ".csvx"):
            os.remove(base_path + str(year) + ".csvx")

    def skip_remove_check(self) -> bool:
        """Check if file removal should be skipped based on last update time.

        Reads the last update date from './config/update_date.txt' and checks
        if less than 1 day has passed since the last update.

        Returns:
            bool: True if less than 1 day since last update (skip removal),
                False otherwise.

        Example:
            >>> if not fmp.skip_remove_check():
            ...     fmp.remove_first_loop()
        """
        today = datetime.datetime.today()
        if os.path.isfile("./config/update_date.txt"):
            fd = open("./config/update_date.txt", "r")
            update_date = fd.readline()
            fd.close()
            update_date = datetime.datetime.strptime(update_date, "%Y-%m-%d")

            # Check if less than 1 day has passed
            if (today - update_date) < datetime.timedelta(days=1):
                self.logger.info('Skip Remove Files')
                return True
        return False

    @staticmethod
    def validation_check() -> bool:
        """Validate downloaded files for API error messages.

        Checks all CSV files in the data directory for FMP API error messages:
        1. "Limit Reach" - API rate limit exceeded
        2. "Error Message" - Generic error from API

        Files containing these messages are deleted as they contain invalid data.

        Returns:
            bool: True if no files were deleted (all valid), False otherwise.

        Note:
            This method uses a hardcoded path 'E:\qt\data'.
            FIXME: Should use configurable path from main_ctx.
        """
        basepath = 'E:\qt\data'
        flag = True
        del_count = 0
        pass_count = 0

        for dir_name in os.listdir(basepath):
            if os.path.isdir(os.path.join(basepath, dir_name)):
                cur_path = os.path.join(basepath, dir_name)
                par_list = [file for file in os.listdir(cur_path) if file.endswith('csv')]

                for p in par_list:
                    df = pd.read_csv(os.path.join(cur_path, p))

                    # Check for error messages in the data
                    if df.filter(regex='Limit').empty is False or df.filter(regex='Error').empty is False:
                        logging.debug(os.path.join(cur_path, p))
                        os.remove(os.path.join(cur_path, p))
                        del_count += 1
                        flag = False
                    else:
                        pass_count += 1

                logging.info("[ {} ] Delete file count : {} / Total file count {} ".format(cur_path, del_count,
                                                                                           del_count + pass_count))
                del_count = 0
                pass_count = 0
        return flag

    def remove_first_loop(self) -> None:
        """Remove symbol list related files in preparation for refresh.

        First phase of the two-phase removal process. Removes files that affect
        the symbol list so they can be re-fetched with current data.

        Removes:
        - allsymbol.csv: Combined symbol list cache
        - current_list.csv: Current symbol list cache
        - delisted_companies/ directory contents
        - stock_list/ directory contents

        See Also:
            remove_second_loop: Second phase after symbol list is rebuilt.
        """
        if os.path.isfile("./allsymbol.csv"):
            os.remove("./allsymbol.csv")
        if os.path.isfile("./current_list.csv"):
            os.remove("./current_list.csv")
        self.remove_files(self.main_ctx.root_path+"/delisted_companies")
        self.remove_files(self.main_ctx.root_path+"/stock_list")

    def remove_second_loop(self) -> None:
        """Remove data files using the refreshed symbol list.

        Second phase of the two-phase removal process. After the symbol list is
        refreshed, removes outdated data files for symbols in the current list.

        Removes:
        - symbol_available_indexes
        - Current year earning calendar files
        - Financial statement files older than 75 days
        - Historical price files for current year
        - DCF and market cap files (performance bottleneck - see FIXME)
        - Profile data (currently not updated)

        FIXME: remove_current_list_files for DCF and market cap is slow.
            Consider alternative approaches for large datasets.

        See Also:
            remove_first_loop: First phase before symbol list refresh.
        """
        self.remove_files(self.main_ctx.root_path+"/symbol_available_indexes")
        self.remove_current_year(self.main_ctx.root_path+"/earning_calendar/earning_calendar_")

        # Remove financial statement files older than 75 days
        self.remove_current_list_files(self.main_ctx.root_path+"/income_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/balance_sheet_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/cash_flow_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/key_metrics")
        self.remove_current_list_files(self.main_ctx.root_path+"/financial_growth")

        # Remove current year historical price files for each symbol
        for symbol in self.current_list:
            self.remove_current_year(self.main_ctx.root_path+"/historical_price_full/" + str(symbol) + "_")

        # FIXME: These two operations are among the slowest. Consider optimization.
        # Top 2 most time-consuming operations - need better approach
        self.remove_current_list_files(self.main_ctx.root_path+"/historical_daily_discounted_cash_flow")
        self.remove_current_list_files(self.main_ctx.root_path+"/historical_market_capitalization", False)

        # Profile data is intentionally not updated
        self.remove_current_list_files(self.main_ctx.root_path+"/profile", False)

    def collect(self) -> None:
        """Execute the complete FMP data collection workflow.

        This is the main entry point for data collection. It orchestrates the entire
        process in the following sequence:

        1. Load API list from configuration
        2. Fetch ticker lists (stock_list and delisted_companies)
        3. Build symbol lists for NASDAQ and NYSE
        4. Fetch data for all other APIs
        5. Record update timestamp
        6. Validate downloaded files

        The method implements a two-pass approach:
        - First pass: Fetch ticker lists to build symbol list
        - Second pass: Use symbol list to fetch all other data

        This design allows for flexible API management without hardcoding API categories.
        When symbol lists don't exist, APIs requiring symbols will return without errors.

        Note:
            File removal steps (remove_first_loop, remove_second_loop) are currently
            commented out. Uncomment them to enable automatic cleanup of outdated files.

        Raises:
            SystemExit: If validation check fails, indicating corrupted downloads
                that need to be re-fetched.
        """

        api_list = self.__get_api_list()

        # First loop: Remove old symbol list files before fetching new ones
        # if self.skip_remove_check() is False:
        #     self.remove_first_loop()

        # Fetch ticker lists and build symbol lists
        self.__fetch_ticker_list(api_list)
        self.__set_symbol()  # TODO: Add ETF symbols to symbol list
        print("after set_symbol : {}".format(self.symbol_list))

        # Second loop: Remove outdated data files based on new symbol list
        # if self.skip_remove_check() is False:
        #     self.remove_second_loop()

        # Fetch all remaining data
        self.__fetch_data(api_list)

        # Record update timestamp to prevent accidental re-runs
        write_fd = open("./config/update_date.txt", "w")
        today = datetime.date.today()
        write_fd.write(str(today))
        write_fd.close()

        # Validate downloaded files for API errors
        if self.validation_check() is False:
            logging.critical("Validation Check False!! Please run the program again after a few minutes!!")
            exit()
