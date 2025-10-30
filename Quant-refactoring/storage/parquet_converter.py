"""Legacy Parquet converter for financial data consolidation.

This module provides the Parquet class for converting and consolidating financial
data from multiple CSV or Parquet source files into unified datasets. It merges
data from different sources (stock lists, price histories, financial statements,
metrics) and creates view tables organized by year.

The converter handles:
    - Merging multiple CSV/Parquet files per data category
    - Building consolidated view tables from raw data sources
    - Date type conversion and data cleaning
    - Creating year-partitioned datasets for efficient querying
    - Error handling for missing or corrupted source files

Example:
    Basic usage for data consolidation::

        from storage import Parquet

        # Initialize converter with main context
        converter = Parquet(main_ctx)

        # Consolidate all CSV files into single files per category
        converter.insert_csv()

        # Build unified view tables from raw data
        converter.rebuild_table_view()

Note:
    This is a legacy class that works with CSV files and converts them to
    consolidated formats. For new code, consider using ParquetStorage instead
    which provides a more modern API with built-in validation.

TODO:
    - Consider migrating to use ParquetStorage for better validation
    - Add progress indicators for long-running operations
    - Implement incremental updates instead of full rebuilds
"""

import datetime
import logging
import os
from typing import Any, List, Optional

import pandas as pd
from tqdm import tqdm


class Parquet:
    """Legacy Parquet data converter and consolidator.

    This class consolidates financial data from multiple source files into
    unified CSV datasets. It handles merging stock lists, price histories,
    financial statements, and metrics data. The class is designed to work
    with a main context object that provides configuration and utilities.

    The converter performs two main operations:
        1. insert_csv(): Consolidates multiple files per category into single files
        2. rebuild_table_view(): Merges related datasets into unified view tables

    Attributes:
        main_ctx: Main context object providing configuration (start_year, end_year,
            root_path) and utilities (create_dir method).
        tables (dict): Dictionary for storing loaded DataFrames (currently unused).
        view_path (str): Path to directory for storing consolidated view tables.
        rawpq_path (str): Path to directory containing raw Parquet/CSV files.

    Example:
        Initialize and run full conversion pipeline::

            # Assuming main_ctx has required attributes
            converter = Parquet(main_ctx)

            # Step 1: Consolidate raw files
            converter.insert_csv()

            # Step 2: Build view tables
            converter.rebuild_table_view()

    Note:
        This class expects specific directory structure and file naming conventions.
        It processes these categories: balance_sheet_statement, cash_flow_statement,
        delisted_companies, earning_calendar, financial_growth,
        historical_daily_discounted_cash_flow, historical_market_capitalization,
        historical_price_full, income_statement, key_metrics, profile,
        stock_list, symbol_available_indexes.
    """

    def __init__(self, main_ctx: Any) -> None:
        """Initialize Parquet converter with main context.

        Creates necessary directory structure for raw Parquet files and view tables.
        Initializes storage for table data.

        Args:
            main_ctx: Main context object that must provide:
                - root_path (str): Root directory for all data
                - start_year (int): Starting year for date range filtering
                - end_year (int): Ending year for date range filtering
                - create_dir(path: str): Method to create directories

        Example:
            Initialize with a context object::

                class MainContext:
                    def __init__(self):
                        self.root_path = "/data/financial"
                        self.start_year = 2015
                        self.end_year = 2024

                    def create_dir(self, path):
                        os.makedirs(path, exist_ok=True)

                main_ctx = MainContext()
                converter = Parquet(main_ctx)
        """
        self.main_ctx = main_ctx
        self.tables = dict()
        self.view_path = self.main_ctx.root_path + "/VIEW/"
        self.rawpq_path = self.main_ctx.root_path + "/parquet/"

        # Create necessary directories
        self.main_ctx.create_dir(self.view_path)
        self.main_ctx.create_dir(self.rawpq_path)

    def rebuild_table_view(self) -> None:
        """Rebuild consolidated view tables from raw data sources.

        Creates unified view tables by merging related datasets. This method
        performs complex joins and data consolidation to create analysis-ready
        tables. The process includes:

        1. Symbol List Table: Merges stock lists, delisted companies, and profiles
        2. Price Table: Combines historical prices with market capitalization
        3. Financial Statement Table: Joins income, balance sheet, and cash flow
        4. Metrics Table: Combines key metrics, growth data, and DCF valuations
        5. Indexes Table: Copies symbol index information

        For financial statements and metrics, also creates year-partitioned files
        for efficient querying of specific time periods.

        Example:
            Rebuild all view tables::

                converter = Parquet(main_ctx)
                converter.rebuild_table_view()

                # This creates the following files in VIEW/:
                # - symbol_list.csv
                # - price.csv
                # - financial_statement.csv
                # - financial_statement_YYYY.csv (per year)
                # - metrics.csv
                # - metrics_YYYY.csv (per year)
                # - indexes.csv

        Note:
            - This is a time-consuming operation for large datasets
            - Requires all source files to be present in rawpq_path
            - Performs extensive data cleaning and type conversions
            - Creates both full and year-partitioned datasets
            - Memory usage can be high due to large DataFrame operations

        Raises:
            May raise exceptions if source files are missing or corrupted.
            Errors are logged but not explicitly caught.
        """
        # 1. Build Symbol List Table
        # Consolidates stock lists, delisted companies, and profile information
        symbol_list = pd.read_csv(
            self.rawpq_path + "stock_list.csv",
            usecols=['symbol', 'exchangeShortName', 'type']
        )
        delisted = pd.read_csv(
            self.rawpq_path + "delisted_companies.csv",
            usecols=['symbol', 'exchange', 'ipoDate', 'delistedDate']
        )
        profile = pd.read_csv(
            self.rawpq_path + "profile.csv",
            usecols=['symbol', 'ipoDate', 'industry', 'exchangeShortName']
        )

        # Rename exchange column for consistency
        delisted.rename(columns={'exchange': 'exchangeShortName'}, inplace=True)

        # Concatenate symbol lists and remove duplicates
        all_symbol = pd.concat([symbol_list, delisted])
        all_symbol = all_symbol.drop_duplicates('symbol', keep='first')

        # Merge with profile data to get industry and IPO information
        all_symbol = all_symbol.merge(
            profile,
            how='left',
            on=['symbol', 'exchangeShortName']
        )

        # Consolidate IPO dates from multiple sources
        all_symbol['ipoDate'] = all_symbol['ipoDate_x'].combine_first(
            all_symbol['ipoDate_y']
        )
        all_symbol = all_symbol.drop(['ipoDate_x', 'ipoDate_y'], axis=1)
        all_symbol = all_symbol.drop_duplicates('symbol', keep='first')

        # Filter for NASDAQ and NYSE only
        all_symbol = all_symbol[
            (all_symbol['exchangeShortName'] == 'NASDAQ') |
            (all_symbol['exchangeShortName'] == 'NYSE')
        ]

        # Convert date columns to datetime
        all_symbol['ipoDate'] = all_symbol['ipoDate'].astype('datetime64[ns]')
        all_symbol['delistedDate'] = all_symbol['delistedDate'].astype('datetime64[ns]')

        all_symbol = all_symbol.reset_index(drop=True)
        all_symbol.to_csv(self.view_path + "symbol_list.csv", index=False)
        logging.info("create symbol_list df")
        del all_symbol

        # 2. Build Price Table
        # Combines historical prices with market capitalization
        price = pd.read_csv(
            self.rawpq_path + "historical_price_full.csv",
            usecols=['date', 'symbol', 'close', 'volume']
        )
        marketcap = pd.read_csv(
            self.rawpq_path + "historical_market_capitalization.csv",
            usecols=['date', 'symbol', 'marketCap']
        )

        # Merge price and market cap data
        price_marketcap = pd.merge(price, marketcap, how='left', on=['symbol', 'date'])
        del price
        del marketcap

        # Validate date format (should be YYYY-MM-DD)
        invalid_dates = price_marketcap[
            ~price_marketcap['date'].str.match(r'^\d{4}-\d{2}-\d{2}$')
        ]

        # Convert date column to datetime
        price_marketcap['date'] = price_marketcap['date'].astype('datetime64[ns]')
        price_marketcap.to_csv(self.view_path + "price.csv", index=False)

        logging.info("create price df")
        del price_marketcap

        # 3. Build Financial Statement Table
        # Joins income statement, balance sheet, and cash flow statement
        income_statement = pd.read_csv(self.rawpq_path + "income_statement.csv")
        balance_sheet_statement = pd.read_csv(
            self.rawpq_path + "balance_sheet_statement.csv"
        )
        cash_flow_statement = pd.read_csv(
            self.rawpq_path + "cash_flow_statement.csv"
        )

        # Perform outer join to preserve all records from all statements
        financial_statement = income_statement.merge(
            balance_sheet_statement,
            how='outer',
            on=['date', 'symbol']
        ).merge(
            cash_flow_statement,
            how='outer',
            on=['date', 'symbol']
        )

        # Convert date columns with error handling
        financial_statement['date'] = pd.to_datetime(
            financial_statement['date'],
            errors='coerce'
        )
        financial_statement['fillingDate'] = financial_statement['fillingDate'].astype(
            'datetime64[ns]'
        )

        # Save full financial statement table
        financial_statement.to_csv(
            self.view_path + "financial_statement.csv",
            index=False
        )
        logging.info("create financial_statement df")

        # Create year-partitioned files for efficient querying
        for year in range(self.main_ctx.start_year - 1, self.main_ctx.end_year + 1):
            fs_peryear = financial_statement[
                financial_statement['date'].between(
                    datetime.datetime(year, 1, 1),
                    datetime.datetime(year, 12, 31)
                )
            ]
            fs_peryear.to_csv(
                self.view_path + f"financial_statement_{year}.csv",
                index=False
            )
        logging.info("create financial_statement parquet per year")

        del income_statement
        del balance_sheet_statement
        del cash_flow_statement
        del financial_statement

        # 4. Build Metrics Table
        # Combines key metrics, financial growth, and DCF valuations
        key_metrics = pd.read_csv(self.rawpq_path + "key_metrics.csv")
        financial_growth = pd.read_csv(self.rawpq_path + "financial_growth.csv")
        historical_daily_discounted_cash_flow = pd.read_csv(
            self.rawpq_path + "historical_daily_discounted_cash_flow.csv"
        )

        # Merge metrics data with outer join to preserve all records
        metrics = key_metrics.merge(
            financial_growth,
            how='outer',
            on=['date', 'symbol']
        ).merge(
            historical_daily_discounted_cash_flow,
            how='left',
            on=['date', 'symbol']
        )

        # Convert date column
        metrics['date'] = metrics['date'].astype('datetime64[ns]')
        metrics.to_csv(self.view_path + "metrics.csv", index=False)
        logging.info("create metrics df")

        # Create year-partitioned metrics files
        for year in range(self.main_ctx.start_year - 1, self.main_ctx.end_year + 1):
            metrics_peryear = metrics[
                metrics['date'].between(
                    datetime.datetime(year, 1, 1),
                    datetime.datetime(year, 12, 31)
                )
            ]
            metrics_peryear.to_csv(
                self.view_path + f"metrics_{year}.csv",
                index=False
            )

        logging.info("create metrics parquet per year")

        del financial_growth
        del key_metrics
        del metrics

        # 5. Copy Indexes Table
        # Simple copy operation for index membership data
        indexes = pd.read_csv(self.rawpq_path + "symbol_available_indexes.csv")
        indexes.to_csv(self.view_path + "indexes.csv", index=False)
        logging.info("create indexes df")

    def insert_csv(self) -> None:
        """Consolidate multiple CSV/Parquet files into single files per category.

        Scans predefined data directories and merges all CSV or Parquet files
        within each category into a single consolidated CSV file. This is
        typically the first step in the data processing pipeline.

        The method processes these data categories:
            - balance_sheet_statement
            - cash_flow_statement
            - delisted_companies
            - earning_calendar
            - financial_growth
            - historical_daily_discounted_cash_flow
            - historical_market_capitalization
            - historical_price_full (with column filtering)
            - income_statement
            - key_metrics
            - profile
            - stock_list (preserved, not regenerated)
            - symbol_available_indexes (preserved, not regenerated)

        For each category:
            1. Scans the category directory for .csv or .parquet files
            2. Reads all files into DataFrames
            3. Concatenates them into a single DataFrame
            4. Saves the consolidated result as a single CSV file

        Example:
            Consolidate all source files::

                converter = Parquet(main_ctx)
                converter.insert_csv()

                # This reads files from:
                # {root_path}/income_statement/*.csv
                # {root_path}/balance_sheet_statement/*.parquet
                # etc.

                # And creates:
                # {root_path}/parquet/income_statement.csv
                # {root_path}/parquet/balance_sheet_statement.csv
                # etc.

        Note:
            - Existing consolidated files are removed before regeneration
              (except stock_list and symbol_available_indexes)
            - For historical_price_full, only specific columns are loaded
              to reduce memory usage (date, symbol, close, volume)
            - Files with read errors are skipped with a warning
            - Progress is displayed with tqdm progress bar
            - Empty directories are handled gracefully

        TODO:
            - Add support for year-partitioned historical price data
            - Implement incremental updates instead of full consolidation
        """
        # Define data categories to process
        dir_list = [
            "balance_sheet_statement",
            "cash_flow_statement",
            "delisted_companies",
            "earning_calendar",
            "financial_growth",
            "historical_daily_discounted_cash_flow",
            "historical_market_capitalization",
            "historical_price_full",
            "income_statement",
            "key_metrics",
            "profile",
            "stock_list",
            "symbol_available_indexes"
        ]

        logging.info("directory list : {}".format(dir_list))

        # Process each category with progress bar
        for directory in tqdm(dir_list):
            csv_save_path = self.rawpq_path + directory + ".csv"

            # Remove existing consolidated file (except for preserved lists)
            if (directory != 'stock_list') and (directory != 'symbol_available_indexes'):
                if os.path.exists(csv_save_path):
                    os.remove(csv_save_path)

            # Find all CSV and Parquet files in category directory
            file_list = [
                self.main_ctx.root_path + "/" + directory + "/" + file
                for file in os.listdir(self.main_ctx.root_path + "/" + directory)
                if (file.endswith(".parquet") or file.endswith(".csv"))
            ]

            # Read and consolidate all files for this category
            df_list = []
            for filename in file_list:
                try:
                    # Special handling for historical_price_full to reduce memory usage
                    if directory == 'historical_price_full':
                        if filename.endswith('.csv'):
                            df = pd.read_csv(
                                filename,
                                usecols=['date', 'symbol', 'close', 'volume']
                            )
                        elif filename.endswith('.parquet'):
                            df = pd.read_parquet(
                                filename,
                                columns=['date', 'symbol', 'close', 'volume']
                            )
                    else:
                        # Load all columns for other categories
                        if filename.endswith('.csv'):
                            df = pd.read_csv(filename, low_memory=False)
                        elif filename.endswith('.parquet'):
                            df = pd.read_parquet(filename)

                    df_list.append(df)

                except Exception as e:
                    # Log warning and continue if file cannot be read
                    logging.warning(f"Error reading {filename}: {str(e)}")
                    continue

            # Concatenate all DataFrames and save to single CSV
            if df_list:
                # Concatenate all at once for efficiency
                df_all_years = pd.concat(df_list, ignore_index=True)
                df_all_years.to_csv(csv_save_path, index=False)
                logging.info("create df in tables dict : {}".format(directory))
            else:
                logging.warning(f"No data found for directory: {directory}")
