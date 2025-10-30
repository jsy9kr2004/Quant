"""
ML Training Data Generator for Quantitative Trading System

This module generates machine learning training datasets from raw financial data.
It performs the complete data preparation pipeline:

1. Loads financial data from VIEW files (symbol list, price, financial statements, metrics)
2. Generates rebalancing dates at specified intervals (e.g., quarterly)
3. Calculates price changes (target variables for ML models)
4. Creates 12-month lookback windows for each stock
5. Extracts time-series features using tsfresh (autocorrelation, FFT, AR coefficients, etc.)
6. Computes custom financial ratios (OverMC_*, adaptiveMC_*)
7. Normalizes features using RobustScaler (outlier-resistant)
8. Saves quarterly ML datasets as Parquet files

Output Files:
    - rnorm_fs_{year}_{quarter}.parquet: Features without target variable (for latest predictions)
    - rnorm_ml_{year}_{quarter}.parquet: Features with target variable (for training/testing)

Example:
    >>> from config.context_loader import load_config, MainContext
    >>> from training.make_mldata import AIDataMaker
    >>>
    >>> config = load_config('config/conf.yaml')
    >>> ctx = MainContext(config)
    >>> maker = AIDataMaker(ctx, config)
    >>> # ML data files created in /data/ml_per_year/

Author: Quant Trading Team
Date: 2025-10-29
"""

import datetime
import logging
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters, MinimalFCParameters
from dateutil.relativedelta import relativedelta
from functools import reduce
from config.g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map, cal_timefeature_col_list
from config.logger import get_logger
from sklearn.preprocessing import StandardScaler, RobustScaler
from warnings import simplefilter
import warnings

# Configure pandas display for debugging
pd.options.display.width = 30

# Suppress performance warnings
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Suppress specific warnings (data validation is performed in code)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*grouping columns.*')


class AIDataMaker:
    """
    Generates ML training data from financial statements and price data.

    This class orchestrates the complete ML data preparation pipeline:
    - Loads raw financial data from VIEW files
    - Generates rebalancing dates
    - Calculates price changes (target variables)
    - Extracts time-series features with tsfresh
    - Computes custom financial ratios
    - Normalizes features with RobustScaler
    - Saves quarterly ML datasets as Parquet files

    The output datasets are ready for training ML models for stock selection.

    Attributes:
        main_ctx (MainContext): System context with configuration and paths
        conf (Dict[str, Any]): Configuration dictionary from YAML
        logger (logging.Logger): Logger instance for this class
        rebalance_period (int): Months between rebalancing (default: 3)
        symbol_table (pd.DataFrame): Stock metadata (symbol, sector, IPO date, etc.)
        price_table (pd.DataFrame): Historical price and volume data
        fs_table (pd.DataFrame): Financial statements (income, balance, cash flow)
        metrics_table (pd.DataFrame): Financial metrics (P/E, ROE, debt ratios, etc.)
        date_table_list (List): List of date tables (unused, legacy)
        trade_date_list (List[pd.Timestamp]): Actual trading dates for rebalancing

    Class Attributes:
        suffixes_dict (Dict[str, List[str]]): Suffix filters for tsfresh feature selection.
            Reduces feature dimensionality by selecting only key suffixes for each
            feature type (e.g., standard_deviation at r=0.0, 0.25, 0.6, 0.9).

    Example:
        >>> config = load_config('config/conf.yaml')
        >>> ctx = MainContext(config)
        >>> # Create ML data for years 2015-2023
        >>> maker = AIDataMaker(ctx, config)
        >>> # Output: ml_per_year/rnorm_ml_2015_Q1.parquet, rnorm_ml_2015_Q2.parquet, ...

    See Also:
        - config.g_variables: Feature lists and sector mapping
        - training.regressor: Uses generated ML data for model training
        - tsfresh documentation: https://tsfresh.readthedocs.io/
    """

    # tsfresh feature suffix filters for dimensionality reduction
    # Only specific suffixes are kept to reduce feature count while maintaining information
    suffixes_dict = {
        "standard_deviation": ["__r_0.0", "__r_0.25", "__r_0.6", "__r_0.9"],  # Different rolling windows
        "quantile": ["__q_0.2", "__q_0.8"],  # 20th and 80th percentiles
        "autocorrelation": ["__lag_0", "__lag_5", "__lag_9"],  # Different lag periods
        "fft_coefficient": ["__coeff_0", "__coeff_33", "__coeff_99"],  # Key Fourier components
        "cwt_coefficients": ["__coeff_0", "__coeff_6", "__coeff_12"],  # Wavelet transform coefficients
        "symmetry_looking": ["__r_0.0", "__r_0.25", "__r_0.65", "__r_0.9"],  # Symmetry at different scales
        "ar_coefficient": ["__coeff_0", "__coeff_3", "__coeff_6", "__coeff_10"]  # Autoregressive coefficients
    }

    def __init__(self, main_ctx: 'MainContext', conf: Dict[str, Any]) -> None:
        """
        Initialize AIDataMaker and execute complete data preparation pipeline.

        Pipeline execution order:
        1. Load data from VIEW files (load_bt_table)
        2. Generate rebalancing dates (set_date)
        3. Calculate price changes (process_price_table_wdate)
        4. Create ML datasets with time-series features (make_ml_data)

        Args:
            main_ctx: System context containing configuration and paths
            conf: Configuration dictionary with DATA, ML, and BACKTEST sections

        Raises:
            FileNotFoundError: If VIEW files are missing
            ValueError: If no valid trading dates are found
        """
        self.main_ctx = main_ctx
        self.conf = conf
        self.logger = get_logger('AIDataMaker')

        # Get rebalancing period from configuration (default: 3 months = quarterly)
        backtest_config = conf.get('BACKTEST', {})
        self.rebalance_period = backtest_config.get('REBALANCE_PERIOD', 3)

        # Initialize data containers
        self.symbol_table = pd.DataFrame()
        self.price_table = pd.DataFrame()
        self.fs_table = pd.DataFrame()
        self.metrics_table = pd.DataFrame()
        self.date_table_list = []
        self.trade_date_list = []

        # Create output directory for date tables
        self.main_ctx.create_dir(self.main_ctx.root_path + "/DATE_TABLE")

        # Execute pipeline
        self.load_bt_table(main_ctx.start_year)
        self.set_date()
        self.process_price_table_wdate()
        self.make_ml_data(main_ctx.start_year, main_ctx.end_year)

    def load_bt_table(self, year: int) -> None:
        """
        Load financial data from VIEW files.

        Loads four types of data:
        1. symbol_table: Stock metadata (symbol, sector, industry, IPO/delisted dates)
        2. price_table: Daily price and volume data
        3. fs_table: Financial statements (income, balance, cash flow) - loads 3 years back
        4. metrics_table: Financial metrics (P/E, ROE, debt ratios) - loads 3 years back

        The 3-year lookback ensures sufficient historical data for time-series feature extraction.

        Args:
            year: Starting year for data loading (unused, loads start_year-3 to end_year)

        Raises:
            FileNotFoundError: If VIEW directory or required files are missing

        Note:
            - Symbol table is deduplicated by symbol (keeps first occurrence)
            - Financial statements and metrics are loaded year by year to handle large datasets
            - Missing files are logged and skipped (allows partial data loading)

        TODO:
            - Add database support as alternative to CSV/Parquet files
            - Implement incremental loading for large datasets
        """
        # Load stock metadata
        self.symbol_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/symbol_list.csv")
        self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
        self.symbol_table['ipoDate'] = pd.to_datetime(self.symbol_table['ipoDate'])
        self.symbol_table['delistedDate'] = pd.to_datetime(self.symbol_table['delistedDate'])

        # Load price data
        self.price_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/price.csv")
        self.price_table['date'] = pd.to_datetime(self.price_table['date'])

        # Load financial statements (3 years back for time-series features)
        self.fs_table = pd.DataFrame()
        for year in range(self.main_ctx.start_year-3, self.main_ctx.end_year+1):
            fs_file = self.main_ctx.root_path + "/VIEW/financial_statement_" + str(year) + ".csv"
            if not os.path.exists(fs_file):
                self.logger.warning(f"Financial statement file not found, skipping: {fs_file}")
                print(f"WARNING: Financial statement file not found for year {year}")
                continue
            tmp_fs = pd.read_csv(fs_file,
                                    parse_dates=['fillingDate_x', 'acceptedDate_x'],
                                    dtype={'reportedCurrency_x': str, 'period_x': str,
                                        'link_x': str, 'finalLink_x': str})
            self.fs_table = pd.concat([tmp_fs, self.fs_table])

        # Convert date columns to datetime
        if not self.fs_table.empty:
            self.fs_table['date'] = pd.to_datetime(self.fs_table['date'])
            self.fs_table['fillingDate'] = pd.to_datetime(self.fs_table['fillingDate'])
            self.fs_table['acceptedDate'] = pd.to_datetime(self.fs_table['acceptedDate'])

        # Load financial metrics (3 years back)
        self.metrics_table = pd.DataFrame()
        for year in range(self.main_ctx.start_year-3, self.main_ctx.end_year+1):
            metrics_file = self.main_ctx.root_path + "/VIEW/metrics_" + str(year) + ".csv"
            if not os.path.exists(metrics_file):
                self.logger.warning(f"Metrics file not found, skipping: {metrics_file}")
                print(f"WARNING: Metrics file not found for year {year}")
                continue
            tmp_metrics = pd.read_csv(metrics_file,
                                        dtype={'period_x': str, 'period_y': str})
            self.metrics_table = pd.concat([tmp_metrics, self.metrics_table])

        # Convert date column to datetime
        if not self.metrics_table.empty:
            self.metrics_table['date'] = pd.to_datetime(self.metrics_table['date'])

    def get_trade_date(self, pdate: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        Convert calendar date to actual trading date.

        Finds the nearest trading date (date with price data) within 10 days before
        the given date. This handles weekends, holidays, and market closures.

        Args:
            pdate: Target calendar date

        Returns:
            Nearest trading date if found, None otherwise

        Example:
            >>> # If pdate is Sunday, returns previous Friday's trading date
            >>> trading_date = maker.get_trade_date(pd.Timestamp('2023-01-01'))
        """
        # Search for trading date within 10 days before target date
        post_date = pdate - relativedelta(days=10)
        res = self.price_table.query("date >= @post_date and date <= @pdate")
        if res.empty:
            return None
        else:
            return res.iloc[0].date

    def generate_date_list(self) -> List[datetime.datetime]:
        """
        Generate list of calendar dates for rebalancing.

        Creates dates at fixed intervals (rebalance_period) from start_year-3
        to end_year. The 3-year lookback ensures sufficient historical data.

        Returns:
            List of calendar dates at rebalance_period intervals

        Example:
            >>> # With REBALANCE_PERIOD=3 (quarterly)
            >>> dates = maker.generate_date_list()
            >>> # Returns: [2012-01-01, 2012-04-01, 2012-07-01, 2012-10-01, 2013-01-01, ...]
        """
        date_list = []

        # Get start month/day from configuration (default: January 1)
        backtest_config = self.conf.get('BACKTEST', {})
        start_month = backtest_config.get('START_MONTH', 1)
        start_date = backtest_config.get('START_DATE', 1)

        # Start 3 years before start_year for historical data
        date = datetime.datetime(int(self.main_ctx.start_year)-3, start_month, start_date)
        print(date)

        # Don't go beyond available price data
        recent_date = self.price_table["date"].max()
        end_date = datetime.datetime(self.main_ctx.end_year, 12, 31)
        if end_date > recent_date:
            end_date = recent_date

        # Generate dates at rebalance_period intervals
        while date <= end_date:
            date_list.append(date)
            date += relativedelta(months=self.rebalance_period)

        return date_list

    def set_trade_date_list(self, date_list: List[datetime.datetime]) -> List[pd.Timestamp]:
        """
        Convert calendar dates to actual trading dates.

        For each calendar date, finds the nearest trading date with available price data.
        Skips dates before price data is available.

        Args:
            date_list: List of calendar dates

        Returns:
            List of actual trading dates

        Raises:
            ValueError: If no valid trading dates are found

        Note:
            - Dates without price data are skipped (not an error)
            - Logs warnings for skipped dates
            - At least one valid trading date is required
        """
        trade_date_list = []
        price_min_date = self.price_table["date"].min()
        price_max_date = self.price_table["date"].max()

        for date in date_list:
            tdate = self.get_trade_date(date)
            if tdate is None:
                # No price data for this date - skip it
                self.logger.warning(f"⚠️  Cannot find tradable date for {date.strftime('%Y-%m-%d')}")
                self.logger.warning(f"   Price data range: {price_min_date.strftime('%Y-%m-%d')} to {price_max_date.strftime('%Y-%m-%d')}")
                self.logger.warning(f"   Skipping dates before price data is available...")
                print(f"⚠️  WARNING: Skipping {date.strftime('%Y-%m-%d')} - no price data available")
                continue
            trade_date_list.append(tdate)

        if not trade_date_list:
            error_msg = f"❌ FATAL: No valid trading dates found! Price data range: {price_min_date} to {price_max_date}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"✅ Found {len(trade_date_list)} valid trading dates")
        return trade_date_list


    def set_date(self) -> None:
        """
        Initialize trading date list for rebalancing.

        Generates calendar dates and converts them to actual trading dates.
        Sets self.trade_date_list which is used throughout data preparation.
        """
        date_list = self.generate_date_list()
        self.trade_date_list = self.set_trade_date_list(date_list)


    def process_price_table_wdate(self) -> None:
        """
        Calculate price changes (target variables) for rebalancing dates.

        Processing steps:
        1. Filter price table to only rebalancing dates
        2. Sort by symbol and date
        3. Calculate price_diff: absolute price change from previous rebalance
        4. Calculate volume_mul_price: liquidity indicator (price × volume)
        5. Calculate price_dev: price change rate (percentage return)

        Target Variable:
            price_dev = (price_t - price_{t-1}) / price_{t-1}
            This is the return achieved by holding the stock from previous rebalance to current rebalance.

        Saves:
            price_diff.csv in VIEW directory for reference

        Note:
            - First row of each symbol will have NaN for price_diff and price_dev (no previous data)
            - These rows are filtered out during ML data creation
        """
        # Filter to rebalancing dates only
        self.price_table = self.price_table[self.price_table['date'].isin(self.trade_date_list)]

        # Sort by symbol and date for sequential calculations
        self.price_table = self.price_table.sort_values(by=['symbol', 'date'])

        # Calculate absolute price change from previous rebalance
        self.price_table['price_diff'] = self.price_table.groupby('symbol')['close'].diff()

        # Calculate liquidity indicator (used for filtering low-liquidity stocks)
        self.price_table['volume_mul_price'] = self.price_table['close'] * self.price_table['volume']

        # Calculate percentage return (target variable for ML models)
        # Using previous close price as denominator (not current close)
        self.price_table['price_dev'] = self.price_table['price_diff'] / self.price_table.groupby('symbol')['close'].shift(1)

        # Rename for clarity
        self.price_table.rename(columns={'close': 'price'}, inplace=True)

        # Save for reference
        self.price_table.to_csv(self.main_ctx.root_path + "/VIEW/price_diff.csv", index=False)


    def filter_columns_by_suffixes(self, df: pd.DataFrame) -> List[str]:
        """
        Filter tsfresh columns by suffix to reduce dimensionality.

        tsfresh generates hundreds of features per input column. This method
        filters them to only keep features with specific suffixes defined in
        suffixes_dict, dramatically reducing feature count while maintaining
        information.

        Args:
            df: DataFrame with tsfresh features

        Returns:
            List of column names to keep

        Example:
            >>> # Before: ['revenue_ts_standard_deviation__r_0.0', '__r_0.05', '__r_0.1', ...]
            >>> # After: ['revenue_ts_standard_deviation__r_0.0', '__r_0.25', '__r_0.6', '__r_0.9']
            >>> filtered_cols = maker.filter_columns_by_suffixes(features_df)
        """
        filtered_cols = []
        for col in df.columns:
            include_col = True

            # Check if column matches any keyword in suffixes_dict
            for keyword, suffixes in self.suffixes_dict.items():
                if keyword in col:
                    # If keyword found, only include if column has a valid suffix
                    if any(suffix in col for suffix in suffixes):
                        break  # Valid suffix found, include column
                    else:
                        include_col = False  # Keyword found but no valid suffix, exclude
                        break

            if include_col:
                filtered_cols.append(col)

        return filtered_cols

    def filter_dates(self, df: pd.DataFrame, target_col_name: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Filter DataFrame by date range.

        Filters data to [start_year-01-01, end_year+1-03-01] to ensure we have
        all Q4 data from end_year.

        Args:
            df: DataFrame to filter
            target_col_name: Name of date column to filter on
            start_year: Start year (inclusive)
            end_year: End year (inclusive, extended to March of next year)

        Returns:
            Filtered DataFrame
        """
        df[target_col_name] = pd.to_datetime(df[target_col_name])
        start_date = pd.Timestamp(year=start_year, month=1, day=1)
        end_date = pd.Timestamp(year=end_year+1, month=3, day=1)  # Extended to capture Q4 filings
        filtered_df = df[(df[target_col_name] >= start_date) & (df[target_col_name] <= end_date)]
        return filtered_df

    def reorder_columns(self, df: pd.DataFrame, keywords: List[str] = ['symbol', 'date']) -> pd.DataFrame:
        """
        Reorder DataFrame columns to put key columns first.

        Args:
            df: DataFrame to reorder
            keywords: Column keywords to move to front

        Returns:
            DataFrame with reordered columns
        """
        key_cols = [col for col in df.columns if any(key.lower() in col.lower() for key in keywords)]
        other_cols = [col for col in df.columns if col not in key_cols]
        new_order = key_cols + other_cols
        return df[new_order]

    def efficient_merge_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge columns with _x and _y suffixes from DataFrame joins.

        After merging DataFrames, duplicate columns get _x and _y suffixes.
        This method merges them back, preferring non-null values from either column.

        Args:
            df: DataFrame with _x and _y suffix columns

        Returns:
            DataFrame with merged columns

        Example:
            >>> # Before: revenue_x, revenue_y
            >>> # After: revenue (combined with revenue_x.combine_first(revenue_y))
        """
        # Extract unique base column names
        base_cols = set(col.rstrip('_x').rstrip('_y') for col in df.columns)

        result_df = pd.DataFrame(index=df.index)
        for base in base_cols:
            # Find all columns starting with this base name
            cols = [col for col in df.columns if col.startswith(base)]
            # Merge using combine_first (prefers first non-null value)
            merged_col = reduce(lambda x, y: df[y].combine_first(x), cols, pd.Series([np.nan]*len(df), index=df.index))
            result_df[base] = merged_col

        result_df = self.reorder_columns(result_df)
        return result_df


    def make_ml_data(self, start_year: int, end_year: int) -> None:
        """
        Generate ML training datasets with time-series features.

        Main pipeline function that creates quarterly ML datasets for specified year range.

        Process for each year:
        1. Filter to high-liquidity stocks (top 50% by trading volume)
        2. Merge symbol, price, financial statements, and metrics
        3. For each quarter (Q1, Q2, Q3, Q4):
           a. Create 12-month lookback window
           b. Extract tsfresh time-series features
           c. Calculate custom financial ratios (OverMC_*, adaptiveMC_*)
           d. Normalize with RobustScaler
           e. Calculate target variables (price_dev, price_dev_subavg)
           f. Save to Parquet files

        Args:
            start_year: First year to process
            end_year: Last year to process

        Output Files:
            For each year and quarter:
            - rnorm_fs_{year}_{quarter}.parquet: Features only (for latest predictions)
            - rnorm_ml_{year}_{quarter}.parquet: Features + target (for training)

        Example:
            >>> maker.make_ml_data(2015, 2023)
            >>> # Creates: rnorm_ml_2015_Q1.parquet, rnorm_ml_2015_Q2.parquet, ...

        Notes:
            - Only processes stocks with 12+ quarters of data
            - Filters out low-liquidity stocks (bottom 50% by volume)
            - Skips quarters with insufficient data (logs warnings)
            - Uses RobustScaler for outlier-resistant normalization
            - Calculates sector-adjusted returns (sec_price_dev_subavg)

        TODO:
            - Add option to customize lookback window (currently fixed at 12)
            - Add option to customize liquidity threshold (currently 50%)
            - Add support for custom feature extraction parameters
        """
        # Create output directory
        ml_dir = os.path.join(self.main_ctx.root_path, "ml_per_year")
        self.main_ctx.create_dir(ml_dir)

        for cur_year in range(start_year, end_year+1):
            # Start with symbol table
            table_for_ai = self.symbol_table.copy()

            # Filter price data to current year ± 4 years (for time-series features)
            cur_price_table = self.price_table.copy()
            cur_price_table = self.filter_dates(cur_price_table, 'date', cur_year-4, cur_year)

            # Filter to high-liquidity stocks (top 50% by average trading value)
            symbol_means = cur_price_table.groupby('symbol')['volume_mul_price'].mean().reset_index()
            top_symbols = symbol_means.nlargest(int(len(symbol_means) * 0.50), 'volume_mul_price')
            cur_price_table = cur_price_table[cur_price_table['symbol'].isin(top_symbols['symbol'])]

            # Merge with symbol table
            table_for_ai = pd.merge(table_for_ai, cur_price_table, how='inner', on='symbol')
            table_for_ai.rename(columns={'date': 'rebalance_date'}, inplace=True)

            # Prepare financial statements
            fs = self.fs_table.copy()
            fs = fs[fs['symbol'].isin(top_symbols['symbol'])]
            fs = self.filter_dates(fs, 'fillingDate', cur_year-4, cur_year)
            fs = fs.drop_duplicates(['symbol', 'date'], keep='first')

            # Prepare metrics
            metrics = self.metrics_table.copy()
            metrics = metrics[metrics['symbol'].isin(top_symbols['symbol'])]
            metrics = metrics.drop_duplicates(['symbol', 'date'], keep='first')

            # Merge financial statements and metrics
            common_columns = fs.columns.intersection(metrics.columns)
            fs = fs.drop(columns=common_columns.difference(['symbol', 'date']))
            fs_metrics = pd.merge(fs, metrics, how='inner', on=['symbol', 'date'])
            fs_metrics = fs_metrics.drop_duplicates(['symbol', 'date'], keep='first')

            # Prepare dates
            fs_metrics['date'] = pd.to_datetime(fs_metrics['date'])
            fs_metrics.rename(columns={'date': 'report_date'}, inplace=True)
            fs_metrics['fillingDate'] = pd.to_datetime(fs_metrics['fillingDate'])

            # Map each filing date to next rebalancing date
            # This ensures we only use information available at each rebalancing
            date_index = np.sort(pd.DatetimeIndex(self.trade_date_list.copy()))
            indices = np.searchsorted(date_index, fs_metrics['fillingDate'], side='right')
            fs_metrics['rebalance_date'] = [date_index[i] if i < len(date_index) else pd.NaT for i in indices]

            # Create year_period for temporal ordering
            # Q1=.2, Q2=.4, Q3=.6, Q4=.8 allows decimal sorting
            fs_metrics = fs_metrics.dropna(subset=['calendarYear'])
            period_map = {'Q1': 0.2, 'Q2': 0.4, 'Q3': 0.6, 'Q4': 0.8}
            fs_metrics['year_period'] = fs_metrics['calendarYear'] + fs_metrics['period'].map(period_map)
            fs_metrics = fs_metrics.sort_values(by=['symbol', 'year_period'])

            # Assign time index for tsfresh (0, 1, 2, ..., 11 for 12-quarter window)
            def assign_time(group):
                group = group.sort_values(by='year_period').reset_index(drop=True)
                group['time_for_sort'] = range(len(group))
                return group

            fs_metrics = fs_metrics.groupby('symbol', group_keys=False).apply(assign_time).reset_index(drop=True)

            # Calculate custom ratios: OverMC_* (metric / market cap)
            for col in meaning_col_list:
                if col not in fs_metrics.columns:
                    continue
                new_col_name = 'OverMC_' + col
                fs_metrics[new_col_name] = np.where(fs_metrics['marketCap'] > 0,
                                                    fs_metrics[col]/fs_metrics['marketCap'], np.nan)

            # Calculate custom ratios: adaptiveMC_* (EV / metric)
            # EV (Enterprise Value) = Market Cap + Net Debt
            fs_metrics["adaptiveMC_ev"] = fs_metrics['marketCap'] + fs_metrics["netDebt"]
            for col in cal_ev_col_list:
                new_col_name = 'adaptiveMC_' + col
                fs_metrics[new_col_name] = np.where(fs_metrics[col] > 0,
                                                    fs_metrics['adaptiveMC_ev']/fs_metrics[col], np.nan)

            # Save snapshot for debugging
            print("*** fs_metrics w/ rebalance_date")
            print(fs_metrics)
            fs_metrics.head(1000).to_csv(self.main_ctx.root_path + f"/fs_metric_wdate_{str(cur_year)}.csv", index=False)

            # Process each quarter
            for quarter_str, quarter in [('Q1', 0.2), ('Q2', 0.4), ('Q3', 0.6), ('Q4', 0.8)]:
                base_year_period = cur_year + quarter

                # Output file paths
                file_path = os.path.join(self.main_ctx.root_path, "ml_per_year", f"rnorm_fs_{str(cur_year)}_{quarter_str}.parquet")
                file2_path = os.path.join(self.main_ctx.root_path, "ml_per_year", f"rnorm_ml_{str(cur_year)}_{quarter_str}.parquet")

                # Skip if already exists
                if os.path.isfile(file2_path):
                    print(f"*** there is parquet file {str(cur_year)}_{quarter_str}")
                    continue

                print(base_year_period)

                # Get data up to current quarter
                filtered_data = fs_metrics[fs_metrics['year_period'] <= float(base_year_period)]

                # Create 12-quarter lookback windows
                def get_last_12_rows(group):
                    return group.tail(12)

                window_data = filtered_data.groupby('symbol', group_keys=False).apply(get_last_12_rows).reset_index(drop=True)
                print(window_data)

                # Skip if no data
                if window_data.empty:
                    self.logger.warning(f"No data available for {cur_year}_{quarter_str}. Skipping...")
                    print(f"⚠️  WARNING: No data for {cur_year}_{quarter_str} - skipping file generation")
                    continue

                # Filter out symbols with less than 12 quarters of data
                symbol_counts = window_data['symbol'].value_counts()
                symbols_to_remove = symbol_counts[symbol_counts < 12].index
                window_data = window_data[~window_data['symbol'].isin(symbols_to_remove)]

                if window_data.empty:
                    self.logger.warning(f"No symbols with sufficient data (12+ rows) for {cur_year}_{quarter_str}. Skipping...")
                    print(f"⚠️  WARNING: No symbols with 12+ data points for {cur_year}_{quarter_str} - skipping")
                    continue

                # Extract time-series features with tsfresh
                df_for_extract_feature = pd.DataFrame()

                for target_col in cal_timefeature_col_list:
                    # Only process columns with valid (non-NaN, finite) data
                    if (target_col in window_data.columns and
                        not window_data[target_col].isna().any() and
                        np.isfinite(window_data[target_col]).all()):

                        # Prepare data in tsfresh format
                        temp_df = pd.DataFrame({
                            'id': window_data['symbol'],
                            'kind': target_col,
                            'time': window_data['time_for_sort'],
                            'value': window_data[target_col].values,
                            'year_period': window_data['year_period']
                        })
                        df_for_extract_feature = pd.concat([df_for_extract_feature, temp_df])

                if not df_for_extract_feature.empty:
                    # Extract features with tsfresh
                    features = extract_features(df_for_extract_feature,
                                               column_id='id',
                                               column_kind='kind',
                                               column_sort='time',
                                               column_value='value',
                                               default_fc_parameters=EfficientFCParameters())

                    # Rename columns to include '_ts_' marker
                    features = features.rename(columns=lambda x: f"{x.partition('__')[0]}_ts_{x.partition('__')[2]}")

                    # Filter features by suffix to reduce dimensionality
                    filtered_columns = self.filter_columns_by_suffixes(features)
                    df_w_time_feature = features[filtered_columns].copy()
                    df_w_time_feature['symbol'] = features.index

                    # Filter to current quarter only
                    window_data = window_data[window_data['year_period'] == float(base_year_period)]
                    df_w_time_feature = pd.merge(window_data, df_w_time_feature, how='inner', on='symbol')

                    # Remove absolute value columns (not useful for ML, only ratios matter)
                    abs_col_list = list(set(meaning_col_list) - set(ratio_col_list))
                    for col in abs_col_list:
                        df_w_time_feature = df_w_time_feature.drop([col], axis=1, errors='ignore')

                    # Separate columns that shouldn't be normalized
                    excluded_columns = ['symbol', 'rebalance_date', 'report_date', 'fillingDate_x', 'year_period']
                    excluded_df = df_w_time_feature[excluded_columns]

                    # Select columns for normalization
                    filtered_columns = [
                        col for col in df_w_time_feature.columns
                        if ('_ts_' in col) or  # Time-series features
                        (col in ratio_col_list) or  # Financial ratios
                        col.startswith('OverMC_') or  # Custom ratios
                        col.startswith('adaptiveMC_')  # Custom EV ratios
                    ]
                    filtered_df = df_w_time_feature[filtered_columns]

                    # Normalize with RobustScaler (resistant to outliers)
                    scaler = RobustScaler()
                    scaled_data = scaler.fit_transform(filtered_df)
                    scaled_df = pd.DataFrame(scaled_data, columns=filtered_df.columns)

                    # Combine normalized and excluded columns
                    scaled_df = pd.concat([excluded_df, scaled_df], axis=1)

                    # Save features without target variable (for latest predictions)
                    symbol_industry = table_for_ai[['symbol', 'industry', 'volume_mul_price']]
                    symbol_industry = symbol_industry.drop_duplicates('symbol', keep='first')
                    fs_df = pd.merge(symbol_industry, scaled_df, how='inner', on=['symbol'])
                    fs_df["sector"] = fs_df["industry"].map(sector_map)
                    fs_df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)

                    # Add target variables for training data
                    cur_table_for_ai = pd.merge(table_for_ai, scaled_df, how='inner', on=['symbol','rebalance_date'])
                    cur_table_for_ai["sector"] = cur_table_for_ai["industry"].map(sector_map)

                    # Calculate market-adjusted return
                    cur_table_for_ai['price_dev_subavg'] = cur_table_for_ai['price_dev'] - cur_table_for_ai['price_dev'].mean()

                    # Calculate sector-adjusted return
                    sector_list = list(cur_table_for_ai['sector'].unique())
                    sector_list = [x for x in sector_list if str(x) != 'nan']
                    for sec in sector_list:
                        sec_mask = cur_table_for_ai['sector'] == sec
                        sec_mean = cur_table_for_ai.loc[sec_mask, 'price_dev'].mean()
                        cur_table_for_ai.loc[sec_mask, 'sec_price_dev_subavg'] = cur_table_for_ai.loc[sec_mask, 'price_dev'] - sec_mean

                    # Save complete dataset with target variables
                    cur_table_for_ai.to_parquet(file2_path, engine='pyarrow', compression='snappy', index=False)
                    self.logger.info(f"✅ Saved ML data: {os.path.basename(file2_path)}")
