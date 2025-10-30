"""Backtesting System for Quantitative Trading Strategies.

This module provides a comprehensive backtesting framework for quantitative trading strategies.
It simulates historical trading by:
1. Loading financial data (prices, fundamentals, metrics) for multiple time periods
2. Applying scoring plans to rank stocks based on various financial indicators
3. Selecting top-k stocks for each rebalancing period
4. Calculating returns and performance metrics (MDD, Sharpe ratio, etc.)

The backtesting workflow consists of 4 main components:
- Backtest: Main controller that orchestrates the entire backtesting process
- PlanHandler: Manages scoring plans and applies them to rank stocks
- DateHandler: Prepares feature-engineered data for a specific date
- EvaluationHandler: Calculates performance metrics and generates reports

Key Features:
- Multi-period backtesting with configurable rebalancing intervals
- Parallel processing support for improved performance
- Feature engineering with time-series analysis (tsfresh)
- Comprehensive reporting (EVAL, RANK, AVG reports)
- Support for both database and parquet storage backends

Plan.csv Format:
The plan.csv file defines scoring rules with the following columns:
- key: Financial metric name (e.g., 'PER', 'PBR', 'operatingIncome')
- key_dir: Direction ('high' = higher is better, 'low' = lower is better)
- weight: Score weight for this metric
- diff: Score difference between adjacent ranks
- base: Threshold value
- base_dir: Threshold direction ('>' or '<')

Example Usage:
    ```python
    from g_variables import Context

    # Initialize context
    ctx = Context(start_year=2020, end_year=2023)

    # Create plan handler
    plan_handler = PlanHandler(k_num=20, absolute_score=100, main_ctx=ctx)
    plan_handler.plan_list = load_plans_from_csv('plan.csv')

    # Run backtest with 3-month rebalancing
    backtest = Backtest(
        main_ctx=ctx,
        conf=config,
        plan_handler=plan_handler,
        rebalance_period=3
    )
    ```

Performance Considerations:
- First run generates DATE_TABLE and PLANED_DATE_TABLE files (slow)
- Subsequent runs read from cached files (fast)
- Memory usage scales with number of stocks and features
- Parallel processing can be enabled but may increase memory usage

Typical Workflow:
1. Backtest.run() iterates through rebalancing periods
2. For each period:
   a. DateHandler loads and engineers features
   b. PlanHandler applies scoring plans
   c. EvaluationHandler selects best-k stocks
3. After all periods:
   a. Calculate prices at buy/sell dates
   b. Calculate returns and performance metrics
   c. Generate EVAL, RANK, and AVG reports

Author: Quantitative Trading Team
Date: 2024
"""

import copy
import csv
import datetime
import logging
import multiprocessing
import sys
import os
from typing import Optional, Dict, List, Any, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

from collections import defaultdict
from dateutil.relativedelta import relativedelta
from functools import reduce
from g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map, cal_timefeature_col_list
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from functools import partial
from warnings import simplefilter

pd.options.display.width = 30
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
CHUNK_SIZE = 20480

# Time periods for historical feature engineering (in months)
time_periods = [3, 6, 9, 12, 15, 18, 21, 24]


class Backtest:
    """Main backtesting controller that orchestrates the entire backtesting process.

    This class manages the end-to-end backtesting workflow including:
    - Loading historical data from database or parquet files
    - Creating DateHandler instances for each rebalancing period
    - Running scoring plans through PlanHandler
    - Collecting best-k stocks through EvaluationHandler
    - Generating performance reports

    The backtesting process follows this sequence:
    1. Load initial data tables (symbol_table, price_table, fs_table, metrics_table)
    2. For each rebalancing period from start_year to end_year:
       a. Create DateHandler with feature-engineered data
       b. Run PlanHandler to score all stocks
       c. Select top-k stocks via EvaluationHandler
    3. Calculate actual returns by looking up prices at buy/sell dates
    4. Generate comprehensive reports (EVAL, RANK, AVG)

    Attributes:
        main_ctx: Main context containing configuration and database connection
        conf: Configuration dictionary with paths and parameters
        plan_handler: PlanHandler instance for scoring stocks
        rebalance_period: Number of months between rebalancing (e.g., 3 for quarterly)
        symbol_table: DataFrame of all stock symbols with metadata
        price_table: DataFrame of daily stock prices
        fs_table: DataFrame of financial statements
        metrics_table: DataFrame of financial metrics
        table_year: Current year for which tables are loaded
        eval_report_path: Path to evaluation report file
        rank_report_path: Path to ranking report file
        avg_report_path: Path to average report file
        eval_handler: EvaluationHandler instance for performance calculation

    Example:
        ```python
        # Create backtest with quarterly rebalancing
        backtest = Backtest(
            main_ctx=context,
            conf=config,
            plan_handler=plan_handler,
            rebalance_period=3
        )
        # Backtest automatically runs upon initialization
        ```
    """

    def __init__(
        self,
        main_ctx: Any,
        conf: Dict[str, Any],
        plan_handler: 'PlanHandler',
        rebalance_period: int
    ) -> None:
        """Initialize Backtest and automatically start the backtesting process.

        Args:
            main_ctx: Main context object containing:
                - start_year: Starting year for backtest
                - end_year: Ending year for backtest
                - conn: Database connection (if STORAGE_TYPE='DB')
                - root_path: Root directory path
            conf: Configuration dictionary containing:
                - STORAGE_TYPE: 'DB' or 'PARQUET'
                - ROOT_PATH: Root directory for data files
                - REPORT_LIST: List of report types to generate
                - START_MONTH: Starting month (default 1)
                - START_DATE: Starting day (default 1)
                - MEMBER_CNT: Number of stocks to select per period
                - TOP_K_NUM: Number of top stocks for ranking report
                - TOTAL_ASSET: Initial asset amount
            plan_handler: PlanHandler instance with configured scoring plans
            rebalance_period: Rebalancing interval in months (e.g., 1, 3, 6, 12)

        Note:
            The backtesting process starts automatically upon initialization.
            Results are written to report files in the ./reports/ directory.
        """
        self.main_ctx = main_ctx
        self.conf = conf
        self.plan_handler = plan_handler
        self.rebalance_period = rebalance_period

        # Initialize data tables (loaded in load_bt_table)
        self.symbol_table = pd.DataFrame()
        self.price_table = pd.DataFrame()
        self.fs_table = pd.DataFrame()
        self.metrics_table = pd.DataFrame()
        self.table_year = main_ctx.start_year

        # Create report paths
        self.eval_report_path = self.create_report("EVAL")
        self.rank_report_path = self.create_report("RANK")
        self.avg_report_path = self.create_report("AVG")

        # Create necessary directories for intermediate results
        self.main_ctx.create_dir(self.conf['ROOT_PATH'] + "/DATE_TABLE")
        self.main_ctx.create_dir(self.conf['ROOT_PATH'] + "/PLANED_DATE_TABLE")

        # Load initial data tables
        self.load_bt_table(main_ctx.start_year)

        # Create evaluation handler for performance calculation
        self.eval_handler = EvaluationHandler(self)

        # Start backtesting automatically
        self.run()

    def create_report(self, report_type: str) -> Optional[str]:
        """Create a report file with appropriate headers.

        This method creates CSV report files with incrementing indices to avoid
        overwriting existing reports. It writes common metadata headers including
        report date, backtest period, and plan parameters.

        Args:
            report_type: Type of report to create. Options:
                - 'EVAL': Evaluation report with performance metrics
                - 'RANK': Ranking report with detailed stock rankings
                - 'AVG': Average report with aggregated statistics

        Returns:
            Path to the created report file, or None if report_type not in REPORT_LIST

        Example:
            >>> path = backtest.create_report('EVAL')
            >>> print(path)
            './reports/EVAL_REPORT_0.csv'
        """
        if report_type in self.conf['REPORT_LIST']:
            path = "./reports/" + report_type + "_REPORT_"
            idx = 0

            # Find next available index to avoid overwriting
            while True:
                if not os.path.exists(path + str(idx) + ".csv"):
                    path = path + str(idx) + ".csv"
                    logging.info('REPORT PATH: "{}" ...'.format(path))
                    break
                else:
                    idx += 1

            # Write common header information
            if report_type == "EVAL" or report_type == "RANK":
                with open(path, 'w', newline='') as file:
                    writer = csv.writer(file, delimiter=",")
                    writer.writerow(["COMMON"])
                    writer.writerow(["Report Date", datetime.datetime.now().strftime('%m-%d %H:%M')])
                    writer.writerow(["Rebalance Period", str(self.rebalance_period) + " Month",
                                     "Start Year", self.main_ctx.start_year,
                                     "End Year", self.main_ctx.end_year])
                    writer.writerow(["K", self.plan_handler.k_num])
                    writer.writerow("")

                    # Write plan parameters for EVAL reports
                    if report_type == "EVAL":
                        writer.writerow(["PLAN HANDLER"])
                        writer.writerow(["key", "key_dir", "weight", "diff", "base", "base_dir"])
                        for plan in self.plan_handler.plan_list:
                            dict_writer = csv.DictWriter(file, fieldnames=plan["params"])
                            dict_writer.writerow(plan["params"])
            return path
        else:
            return None

    def data_from_database(self, query: str) -> pd.DataFrame:
        """Load data from database in chunks to handle large datasets.

        This method reads SQL query results in chunks to avoid memory issues
        with large tables. Chunks are concatenated into a single DataFrame.

        Args:
            query: SQL query string to execute

        Returns:
            DataFrame containing all query results

        Example:
            >>> query = "SELECT * FROM PRICE WHERE date >= '2020-01-01'"
            >>> df = backtest.data_from_database(query)
        """
        logging.info("Query : " + query)
        chunks = pd.read_sql_query(sql=query, con=self.main_ctx.conn, chunksize=CHUNK_SIZE)
        table = pd.DataFrame()
        for df in chunks:
            table = pd.concat([table, df])
        return table

    def load_bt_table(self, year: int) -> None:
        """Load backtesting data tables from storage backend.

        This method loads the four main data tables required for backtesting:
        1. symbol_table: Stock metadata (IPO dates, delisting dates, etc.)
        2. price_table: Daily price data (open, high, low, close, volume)
        3. fs_table: Financial statements (income, balance sheet, cash flow)
        4. metrics_table: Calculated financial metrics (ratios, indicators)

        Data can be loaded from either database or parquet files depending on
        STORAGE_TYPE configuration. For parquet storage, financial statement
        and metrics data are loaded in yearly chunks (current year ± 3 years)
        to manage memory usage.

        Args:
            year: Year for which to load data (affects fs_table and metrics_table range)

        Note:
            - For STORAGE_TYPE='DB': Loads directly from database
            - For STORAGE_TYPE='PARQUET': Loads from CSV files in VIEW directory
            - Financial data is loaded for [year-3, year] range
            - Duplicate symbols are removed, keeping first occurrence
        """
        query = "SELECT * FROM PRICE WHERE date BETWEEN '" \
                + str(datetime.datetime(self.main_ctx.start_year, 1, 1)) + "'" \
                + " AND '" + str(datetime.datetime(self.main_ctx.end_year, 12, 31)) + "'"

        if self.conf['STORAGE_TYPE'] == "DB":
            # Load from database
            self.symbol_table = self.data_from_database("SELECT * FROM symbol_list")
            self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
            self.price_table = self.data_from_database(query)
            self.fs_table = self.data_from_database("SELECT * FROM financial_statement")
            self.metrics_table = self.data_from_database("SELECT * FROM METRICS")

        if self.conf['STORAGE_TYPE'] == "PARQUET":
            # Load from parquet/CSV files
            self.symbol_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/symbol_list.csv")
            self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
            self.symbol_table['ipoDate'] = pd.to_datetime(self.symbol_table['ipoDate'])
            self.symbol_table['delistedDate'] = pd.to_datetime(self.symbol_table['delistedDate'])

            # Load price data
            self.price_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/price.csv")
            self.price_table['date'] = pd.to_datetime(self.price_table['date'])

            # Load financial statements for current year ± 3 years
            self.fs_table = pd.DataFrame()
            for y in range(self.main_ctx.start_year-3, self.main_ctx.start_year+1):
                tmp_fs = pd.read_csv(
                    self.main_ctx.root_path + "/VIEW/financial_statement_" + str(y) + ".csv",
                    parse_dates=['fillingDate_x', 'acceptedDate_x'],
                    dtype={'reportedCurrency_x': str, 'period_x': str,
                           'link_x': str, 'finalLink_x': str}
                )
                self.fs_table = pd.concat([tmp_fs, self.fs_table])
            del tmp_fs
            self.fs_table['date'] = pd.to_datetime(self.fs_table['date'])
            self.fs_table['fillingDate'] = pd.to_datetime(self.fs_table['fillingDate'])
            self.fs_table['acceptedDate'] = pd.to_datetime(self.fs_table['acceptedDate'])

            # Load metrics for current year ± 3 years
            self.metrics_table = pd.DataFrame()
            for y in range(self.main_ctx.start_year-3, self.main_ctx.start_year+1):
                tmp_metrics = pd.read_csv(
                    self.main_ctx.root_path + "/VIEW/metrics_" + str(y) + ".csv",
                    dtype={'period_x': str, 'period_y': str}
                )
                self.metrics_table = pd.concat([tmp_metrics, self.metrics_table])
            del tmp_metrics
            self.metrics_table['date'] = pd.to_datetime(self.metrics_table['date'])

    def reload_bt_table(self, year: int) -> None:
        """Reload financial statement and metrics tables for a new year.

        This method is called when the backtest advances to a new year. It reloads
        only the financial statement and metrics tables (not symbol or price tables)
        to include data for the new year while maintaining the ±3 year window.

        Args:
            year: New year for which to reload data

        Note:
            Only used when STORAGE_TYPE='PARQUET'
            Reloads fs_table and metrics_table for [year-3, year] range
        """
        logging.info("reload_bt_table, year : {}".format(year))

        # Reload financial statements
        self.fs_table = pd.DataFrame()
        for y in range(year-3, year+1):
            tmp_fs = pd.read_csv(
                self.main_ctx.root_path + "/VIEW/financial_statement_" + str(y) + ".csv",
                parse_dates=['fillingDate_x', 'acceptedDate_x'],
                dtype={'reportedCurrency_x': str, 'period_x': str,
                       'link_x': str, 'finalLink_x': str}
            )
            self.fs_table = pd.concat([tmp_fs, self.fs_table])
            del tmp_fs
        self.fs_table['date'] = pd.to_datetime(self.fs_table['date'])
        self.fs_table['fillingDate'] = pd.to_datetime(self.fs_table['fillingDate'])
        self.fs_table['acceptedDate'] = pd.to_datetime(self.fs_table['acceptedDate'])

        # Reload metrics
        self.metrics_table = pd.DataFrame()
        for y in range(year-3, year+1):
            tmp_metrics = pd.read_csv(self.main_ctx.root_path + "/VIEW/metrics_" + str(y) + ".csv")
            self.metrics_table = pd.concat([tmp_metrics, self.metrics_table])
        del tmp_metrics
        self.metrics_table['date'] = pd.to_datetime(self.metrics_table['date'])

    def get_trade_date(self, pdate: datetime.datetime) -> Optional[datetime.datetime]:
        """Find the nearest trading date for a given date.

        Since markets may be closed on weekends and holidays, this method finds
        the nearest actual trading date by looking for price data within 10 days
        before the given date.

        Args:
            pdate: Target date to find trading date for

        Returns:
            Nearest trading date, or None if no trading date found within 10 days

        Example:
            >>> trade_date = backtest.get_trade_date(datetime.datetime(2023, 7, 4))
            >>> # Returns 2023-07-03 if July 4th is a holiday
        """
        post_date = pdate - relativedelta(days=10)
        res = self.price_table.query("date >= @post_date and date <= @pdate")
        if res.empty:
            return None
        else:
            return res.iloc[0].date

    def run(self) -> None:
        """Execute the main backtesting loop across all rebalancing periods.

        This is the core method that orchestrates the entire backtesting process:

        1. Iterate through time from start_year to end_year by rebalance_period
        2. For each period:
           a. Find actual trading date (markets may be closed)
           b. Create DateHandler with feature-engineered data
           c. Run PlanHandler to score all stocks
           d. Select best-k stocks via EvaluationHandler
        3. Handle year transitions by reloading data tables
        4. Print current best stocks for the most recent date
        5. After all periods, run EvaluationHandler to calculate returns

        The method handles edge cases like:
        - Non-trading dates (weekends, holidays)
        - Year transitions (reload data tables)
        - End date exceeding available data
        - Current date recommendations (no future data)

        Note:
            This method is called automatically during __init__
            Results are stored in self.eval_handler.best_k

        TODO:
            Separate the loop logic when plan is not used to avoid confusion.
            Currently plan_handler.run() is called even when not using plans,
            just to set up best_k list.
        """
        date = datetime.datetime(
            self.main_ctx.start_year,
            self.conf['START_MONTH'],
            self.conf['START_DATE']
        )
        recent_date = self.price_table["date"].max()
        end_date = datetime.datetime(self.main_ctx.end_year, 12, 31)

        # Adjust end date if it exceeds available data
        if end_date > recent_date:
            end_date = recent_date

        cur_table_year = self.main_ctx.start_year

        # === START OF MAIN BACKTESTING LOOP ===
        while True:
            # Find actual trading date (markets may be closed)
            tdate = self.get_trade_date(date)
            if tdate is None:
                logging.info("tradable date is None. break")
                break

            logging.info("Backtest Run : " + str(tdate.strftime("%Y-%m-%d")))

            # Create DateHandler with feature-engineered data for this date
            self.plan_handler.date_handler = DateHandler(self, tdate)
            logging.info("complete set date_handler date : {}".format(tdate.strftime("%Y-%m-%d")))

            # Run scoring plans to rank stocks
            self.plan_handler.run(self.conf)

            # Select best-k stocks or print current recommendations
            if date != recent_date:
                # Regular period: select best-k stocks for backtesting
                self.eval_handler.set_best_k(
                    tdate,
                    date + relativedelta(months=self.rebalance_period),
                    self.plan_handler.date_handler
                )
            else:
                # Most recent date: print current best recommendations
                self.eval_handler.print_current_best(self.plan_handler.date_handler)
                break

            # Advance to next rebalancing period
            if (date + relativedelta(months=self.rebalance_period)) <= end_date:
                date += relativedelta(months=self.rebalance_period)

                # Reload data tables if year changed
                if date.year != cur_table_year:
                    cur_table_year = date.year
                    self.reload_bt_table(cur_table_year)
            else:
                # Last loop: run one more time with most recent date to print recommendations
                if self.eval_report_path is not None:
                    date = recent_date
                    if date.year != cur_table_year:
                        cur_table_year = date.year
                        self.reload_bt_table(cur_table_year)
                else:
                    # Skip current recommendations if not generating eval report
                    break
        # === END OF MAIN BACKTESTING LOOP ===

        # Log sparse column statistics (columns with too many missing values)
        logging.info("DateHandler.global_sparse_col : ")
        for k, v in DateHandler.global_sparse_col.items():
            logging.debug(f"{k} - {v}")

        # Calculate returns and generate reports
        logging.info("START Evaluation")
        self.eval_handler.run(self.price_table)


class PlanHandler:
    """Manages scoring plans and applies them to rank stocks.

    This class handles the application of scoring plans to rank stocks based on
    various financial metrics. Each plan defines rules for scoring stocks, and
    the total score determines which stocks are selected for the portfolio.

    A scoring plan consists of:
    - key: Financial metric to use (e.g., 'PER', 'PBR', 'operatingIncome')
    - key_dir: Direction ('high' = higher is better, 'low' = lower is better)
    - weight: Importance weight for this metric
    - diff: Score difference between adjacent ranks
    - base: Threshold value for filtering stocks
    - base_dir: Threshold comparison operator ('>' or '<')

    The scoring process:
    1. For each plan, rank stocks by the metric (after direction adjustment)
    2. Assign scores to top-k stocks: best gets absolute_score,
       second gets absolute_score - diff, etc.
    3. Sum scores across all plans to get total score per stock
    4. Select stocks with highest total scores

    Attributes:
        plan_list: List of plan dictionaries, each containing:
            - f_name: Function to execute for this plan
            - params: Plan parameters (key, key_dir, weight, diff, base, base_dir)
        date_handler: DateHandler instance with current date's data
        k_num: Number of top stocks to consider per plan (e.g., 20)
        absolute_score: Maximum score for best-ranked stock (e.g., 100)
        main_ctx: Main context object

    Example:
        ```python
        plan_handler = PlanHandler(k_num=20, absolute_score=100, main_ctx=ctx)
        plan_handler.plan_list = [
            {
                'f_name': plan_handler.single_metric_plan,
                'params': {
                    'key': 'PER',
                    'key_dir': 'low',
                    'weight': 1.0,
                    'diff': 5,
                    'base': 0,
                    'base_dir': '>'
                }
            }
        ]
        ```
    """

    def __init__(
        self,
        k_num: int,
        absolute_score: int,
        main_ctx: Any
    ) -> None:
        """Initialize PlanHandler with scoring parameters.

        Args:
            k_num: Number of top stocks to consider per plan (e.g., 20)
            absolute_score: Maximum score for best-ranked stock (e.g., 100)
            main_ctx: Main context object with logger and configuration
        """
        self.plan_list: Optional[List[Dict[str, Any]]] = None
        self.date_handler: Optional['DateHandler'] = None
        self.k_num = k_num
        self.absolute_score = absolute_score
        self.main_ctx = main_ctx

    def run(self, conf: Dict[str, Any]) -> None:
        """Execute all scoring plans and calculate total scores.

        This method runs all configured plans to score stocks:
        1. Check if planed_dtable CSV exists for this date
        2. If exists: load from file (fast path)
        3. If not exists: run all plans and save results (slow path)
        4. Sum all plan scores to get final stock scores

        The planed_dtable is cached to disk to avoid recomputing scores on
        subsequent runs, significantly improving performance.

        Args:
            conf: Configuration dictionary containing ROOT_PATH

        Raises:
            AssertionError: If plan_list or date_handler is None

        Note:
            First run is slow (creates planed_dtable), subsequent runs are fast
        """
        assert self.plan_list is not None, "Empty Plan List"
        assert self.date_handler is not None, "Empty Date Handler"

        # Check if planed_dtable already exists for this date
        pdate = self.date_handler.date
        planed_dtable_path = (
            conf['ROOT_PATH'] +
            "/PLANED_DATE_TABLE/planed_dtable_{}.csv".format(pdate.strftime('%Y-%m-%d'))
        )

        if not os.path.exists(planed_dtable_path):
            # Planed table doesn't exist - need to create it
            logging.info("there is no planed date_table : " + planed_dtable_path)
            logging.info("start to create planed_date_table : " + planed_dtable_path)

            # TODO: Multiprocessing causes abnormal table size increase
            # Currently processing serially. After first run, will read from cache.
            i = 0
            for plan in self.plan_list:
                logging.debug(
                    "[{}/{}] {} processing....".format(
                        i, len(self.plan_list), str(plan["params"]["key"])
                    )
                )
                self.plan_run(plan)
                i += 1

            # Sum all plan scores to get total score
            score_col_list = self.date_handler.dtable.columns.str.contains("_score")
            self.date_handler.dtable['score'] = self.date_handler.dtable.loc[:, score_col_list].sum(axis=1)

            # Save for future use
            self.date_handler.dtable.to_csv(
                conf['ROOT_PATH'] + "/PLANED_DATE_TABLE/" +
                'planed_dtable_{}.csv'.format(pdate.strftime('%Y-%m-%d')),
                index=False
            )
        else:
            # Planed table exists - load from cache
            logging.info("there is csv file for this date. read planed date table from csv")
            self.date_handler.dtable = pd.read_csv(planed_dtable_path)

    @staticmethod
    def plan_run(plan: Dict[str, Any]) -> pd.DataFrame:
        """Execute a single plan's scoring function.

        Args:
            plan: Plan dictionary containing 'f_name' (function) and 'params'

        Returns:
            DataFrame with score columns added
        """
        return plan["f_name"](plan["params"])

    def single_metric_plan_no_parallel(self, params: Dict[str, Any]) -> None:
        """Score stocks based on a single metric (non-parallel version).

        This method ranks stocks by a single financial metric and assigns scores
        to the top-k stocks. The scoring is done in-place on date_handler.dtable.

        Scoring Logic:
        1. Rank all stocks by the metric (after direction adjustment)
        2. Assign scores to top-k stocks:
           - Rank 1: absolute_score
           - Rank 2: absolute_score - diff
           - Rank 3: absolute_score - 2*diff
           - ...
        3. All other stocks get score 0

        Args:
            params: Plan parameters containing:
                - key: Metric name (e.g., 'PER', 'PBR')
                - key_dir: 'high' or 'low' (already adjusted in _sorted column)
                - diff: Score difference between ranks

        Warning:
            Logs warning if diff is too large (causes negative scores)
        """
        if self.absolute_score - params["diff"] * self.k_num < 0:
            logging.warning(
                "Wrong params['diff'] : TOO BIG! SET UNDER " +
                str(self.absolute_score / self.k_num)
            )

        key = str(params["key"])

        # All features are preprocessed in DateHandler (_sorted columns: high is good)
        rank_name = key + '_rank'
        self.date_handler.dtable[rank_name] = self.date_handler.dtable[key+"_sorted"].rank(
            ascending=False,
            method='min',
            na_option='bottom'
        )

        # Assign scores to top-k ranked stocks
        score_name = key + '_score'
        self.date_handler.dtable.loc[
            self.date_handler.dtable[rank_name] <= self.k_num,
            score_name
        ] = self.absolute_score - (
            (self.date_handler.dtable.loc[
                self.date_handler.dtable[rank_name] <= self.k_num,
                rank_name
            ] - 1) * params["diff"]
        )

        # Fill remaining rows with 0
        self.date_handler.dtable[score_name] = self.date_handler.dtable[score_name].fillna(0)

    def single_metric_plan(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Score stocks based on a single metric (parallel-compatible version).

        This method ranks stocks by a single financial metric and returns a DataFrame
        with score columns. Designed to be used with multiprocessing.

        Args:
            params: Plan parameters containing:
                - key: Metric name (e.g., 'PER', 'PBR')
                - key_dir: Direction ('high' or 'low')
                - weight: Score multiplier
                - diff: Score difference between ranks
                - base: Threshold value
                - base_dir: Threshold operator

        Returns:
            DataFrame with columns ['symbol', '{key}_score', '{key}_rank']

        Warning:
            Logs warning if diff is too large (causes negative scores)
        """
        logger = self.main_ctx.get_multi_logger()

        if self.absolute_score - params["diff"] * self.k_num < 0:
            logger.warning(
                "Wrong params['diff'] : TOO BIG! SET UNDER " +
                str(self.absolute_score / self.k_num)
            )

        key = str(params["key"])

        # All features are preprocessed in DateHandler (_sorted columns: high is good)
        # Select top k stocks by sorted value
        top_k_df = self.date_handler.dtable.sort_values(
            by=[key+"_sorted"],
            ascending=False,
            na_position="last"
        )[:self.k_num]

        symbols = top_k_df['symbol']
        del top_k_df

        # Create return DataFrame with scores
        return_df = self.date_handler.dtable[['symbol']].copy()
        delta = self.absolute_score

        # Suppress chained assignment warning
        pd.set_option('mode.chained_assignment', None)

        local_score_name = key + '_score'
        # Assign scores to top-k stocks (decreasing by diff)
        for sym in symbols:
            return_df.loc[
                (self.date_handler.dtable.symbol == sym),
                local_score_name
            ] = params["weight"] * delta
            delta = delta - params["diff"]

        # Create rank column
        local_rank_name = key+'_rank'
        return_df[local_rank_name] = return_df[local_score_name].rank(
            method='min',
            ascending=False
        )
        return_df[local_rank_name] = return_df[local_rank_name].fillna(-1)
        return_df[local_score_name] = return_df[local_score_name].fillna(0)

        return return_df


class DateHandler:
    """Prepares feature-engineered data for a specific date.

    This class is responsible for creating a comprehensive feature table (dtable)
    for a specific date by:
    1. Loading stocks that were tradable on that date
    2. Merging price data, financial statements, and metrics
    3. Creating derived features:
       - Relative features (normalized by market cap)
       - Time-series features (changes over 3, 6, 12, 24 months)
       - Time-series patterns (using tsfresh)
       - Enterprise value ratios
    4. Preprocessing features (handling directions, normalization)
    5. Filtering sparse features and rows

    The dtable is cached to disk to avoid recomputing features on subsequent runs.

    Class Attributes:
        global_sparse_col: Dictionary tracking columns that are frequently sparse
            across different dates (columns with >50% missing values)

    Instance Attributes:
        date: Target date for which to prepare data
        dtable: Main feature table containing all stocks and features for this date

    Feature Engineering Process:
    1. Get current prices and filter by trading volume
    2. Merge latest financial statements (within last 6 months)
    3. Create relative features (divide by market cap)
    4. Create time-series features:
       - Load historical financial data (3, 6, 9, 12, 15, 18, 21, 24 months ago)
       - Calculate percentage changes
       - Extract time-series patterns using tsfresh
    5. Create enterprise value ratios
    6. Preprocess all features:
       - Adjust direction (make "high is good" uniform)
       - Handle missing values
    7. Filter sparse columns and rows
    8. Save to cache for future use

    Example:
        ```python
        # Create DateHandler for 2023-01-01
        date_handler = DateHandler(backtest, datetime.datetime(2023, 1, 1))

        # Access feature table
        print(date_handler.dtable.shape)  # (2000, 500) - 2000 stocks, 500 features
        print(date_handler.dtable.columns)  # Feature names
        ```
    """

    # Class variable to track sparse columns across all dates
    global_sparse_col: Dict[str, int] = defaultdict(int)

    def __init__(self, backtest: 'Backtest', date: datetime.datetime) -> None:
        """Initialize DateHandler and create feature table for given date.

        Args:
            backtest: Backtest instance with data tables
            date: Target date for feature engineering

        Note:
            Feature table creation is automatic during initialization.
            If cached dtable exists, it will be loaded instead of recomputed.
        """
        # Suppress chained assignment warnings
        pd.set_option('mode.chained_assignment', None)

        logging.info("in datehandler date : " + date.strftime("%Y-%m-%d"))
        self.date = date

        # Main feature table (will be populated by init_data)
        self.dtable: Optional[pd.DataFrame] = None

        # Create or load feature table
        self.init_data(backtest)

    def init_data(self, backtest: 'Backtest') -> None:
        """Initialize feature table by loading from cache or creating new.

        This method checks if a cached dtable exists for this date. If yes,
        loads from cache. If no, creates new dtable with feature engineering.

        Args:
            backtest: Backtest instance with data tables

        Note:
            Cached dtables significantly improve performance (10-100x faster)
        """
        logging.info("START init_data in date handler ")

        # Find actual trading date
        trade_date = backtest.get_trade_date(self.date)
        dtable_path = (
            backtest.conf['ROOT_PATH'] + "/DATE_TABLE/dtable_" +
            str(trade_date.year) + '_' + str(trade_date.month) + '_' +
            str(trade_date.day) + '.csv'
        )

        if not os.path.exists(dtable_path):
            logging.info("there is no date_table : ")
            # Create new dtable with feature engineering
            self.create_dtable(backtest)
        else:
            logging.info(
                "there is parquet file for this date. read date table from parquet. "
                "please check dtable file is recent version"
            )
            # Load cached dtable
            self.dtable = pd.read_csv(dtable_path)

        # Map industry to sector (required for some strategies)
        self.dtable["sector"] = self.dtable["industry"].map(sector_map)

    def get_price_for_dtable(self, backtest: 'Backtest') -> None:
        """Load and filter price data for active stocks.

        This method:
        1. Filters stocks that are not delisted as of current date
        2. Merges with current date's price data
        3. Calculates total trading value (volume * price)
        4. Filters to top 10% by trading value to avoid illiquid stocks

        Args:
            backtest: Backtest instance with symbol_table and price_table

        Note:
            Trading value filter ensures we only consider liquid stocks
            Threshold can be adjusted via TRADING_VALUE_THRESHOLD config
        """
        # Filter stocks that are still active (not delisted)
        query = '(delistedDate >= "{}") or (delistedDate == "NaT") or (delistedDate == "None")'.format(
            self.date
        )
        self.dtable = backtest.symbol_table.query(query).copy()
        self.dtable = self.dtable.assign(score=0)

        # Get price data for this date
        trade_date = backtest.get_trade_date(self.date)
        price = backtest.price_table.query("date == @trade_date")
        price = price[['symbol', 'date', 'close', 'volume', 'marketCap']].copy()
        price = price.drop_duplicates('symbol', keep='first')

        # Merge price data
        self.dtable = pd.merge(self.dtable, price, how='left', on='symbol')

        # Calculate total trading value (volume * price)
        self.dtable['volume_mul_price'] = self.dtable['close'] * self.dtable['volume']

        # Remove stocks with no trading data
        self.dtable = self.dtable.dropna(subset='volume_mul_price')

        # Keep only top 10% by trading value to ensure liquidity
        self.dtable = self.dtable.nlargest(
            int(len(self.dtable) * 0.10),
            'volume_mul_price',
            keep='all'
        )

    def remove_x_y_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove and merge duplicate columns from pandas merge operations.

        When merging DataFrames, pandas creates _x and _y suffixes for duplicate
        column names. This method:
        1. Removes _x_x, _y_y, _x_y, _y_x columns (double suffixes)
        2. Merges _x and _y column pairs by preferring _x values
        3. Renames merged columns to original name (without suffix)

        Args:
            df: DataFrame with potential _x and _y suffixed columns

        Returns:
            DataFrame with duplicate columns merged and cleaned

        Example:
            >>> df = pd.merge(df1, df2, on='symbol')
            >>> # Results in 'revenue_x' and 'revenue_y' columns
            >>> df = remove_x_y_columns(df)
            >>> # Now has single 'revenue' column
        """
        new_df = df.copy()

        # Remove double-suffixed columns
        columns_to_drop = [
            col for col in new_df.columns
            if col.endswith('_x_x') or col.endswith('_y_y') or
               col.endswith('_x_y') or col.endswith('_y_x')
        ]
        new_df = new_df.drop(columns=columns_to_drop)

        # Find _x and _y column pairs
        columns_pairs = {}
        for col in new_df.columns:
            if col.endswith('_x'):
                columns_pairs[col] = col.replace('_x', '_y')

        # Merge each pair
        for col_x, col_y in columns_pairs.items():
            # Prefer _x column, use _y if _x is null
            new_df[col_x] = np.where(
                new_df[col_x].notnull(),
                new_df[col_x],
                new_df[col_y]
            )

            # Rename to original name (remove _x suffix)
            new_col_name = col_x[:-2]
            new_df = new_df.rename(columns={col_x: new_col_name})

            # Drop _y column
            new_df = new_df.drop(columns=[col_y])

        return new_df

    def get_fs_metrics(self, backtest: 'Backtest') -> pd.DataFrame:
        """Load and merge financial statements and metrics for current date.

        This method:
        1. Loads most recent financial statements (within last 6 months)
        2. Merges with financial metrics
        3. Creates relative features by dividing by market cap

        The 6-month window ensures we get the latest available financial data
        while allowing time for filing delays.

        Args:
            backtest: Backtest instance with fs_table and metrics_table

        Returns:
            DataFrame with financial statements, metrics, and derived features

        Note:
            Creates 'OverMC_' prefixed columns for market-cap-normalized features
        """
        # Get financial statements from last 6 months
        prev = self.date - relativedelta(months=6)
        fs = backtest.fs_table.copy()

        # Filter to recent filings and keep most recent per symbol
        fs = fs[fs.fillingDate <= self.date]
        fs = fs[prev <= fs.fillingDate]
        fs = fs.drop_duplicates('symbol', keep='first')

        # Keep only symbols in our dtable (active stocks)
        dsymbol = self.dtable[['symbol']].copy()
        fs = pd.merge(dsymbol, fs, how='inner', on=['symbol'])

        # Merge with metrics
        metrics = backtest.metrics_table.copy()
        fs_metrics = pd.merge(fs, metrics, how='left', on=['symbol', 'date'])

        # Add market cap for relative feature calculation
        cap = self.dtable[['symbol', 'marketCap']].copy()
        cap.rename(columns={'marketCap': 'cal_marketCap'}, inplace=True)
        fs_metrics = pd.merge(fs_metrics, cap, how='left', on=['symbol'])

        # Clean up duplicate columns from merges
        fs_metrics = self.remove_x_y_columns(fs_metrics)

        # Create relative features (normalized by market cap)
        # This makes features comparable across different-sized companies
        for col in meaning_col_list:
            if col not in fs_metrics.columns:
                logging.warn(f"there is no {col} column in fs_metrics table")
                continue

            new_col_name = 'OverMC_' + col
            fs_metrics[new_col_name] = np.where(
                fs_metrics['cal_marketCap'] > 0,
                fs_metrics[col] / fs_metrics['cal_marketCap'],
                np.nan
            )

        return fs_metrics

    def create_dtable(self, backtest: 'Backtest') -> None:
        """Create comprehensive feature table with time-series features.

        This is the main feature engineering pipeline that:
        1. Loads current prices and filters by trading volume
        2. Loads latest financial statements and metrics
        3. Creates time-series features for multiple periods
        4. Extracts time-series patterns using tsfresh
        5. Creates enterprise value ratios
        6. Preprocesses all features (direction, normalization)
        7. Filters sparse columns and rows
        8. Saves to cache

        Time-Series Feature Engineering:
        - For each period (3, 6, 9, 12, 15, 18, 21, 24 months):
          - Load historical financial data
          - Calculate percentage changes
        - Use tsfresh to extract statistical patterns from time series

        Feature Preprocessing:
        - Adjust feature directions so "high is good" uniformly
        - Handle missing values
        - Filter very sparse features (>50% missing)

        Args:
            backtest: Backtest instance with all data tables

        Note:
            First run is slow (feature engineering + tsfresh)
            Creates cached dtable for fast loading next time

        TODO:
            Consider removing diff features since time-series patterns may be sufficient
        """
        # Step 1: Get current prices and filter by trading volume
        self.get_price_for_dtable(backtest)

        # Step 2: Get latest financial statements and metrics
        fs_metrics = self.get_fs_metrics(backtest)

        # Step 3: Prepare for time-series feature extraction
        # Rename current values to prev0 (most recent)
        for col_name in cal_timefeature_col_list:
            fs_metrics[f'prev0_{col_name}'] = fs_metrics[col_name]

        # Step 4: Load historical data for each time period
        for prev_n in time_periods:
            prefix_col_name = "prev" + str(prev_n) + "_"

            # Get financial data from prev_n months ago
            prev_date = self.date - relativedelta(months=prev_n)
            prev_prev_date = self.date - relativedelta(months=prev_n+4)
            prev_fs = backtest.fs_table.copy()
            prev_fs = prev_fs[prev_fs.fillingDate <= prev_date]
            prev_fs = prev_fs[prev_prev_date <= prev_fs.fillingDate]
            prev_fs = prev_fs.drop_duplicates('symbol', keep='first')

            # Merge with metrics
            metrics = backtest.metrics_table.copy()
            prev_fs_metrics = pd.merge(prev_fs, metrics, how='left', on=['symbol', 'date'])
            prev_fs_metrics = self.remove_x_y_columns(prev_fs_metrics)

            # Rename columns with prefix and merge
            symbols = prev_fs_metrics['symbol']
            prev_fs_metrics = prev_fs_metrics[meaning_col_list].copy()
            prev_fs_metrics = prev_fs_metrics.rename(columns=lambda x: prefix_col_name + x)
            prev_fs_metrics['symbol'] = symbols
            fs_metrics = pd.merge(fs_metrics, prev_fs_metrics, how='left', on=['symbol'])

        # Step 5: Calculate percentage changes for key periods
        for prev_n in [3, 6, 12, 24]:
            prefix_col_name = "prev" + str(prev_n) + "_"

            for col in meaning_col_list:
                # Calculate percentage change from prev_n months ago
                new_col_name = "diff" + str(prev_n) + "_" + col
                fs_metrics[new_col_name] = np.where(
                    fs_metrics[prefix_col_name+col] > 0,
                    (fs_metrics[col] - fs_metrics[prefix_col_name+col]) /
                    fs_metrics[prefix_col_name+col],
                    np.nan
                )

        # Step 6: Extract time-series patterns using tsfresh
        long_format_list = []
        time_periods_w0 = [0] + time_periods  # Include current period

        # Convert to long format for tsfresh
        for prev_n in time_periods_w0:
            for col in cal_timefeature_col_list:
                temp_df = pd.DataFrame({
                    "id": fs_metrics.symbol,
                    "time": prev_n,
                    "value": fs_metrics["prev" + str(prev_n) + "_" + col],
                    "kind": col
                })
                long_format_list.append(temp_df)

        # Concatenate and extract features
        long_format_df = pd.concat(long_format_list, ignore_index=True)
        print(long_format_df)
        long_format_df = long_format_df.dropna(subset=['value'])

        # Use tsfresh to extract time-series features
        settings = EfficientFCParameters()
        try:
            extracted_features = extract_features(
                long_format_df,
                column_id='id',
                column_sort='time',
                column_kind='kind',
                column_value='value',
                default_fc_parameters=settings
            )
            print(extracted_features.head())

            # Add 'fresh_' prefix to distinguish from original features
            extracted_features.columns = ['fresh_' + col for col in extracted_features.columns]
            fs_metrics = fs_metrics.reset_index().merge(
                extracted_features,
                left_on='symbol',
                right_index=True
            ).set_index('index')
        except Exception as e:
            print("failed extract features : ")
            print(str(self.date.year) + '_' + str(self.date.month) + '_' + str(self.date.day))
            pass

        # Step 7: Remove prev columns (only needed for diff/tsfresh)
        for prev_n in time_periods:
            prefix_col_name = "prev" + str(prev_n) + "_"
            for col in meaning_col_list:
                fs_metrics = fs_metrics.drop([prefix_col_name+col], axis=1)

        # Step 8: Remove absolute value columns (keep only ratios)
        abs_col_list = list(set(meaning_col_list) - set(ratio_col_list))
        for col in abs_col_list:
            fs_metrics = fs_metrics.drop([col], axis=1)

        # Step 9: Filter rows with too many missing values (>70%)
        logging.info("before fs_metric len : {}".format(len(fs_metrics)))
        fs_metrics['nan_count_per_row'] = fs_metrics.isnull().sum(axis=1)
        filtered_row = fs_metrics['nan_count_per_row'] < int(len(fs_metrics.columns)*0.7)
        fs_metrics = fs_metrics.loc[filtered_row, :]
        logging.info("after fs_metric len : {}".format(len(fs_metrics)))

        # Step 10: Create enterprise value (EV) based features
        # EV = Market Cap + Net Debt
        fs_metrics["adaptiveMC_ev"] = fs_metrics['cal_marketCap'] + fs_metrics["netDebt"]

        for col in cal_ev_col_list:
            new_col_name = 'adaptiveMC_' + col
            fs_metrics[new_col_name] = np.where(
                fs_metrics[col] > 0,
                fs_metrics['adaptiveMC_ev'] / fs_metrics[col],
                np.nan
            )

        # Step 11: Preprocess features (adjust directions)
        # Load feature direction information (high/low is good)
        highlow = pd.read_csv('./sort.csv', header=0)

        for feature in fs_metrics.columns:
            # Skip derived features (they'll use original feature's direction)
            if (str(feature).startswith("diff") or
                str(feature).startswith("fresh_") or
                str(feature).startswith("OverMC_") or
                str(feature).startswith("adaptiveMC_")):
                continue

            feature_name = feature

            # Only process features in our known lists
            if (feature_name not in meaning_col_list) and (feature_name not in ratio_col_list):
                continue

            # Get direction for this feature
            f = highlow.query("name == @feature_name")
            if f.empty:
                continue

            # Create _sorted column (high is always good)
            feature_sortedvalue_col_name = feature + "_sorted"
            fs_metrics[feature_sortedvalue_col_name] = fs_metrics[feature].copy()

            # If "low is good", invert the values
            if f.iloc[0].sort == "low":
                try:
                    feat_max = fs_metrics[feature].max()
                    # Positive values become negative (inverted)
                    # Negative values become even more negative (worse than positive)
                    fs_metrics[feature_sortedvalue_col_name] = [
                        s*(-1) if s >= 0 else (s - feat_max)
                        for s in fs_metrics[feature]
                    ]
                except Exception as e:
                    logging.info(str(e))
                    continue

        # Step 12: Merge features into main dtable
        self.dtable = pd.merge(self.dtable, fs_metrics, how='left', on='symbol')

        # Step 13: Track sparse columns (>50% missing) globally
        columns_with_nan_above_threshold = self.dtable.columns[
            self.dtable.isnull().sum(axis=0) >= int(len(self.dtable)*0.5)
        ].tolist()

        for c in columns_with_nan_above_threshold:
            DateHandler.global_sparse_col[c] += 1

        logging.info("DateHandler.global_sparse_col : ")
        for k, v in DateHandler.global_sparse_col.items():
            logging.debug(f"{k} - {v}")

        # Step 14: Map industry to sector and remove duplicates
        self.dtable["sector"] = self.dtable["industry"].map(sector_map)
        self.dtable = self.dtable.drop_duplicates('symbol', keep='first')

        # Step 15: Save to cache for future use
        self.dtable.to_csv(
            backtest.conf['ROOT_PATH'] + "/DATE_TABLE/" +
            'dtable_' + str(self.date.year) + '_' +
            str(self.date.month) + '_' + str(self.date.day) + '.csv',
            index=False
        )

        logging.info("END create dtable in date handler ")


class EvaluationHandler:
    """Calculates performance metrics and generates backtesting reports.

    This class is responsible for:
    1. Collecting best-k stocks for each rebalancing period
    2. Looking up actual prices at buy and sell dates
    3. Calculating returns for each period
    4. Computing performance metrics (MDD, Sharpe ratio, total return)
    5. Generating comprehensive reports (EVAL, RANK, AVG)

    The evaluation process:
    1. During backtesting, set_best_k() collects top stocks for each period
    2. After all periods, run() performs:
       a. cal_price(): Look up prices at buy/sell dates
       b. cal_earning(): Calculate returns per period
       c. cal_mdd(): Calculate maximum drawdown
       d. cal_sharp(): Calculate Sharpe ratio (TODO)
       e. print_report(): Generate and save reports

    Reports Generated:
    - EVAL Report: Performance metrics per period and overall statistics
    - RANK Report: Detailed stock rankings with features for each period
    - AVG Report: Aggregated statistics across all periods

    Attributes:
        best_k: List of tuples for each rebalancing period:
            [0] date: Start date (buy date)
            [1] rebalance_date: End date (sell date)
            [2] best_group: DataFrame of selected stocks with scores
            [3] reference_group: DataFrame of all stocks with rankings
            [4] period_earning_rate: Return for this period
        historical_earning_per_rebalanceday: Historical earnings data (deprecated)
        backtest: Reference to parent Backtest instance
        member_cnt: Number of stocks to actually purchase per period
        accumulated_earning: Cumulative earnings across all periods
        MDD: Maximum drawdown percentage
        sharp: Sharpe ratio (TODO: not implemented)
        total_asset: Initial asset amount

    Example:
        ```python
        eval_handler = EvaluationHandler(backtest)

        # During backtesting
        eval_handler.set_best_k(date, rebalance_date, scored_datehandler)

        # After backtesting
        eval_handler.run(price_table)
        ```
    """

    def __init__(self, backtest: 'Backtest') -> None:
        """Initialize EvaluationHandler.

        Args:
            backtest: Parent Backtest instance
        """
        self.best_k: List[List[Any]] = []
        self.historical_earning_per_rebalanceday: List[Any] = []
        self.backtest = backtest
        self.member_cnt = self.cal_member_cnt()
        self.accumulated_earning: float = 0
        self.MDD: float = 0
        self.sharp: float = 0
        self.total_asset = backtest.conf['TOTAL_ASSET']

    def cal_member_cnt(self) -> int:
        """Calculate number of stocks to purchase per period.

        Returns:
            Number of stocks from MEMBER_CNT config (default: 4)

        Note:
            This determines portfolio diversification.
            E.g., MEMBER_CNT=4 means buying 4 stocks each period.
        """
        return self.backtest.conf['MEMBER_CNT']

    def print_current_best(self, scored_dh: 'DateHandler') -> None:
        """Print current top stock recommendations to result.csv.

        This method is called for the most recent date to generate
        current recommendations (no historical returns available).

        Args:
            scored_dh: DateHandler with scored stocks

        Note:
            Saves to ./result.csv with top MEMBER_CNT stocks
        """
        # Select top stocks by score
        best_symbol = scored_dh.dtable.sort_values(
            by=["score"],
            axis=0,
            ascending=False
        ).head(self.member_cnt)
        best_symbol = best_symbol.assign(count=0)
        best_symbol.to_csv('./result.csv')

    def set_best_k(
        self,
        date: datetime.datetime,
        rebalance_date: datetime.datetime,
        scored_dh: 'DateHandler'
    ) -> None:
        """Add best-k stocks for a rebalancing period to evaluation list.

        This method is called for each rebalancing period during backtesting.
        It selects the top stocks by score and stores them for later evaluation.

        Args:
            date: Start date (buy date) for this period
            rebalance_date: End date (sell date) for this period
            scored_dh: DateHandler with stock scores for this date

        Note:
            Selects MEMBER_CNT * 2 stocks to allow for price lookup failures
            Actual portfolio will be reduced to MEMBER_CNT after price lookup
        """
        if self.backtest.eval_report_path is not None:
            # Select top stocks by score (2x member count for safety)
            best_symbol = scored_dh.dtable.sort_values(
                by=["score"],
                axis=0,
                ascending=False
            ).head(self.member_cnt * 2)
            best_symbol = best_symbol.assign(count=0)
        else:
            best_symbol = pd.DataFrame()

        reference_group = pd.DataFrame()
        period_earning_rate = 0

        # Store for later evaluation
        self.best_k.append([
            date,
            rebalance_date,
            best_symbol,
            reference_group,
            period_earning_rate
        ])

    def cal_price(self) -> None:
        """Look up actual prices for selected stocks at buy and sell dates.

        This method fills in the 'price' and 'rebalance_day_price' columns
        in best_k by looking up actual prices from the price_table.

        For each period:
        1. Create DateHandler for start date (buy date)
        2. Create DateHandler for end date (sell date)
        3. Look up prices for each selected stock
        4. Calculate period_price_diff for RANK report

        The process handles:
        - Missing prices (stocks with no data)
        - Zero prices (delisted stocks)
        - Year transitions (reload data tables)

        Note:
            Prices are looked up using DateHandler which ensures proper
            feature-engineered data context for each date
        """
        pd.set_option('mode.chained_assignment', None)
        logging.info("best k length : %d", len(self.best_k))

        for idx, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            # Reload data tables if year changed
            if date.year != self.backtest.table_year:
                logging.info("Reload BackTest Table. year : {} -> {}".format(
                    self.backtest.table_year, date.year
                ))
                self.backtest.load_bt_table(date.year)
                self.backtest.table_year = date.year

            # Handle latest date (current recommendations)
            if (idx == len(self.best_k) - 1) and (len(self.best_k) != 1):
                logging.info("print latest data : " + date.strftime("%Y-%m-%d"))
                self.best_k[idx][3] = start_dh.dtable
                self.best_k[idx][3] = self.best_k[idx][3][self.best_k[idx][3].close > 0.000001]
                self.best_k[idx][3].rename(columns={'close': 'price'}, inplace=True)
                break

            # Create DateHandlers for start and end dates
            if idx == 0:
                start_dh = DateHandler(self.backtest, date)
            end_dh = DateHandler(self.backtest, rebalance_date)

            # Generate RANK report if requested
            if self.backtest.rank_report_path is not None:
                logging.info("rank/ai report cur date : {}".format(date))

                # Get all stocks with their features
                self.best_k[idx][3] = start_dh.dtable

                # Look up rebalance date prices
                rebalance_date_price_df = end_dh.dtable[['symbol', 'close']].copy()
                rebalance_date_price_df.rename(
                    columns={'close': 'rebalance_day_price'},
                    inplace=True
                )
                self.best_k[idx][3] = pd.merge(
                    self.best_k[idx][3],
                    rebalance_date_price_df,
                    how='outer',
                    on='symbol'
                )

                # Filter valid prices
                self.best_k[idx][3] = self.best_k[idx][3][self.best_k[idx][3].close > 0.000001]

                # Calculate period return
                diff = self.best_k[idx][3]['rebalance_day_price'] - self.best_k[idx][3]['close']
                self.best_k[idx][3]['period_price_diff'] = diff / self.best_k[idx][3]['close']

                # Create rank columns for all features
                for feature in self.best_k[idx][3].columns:
                    if '_sorted' in feature:
                        feature_rank_col_name = feature + "_rank"
                        self.best_k[idx][3][feature_rank_col_name] = \
                            self.best_k[idx][3][feature].rank(method='max', ascending=False)

                # Keep only top-k for report
                self.best_k[idx][3] = self.best_k[idx][3].sort_values(
                    by=["period_price_diff"],
                    axis=0,
                    ascending=False
                )[:self.backtest.conf['TOP_K_NUM']]
            else:
                # No RANK report
                self.best_k[idx][3] = pd.DataFrame()

            # Look up prices for selected stocks (EVAL report)
            if self.backtest.eval_report_path is not None:
                syms = best_group['symbol']

                for sym in syms:
                    # Look up start date price
                    if start_dh.dtable.loc[(start_dh.dtable['symbol'] == sym), 'close'].empty:
                        logging.debug(
                            "there is no price in start_dh FMP API  symbol : {}".format(sym)
                        )
                        self.best_k[idx][2].loc[
                            (self.best_k[idx][2].symbol == sym),
                            'price'
                        ] = 0
                    else:
                        self.best_k[idx][2].loc[
                            (self.best_k[idx][2].symbol == sym),
                            'price'
                        ] = start_dh.dtable.loc[
                            (start_dh.dtable['symbol'] == sym),
                            'close'
                        ].values[0]

                    # Look up rebalance date price
                    if end_dh.dtable.loc[(end_dh.dtable['symbol'] == sym), 'close'].empty:
                        logging.debug(
                            "there is no price in end_dh FMP API  symbol : {}".format(sym)
                        )
                        self.best_k[idx][2].loc[
                            (self.best_k[idx][2].symbol == sym),
                            'rebalance_day_price'
                        ] = 0
                    else:
                        self.best_k[idx][2].loc[
                            (self.best_k[idx][2].symbol == sym),
                            'rebalance_day_price'
                        ] = end_dh.dtable.loc[
                            (end_dh.dtable['symbol'] == sym),
                            'close'
                        ].values[0]

                        # Check for delisted stocks (price near 0)
                        if end_dh.dtable.loc[(end_dh.dtable['symbol'] == sym), 'close'].values[0] < 0.01:
                            logging.debug("close price already 0 : {}".format(sym))

                logging.debug(str(self.best_k[idx][2]))

                # Filter out stocks with invalid prices
                self.best_k[idx][2] = self.best_k[idx][2][
                    self.best_k[idx][2].rebalance_day_price > 0.000001
                ]

            # Reduce to actual portfolio size
            self.best_k[idx][2] = self.best_k[idx][2].head(self.member_cnt)

            # Carry over end_dh to next iteration as start_dh
            start_dh = copy.deepcopy(end_dh)

            logging.info(str(idx) + " " + str(date))

    def cal_earning_no_parallel(self) -> None:
        """Calculate returns for each period (non-parallel version).

        This method calculates portfolio returns by:
        1. Allocating equal amounts to each selected stock
        2. Calculating shares to buy based on start price
        3. Calculating portfolio value at rebalance date
        4. Computing period return

        Note:
            This is a serial version (no multiprocessing)
            Used for debugging or when multiprocessing causes issues
        """
        logging.info("START cal_earning")
        params = copy.deepcopy(self.best_k)
        logging.info("in cal_earning : params : ")

        for best in self.best_k:
            (date, rebalance_date, best_group, reference_group, period_earning_rate) = best

            # Check if price data is available
            if 'price' not in best_group.columns:
                logging.warning("No Price Column!!")
                continue

            # Allocate total asset equally among selected stocks
            total_asset = 100000000  # TODO: Use config value
            stock_cnt = (total_asset / len(best_group)) / best_group['price']
            stock_cnt = stock_cnt.replace([np.inf, -np.inf], 0)
            stock_cnt = stock_cnt.fillna(0)

            # Calculate invested amount
            price_mul_stock_cnt = best_group['price'] * stock_cnt
            my_asset_period = price_mul_stock_cnt.sum()
            remain_asset = total_asset - price_mul_stock_cnt.sum()

            if my_asset_period == 0:
                return

            # Store stock counts for MDD calculation
            best[2]['count'] = stock_cnt

            # Calculate period earnings
            rebalance_day_price_mul_stock_cnt = best_group['rebalance_day_price'] * stock_cnt
            best[2]['period_earning'] = rebalance_day_price_mul_stock_cnt - price_mul_stock_cnt
            period_earning = rebalance_day_price_mul_stock_cnt.sum() - price_mul_stock_cnt.sum()
            best[4] = period_earning

    @staticmethod
    def cal_earning_func(best_k: List[Any]) -> Optional[List[Any]]:
        """Calculate returns for a single period (static method for multiprocessing).

        This is the worker function for parallel return calculation.

        Args:
            best_k: Single element from self.best_k list containing:
                [0] date: Start date
                [1] rebalance_date: End date
                [2] best_group: Selected stocks
                [3] reference_group: All stocks
                [4] period_earning_rate: Return (to be calculated)

        Returns:
            Updated best_k element with calculated returns, or None if error

        Note:
            This must be a static method to work with multiprocessing
        """
        (date, rebalance_date, best_group, reference_group, period_earning_rate) = best_k

        # Check if price data is available
        if 'price' not in best_group.columns:
            return None

        # Allocate total asset equally among selected stocks
        total_asset = 100000000
        stock_cnt = (total_asset / len(best_group)) / best_group['price']
        stock_cnt = stock_cnt.replace([np.inf, -np.inf], 0)
        stock_cnt = stock_cnt.fillna(0)

        # Calculate invested amount
        price_mul_stock_cnt = best_group['price'] * stock_cnt
        my_asset_period = price_mul_stock_cnt.sum()
        remain_asset = total_asset - price_mul_stock_cnt.sum()

        if my_asset_period == 0:
            return None

        # Store stock counts for MDD calculation
        best_k[2]['count'] = stock_cnt

        # Calculate period earnings
        rebalance_day_price_mul_stock_cnt = best_group['rebalance_day_price'] * stock_cnt
        best_k[2]['period_earning'] = rebalance_day_price_mul_stock_cnt - price_mul_stock_cnt
        period_earning = rebalance_day_price_mul_stock_cnt.sum() - price_mul_stock_cnt.sum()
        best_k[4] = period_earning

        return best_k

    def cal_earning(self) -> None:
        """Calculate returns for all periods using multiprocessing.

        This method uses multiprocessing to calculate returns in parallel,
        significantly improving performance for long backtests.

        Note:
            Uses (CPU count - 4) processes to leave resources for other tasks
            Results are stored back into self.best_k
        """
        logging.info("START cal_earning")
        params = copy.deepcopy(self.best_k)

        logging.info("in cal_earning : params : ")

        # Use multiprocessing to calculate returns in parallel
        with Pool(processes=multiprocessing.cpu_count()-4, initializer=install_mp_handler()) as pool:
            df_list = pool.map(self.cal_earning_func, params)

        # Filter out None results (failed calculations)
        df_list = list(filter(None.__ne__, df_list))

        logging.info("in cal_earning : df_list : ")

        # Update self.best_k with calculated returns
        self.best_k = df_list

        for elem in df_list:
            if elem == None:
                continue
            (date, rebalance_date, best_group, reference_group, period_earning) = elem
            logging.debug(date)
            logging.debug(period_earning)

            # Save debug output if log level is DEBUG
            if self.backtest.main_ctx.log_lvl == 10:
                best_group.to_csv("./earning_test.csv")

    def cal_mdd(self, price_table: pd.DataFrame) -> None:
        """Calculate Maximum Drawdown (MDD) across the entire backtest period.

        MDD measures the largest peak-to-trough decline in portfolio value,
        indicating the worst possible loss an investor could have experienced.

        The calculation:
        1. For each day between rebalancing periods:
           a. Calculate portfolio value (shares * current price)
           b. Track maximum value seen so far (peak)
           c. Track minimum value seen so far (trough)
        2. MDD = (trough - peak) / peak * 100

        Args:
            price_table: DataFrame with daily price data

        Note:
            Requires 'count' column in best_group (shares owned per stock)
            Updates self.MDD with calculated maximum drawdown percentage

        TODO:
            Fix SettingWithCopyWarning in line with my_asset calculation
        """
        best_asset = -1
        worst_asset = self.total_asset * 100000

        for i, (date, rebalance_date, best_group, reference_group, period_earning_rate) in enumerate(self.best_k):
            if i == 0:
                prev_date = date
                continue
            else:
                # Get all prices for all days in this period
                allday_price_allsymbol = []
                syms = best_group['symbol']

                # For each held stock, get daily prices
                for sym in syms:
                    allday_price_per_symbol = price_table.query(
                        "(symbol == @sym) and (date <= @date and date >= @prev_date)"
                    )

                    if allday_price_per_symbol.empty:
                        continue
                    else:
                        # Calculate daily value for this stock (shares * price)
                        count_per_sym = best_group.loc[(best_group.symbol == sym), 'count'].values
                        allday_price_per_symbol = allday_price_per_symbol.assign(
                            my_asset=lambda x: x.close * count_per_sym
                        )
                        allday_price_allsymbol.append(allday_price_per_symbol)

                if allday_price_allsymbol == "":
                    logging.warning("allday_price_allsymbol is empty. can't calc MDD.")
                    return

                # Sum portfolio value across all stocks for each day
                accum_df = pd.DataFrame()
                for j, df in enumerate(allday_price_allsymbol):
                    df = df.reset_index(drop=True)
                    if j == 0:
                        accum_df = df[['date', 'my_asset']].copy()
                    else:
                        accum_df = accum_df[['my_asset']] + df[['my_asset']]

                # Add date column back
                accum_df['date'] = df['date']

                # Track peak and trough
                if accum_df['my_asset'].max(axis=0) > best_asset:
                    best_asset = accum_df['my_asset'].max(axis=0)
                    best_date = accum_df.loc[accum_df['my_asset'].idxmax(), 'date']

                if accum_df['my_asset'].min(axis=0) < worst_asset:
                    worst_asset = accum_df['my_asset'].min(axis=0)
                    worst_date = accum_df.loc[accum_df['my_asset'].idxmin(), 'date']

                prev_date = date

        # Calculate MDD percentage
        mdd = ((worst_asset / best_asset) - 1) * 100
        logging.info(
            "MDD : {:.2f}%, best date : {}, worst date : {}".format(
                mdd, best_date, worst_date
            )
        )
        self.MDD = mdd

    def cal_sharp(self) -> None:
        """Calculate Sharpe ratio for the strategy.

        Sharpe ratio measures risk-adjusted returns:
        Sharpe = (Return - RiskFreeRate) / StdDev

        Higher values indicate better risk-adjusted performance.

        TODO:
            This method is not yet implemented
            Need to calculate annualized return and standard deviation
        """
        sharp = 0
        self.sharp = sharp

    @staticmethod
    def write_csv(
        path: str,
        date: datetime.datetime,
        rebalance_date: datetime.datetime,
        elem: pd.DataFrame
    ) -> None:
        """Write a period's results to CSV report.

        Args:
            path: Report file path
            date: Period start date
            rebalance_date: Period end date
            elem: DataFrame with period results
        """
        fd = open(path, 'a')
        writer = csv.writer(fd, delimiter=",")
        writer.writerow("")
        writer.writerow(["start", date, "end", rebalance_date])
        fd.close()
        elem.to_csv(path, mode="a")

    def print_report(self) -> None:
        """Generate and save comprehensive backtesting reports.

        This method generates three types of reports:

        1. EVAL Report:
           - Period-by-period performance
           - Selected stocks and their returns
           - Cumulative returns
           - Comparison with reference indices (S&P 500, QQQ, etc.)
           - Overall statistics

        2. RANK Report:
           - Detailed stock rankings for each period
           - All features and their ranks
           - Actual returns for each stock
           - Top-k stocks by actual performance

        3. AVG Report:
           - Aggregated statistics across all periods
           - Average feature values
           - Average returns by characteristics

        The reports are saved to files specified in create_report().

        Note:
            Compares strategy performance with reference symbols (benchmarks)
            Calculates accumulated earnings using compound returns
        """
        plan_earning = 1
        total_asset = 100000000
        accumulated_earning = 100  # Start at 100% (1.0x)
        max_local_plan_earning = -9999999999999
        min_local_plan_earning = 9999999999999

        # Generate reports for each period
        for idx, (date, rebalance_date, eval_elem, rank_elem, period_earning_rate) in enumerate(self.best_k):
            # Calculate local return for this period
            local_plan_earning = period_earning_rate / total_asset
            accumulated_earning = accumulated_earning * (1.0 + local_plan_earning)

            # Write EVAL report
            if self.backtest.eval_report_path is not None:
                # Track best and worst periods
                if max_local_plan_earning < local_plan_earning:
                    max_local_plan_earning = local_plan_earning
                if min_local_plan_earning > local_plan_earning:
                    min_local_plan_earning = local_plan_earning

                # Write period results
                self.write_csv(
                    self.backtest.eval_report_path,
                    date,
                    rebalance_date,
                    eval_elem
                )

                # Append earnings
                fd = open(self.backtest.eval_report_path, 'a', newline='')
                writer = csv.writer(fd, delimiter=",")
                writer.writerow([str(period_earning_rate)])
                writer.writerow([str(accumulated_earning)])
                fd.close()

            # Write RANK report (only for first few periods)
            if self.backtest.rank_report_path is not None:
                if idx <= self.backtest.conf['RANK_PERIOD']:
                    # Save individual period file
                    rank_partial_path = (
                        self.backtest.rank_report_path[:-4] + '_' +
                        str(date.year) + '_' + str(date.month) + '_' +
                        str(date.day) + '.csv'
                    )
                    rank_elem.to_csv(rank_partial_path, index=False)

                    # Append to main rank report
                    self.write_csv(
                        self.backtest.rank_report_path,
                        date,
                        rebalance_date,
                        rank_elem
                    )

            # Write AVG report
            if self.backtest.avg_report_path is not None:
                rank_elem.to_csv(
                    self.backtest.avg_report_path,
                    mode="a",
                    index=False,
                    header=False
                )

        # Calculate benchmark returns and final statistics
        if self.backtest.eval_report_path is not None:
            ref_total_earning_rates = dict()

            # Calculate returns for reference symbols (benchmarks)
            for ref_sym in self.backtest.conf['REFERENCE_SYMBOL']:
                start_date = self.backtest.get_trade_date(
                    datetime.datetime(self.backtest.main_ctx.start_year, 1, 1)
                )
                end_date = self.backtest.get_trade_date(
                    datetime.datetime(self.backtest.main_ctx.end_year, 12, 31)
                )

                # Handle case where end date is in future
                if end_date is None:
                    end_date = self.backtest.price_table["date"].max()

                logging.debug(ref_sym)
                logging.debug(
                    "start_date : " + start_date.strftime("%Y-%m-%d") +
                    "    end_date : " + end_date.strftime("%Y-%m-%d")
                )

                # Get start and end prices
                reference_earning_df = self.backtest.price_table.query(
                    "(symbol == @ref_sym) and ((date == @start_date) or (date == @end_date))"
                )
                logging.debug(reference_earning_df)

                # Calculate return
                if len(reference_earning_df) == 2:
                    reference_earning = (
                        reference_earning_df.iloc[1]['close'] -
                        reference_earning_df.iloc[0]['close']
                    )
                    ref_total_earning_rate = (
                        reference_earning / reference_earning_df.iloc[0]['close']
                    ) * 100
                    ref_total_earning_rates[ref_sym] = ref_total_earning_rate
                else:
                    logging.info(
                        "REFERENCE_SYMBOL [ " + str(ref_sym) + " ] ( " +
                        start_date.strftime("%Y-%m-%d") + " ~ " +
                        end_date.strftime("%Y-%m-%d") +
                        " ) is Strange Value!!! NEED CHECK!!!"
                    )
                    ref_total_earning_rates[ref_sym] = 0

            # Calculate our strategy's total return
            plan_total_earning_rate = (accumulated_earning - 100) / 100

            # Log summary statistics
            logging.warning(
                "TOP_K_NUM : " + str(self.backtest.conf['TOP_K_NUM']) +
                ", MEMBER_CNT : " + str(self.backtest.conf['MEMBER_CNT']) +
                ", ABSOLUTE_SCORE : " + str(self.backtest.conf['ABSOLUTE_SCORE']) +
                ", Our_Earning : " + str(plan_total_earning_rate) +
                ", MAX_LOCAL_PLAN_EARNING : " + str(max_local_plan_earning) +
                ", MIN_LOCAL_PLAN_EARNING : " + str(min_local_plan_earning)
            )

            # Write final summary to report
            fd = open(self.backtest.eval_report_path, 'a')
            writer = csv.writer(fd, delimiter=",")
            writer.writerow("")
            writer.writerow(["ours", plan_total_earning_rate])

            # Write benchmark returns
            for ref_sym, total_earning_rate in ref_total_earning_rates.items():
                writer.writerow([ref_sym, total_earning_rate])
            fd.close()

    def run(self, price_table: pd.DataFrame) -> None:
        """Execute the complete evaluation pipeline.

        This is the main entry point for evaluation after backtesting.
        It runs the complete evaluation pipeline:

        1. cal_price(): Look up actual prices at buy/sell dates
        2. cal_earning(): Calculate returns for each period (if EVAL report)
        3. cal_mdd(): Calculate maximum drawdown (commented out)
        4. cal_sharp(): Calculate Sharpe ratio (commented out)
        5. print_report(): Generate and save reports

        Args:
            price_table: DataFrame with daily price data

        Note:
            MDD and Sharpe calculations are currently commented out
            for performance reasons but can be enabled if needed
        """
        # Look up prices for all periods
        self.cal_price()

        # Calculate returns if generating EVAL report
        if self.backtest.eval_report_path is not None:
            self.cal_earning()
            # self.cal_mdd(price_table)  # Uncomment to calculate MDD
            # self.cal_sharp()  # Uncomment to calculate Sharpe ratio

        # Generate all reports
        self.print_report()
