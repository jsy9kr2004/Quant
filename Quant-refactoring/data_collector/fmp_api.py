"""FMP API URL Builder and Configuration Module.

This module provides the FMPAPI class for parsing, configuring, and building
Financial Modeling Prep API URLs with various query parameters. It handles URL
construction for different API versions (v3, v4), query parameter management,
and generates batches of API URLs for parallel fetching.

The FMPAPI class supports:
- URL parsing and validation
- Query parameter extraction and manipulation
- Symbol-based URL generation
- Time-based URL generation (year, quarter, date ranges)
- Pagination for large datasets

Typical usage example:
    from context import MainContext

    main_ctx = MainContext()
    url = "https://financialmodelingprep.com/api/v3/income-statement/AAPL?period=quarter&apikey=xxx"
    api = FMPAPI(main_ctx, url)
    api.symbol_list = ['AAPL', 'MSFT', 'GOOGL']
    url_list = api.make_api_list()
"""

from urllib.parse import urlparse, parse_qs, urlencode
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any


class FMPAPI:
    """FMP API URL builder and configuration manager.

    Parses FMP API URLs, extracts query parameters, and generates batches of
    API call URLs based on symbols, time periods, or pagination requirements.

    This class handles both v3 and v4 API versions, manages query parameters,
    and creates file paths for storing downloaded data.

    Attributes:
        main_ctx: Main context object with configuration.
        fmp_url (str): Base FMP API URL.
        api_key (str): FMP API key for authentication.
        url (str): Original full URL provided during initialization.
        main_url (str): API endpoint path without query parameters.
        category (str): API category extracted from URL path.
        condition (Dict[str, List[str]]): Query parameters as key-value pairs.
        need_symbol (bool): Whether this API requires stock symbols.
        symbol_list (List[str]): List of stock symbols to fetch data for.
        is_v4 (bool): True if API version 4, False if version 3.
        file_path (str): Directory path for storing downloaded data.
        page_num (int): Current page number for paginated APIs.
        page_set_num (int): Number of pages to fetch per batch.
        fmp_api_logger: Logger instance for this API.

    Example:
        >>> url = "https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=xxx"
        >>> api = FMPAPI(main_ctx, url)
        >>> api.symbol_list = ['AAPL', 'MSFT']
        >>> urls = api.make_api_list()
    """

    def __init__(self, main_ctx, url: str) -> None:
        """Initialize FMPAPI with URL parsing and configuration.

        Parses the provided URL to extract the API endpoint, query parameters,
        and determines if symbols are needed. Creates the output directory structure.

        Args:
            main_ctx: Main context object containing configuration.
            url (str): Full FMP API URL to parse and configure.

        Raises:
            Exception: If the output directory cannot be created.

        Note:
            URLs with multiple lines (separated by backslash) will use only
            the first line. This handles potential formatting issues in CSV files.
        """
        # Handle multi-line URLs by taking only the first line
        # Backslash might indicate line continuation in CSV files
        if url.find('\\') != -1:
            url = url.split('\\')[0]

        # Parse URL into components
        parsed_url = urlparse(url)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        # Remove API key and limit from query parameters
        # These will be re-added when building final URLs
        if 'apikey' in query_params.keys():
            del query_params['apikey']
        if 'limit' in query_params.keys():
            del query_params['limit']

        # Determine API version (v3 or v4)
        is_v4 = False if 'api/v3/' in path else True

        # Extract category and check if symbols are needed
        if is_v4:
            category = path.split('api/v4/')[1] if is_v4 else path.split('api/v3/')[1]
            need_symbol = False
            if 'symbol' in query_params.keys():
                need_symbol = True
        else:
            # v3 API: symbols are in the path, not query parameters
            category = path.split('api/v4/')[1] if is_v4 else path.split('api/v3/')[1]
            need_symbol = False
            if main_ctx.ex_symbol in category:
                # Remove example symbol from category and path
                category = category.split(f'/{main_ctx.ex_symbol}')[0]
                path = path.split(f'/{main_ctx.ex_symbol}')[0]
                need_symbol = True

        self.main_ctx = main_ctx
        self.fmp_url: str = self.main_ctx.fmp_url
        self.api_key: str = self.main_ctx.api_key
        self.url: str = url
        self.main_url: str = path
        self.category: str = category
        self.condition: Dict[str, List[str]] = query_params
        self.need_symbol: bool = need_symbol
        self.symbol_list: List[str] = [] if self.need_symbol else [self.converted_category]
        self.is_v4: bool = is_v4
        self.file_path: str = f'{self.main_ctx.root_path}/{self.converted_category}'

        self.fmp_api_logger = self.main_ctx.get_logger(f'fmp_api({self.converted_category})')
        self.fmp_api_logger.info(f'FMPAPI {self.converted_category} initialized')

        # Create output directory
        if self.main_ctx.create_dir(self.file_path):
            self.fmp_api_logger.info(f'Directory {self.file_path} created successfully')
        else:
            self.fmp_api_logger.error(f'Directory {self.file_path} is not created')
            raise Exception(f'Directory {self.file_path} is not created')

        self.page_num: int = 0
        # Limit page batch size to avoid API rate limits
        # This controls how many pages are fetched per batch, not worker count
        self.page_set_num: int = min(8, os.cpu_count())

    @property
    def converted_category(self) -> str:
        """Convert category path to filesystem-safe directory name.

        Replaces slashes and dashes with underscores for use in file paths.

        Returns:
            str: Filesystem-safe category name.

        Example:
            >>> api.category = "income-statement/annual"
            >>> api.converted_category
            'income_statement_annual'
        """
        return self.category.replace("/", "-").replace("-", "_")

    @property
    def query_params_str(self) -> str:
        """Build query parameter string from condition dictionary.

        Converts the condition dictionary to URL-encoded query string format.
        Takes the first value from each parameter's list.

        Returns:
            str: URL-encoded query parameter string.

        Example:
            >>> api.condition = {'period': ['quarter'], 'year': [2023]}
            >>> api.query_params_str
            'period=quarter&year=2023'
        """
        return urlencode({k: v[0] for k, v in self.condition.items()}, doseq=True)

    @property
    def page_in_condition(self) -> bool:
        """Check if pagination is used in query parameters.

        Returns:
            bool: True if 'page' parameter exists in conditions.
        """
        return 'page' in self.condition.keys()

    def __full_url(self, symbol: str) -> str:
        """Build complete FMP API URL with all parameters.

        Constructs the full URL including base URL, endpoint path, query parameters,
        symbol (if needed), and API key. Handles both v3 and v4 API formats.

        Args:
            symbol (str): Stock symbol to include in URL (or category name if no symbol needed).

        Returns:
            str: Complete FMP API URL ready for HTTP request.

        Example:
            v3: https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=xxx
            v4: https://financialmodelingprep.com/api/v4/shares_float?symbol=AAPL&apikey=xxx
        """
        if self.is_v4:
            # v4 API: symbol in query parameters
            if self.need_symbol:
                self.condition['symbol'] = [symbol]
            return f'{self.fmp_url}{self.main_url}?{self.query_params_str}&apikey={self.api_key}'
        else:
            # v3 API: symbol in path
            if self.need_symbol:
                return f'{self.fmp_url}{self.main_url}/{symbol}?{self.query_params_str}&apikey={self.api_key}'
            else:
                return f'{self.fmp_url}{self.main_url}?{self.query_params_str}&apikey={self.api_key}'

    def __url_from_symbols(self, file_postfix: str) -> List[List[Any]]:
        """Generate API call parameters for all symbols in symbol_list.

        Creates a list of parameters for each symbol, including file path,
        symbol name, file postfix, and full URL. Checks for existing files
        to avoid duplicate downloads.

        Args:
            file_postfix (str): Suffix to append to output filename (e.g., '_2023_Q1').

        Returns:
            List[List[Any]]: List of parameter lists [file_path, symbol, postfix, url]
                for each symbol.

        Raises:
            Exception: If output file already exists for any symbol.

        Example:
            >>> api.symbol_list = ['AAPL', 'MSFT']
            >>> params = api.__url_from_symbols('_2023')
            >>> params[0]
            ['/data/profile', 'AAPL', '_2023', 'https://...']
        """
        api_list = []
        for symbol in self.symbol_list:
            # Check if file already exists (.csv or .csvx for empty data)
            if os.path.isfile(f'{self.file_path}/{symbol+file_postfix}.csv') \
               or os.path.isfile(f'{self.file_path}/{symbol+file_postfix}.csvx'):
                self.fmp_api_logger.info(f'Skip existing file: {self.file_path}/{symbol+file_postfix}.csv/csvx')
                continue

            api_list.append([self.file_path, symbol, file_postfix, self.__full_url(symbol)])
        return api_list

    def set_symbol_list(self, symbol_list: List[str]) -> None:
        """Set the list of stock symbols for data fetching.

        Validates that the API requires symbols before setting the list.
        For historical-price-full category, ETF symbols could be added here.

        Args:
            symbol_list (List[str]): List of stock symbols to fetch.

        Raises:
            Exception: If API doesn't require symbols but set_symbol_list is called.

        Note:
            TODO: Add ETF symbols (SPY, IVV, VTI, etc.) for historical-price-full.
        """
        if not self.need_symbol:
            self.fmp_api_logger.info('symbol이 필요하지 않은데 set하려함')
            raise Exception('symbol이 필요하지 않은데 set하려함')
        self.symbol_list = symbol_list

        # TODO: Add ETF symbols for historical price data
        if self.category == "historical-price-full":
            self.fmp_api_logger.info("add ETF symbols to get historical-price-full data")
            # self.symbol_list.append(pd.Series(["SPY", "IVV", "VTI", "VOO", "QQQ", "VEA", "IEFA"]))

        self.symbol_list = self.symbol_list.dropna()

    def __make_api_list_year(self) -> List[List[Any]]:
        """Generate API URLs for year-based or quarterly data.

        Creates URLs for each year in the configured range (start_year to end_year).
        Supports three modes based on query parameters:
        1. period: Creates URLs for 4 periods per year (P1-P4)
        2. quarter: Creates URLs for 4 quarters per year (Q1-Q4)
        3. neither: Creates one URL per year

        Returns:
            List[List[Any]]: List of API call parameters for all years/periods.

        Example:
            With year=2023 and period parameter:
            - AAPL_2023_P1.csv
            - AAPL_2023_P2.csv
            - ...
        """
        ret = []
        for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
            if 'period' in self.condition.keys():
                # Generate 4 periods per year
                for period in range(1, 5):
                    file_postfix = f'_{year}_P{period}'
                    self.condition['year'] = [year]
                    self.condition['period'] = [period]
                    ret.extend(self.__url_from_symbols(file_postfix))
            elif 'quarter' in self.condition.keys():
                # Generate 4 quarters per year
                for quarter in range(1, 5):
                    file_postfix = f'_{year}_Q{quarter}'
                    self.condition['year'] = [year]
                    self.condition['quarter'] = [quarter]
                    ret.extend(self.__url_from_symbols(file_postfix))
            else:
                # Generate one file per year
                file_postfix = f'_{year}'
                self.condition['year'] = [year]
                ret.extend(self.__url_from_symbols(file_postfix))
        return ret

    def __make_api_list_from_to(self) -> List[List[Any]]:
        """Generate API URLs with from/to date range parameters.

        Creates URLs for each year with date range from Jan 1 to Dec 31.
        Used for APIs that accept 'from' and 'to' date parameters.

        Returns:
            List[List[Any]]: List of API call parameters for all years.

        Example:
            For 2023: from=2023-01-01&to=2023-12-31
            Output: AAPL_2023.csv
        """
        ret = []
        for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
            file_postfix = f'_{year}'
            self.condition['from'] = [f'{year}-01-01']
            self.condition['to'] = [f'{year}-12-31']
            ret.extend(self.__url_from_symbols(file_postfix))
        return ret

    def __make_api_list_date(self) -> List[List[Any]]:
        """Generate API URLs for each day in the configured date range.

        Creates one URL per day from start_year/01/01 to end_year/12/31.
        Used for APIs that require a specific date parameter.

        Returns:
            List[List[Any]]: List of API call parameters for all dates.

        Warning:
            This can generate a very large number of URLs (365+ days per year).
            Use carefully to avoid rate limiting.

        Example:
            Output files: AAPL_2023-01-01.csv, AAPL_2023-01-02.csv, ...
        """
        ret = []
        start_date = datetime(self.main_ctx.start_year, 1, 1)
        end_date = datetime(self.main_ctx.end_year, 12, 31)
        current_date = start_date

        # Iterate through each day in the range
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            file_postfix = f'_{date_str}'
            self.condition['date'] = [date_str]
            ret.extend(self.__url_from_symbols(file_postfix))
            current_date += timedelta(days=1)
        return ret

    def __make_api_list_page(self) -> List[List[Any]]:
        """Generate API URLs for paginated data.

        Creates URLs for a batch of pages starting from current page_num.
        Automatically increments page_num for the next batch.

        Returns:
            List[List[Any]]: List of API call parameters for page batch.

        Note:
            Batch size is controlled by page_set_num (default: 8).
            Call this method repeatedly until workers return no data.

        Example:
            First call: pages 0-7
            Second call: pages 8-15
            Output files: stock_list_0.csv, stock_list_1.csv, ...
        """
        ret = []
        for i in range(self.page_num, self.page_num + self.page_set_num):
            file_postfix = f'_{i}'
            self.condition['page'] = [i]
            ret.extend(self.__url_from_symbols(file_postfix))

        # Increment page number for next batch
        self.page_num += self.page_set_num
        return ret

    def make_api_list(self) -> List[List[Any]]:
        """Generate API call parameters based on query parameter types.

        Determines the appropriate URL generation method based on which
        query parameters are present in the API configuration.

        Priority order:
        1. year parameter -> __make_api_list_year()
        2. from parameter -> __make_api_list_from_to()
        3. date parameter -> __make_api_list_date()
        4. page parameter -> __make_api_list_page()
        5. none of above -> simple symbol-based URLs

        Returns:
            List[List[Any]]: List of [file_path, symbol, postfix, url] for fetching.

        Example:
            >>> api_list = api.make_api_list()
            >>> for params in api_list[:3]:
            ...     print(params[1], params[3])  # Print symbol and URL
            AAPL https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=xxx
            MSFT https://financialmodelingprep.com/api/v3/profile/MSFT?apikey=xxx
            GOOGL https://financialmodelingprep.com/api/v3/profile/GOOGL?apikey=xxx
        """
        if 'year' in self.condition.keys():
            return self.__make_api_list_year()
        elif 'from' in self.condition.keys():
            return self.__make_api_list_from_to()
        elif 'date' in self.condition.keys():
            return self.__make_api_list_date()
        elif 'page' in self.condition.keys():
            return self.__make_api_list_page()
        else:
            # Simple symbol-based URLs without time parameters
            return self.__url_from_symbols('')
