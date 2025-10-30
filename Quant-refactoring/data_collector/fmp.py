"""Financial Modeling Prep (FMP) 데이터 수집 모듈입니다.

이 모듈은 Financial Modeling Prep API에서 금융 데이터를 가져오기 위한 메인 FMP 데이터
수집 인터페이스를 제공합니다. 심볼 리스트 관리, 데이터 수집, 파일 정리, 다운로드된
데이터 검증을 처리합니다.

FMP 클래스는 전체 데이터 수집 워크플로우를 조율합니다:
1. 주식 및 상장폐지 회사 리스트 가져오기
2. NASDAQ 및 NYSE 거래소의 심볼 리스트 구축
3. 모든 심볼에 대한 금융 데이터 수집
4. 오래된 파일 검증 및 정리

사용 예제:
    from context import MainContext

    main_ctx = MainContext()
    fmp = FMP(main_ctx)
    fmp.collect()  # 데이터 수집 프로세스 시작
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
    """Financial Modeling Prep API에서 금융 데이터를 수집하는 메인 클래스입니다.

    이 클래스는 티커 리스트 가져오기, NASDAQ 및 NYSE 거래소의 심볼 리스트 구축,
    금융 데이터 수집, 오래된 파일 정리를 포함한 전체 데이터 수집 워크플로우를
    관리합니다.

    Attributes:
        main_ctx: 설정과 공유 리소스를 포함하는 메인 컨텍스트 객체.
        symbol_list (list): 상장폐지 회사를 포함한 전체 주식 심볼 리스트.
        current_list (list): 현재 활성 주식 심볼 리스트.
        logger: 이 클래스의 로거 인스턴스.

    Example:
        >>> from context import MainContext
        >>> main_ctx = MainContext()
        >>> fmp = FMP(main_ctx)
        >>> fmp.collect()
    """

    def __init__(self, main_ctx) -> None:
        """FMP 데이터 수집기를 초기화합니다.

        Args:
            main_ctx: API 키, 루트 경로, 로깅 설정 등의 설정을 포함하는
                메인 컨텍스트 객체.
        """
        self.main_ctx = main_ctx
        self.symbol_list: List[str] = pd.DataFrame()
        self.current_list: List[str] = pd.DataFrame()

        self.logger = self.main_ctx.get_logger('fmp')

    def __get_api_list(self) -> List[FMPAPI]:
        """타겟 API 리스트 CSV에서 FMPAPI 객체 리스트를 생성합니다.

        설정된 CSV 파일에서 API 리스트를 읽고 각 URL 항목에 대해 FMPAPI 객체를
        생성합니다. 빈 행은 제거됩니다.

        Returns:
            List[FMPAPI]: 데이터 수집을 위해 초기화된 FMPAPI 객체 리스트.

        Raises:
            FileNotFoundError: 타겟 API 리스트 파일이 존재하지 않는 경우.
            KeyError: CSV에 'URL' 컬럼이 없는 경우.
        """
        url_df = pd.read_csv(self.main_ctx.target_api_list, header=0, usecols=["URL"])
        url_df = url_df.dropna()
        url_list = url_df['URL'].tolist()
        api_list = [FMPAPI(self.main_ctx, url) for url in url_list]

        return api_list

    def __fetch_ticker_list(self, api_list: List[FMPAPI]) -> None:
        """FMP API에서 주식 리스트 및 상장폐지 회사 데이터를 가져옵니다.

        API 리스트를 필터링하여 stock_list 및 delisted_companies 엔드포인트를
        찾고, 둘 다 존재하는지 검증한 후 데이터를 가져옵니다.

        Args:
            api_list (List[FMPAPI]): 필터링할 FMPAPI 객체의 전체 리스트.

        Raises:
            Exception: 리스트에서 stock_list 또는 delisted_companies API를 찾을 수 없는 경우.
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
        """stock_list 및 delisted_companies 데이터에서 심볼 리스트를 구축합니다.

        이 메서드는:
        1. stock_list CSV 파일 로드
        2. NASDAQ 및 NYSE 거래소의 주식만 필터링
        3. 동일한 거래소의 상장폐지 회사와 병합
        4. 두 개의 리스트 생성:
           - symbol_list: 모든 심볼 (상장폐지 포함)
           - current_list: 최근 활성 심볼 (1개월 내 상장폐지 또는 현재 활성)

        심볼 리스트는 디버깅 및 검증 목적으로 CSV 파일(allsymbol.csv 및 current_list.csv)에
        저장됩니다.

        Raises:
            FileNotFoundError: stock_list.csv가 존재하지 않는 경우.
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
        """stock_list 및 delisted_companies를 제외한 모든 API에서 데이터를 가져옵니다.

        티커 리스트 API를 필터링하고 나머지 엔드포인트에서 데이터를 가져옵니다.
        심볼이 필요한 API에는 가져오기 전에 symbol_list가 할당됩니다.

        Args:
            api_list (List[FMPAPI]): FMPAPI 객체의 전체 리스트.
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
        """지정된 디렉토리에서 파일을 제거합니다.

        Args:
            path (str): 정리할 디렉토리 경로.
            only_csv (bool, optional): True인 경우 CSV 파일만 제거.
                False인 경우 모든 파일 제거. 기본값은 True.
        """
        if os.path.isdir(path) is False:
            return
        for file in os.listdir(path):
            if only_csv is True and not (file.endswith(".csv") or file.endswith(".csvx")):
                continue
            os.remove(os.path.join(path, file))

    def remove_current_list_files(self, base_path: str, check_target: bool = True) -> None:
        """현재 리스트의 심볼에 대한 오래된 파일을 제거합니다.

        current_list 심볼을 순회하며 다음 경우 데이터 파일을 제거합니다:
        1. check_target이 False: 무조건 모든 파일 제거
        2. check_target이 True: 75일보다 오래된 파일만 제거

        Args:
            base_path (str): 심볼 데이터 파일이 포함된 기본 디렉토리 경로.
            check_target (bool, optional): True인 경우 제거 전에 파일 날짜 확인.
                기본값은 True.

        Note:
            파일의 날짜를 확인하려면 'date' 컬럼이 있어야 합니다. date 컬럼이 없거나
            빈 date 값을 가진 파일은 즉시 제거됩니다.
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
        """지정된 기본 경로에서 현재 연도의 파일을 제거합니다.

        현재 연도 접미사를 가진 .csv 및 .csvx 파일을 모두 제거합니다.

        Args:
            base_path (str): 기본 경로 패턴 (예: 'path/to/data_').
                현재 연도가 추가되어 'path/to/data_2025.csv' 형태가 됩니다.

        Example:
            >>> FMP.remove_current_year('/data/historical_price_full/AAPL_')
            # 제거: /data/historical_price_full/AAPL_2025.csv
            #       /data/historical_price_full/AAPL_2025.csvx
        """
        today = dateutil.utils.today()
        year = today.strftime("%Y")
        if os.path.isfile(base_path + str(year) + ".csv"):
            os.remove(base_path + str(year) + ".csv")
        if os.path.isfile(base_path + str(year) + ".csvx"):
            os.remove(base_path + str(year) + ".csvx")

    def skip_remove_check(self) -> bool:
        """마지막 업데이트 시간을 기준으로 파일 제거를 건너뛸지 확인합니다.

        './config/update_date.txt'에서 마지막 업데이트 날짜를 읽고
        마지막 업데이트 이후 1일 미만이 경과했는지 확인합니다.

        Returns:
            bool: 마지막 업데이트 이후 1일 미만인 경우 True (제거 건너뛰기),
                그렇지 않으면 False.

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
        """다운로드된 파일에 API 오류 메시지가 있는지 검증합니다.

        데이터 디렉토리의 모든 CSV 파일에서 FMP API 오류 메시지를 확인합니다:
        1. "Limit Reach" - API 요청 제한 초과
        2. "Error Message" - API의 일반 오류

        이러한 메시지가 포함된 파일은 유효하지 않은 데이터를 포함하므로 삭제됩니다.

        Returns:
            bool: 삭제된 파일이 없으면 True (모두 유효), 그렇지 않으면 False.

        Note:
            이 메서드는 하드코딩된 경로 'E:\qt\data'를 사용합니다.
            FIXME: main_ctx에서 설정 가능한 경로를 사용해야 합니다.
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
        if self.skip_remove_check() is False:
            self.remove_first_loop()

        # Fetch ticker lists and build symbol lists
        self.__fetch_ticker_list(api_list)
        self.__set_symbol()  # TODO: Add ETF symbols to symbol list
        print("after set_symbol : {}".format(self.symbol_list))

        # Second loop: Remove outdated data files based on new symbol list
        if self.skip_remove_check() is False:
            self.remove_second_loop()

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
