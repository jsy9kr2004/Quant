"""Financial Modeling Prep (FMP) 데이터 수집 모듈입니다.

이 모듈은 Financial Modeling Prep API에서 금융 데이터를 가져오기 위한 메인 FMP 데이터
수집 인터페이스를 제공합니다. 심볼 리스트 관리, 데이터 수집, 파일 정리, 다운로드된
데이터 검증을 처리합니다.

FMP 클래스는 전체 데이터 수집 워크플로우를 조율합니다:
1. 주식 및 상장폐지 회사 리스트 가져오기
2. NASDAQ 및 NYSE 거래소의 심볼 리스트 구축
3. 모든 심볼에 대한 금융 데이터 수집
4. 오래된 파일 검증 및 정리

사용 예시:
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

    사용 예시:
        from context import MainContext
        main_ctx = MainContext()
        fmp = FMP(main_ctx)
        fmp.collect()
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

        # stock_list API 찾기 및 검증
        stock_list_api = [api for api in api_list if api.converted_category == 'stock_list']
        if len(stock_list_api) == 0:
            self.logger.error('stock list는 받아온다는 전제')
            raise Exception('stock list는 받아온다는 전제')
        stock_list_api = stock_list_api[0]

        # delisted_companies API 찾기 및 검증
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

        # NASDAQ 및 NYSE 거래소의 주식만 필터링
        # 참고: read_csv는 인덱스를 포함하지 않으므로 첫 번째 열을 삭제하지 않습니다
        filtered_symbol = symbol_list[(symbol_list['type'] == "stock")
                                      & ((symbol_list['exchangeShortName'] == 'NASDAQ')
                                         | (symbol_list['exchangeShortName'] == 'NYSE'))]
        filtered_symbol = filtered_symbol.dropna(subset=['symbol'])
        filtered_symbol = filtered_symbol.reset_index(drop=True)
        filtered_symbol = filtered_symbol.drop(['price', 'exchange', 'name'], axis=1)
        all_symbol = filtered_symbol

        # NASDAQ 및 NYSE의 상장폐지 회사와 병합
        file_list = os.listdir(self.main_ctx.root_path + "/delisted_companies/")
        for file in file_list:
            if os.path.splitext(file)[1] == ".csv":
                delisted = pd.read_csv(self.main_ctx.root_path + "/delisted_companies/" + file)
                if delisted.empty == True:
                    continue
                # NASDAQ 및 NYSE 거래소만 필터링
                delisted = delisted[((delisted['exchange'] == 'NASDAQ') | (delisted['exchange'] == 'NYSE'))]
                delisted = delisted.dropna(subset=['symbol'])
                delisted = delisted.reset_index(drop=True)
                delisted.rename(columns={'exchange':'exchangeShortName'}, inplace=True)
                delisted = delisted.drop(['companyName'], axis=1)
                all_symbol = pd.concat([all_symbol, delisted])

        # 모든 심볼을 CSV에 저장하고 완전한 심볼 리스트 생성
        all_symbol.to_csv('./allsymbol.csv', index=False)
        all_symbol = all_symbol.drop_duplicates('symbol', keep='first')
        all_symbol = all_symbol.reset_index(drop=True)
        self.symbol_list = all_symbol["symbol"].to_list()

        # 현재 심볼 리스트 생성 (최근 활성 주식)
        # TODO: 가장 최근 상장폐지 날짜에서 1개월을 빼는 목적을 명확히 할 것
        all_symbol["delistedDate"] = pd.to_datetime(all_symbol["delistedDate"])
        recent_date = all_symbol["delistedDate"].max()
        recent_date -= relativedelta(months=1)  # 가장 최근 상장폐지 날짜에서 1개월 빼기

        # 다음 중 하나에 해당하는 심볼 쿼리:
        # 1. 최근에 상장폐지됨 (1개월 이내)
        # 2. 상장폐지되지 않음 (NaT 또는 None)
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

        # 티커 리스트가 아닌 API만 필터링
        rest_api_list = [api for api in api_list if api.converted_category not in ['stock_list', 'delisted_companies']]

        # 심볼이 필요한 API에 심볼 리스트 할당
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
                    # 날짜를 확인하기 위해 전체 파일 읽기
                    # TODO: 첫 번째/마지막 행만 읽는 더 효율적인 방법 찾기
                    row = pd.read_csv(path)

                    # 파일에 date 컬럼이 있는지 확인
                    if "date" in row.columns:
                        if row["date"].empty is True:
                            os.remove(path)
                            continue
                    else:
                        # date 컬럼이 없으면 파일 제거
                        os.remove(path)
                        continue

                    # 파일이 75일보다 오래되었는지 확인
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

        사용 예시:
            FMP.remove_current_year('/data/historical_price_full/AAPL_')
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

        사용 예시:
            if not fmp.skip_remove_check():
                fmp.remove_first_loop()
        """
        today = datetime.datetime.today()
        if os.path.isfile("./config/update_date.txt"):
            fd = open("./config/update_date.txt", "r")
            update_date = fd.readline()
            fd.close()
            update_date = datetime.datetime.strptime(update_date, "%Y-%m-%d")

            # 1일 미만 경과했는지 확인
            if (today - update_date) < datetime.timedelta(days=1):
                self.logger.info('Skip Remove Files')
                return True
        return False

    def validation_check(self) -> bool:
        """다운로드된 파일에 API 오류 메시지가 있는지 검증합니다.

        데이터 디렉토리의 모든 CSV 파일에서 FMP API 오류 메시지를 확인합니다:
        1. "Limit Reach" - API 요청 제한 초과
        2. "Error Message" - API의 일반 오류

        이러한 메시지가 포함된 파일은 유효하지 않은 데이터를 포함하므로 _quarantine 폴더로 이동됩니다.
        또한 mixed type warnings를 감지하여 로그에 기록합니다.

        Returns:
            bool: 격리된 파일이 없으면 True (모두 유효), 그렇지 않으면 False.
        """
        import shutil
        import warnings

        basepath = self.main_ctx.root_path
        quarantine_base = os.path.join(basepath, '_quarantine')
        os.makedirs(quarantine_base, exist_ok=True)

        flag = True
        quarantine_count = 0
        pass_count = 0
        mixed_type_count = 0
        retry_list = []

        logging.info("🔍 Starting validation check with quarantine system")
        logging.info(f"   Base path: {basepath}")
        logging.info(f"   Quarantine path: {quarantine_base}")

        for dir_name in os.listdir(basepath):
            dir_path = os.path.join(basepath, dir_name)

            # Skip quarantine directory and non-directories
            if dir_name == '_quarantine' or not os.path.isdir(dir_path):
                continue

            cur_path = dir_path
            par_list = [file for file in os.listdir(cur_path) if file.endswith('csv')]

            category_quarantine_count = 0
            category_mixed_type_count = 0

            for p in par_list:
                file_path = os.path.join(cur_path, p)
                has_error = False
                has_mixed_type = False

                try:
                    # Catch DtypeWarning for mixed types
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", category=pd.errors.DtypeWarning)
                        df = pd.read_csv(file_path, low_memory=True)

                        # Check for mixed type warnings
                        if len(w) > 0:
                            has_mixed_type = True
                            mixed_type_count += 1
                            category_mixed_type_count += 1

                            # Log detailed mixed type info
                            for warning_item in w:
                                logging.warning(f"⚠️  Mixed type detected: {file_path}")
                                logging.warning(f"    Warning: {warning_item.message}")

                                # Sample first few rows for debugging
                                if len(df) > 0:
                                    logging.debug(f"    First row sample: {df.iloc[0].to_dict()}")

                    # 데이터에서 오류 메시지 확인
                    if df.filter(regex='Limit').empty is False or df.filter(regex='Error').empty is False:
                        has_error = True

                        # Extract symbol from filename (usually format: SYMBOL.csv)
                        symbol = os.path.splitext(p)[0]

                        # Create quarantine subdirectory
                        category_quarantine_path = os.path.join(quarantine_base, dir_name)
                        os.makedirs(category_quarantine_path, exist_ok=True)

                        # Move file to quarantine
                        quarantine_file_path = os.path.join(category_quarantine_path, p)
                        shutil.move(file_path, quarantine_file_path)

                        quarantine_count += 1
                        category_quarantine_count += 1
                        flag = False

                        # Determine error type
                        error_type = []
                        if not df.filter(regex='Limit').empty:
                            error_type.append('Limit Reach')
                        if not df.filter(regex='Error').empty:
                            error_type.append('Error Message')
                        if has_mixed_type:
                            error_type.append('Mixed Type')

                        # Add to retry list
                        retry_list.append({
                            'category': dir_name,
                            'symbol': symbol,
                            'original_path': file_path,
                            'quarantine_path': quarantine_file_path,
                            'error_type': ', '.join(error_type)
                        })

                        logging.info(f"🔒 Quarantined: {dir_name}/{p} -> {error_type}")
                    else:
                        pass_count += 1

                        # Log mixed type files that passed API error check
                        if has_mixed_type:
                            logging.info(f"⚠️  Mixed type (no API error): {dir_name}/{p}")

                except Exception as e:
                    logging.error(f"❌ Error processing {file_path}: {e}")
                    continue

            logging.info("[ {} ] Quarantined: {} | Mixed types: {} | Valid: {} | Total: {}".format(
                cur_path,
                category_quarantine_count,
                category_mixed_type_count,
                pass_count,
                category_quarantine_count + pass_count
            ))
            quarantine_count = 0
            pass_count = 0

        # Save retry list to CSV
        if retry_list:
            retry_csv_path = os.path.join(basepath, '_retry_list.csv')
            retry_df = pd.DataFrame(retry_list)
            retry_df.to_csv(retry_csv_path, index=False)
            logging.info(f"📝 Created retry list: {retry_csv_path} ({len(retry_list)} files)")
            logging.info(f"   Use this list to re-fetch problematic files")
        else:
            logging.info("✅ No files needed quarantine - all files valid!")

        logging.info(f"🏁 Validation complete: Mixed type warnings: {mixed_type_count}")

        return flag

    def remove_first_loop(self) -> None:
        """새로고침을 준비하기 위해 심볼 리스트 관련 파일을 제거합니다.

        2단계 제거 프로세스의 첫 번째 단계입니다. 심볼 리스트에 영향을 주는 파일을
        제거하여 현재 데이터로 다시 가져올 수 있도록 합니다.

        제거 대상:
        - allsymbol.csv: 결합된 심볼 리스트 캐시
        - current_list.csv: 현재 심볼 리스트 캐시
        - delisted_companies/ 디렉토리 내용
        - stock_list/ 디렉토리 내용

        See Also:
            remove_second_loop: 심볼 리스트가 재구축된 후의 두 번째 단계.
        """
        if os.path.isfile("./allsymbol.csv"):
            os.remove("./allsymbol.csv")
        if os.path.isfile("./current_list.csv"):
            os.remove("./current_list.csv")
        self.remove_files(self.main_ctx.root_path+"/delisted_companies")
        self.remove_files(self.main_ctx.root_path+"/stock_list")

    def remove_second_loop(self) -> None:
        """새로고침된 심볼 리스트를 사용하여 데이터 파일을 제거합니다.

        2단계 제거 프로세스의 두 번째 단계입니다. 심볼 리스트가 새로고침된 후,
        현재 리스트의 심볼에 대한 오래된 데이터 파일을 제거합니다.

        제거 대상:
        - symbol_available_indexes
        - 현재 연도 수익 캘린더 파일
        - 75일보다 오래된 재무제표 파일
        - 현재 연도 히스토리컬 가격 파일
        - DCF 및 시가총액 파일 (성능 병목 - FIXME 참조)
        - 프로필 데이터 (현재 업데이트되지 않음)

        FIXME: DCF 및 시가총액에 대한 remove_current_list_files가 느립니다.
            대규모 데이터셋에 대한 대체 접근 방식을 고려하십시오.

        See Also:
            remove_first_loop: 심볼 리스트 새로고침 전의 첫 번째 단계.
        """
        self.remove_files(self.main_ctx.root_path+"/symbol_available_indexes")
        self.remove_current_year(self.main_ctx.root_path+"/earning_calendar/earning_calendar_")

        # 75일보다 오래된 재무제표 파일 제거
        self.remove_current_list_files(self.main_ctx.root_path+"/income_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/balance_sheet_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/cash_flow_statement")
        self.remove_current_list_files(self.main_ctx.root_path+"/key_metrics")
        self.remove_current_list_files(self.main_ctx.root_path+"/financial_growth")

        # 각 심볼에 대한 현재 연도 히스토리컬 가격 파일 제거
        for symbol in self.current_list:
            self.remove_current_year(self.main_ctx.root_path+"/historical_price_full/" + str(symbol) + "_")

        # FIXME: 이 두 작업은 가장 느린 작업 중 하나입니다. 최적화를 고려하십시오.
        # 가장 시간 소모적인 작업 상위 2개 - 더 나은 접근 방식 필요
        self.remove_current_list_files(self.main_ctx.root_path+"/historical_daily_discounted_cash_flow")
        self.remove_current_list_files(self.main_ctx.root_path+"/historical_market_capitalization", False)

        # 프로필 데이터는 의도적으로 업데이트되지 않습니다
        self.remove_current_list_files(self.main_ctx.root_path+"/profile", False)

    def collect(self) -> None:
        """완전한 FMP 데이터 수집 워크플로우를 실행합니다.

        이것은 데이터 수집을 위한 메인 진입점입니다. 다음 순서로 전체
        프로세스를 조율합니다:

        1. 설정에서 API 리스트 로드
        2. 티커 리스트 가져오기 (stock_list 및 delisted_companies)
        3. NASDAQ 및 NYSE를 위한 심볼 리스트 구축
        4. 다른 모든 API에서 데이터 가져오기
        5. 업데이트 타임스탬프 기록
        6. 다운로드된 파일 검증

        이 메서드는 2회 처리 접근 방식을 구현합니다:
        - 첫 번째 회: 심볼 리스트를 구축하기 위해 티커 리스트 가져오기
        - 두 번째 회: 심볼 리스트를 사용하여 다른 모든 데이터 가져오기

        이 설계는 API 카테고리를 하드코딩하지 않고 유연한 API 관리를 가능하게 합니다.
        심볼 리스트가 존재하지 않을 때 심볼이 필요한 API는 오류 없이 반환됩니다.

        Note:
            파일 제거 단계(remove_first_loop, remove_second_loop)는 현재 주석 처리되어
            있습니다. 오래된 파일의 자동 정리를 활성화하려면 주석을 해제하십시오.

        Raises:
            SystemExit: 검증 확인이 실패하면 다시 가져와야 하는 손상된 다운로드를
                나타냅니다.
        """

        api_list = self.__get_api_list()

        # 첫 번째 루프: 새 티커 리스트를 가져오기 전에 이전 심볼 리스트 파일 제거
        if self.skip_remove_check() is False:
            self.remove_first_loop()

        # 티커 리스트를 가져오고 심볼 리스트 구축
        self.__fetch_ticker_list(api_list)
        self.__set_symbol()  # TODO: 심볼 리스트에 ETF 심볼 추가
        print("after set_symbol : {}".format(self.symbol_list))

        # 두 번째 루프: 새 심볼 리스트를 기반으로 오래된 데이터 파일 제거
        if self.skip_remove_check() is False:
            self.remove_second_loop()

        # 나머지 모든 데이터 가져오기
        self.__fetch_data(api_list)

        # 실수로 재실행하는 것을 방지하기 위해 업데이트 타임스탬프 기록
        write_fd = open("./config/update_date.txt", "w")
        today = datetime.date.today()
        write_fd.write(str(today))
        write_fd.close()

        # API 오류에 대한 다운로드된 파일 검증
        if self.validation_check() is False:
            logging.critical("Validation Check False!! Please run the program again after a few minutes!!")
            exit()
