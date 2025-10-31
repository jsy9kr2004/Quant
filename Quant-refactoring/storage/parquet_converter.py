"""금융 데이터 통합을 위한 레거시 Parquet 변환기입니다.

이 모듈은 여러 CSV 또는 Parquet 소스 파일의 금융 데이터를 통합된 데이터셋으로
변환하고 통합하는 Parquet 클래스를 제공합니다. 다양한 소스(주식 목록, 가격 기록,
재무제표, 메트릭)의 데이터를 병합하고 연도별로 구성된 뷰 테이블을 생성합니다.

변환기가 처리하는 작업:
    - 데이터 카테고리별로 여러 CSV/Parquet 파일 병합
    - 원시 데이터 소스에서 통합 뷰 테이블 구축
    - 날짜 타입 변환 및 데이터 정제
    - 효율적인 쿼리를 위한 연도별 파티션 데이터셋 생성
    - 누락되거나 손상된 소스 파일에 대한 오류 처리

사용 예시:
    데이터 통합을 위한 기본 사용법::

        from storage import Parquet

        # 메인 컨텍스트로 변환기 초기화
        converter = Parquet(main_ctx)

        # 모든 CSV 파일을 카테고리별 단일 파일로 통합
        converter.insert_csv()

        # 원시 데이터에서 통합 뷰 테이블 구축
        converter.rebuild_table_view()

Note:
    이것은 CSV 파일을 다루고 통합 형식으로 변환하는 레거시 클래스입니다.
    새로운 코드의 경우, 내장 검증 기능이 있는 보다 현대적인 API를 제공하는
    ParquetStorage 사용을 고려하십시오.

TODO:
    - 더 나은 검증을 위해 ParquetStorage 사용으로 마이그레이션 고려
    - 장시간 실행 작업에 대한 진행률 표시기 추가
    - 전체 재구축 대신 증분 업데이트 구현
"""

import datetime
import logging
import os
from typing import Any, List, Optional

import pandas as pd
from tqdm import tqdm


class Parquet:
    """레거시 Parquet 데이터 변환기 및 통합기입니다.

    이 클래스는 여러 소스 파일의 금융 데이터를 통합된 CSV 데이터셋으로 통합합니다.
    주식 목록, 가격 기록, 재무제표, 메트릭 데이터 병합을 처리합니다. 이 클래스는
    구성과 유틸리티를 제공하는 메인 컨텍스트 객체와 함께 작동하도록 설계되었습니다.

    변환기는 두 가지 주요 작업을 수행합니다:
        1. insert_csv(): 카테고리별 여러 파일을 단일 파일로 통합
        2. rebuild_table_view(): 관련 데이터셋을 통합 뷰 테이블로 병합

    Attributes:
        main_ctx: 구성(start_year, end_year, root_path)과 유틸리티(create_dir 메서드)를
            제공하는 메인 컨텍스트 객체.
        tables (dict): 로드된 DataFrame을 저장하는 딕셔너리 (현재 미사용).
        view_path (str): 통합 뷰 테이블을 저장하는 디렉토리 경로.
        rawpq_path (str): 원시 Parquet/CSV 파일이 있는 디렉토리 경로.

    사용 예시:
        초기화 및 전체 변환 파이프라인 실행::

            # main_ctx가 필요한 속성을 가지고 있다고 가정
            converter = Parquet(main_ctx)

            # 1단계: 원시 파일 통합
            converter.insert_csv()

            # 2단계: 뷰 테이블 구축
            converter.rebuild_table_view()

    Note:
        이 클래스는 특정 디렉토리 구조와 파일 명명 규칙을 기대합니다.
        다음 카테고리를 처리합니다: balance_sheet_statement, cash_flow_statement,
        delisted_companies, earning_calendar, financial_growth,
        historical_daily_discounted_cash_flow, historical_market_capitalization,
        historical_price_full, income_statement, key_metrics, profile,
        stock_list, symbol_available_indexes.
    """

    def __init__(self, main_ctx: Any) -> None:
        """메인 컨텍스트로 Parquet 변환기를 초기화합니다.

        원시 Parquet 파일과 뷰 테이블을 위한 필요한 디렉토리 구조를 생성합니다.
        테이블 데이터를 위한 저장소를 초기화합니다.

        Args:
            main_ctx: 다음을 제공해야 하는 메인 컨텍스트 객체:
                - root_path (str): 모든 데이터를 위한 루트 디렉토리
                - start_year (int): 날짜 범위 필터링을 위한 시작 연도
                - end_year (int): 날짜 범위 필터링을 위한 종료 연도
                - create_dir(path: str): 디렉토리를 생성하는 메서드

        사용 예시:
            컨텍스트 객체로 초기화::

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
        """원시 데이터 소스에서 통합 뷰 테이블을 재구축합니다.

        관련 데이터셋을 병합하여 통합 뷰 테이블을 생성합니다. 이 메서드는
        분석 준비된 테이블을 생성하기 위해 복잡한 조인과 데이터 통합을 수행합니다.
        프로세스는 다음을 포함합니다:

        1. 심볼 리스트 테이블: 주식 목록, 상장 폐지 기업, 프로필 병합
        2. 가격 테이블: 과거 가격과 시가총액 결합
        3. 재무제표 테이블: 손익계산서, 재무상태표, 현금흐름표 조인
        4. 메트릭 테이블: 주요 메트릭, 성장 데이터, DCF 평가 결합
        5. 인덱스 테이블: 심볼 인덱스 정보 복사

        재무제표와 메트릭의 경우, 특정 기간의 효율적인 쿼리를 위해
        연도별 파티션 파일도 생성합니다.

        사용 예시:
            모든 뷰 테이블 재구축::

                converter = Parquet(main_ctx)
                converter.rebuild_table_view()

                # VIEW/에 다음 파일을 생성합니다:
                # - symbol_list.csv
                # - price.csv
                # - financial_statement.csv
                # - financial_statement_YYYY.csv (연도별)
                # - metrics.csv
                # - metrics_YYYY.csv (연도별)
                # - indexes.csv

        Note:
            - 대용량 데이터셋의 경우 시간이 오래 걸리는 작업입니다
            - rawpq_path에 모든 소스 파일이 있어야 합니다
            - 광범위한 데이터 정제 및 타입 변환을 수행합니다
            - 전체 및 연도별 파티션 데이터셋을 모두 생성합니다
            - 대용량 DataFrame 작업으로 인해 메모리 사용량이 높을 수 있습니다

        Raises:
            소스 파일이 누락되거나 손상된 경우 예외가 발생할 수 있습니다.
            오류는 로깅되지만 명시적으로 잡히지 않습니다.
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
        """여러 CSV/Parquet 파일을 카테고리별 단일 파일로 통합합니다.

        사전 정의된 데이터 디렉토리를 스캔하고 각 카테고리 내의 모든 CSV 또는
        Parquet 파일을 단일 통합 CSV 파일로 병합합니다. 이것은 일반적으로
        데이터 처리 파이프라인의 첫 번째 단계입니다.

        메서드가 처리하는 데이터 카테고리:
            - balance_sheet_statement
            - cash_flow_statement
            - delisted_companies
            - earning_calendar
            - financial_growth
            - historical_daily_discounted_cash_flow
            - historical_market_capitalization
            - historical_price_full (컬럼 필터링 포함)
            - income_statement
            - key_metrics
            - profile
            - stock_list (보존됨, 재생성 안 함)
            - symbol_available_indexes (보존됨, 재생성 안 함)

        각 카테고리에 대해:
            1. 카테고리 디렉토리에서 .csv 또는 .parquet 파일 스캔
            2. 모든 파일을 DataFrame으로 읽기
            3. 단일 DataFrame으로 연결
            4. 통합 결과를 단일 CSV 파일로 저장

        사용 예시:
            모든 소스 파일 통합::

                converter = Parquet(main_ctx)
                converter.insert_csv()

                # 다음에서 파일을 읽습니다:
                # {root_path}/income_statement/*.csv
                # {root_path}/balance_sheet_statement/*.parquet
                # 등.

                # 다음을 생성합니다:
                # {root_path}/parquet/income_statement.csv
                # {root_path}/parquet/balance_sheet_statement.csv
                # 등.

        Note:
            - 기존 통합 파일은 재생성 전에 제거됩니다
              (stock_list와 symbol_available_indexes 제외)
            - historical_price_full의 경우, 메모리 사용량을 줄이기 위해
              특정 컬럼만 로드됩니다 (date, symbol, close, volume)
            - 읽기 오류가 있는 파일은 경고와 함께 건너뜁니다
            - tqdm 진행률 표시줄로 진행 상황이 표시됩니다
            - 빈 디렉토리는 우아하게 처리됩니다

        TODO:
            - 연도별 파티션 과거 가격 데이터 지원 추가
            - 전체 통합 대신 증분 업데이트 구현
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
