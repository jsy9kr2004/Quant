"""FMP API URL 빌더 및 설정 모듈입니다.

이 모듈은 Financial Modeling Prep API URL을 파싱, 설정, 빌드하기 위한 FMPAPI 클래스를
제공하며, 다양한 쿼리 파라미터를 처리합니다. 다양한 API 버전(v3, v4)에 대한 URL 생성,
쿼리 파라미터 관리, 병렬 가져오기를 위한 API URL 배치 생성을 처리합니다.

FMPAPI 클래스가 지원하는 기능:
- URL 파싱 및 검증
- 쿼리 파라미터 추출 및 조작
- 심볼 기반 URL 생성
- 시간 기반 URL 생성 (연도, 분기, 날짜 범위)
- 대규모 데이터셋을 위한 페이지네이션

사용 예시:
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
    """FMP API URL 빌더 및 설정 관리자입니다.

    FMP API URL을 파싱하고, 쿼리 파라미터를 추출하며, 심볼, 시간 기간 또는
    페이지네이션 요구사항에 따라 API 호출 URL 배치를 생성합니다.

    이 클래스는 v3 및 v4 API 버전을 모두 처리하고, 쿼리 파라미터를 관리하며,
    다운로드된 데이터를 저장하기 위한 파일 경로를 생성합니다.

    Attributes:
        main_ctx: 설정이 포함된 메인 컨텍스트 객체.
        fmp_url (str): 기본 FMP API URL.
        api_key (str): 인증을 위한 FMP API 키.
        url (str): 초기화 시 제공된 원본 전체 URL.
        main_url (str): 쿼리 파라미터 없는 API 엔드포인트 경로.
        category (str): URL 경로에서 추출한 API 카테고리.
        condition (Dict[str, List[str]]): 키-값 쌍으로 된 쿼리 파라미터.
        need_symbol (bool): 이 API가 주식 심볼을 필요로 하는지 여부.
        symbol_list (List[str]): 데이터를 가져올 주식 심볼 리스트.
        is_v4 (bool): API 버전 4인 경우 True, 버전 3인 경우 False.
        file_path (str): 다운로드된 데이터를 저장할 디렉토리 경로.
        page_num (int): 페이지네이션 API의 현재 페이지 번호.
        page_set_num (int): 배치당 가져올 페이지 수.
        fmp_api_logger: 이 API의 로거 인스턴스.

    사용 예시:
        url = "https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=xxx"
        api = FMPAPI(main_ctx, url)
        api.symbol_list = ['AAPL', 'MSFT']
        urls = api.make_api_list()
    """

    def __init__(self, main_ctx, url: str) -> None:
        """URL 파싱 및 설정으로 FMPAPI를 초기화합니다.

        제공된 URL을 파싱하여 API 엔드포인트, 쿼리 파라미터를 추출하고,
        심볼이 필요한지 판단합니다. 출력 디렉토리 구조를 생성합니다.

        Args:
            main_ctx: 설정을 포함하는 메인 컨텍스트 객체.
            url (str): 파싱하고 설정할 전체 FMP API URL.

        Raises:
            Exception: 출력 디렉토리를 생성할 수 없는 경우.

        Note:
            여러 줄로 된 URL(백슬래시로 구분)은 첫 번째 줄만 사용합니다.
            이는 CSV 파일의 잠재적 형식 문제를 처리합니다.
        """
        # 백슬래시가 포함된 여러 줄 URL의 경우 첫 번째 줄만 사용
        # CSV 파일에서 줄 연속을 나타낼 수 있습니다
        if url.find('\\') != -1:
            url = url.split('\\')[0]

        # URL을 구성 요소로 파싱
        parsed_url = urlparse(url)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        # 쿼리 파라미터에서 API 키와 제한 제거
        # 최종 URL 빌드 시 다시 추가됩니다
        if 'apikey' in query_params.keys():
            del query_params['apikey']
        if 'limit' in query_params.keys():
            del query_params['limit']

        # API 버전(v3 또는 v4) 결정
        is_v4 = False if 'api/v3/' in path else True

        # 카테고리 추출 및 심볼 필요 여부 확인
        if is_v4:
            category = path.split('api/v4/')[1] if is_v4 else path.split('api/v3/')[1]
            need_symbol = False
            if 'symbol' in query_params.keys():
                need_symbol = True
        else:
            # v3 API: 심볼은 쿼리 파라미터가 아닌 경로에 있습니다
            category = path.split('api/v4/')[1] if is_v4 else path.split('api/v3/')[1]
            need_symbol = False
            if main_ctx.ex_symbol in category:
                # 카테고리와 경로에서 예제 심볼 제거
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

        # 출력 디렉토리 생성
        if self.main_ctx.create_dir(self.file_path):
            self.fmp_api_logger.info(f'Directory {self.file_path} created successfully')
        else:
            self.fmp_api_logger.error(f'Directory {self.file_path} is not created')
            raise Exception(f'Directory {self.file_path} is not created')

        self.page_num: int = 0
        # API 속도 제한을 피하기 위해 페이지 배치 크기 제한
        # 워커 수가 아닌 배치당 가져올 페이지 수를 제어합니다
        self.page_set_num: int = min(8, os.cpu_count())

    @property
    def converted_category(self) -> str:
        """카테고리 경로를 파일시스템 안전 디렉토리 이름으로 변환합니다.

        파일 경로에 사용하기 위해 슬래시와 대시를 언더스코어로 대체합니다.

        Returns:
            str: 파일시스템 안전 카테고리 이름.

        사용 예시:
            api.category = "income-statement/annual"
            api.converted_category
            'income_statement_annual'
        """
        return self.category.replace("/", "-").replace("-", "_")

    @property
    def query_params_str(self) -> str:
        """condition 딕셔너리에서 쿼리 파라미터 문자열을 빌드합니다.

        condition 딕셔너리를 URL 인코딩된 쿼리 문자열 형식으로 변환합니다.
        각 파라미터 리스트의 첫 번째 값을 가져옵니다.

        Returns:
            str: URL 인코딩된 쿼리 파라미터 문자열.

        사용 예시:
            api.condition = {'period': ['quarter'], 'year': [2023]}
            api.query_params_str
            'period=quarter&year=2023'
        """
        return urlencode({k: v[0] for k, v in self.condition.items()}, doseq=True)

    @property
    def page_in_condition(self) -> bool:
        """쿼리 파라미터에서 페이지네이션이 사용되는지 확인합니다.

        Returns:
            bool: condition에 'page' 파라미터가 있으면 True.
        """
        return 'page' in self.condition.keys()

    def __full_url(self, symbol: str) -> str:
        """모든 파라미터를 포함한 완전한 FMP API URL을 빌드합니다.

        기본 URL, 엔드포인트 경로, 쿼리 파라미터, 심볼(필요한 경우), API 키를
        포함한 전체 URL을 구성합니다. v3 및 v4 API 형식을 모두 처리합니다.

        Args:
            symbol (str): URL에 포함할 주식 심볼 (심볼이 필요 없는 경우 카테고리 이름).

        Returns:
            str: HTTP 요청을 위한 완전한 FMP API URL.

        사용 예시:
            v3: https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=xxx
            v4: https://financialmodelingprep.com/api/v4/shares_float?symbol=AAPL&apikey=xxx
        """
        if self.is_v4:
            # v4 API: 쿼리 파라미터에 심볼 포함
            if self.need_symbol:
                self.condition['symbol'] = [symbol]
            return f'{self.fmp_url}{self.main_url}?{self.query_params_str}&apikey={self.api_key}'
        else:
            # v3 API: 경로에 심볼 포함
            if self.need_symbol:
                return f'{self.fmp_url}{self.main_url}/{symbol}?{self.query_params_str}&apikey={self.api_key}'
            else:
                return f'{self.fmp_url}{self.main_url}?{self.query_params_str}&apikey={self.api_key}'

    def __url_from_symbols(self, file_postfix: str) -> List[List[Any]]:
        """symbol_list의 모든 심볼에 대한 API 호출 파라미터를 생성합니다.

        각 심볼에 대한 파라미터 리스트를 생성하며, 파일 경로, 심볼 이름,
        파일 접미사, 전체 URL을 포함합니다. 중복 다운로드를 피하기 위해
        기존 파일을 확인합니다.

        Args:
            file_postfix (str): 출력 파일명에 추가할 접미사 (예: '_2023_Q1').

        Returns:
            List[List[Any]]: 각 심볼에 대한 파라미터 리스트 [file_path, symbol, postfix, url].

        Raises:
            Exception: 심볼에 대한 출력 파일이 이미 존재하는 경우.

        사용 예시:
            api.symbol_list = ['AAPL', 'MSFT']
            params = api.__url_from_symbols('_2023')
            params[0]
            ['/data/profile', 'AAPL', '_2023', 'https://...']
        """
        api_list = []
        for symbol in self.symbol_list:
            # 파일이 이미 존재하는지 확인 (.csv 또는 빈 데이터를 나타내는 .csvx)
            if os.path.isfile(f'{self.file_path}/{symbol+file_postfix}.csv') \
               or os.path.isfile(f'{self.file_path}/{symbol+file_postfix}.csvx'):
                self.fmp_api_logger.info(f'Skip existing file: {self.file_path}/{symbol+file_postfix}.csv/csvx')
                continue

            api_list.append([self.file_path, symbol, file_postfix, self.__full_url(symbol)])
        return api_list

    def set_symbol_list(self, symbol_list: List[str]) -> None:
        """데이터 가져오기를 위한 주식 심볼 리스트를 설정합니다.

        리스트를 설정하기 전에 API가 심볼을 필요로 하는지 검증합니다.
        historical-price-full 카테고리의 경우 여기에 ETF 심볼을 추가할 수 있습니다.

        Args:
            symbol_list (List[str]): 가져올 주식 심볼 리스트.

        Raises:
            Exception: API가 심볼을 필요로 하지 않는데 set_symbol_list가 호출된 경우.

        Note:
            TODO: historical-price-full에 ETF 심볼(SPY, IVV, VTI 등) 추가.
        """
        if not self.need_symbol:
            self.fmp_api_logger.info('symbol이 필요하지 않은데 set하려함')
            raise Exception('symbol이 필요하지 않은데 set하려함')
        self.symbol_list = symbol_list

        # TODO: 히스토리컬 가격 데이터를 위한 ETF 심볼 추가
        if self.category == "historical-price-full":
            self.fmp_api_logger.info("add ETF symbols to get historical-price-full data")
            # self.symbol_list.append(pd.Series(["SPY", "IVV", "VTI", "VOO", "QQQ", "VEA", "IEFA"]))

        self.symbol_list = self.symbol_list.dropna()

    def __make_api_list_year(self) -> List[List[Any]]:
        """연도 기반 또는 분기별 데이터에 대한 API URL을 생성합니다.

        설정된 범위(start_year부터 end_year까지)의 각 연도에 대한 URL을 생성합니다.
        쿼리 파라미터에 따라 세 가지 모드를 지원합니다:
        1. period: 연도당 4개 기간(P1-P4) URL 생성
        2. quarter: 연도당 4개 분기(Q1-Q4) URL 생성
        3. 둘 다 없음: 연도당 하나의 URL 생성

        Returns:
            List[List[Any]]: 모든 연도/기간에 대한 API 호출 파라미터 리스트.

        사용 예시:
            year=2023 및 period 파라미터가 있는 경우:
            - AAPL_2023_P1.csv
            - AAPL_2023_P2.csv
            - ...
        """
        ret = []
        for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
            if 'period' in self.condition.keys():
                # 연도당 4개 기간 생성
                for period in range(1, 5):
                    file_postfix = f'_{year}_P{period}'
                    self.condition['year'] = [year]
                    self.condition['period'] = [period]
                    ret.extend(self.__url_from_symbols(file_postfix))
            elif 'quarter' in self.condition.keys():
                # 연도당 4개 분기 생성
                for quarter in range(1, 5):
                    file_postfix = f'_{year}_Q{quarter}'
                    self.condition['year'] = [year]
                    self.condition['quarter'] = [quarter]
                    ret.extend(self.__url_from_symbols(file_postfix))
            else:
                # 연도당 하나의 파일 생성
                file_postfix = f'_{year}'
                self.condition['year'] = [year]
                ret.extend(self.__url_from_symbols(file_postfix))
        return ret

    def __make_api_list_from_to(self) -> List[List[Any]]:
        """from/to 날짜 범위 파라미터를 사용하여 API URL을 생성합니다.

        1월 1일부터 12월 31일까지의 날짜 범위로 각 연도에 대한 URL을 생성합니다.
        'from' 및 'to' 날짜 파라미터를 받는 API에 사용됩니다.

        Returns:
            List[List[Any]]: 모든 연도에 대한 API 호출 파라미터 리스트.

        사용 예시:
            2023년의 경우: from=2023-01-01&to=2023-12-31
            출력: AAPL_2023.csv
        """
        ret = []
        for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
            file_postfix = f'_{year}'
            self.condition['from'] = [f'{year}-01-01']
            self.condition['to'] = [f'{year}-12-31']
            ret.extend(self.__url_from_symbols(file_postfix))
        return ret

    def __make_api_list_date(self) -> List[List[Any]]:
        """설정된 날짜 범위의 각 날짜에 대한 API URL을 생성합니다.

        start_year/01/01부터 end_year/12/31까지 하루에 하나씩 URL을 생성합니다.
        특정 날짜 파라미터를 필요로 하는 API에 사용됩니다.

        Returns:
            List[List[Any]]: 모든 날짜에 대한 API 호출 파라미터 리스트.

        Warning:
            이는 매우 많은 수의 URL을 생성할 수 있습니다(연도당 365일 이상).
            속도 제한을 피하기 위해 주의해서 사용하십시오.

        사용 예시:
            출력 파일: AAPL_2023-01-01.csv, AAPL_2023-01-02.csv, ...
        """
        ret = []
        start_date = datetime(self.main_ctx.start_year, 1, 1)
        end_date = datetime(self.main_ctx.end_year, 12, 31)
        current_date = start_date

        # 범위의 각 날짜를 반복
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            file_postfix = f'_{date_str}'
            self.condition['date'] = [date_str]
            ret.extend(self.__url_from_symbols(file_postfix))
            current_date += timedelta(days=1)
        return ret

    def __make_api_list_page(self) -> List[List[Any]]:
        """페이지네이션된 데이터에 대한 API URL을 생성합니다.

        현재 page_num에서 시작하는 페이지 배치에 대한 URL을 생성합니다.
        다음 배치를 위해 page_num을 자동으로 증가시킵니다.

        Returns:
            List[List[Any]]: 페이지 배치에 대한 API 호출 파라미터 리스트.

        Note:
            배치 크기는 page_set_num으로 제어됩니다(기본값: 8).
            워커가 데이터를 반환하지 않을 때까지 이 메서드를 반복적으로 호출하십시오.

        사용 예시:
            첫 번째 호출: 페이지 0-7
            두 번째 호출: 페이지 8-15
            출력 파일: stock_list_0.csv, stock_list_1.csv, ...
        """
        ret = []
        for i in range(self.page_num, self.page_num + self.page_set_num):
            file_postfix = f'_{i}'
            self.condition['page'] = [i]
            ret.extend(self.__url_from_symbols(file_postfix))

        # 다음 배치를 위해 페이지 번호 증가
        self.page_num += self.page_set_num
        return ret

    def make_api_list(self) -> List[List[Any]]:
        """쿼리 파라미터 유형에 따라 API 호출 파라미터를 생성합니다.

        API 설정에 존재하는 쿼리 파라미터에 따라 적절한 URL 생성 메서드를
        결정합니다.

        우선 순위:
        1. year 파라미터 -> __make_api_list_year()
        2. from 파라미터 -> __make_api_list_from_to()
        3. date 파라미터 -> __make_api_list_date()
        4. page 파라미터 -> __make_api_list_page()
        5. 위의 모두 해당 없음 -> 단순 심볼 기반 URL

        Returns:
            List[List[Any]]: 가져오기를 위한 [file_path, symbol, postfix, url] 리스트.

        사용 예시:
            api_list = api.make_api_list()
            for params in api_list[:3]:
                print(params[1], params[3])  # 심볼과 URL 출력
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
            # 시간 파라미터 없는 단순 심볼 기반 URL
            return self.__url_from_symbols('')
