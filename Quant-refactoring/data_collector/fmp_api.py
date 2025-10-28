from urllib.parse import urlparse, parse_qs, urlencode
from datetime import datetime, timedelta
import os

class FMPAPI():
    def __init__(self, main_ctx, url):
        ######### ???? #########
        # 여러줄이 들어간 경우, 가장 앞에 써진 url 만을 돌린다. \n을 찾아서 뒤를 전부 지워주는 작업
        if url.find('\\') != -1: # \\???? \n????
            url = url.split('\\')[0]
        ######### ???? #########

        parsed_url = urlparse(url)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        # query_params에서 삭제해야 하는 것
        # 1) apikey 삭제 
        if 'apikey' in query_params.keys():
            del query_params['apikey']
        # 2) limit 삭제
        if 'limit' in query_params.keys():
            del query_params['limit']
        
        # v3/v4 체크
        is_v4 = False if 'api/v3/' in path else True
        if is_v4:
            category = path.split('api/v4/')[1] if is_v4 else path.split('api/v3/')[1]
            need_symbol = False
            if 'symbol' in query_params.keys():
                need_symbol = True
        else:
            category = path.split('api/v4/')[1] if is_v4 else path.split('api/v3/')[1]
            need_symbol = False
            if main_ctx.ex_symbol in category:
                category = category.split(f'/{main_ctx.ex_symbol}')[0]
                path = path.split(f'/{main_ctx.ex_symbol}')[0]
                need_symbol = True

        self.main_ctx = main_ctx
        self.fmp_url = self.main_ctx.fmp_url
        self.api_key = self.main_ctx.api_key
        self.url = url
        self.main_url = path
        self.category = category
        self.condition = query_params
        self.need_symbol = need_symbol
        self.symbol_list = [] if self.need_symbol else [self.converted_category]
        self.is_v4 = is_v4
        self.file_path = f'{self.main_ctx.root_path}/{self.converted_category}'


        self.fmp_api_logger = self.main_ctx.get_logger(f'fmp_api({self.converted_category})')
        self.fmp_api_logger.info(f'FMPAPI {self.converted_category} initialized')

        if self.main_ctx.create_dir(self.file_path):
            self.fmp_api_logger.info(f'Directory {self.file_path} created successfully')
        else:
            self.fmp_api_logger.error(f'Directory {self.file_path} is not created')
            raise Exception(f'Directory {self.file_path} is not created')

        self.page_num = 0
        # Limit page batch size to avoid API rate limits (was: cpu_count())
        # This controls how many pages are fetched per batch, not worker count
        self.page_set_num = min(8, os.cpu_count())

    @property
    def converted_category(self):
        return self.category.replace("/", "-").replace("-", "_")

    @property
    def query_params_str(self):
        return urlencode({k: v[0] for k, v in self.condition.items()}, doseq=True)
    
    @property
    def page_in_condition(self):
        return 'page' in self.condition.keys()

    def __full_url(self, symbol):
        if self.is_v4:
            if self.need_symbol:
                self.query_params['symbol'] = symbol
            return f'{self.fmp_url}{self.main_url}?{self.query_params_str}&apikey={self.api_key}'
        else:
            if self.need_symbol:
                return f'{self.fmp_url}{self.main_url}/{symbol}?{self.query_params_str}&apikey={self.api_key}'
            else:
                return f'{self.fmp_url}{self.main_url}?{self.query_params_str}&apikey={self.api_key}'
            
    def __url_from_symbols(self, file_postfix):
        api_list = []
        for symbol in self.symbol_list:
            if os.path.isfile(f'{self.file_path}/{symbol+file_postfix}.csv') \
               or os.path.isfile(f'{self.file_path}/{symbol+file_postfix}.csvx'):
                self.fmp_api_logger.info(f'Skip existing file: {self.file_path}/{symbol+file_postfix}.csv/csvx')
                continue

            api_list.append([self.file_path, symbol, file_postfix, self.__full_url(symbol)])
        return api_list
            
    def set_symbol_list(self, symbol_list):
        if not self.need_symbol:
            self.fmp_api_logger.info('symbol이 필요하지 않은데 set하려함')
            raise Exception('symbol이 필요하지 않은데 set하려함')
        self.symbol_list = symbol_list
        if self.category == "historical-price-full":
            self.fmp_api_logger.info("add ETF symbols to get historical-price-full data")
            # self.symbol_list.append(pd.Series(["SPY", "IVV", "VTI", "VOO", "QQQ", "VEA", "IEFA"]))
        self.symbol_list = self.symbol_list.dropna()
    
    def __make_api_list_year(self):
        ret = []
        for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
            if 'period' in self.condition.keys():
                for period in range(1, 5):
                    file_postfix = f'_{year}_P{period}'
                    self.condition['year'] = [year]
                    self.condition['period'] = [period]
                    ret.extend(self.__url_from_symbols(file_postfix))
            elif 'quarter' in self.condition.keys():
                for quarter in range(1, 5):
                    file_postfix = f'_{year}_Q{quarter}'
                    self.condition['year'] = [year]
                    self.condition['quarter'] = [quarter]
                    ret.extend(self.__url_from_symbols(file_postfix))
            else:
                file_postfix = f'_{year}'
                self.condition['year'] = [year]
                ret.extend(self.__url_from_symbols(file_postfix))
        return ret

    def __make_api_list_from_to(self):
        ret = []
        for year in range(self.main_ctx.start_year, self.main_ctx.end_year + 1):
            file_postfix = f'_{year}'
            self.condition['from'] = [f'{year}-01-01']
            self.condition['to'] = [f'{year}-12-31']
            ret.extend(self.__url_from_symbols(file_postfix))
        return ret

    def __make_api_list_date(self):
        ret = []
        start_date = datetime(self.main_ctx.start_year, 1, 1)
        end_date = datetime(self.main_ctx.end_year, 12, 31)
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            file_postfix = f'_{date_str}'
            self.condition['date'] = [date_str]
            ret.extend(self.__url_from_symbols(file_postfix))
            current_date += timedelta(days=1)
        return ret

    def __make_api_list_page(self):
        ret = []
        for i in range(self.page_num, self.page_num+self.page_set_num):
            file_postfix = f'_{i}'
            self.condition['page'] = [i]
            ret.extend(self.__url_from_symbols(file_postfix))
        self.page_num += self.page_set_num # when recalled, it'll return next set
        return ret

    def make_api_list(self):
        # logging.info("main url : {}, extra url: {}".format(main_url, extra_url))
        if 'year' in self.condition.keys():
            return self.__make_api_list_year()
        elif 'from' in self.condition.keys():
            return self.__make_api_list_from_to()
        elif 'date' in self.condition.keys():
            return self.__make_api_list_date()
        elif 'page' in self.condition.keys():
            return self.__make_api_list_page()
        else:
            return self.__url_from_symbols('')
