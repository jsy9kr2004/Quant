import copy
import csv
import datetime
import logging
import multiprocessing
import sys
import os
import ray

import numpy as np
import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters, MinimalFCParameters        
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from collections import defaultdict
from dateutil.relativedelta import relativedelta
from functools import reduce
from g_variables import ratio_col_list, meaning_col_list, cal_ev_col_list, sector_map, cal_timefeature_col_list
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import defaultdict
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from functools import partial
from warnings import simplefilter

pd.options.display.width = 30
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class AIDataMaker:
    # 각 키워드 쌍에 대한 접미사 리스트
    suffixes_dict = {
        "standard_deviation": ["__r_0.0", "__r_0.25", "__r_0.6", "__r_0.9"],
        "quantile": ["__q_0.2", "__q_0.8"],
        "autocorrelation": ["__lag_0", "__lag_5", "__lag_9"],
        "fft_coefficient": ["__coeff_0", "__coeff_33", "__coeff_99"],
        "cwt_coefficients": ["__coeff_0", "__coeff_6", "__coeff_12"],
        "symmetry_looking": ["__r_0.0", "__r_0.25", "__r_0.65", "__r_0.9"],
        "ar_coefficient": ["__coeff_0", "__coeff_3", "__coeff_6", "__coeff_10"]
    }
    
    def __init__(self, main_ctx, conf):
        """
        1) 각 start~end 까지 rebalance period만큼 뛰며 date table 만들기 -> 
        rebalance period 에서 이전 N개월을 date table을 한 번에 보면서 시계열 feature 및 y(price 변화) 채우기
        -> 
        """
        self.main_ctx = main_ctx
        self.conf = conf
        self.rebalance_period = conf['REBALANCE_PERIOD']
        
        self.symbol_table = pd.DataFrame()
        self.price_table = pd.DataFrame()
        self.fs_table = pd.DataFrame()
        self.metrics_table = pd.DataFrame()
        self.date_table_list = []
        self.trade_date_list = []
       
        # backtest 시 이용될 디렉토리를 만들고 시작
        self.main_ctx.create_dir(self.conf['ROOT_PATH'] + "/DATE_TABLE")

        # VIEW에서 필요한 데이터 읽어오기        
        self.load_bt_table(main_ctx.start_year)
        # rebalance date 가져오기
        self.set_date()
        # rebalace date에 해당하는 price table 가져와서 가격 변동 column 채우기
        self.process_price_table_wdate()
        # fs_metric에 시계열  특성까지 추가해서 ml 데이터 만들기
        self.make_ml_data(main_ctx.start_year, main_ctx.end_year)

    def load_bt_table(self, year):
        """
        추후에 database에서 가져올 데이터가 많을 걸 대비해서 __init__ 함수에서 세팅하지 않고, 해당 함수에서 세팅토록 함
        일부 필요한 내용한 init하거나 분할해서 가져오려고 한다면 쿼리가 더 복잡해질 수 있기에 따로 빼놓음
        init에서 세팅하지 않은 이유를 코드에 도입. 해당하는 year에 값을 가져오도록 변경
        """

        self.symbol_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/symbol_list.csv")
        self.symbol_table = self.symbol_table.drop_duplicates('symbol', keep='first')
        self.symbol_table['ipoDate'] = pd.to_datetime(self.symbol_table['ipoDate'])
        self.symbol_table['delistedDate'] = pd.to_datetime(self.symbol_table['delistedDate'])

        self.price_table = pd.read_csv(self.main_ctx.root_path + "/VIEW/price.csv")
        self.price_table['date'] = pd.to_datetime(self.price_table['date'])

        self.fs_table = pd.DataFrame()
        for year in range(self.main_ctx.start_year-3, self.main_ctx.end_year+1):
            tmp_fs = pd.read_csv(self.main_ctx.root_path + "/VIEW/financial_statement_" + str(year) + ".csv",
                                    parse_dates=['fillingDate_x', 'acceptedDate_x'],
                                    dtype={'reportedCurrency_x': str, 'period_x': str,
                                        'link_x': str, 'finalLink_x': str})
            self.fs_table = pd.concat([tmp_fs, self.fs_table])
        del tmp_fs
        self.fs_table['date'] = pd.to_datetime(self.fs_table['date'])
        self.fs_table['fillingDate'] = pd.to_datetime(self.fs_table['fillingDate'])
        self.fs_table['acceptedDate'] = pd.to_datetime(self.fs_table['acceptedDate'])

        self.metrics_table = pd.DataFrame()
        for year in range(self.main_ctx.start_year-3, self.main_ctx.end_year+1):
            tmp_metrics = pd.read_csv(self.main_ctx.root_path + "/VIEW/metrics_" + str(year) + ".csv",
                                        dtype={'period_x': str, 'period_y': str})
            self.metrics_table = pd.concat([tmp_metrics, self.metrics_table])
        del tmp_metrics
        self.metrics_table['date'] = pd.to_datetime(self.metrics_table['date'])
        
    def get_trade_date(self, pdate):
        """개장일이 아닐 수도 있기에 보정해주는 함수"""
        # pdate =  pdate.date()
        post_date = pdate - relativedelta(days=10)
        res = self.price_table.query("date >= @post_date and date <= @pdate")
        if res.empty:
            return None
        else:
            return res.iloc[0].date

    def generate_date_list(self):
        """rebalance date list 뽑기"""
        date_list = []
        date = datetime.datetime(int(self.main_ctx.start_year)-3, self.conf['START_MONTH'], self.conf['START_DATE'])
        print(date)
        recent_date = self.price_table["date"].max()
        end_date = datetime.datetime(self.main_ctx.end_year, 12, 31)
        if end_date > recent_date:
            end_date = recent_date   
        
        while date <= end_date:
            date_list.append(date)
            date += relativedelta(months=self.rebalance_period)
        
        return date_list

    def set_trade_date_list(self, date_list):
        """rebalance date list를 실제 거래일로 보정"""
        trade_date_list = []
        for date in date_list:
            tdate = self.get_trade_date(date)
            if tdate is None:
                # trade date를 못찾은 경우인데 있으면 안됨
                logging.error("tradable date is None. break")
                print("tradable date is None. break")
                break  
            trade_date_list.append(tdate)        
        return trade_date_list
        
        
    def set_date(self):
        date_list = self.generate_date_list()
        self.trade_date_list = self.set_trade_date_list(date_list)
    
    
    def process_price_table_wdate(self):
        # 원본 price VIEW에서 관련 날짜만 필터링
        self.price_table = self.price_table[self.price_table['date'].isin(self.trade_date_list)]
        # 데이터를 'symbol'과 'date' 기준으로 정렬
        self.price_table = self.price_table.sort_values(by=['symbol', 'date'])
        # 각 'symbol' 내에서 인접한 행들의 'close' 값 차이를 계산하여 'price_diff' 열에 추가
        self.price_table['price_diff'] = self.price_table.groupby('symbol')['close'].diff()
        self.price_table['volume_mul_price'] = self.price_table['close'] * self.price_table['volume']
        
        # 변화량은 이전 row(이전 period)기준으로 얼마나 올랐나이므로 이전 price로 나눔
        # self.price_table['price_dev'] = self.price_table['price_diff'] / self.price_table['close']
        self.price_table['price_dev'] = self.price_table['price_diff'] / self.price_table.groupby('symbol')['close'].shift(1)
        self.price_table.rename(columns={'close': 'price'}, inplace=True)
        self.price_table.to_csv(self.main_ctx.root_path + "/VIEW/price_diff.csv", index=False)


    def filter_columns_by_suffixes(self, df):
        """
        주어진 DataFrame에서 각 컬럼을 확인하고, suffixes_dict에 정의된 키워드와 접미사가 포함되어 있으면 그 컬럼을 포함시킵니다.
        키워드가 포함되지 않은 컬럼은 무조건 포함시키고, 키워드가 포함된 컬럼은 해당 접미사가 하나라도 있을 때만 포함합니다.
        """
        filtered_cols = []  # 결과 컬럼을 저장할 리스트
        for col in df.columns:
            include_col = True  # 컬럼을 포함시킬지 여부를 결정하는 플래그
            for keyword, suffixes in self.suffixes_dict.items():
                if keyword in col:  # 컬럼에 키워드가 포함되어 있는지 확인
                    # 키워드가 포함된 경우, 접미사 중 하나가 있는지 확인
                    if any(suffix in col for suffix in suffixes):
                        break  # 접미사가 하나라도 있으면 루프 중단하고 컬럼 포함
                    else:
                        include_col = False  # 접미사가 없으면 컬럼을 포함시키지 않음
                        break
            if include_col:  # 컬럼을 포함시킬지 여부에 따라 추가
                filtered_cols.append(col)
        return filtered_cols

    def filter_dates(self, df, target_col_name, start_year, end_year):
        # column 명 확인용
        # date_columns = [col for col in df.columns if 'date' in col.lower()]  # 모든 열 이름을 소문자로 변경하여 검사
        # print(date_columns)
        
        # 'fillingDate'를 datetime 타입으로 변환
        df[target_col_name] = pd.to_datetime(df[target_col_name])
        start_date = pd.Timestamp(year=start_year, month=1, day=1)
        end_date = pd.Timestamp(year=end_year+1, month=3, day=1)
        filtered_df = df[(df[target_col_name] >= start_date) & (df[target_col_name] <= end_date)]
        return filtered_df

    def reorder_columns(self, df, keywords = ['symbol', 'date']):
        # 키워드를 포함하는 컬럼과 그렇지 않은 컬럼을 분리
        key_cols = [col for col in df.columns if any(key.lower() in col.lower() for key in keywords)]
        other_cols = [col for col in df.columns if col not in key_cols]
        # 키워드를 포함하는 컬럼을 앞으로, 그 외의 컬럼을 뒤로 배치
        new_order = key_cols + other_cols
        return df[new_order]
    
    def efficient_merge_columns(self, df):
        # 컬럼 이름에서 고유한 기본 이름 추출
        base_cols = set(col.rstrip('_x').rstrip('_y') for col in df.columns)

        # 각 기본 이름에 대해 해당하는 모든 컬럼을 combine_first로 병합
        result_df = pd.DataFrame(index=df.index)
        for base in base_cols:
            # 해당 기본 이름을 포함하는 모든 컬럼을 찾기
            cols = [col for col in df.columns if col.startswith(base)]
            # reduce를 사용하여 순차적으로 combine_first 적용
            merged_col = reduce(lambda x, y: df[y].combine_first(x), cols, pd.Series([np.nan]*len(df), index=df.index))
            result_df[base] = merged_col
        result_df = self.reorder_columns(result_df)
        return result_df      
    

    def make_ml_data(self, start_year, end_year):
        for cur_year in range(start_year, end_year+1):
            table_for_ai = self.symbol_table.copy()
            print(table_for_ai)
            cur_price_table = self.price_table.copy()
            cur_price_table = self.filter_dates(cur_price_table, 'date', cur_year-3, cur_year)
            
            print("***price table after filter_dates")
            print(cur_price_table)
            # 각 symbol 별로 volume_mul_price의 평균을 계산
            symbol_means = cur_price_table.groupby('symbol')['volume_mul_price'].mean().reset_index()
            # 평균 값이 상위 50%에 해당하는 symbol만 선택
            top_symbols = symbol_means.nlargest(int(len(symbol_means) * 0.50), 'volume_mul_price')
            # 원래 데이터프레임에서 상위 50%에 해당하는 symbol의 데이터만 남김
            cur_price_table = cur_price_table[cur_price_table['symbol'].isin(top_symbols['symbol'])]
            print("***price table after filter_volume")
            print(cur_price_table)
            
            table_for_ai = pd.merge(table_for_ai, cur_price_table, how='inner', on='symbol')
            table_for_ai.rename(columns={'date': 'rebalance_date'}, inplace=True)
            print("***table for ai after merge(table, price)")
            print(table_for_ai)
            
            fs = self.fs_table.copy()
            print("***fs origin")
            print(fs)
            fs = fs[fs['symbol'].isin(top_symbols['symbol'])]
            fs = self.filter_dates(fs, 'fillingDate', cur_year-3, cur_year)
            fs = fs.drop_duplicates(['symbol', 'date'], keep='first')
            print("***fs after filter, dedup")
            print(fs)
            
            metrics = self.metrics_table
            metrics = metrics[metrics['symbol'].isin(top_symbols['symbol'])]
            metrics = metrics.drop_duplicates(['symbol', 'date'], keep='first')

            # fs와 metrics에서 겹치는 컬럼 찾기
            common_columns = fs.columns.intersection(metrics.columns)
            # 겹치는 컬럼을 metrics에서 제외
            fs = fs.drop(columns=common_columns.difference(['symbol', 'date']))
            
            print("222")
            print("*** metrics")
            print(metrics)
            fs_metrics = pd.merge(fs, metrics, how='inner', on=['symbol', 'date'])
            print("*** fs_metrics")
            print(fs_metrics)
            print("333")
            fs_metrics = fs_metrics.drop_duplicates(['symbol', 'date'], keep='first')
            # fs_metrics = efficient_merge_columns(fs_metrics)

            print("444")        
            # df1의 'date' 컬럼을 datetime 타입으로 변환합니다.
            fs_metrics['date'] = pd.to_datetime(fs_metrics['date'])
            fs_metrics.rename(columns={'date': 'report_date'}, inplace=True)
            fs_metrics['fillingDate'] = pd.to_datetime(fs_metrics['fillingDate'])

            # 각 행의 'date'보다 크면서 가장 가까운 'date2'를 찾아 'date3' 컬럼에 할당합니다.
            def find_nearest_date(row_date, date_index):
                # 'date'보다 큰 'date_list' 내의 날짜들만 필터링
                future_dates = date_index[date_index > row_date]
                # 필터링된 날짜들 중 가장 이른 날짜를 반환
                return future_dates.min() if not future_dates.empty else pd.NaT
            date_index = pd.DatetimeIndex(self.trade_date_list.copy())
            # 'date3' 컬럼에 적용합니다.
            # fs_metrics['rebalance_date'] = fs_metrics['fillingDate_x'].apply(lambda x: find_nearest_date(x, date_index))
            # 날짜 인덱스 생성 및 정렬
            date_index = np.sort(date_index)
            # 각 'fillingDate_x'에 대해 다음 'date2'를 한 번에 찾기
            indices = np.searchsorted(date_index, fs_metrics['fillingDate'], side='right')
            fs_metrics['rebalance_date'] = [date_index[i] if i < len(date_index) else pd.NaT for i in indices]        
            
            # calendarYear_x와 period_x를 결합하여 새로운 컬럼 'year_period'를 생성
            fs_metrics = fs_metrics.dropna(subset=['calendarYear'])
            period_map = {'Q1': 0.2, 'Q2': 0.4, 'Q3': 0.6, 'Q4': 0.8}
            fs_metrics['year_period'] = fs_metrics['calendarYear'] + fs_metrics['period'].map(period_map)
            fs_metrics = fs_metrics.sort_values(by=['symbol', 'year_period'])
            
            # 각 symbol 별로 time 값을 0부터 11까지 할당하는 함수
            def assign_time(group):
                group = group.sort_values(by='year_period').reset_index(drop=True)
                group['time_for_sort'] = range(len(group))
                return group

            # symbol 별로 groupby 한 후, time 값을 할당(시계열 feature 추출 시 사용)
            fs_metrics = fs_metrics.groupby('symbol').apply(assign_time).reset_index(drop=True)
            
            for col in meaning_col_list:
                if col not in fs_metrics.columns:
                    continue
                new_col_name = 'OverMC_' + col
                fs_metrics[new_col_name] = np.where(fs_metrics['marketCap'] > 0,
                                                    fs_metrics[col]/fs_metrics['marketCap'], np.nan)
                                                                    
            # "bookValuePerShare", # 시총 / bookValuePerShare = PBR
            # "eps", # (시총/eps = 유사 PER)
            # "netdebt", # (netdebt + 시총) = EV
            # "operatingCashflow", # ev / operatingCashflow = evToOperatingCashFlow
            # "FreeCashflow", # ev / FreeCashflow = evToFreeCashflow
            # "ebitda", #  ev / ebitda = enterpriseValueOverEBITDA
            # "revenues" # ev/revenues =  evToSales
            fs_metrics["adaptiveMC_ev"] = fs_metrics['marketCap'] + fs_metrics["netDebt"]
            for col in cal_ev_col_list:
                new_col_name = 'adaptiveMC_' + col
                fs_metrics[new_col_name] = np.where(fs_metrics[col] > 0,
                                                    fs_metrics['adaptiveMC_ev']/fs_metrics[col], np.nan)            
            print("*** fs_metrics w/ rebalance_date")
            print(fs_metrics)
            fs_metrics.head(1000).to_csv(self.main_ctx.root_path + f"/fs_metric_wdate_{str(cur_year)}.csv", index=False)
            # fs_metrics_sorted.to_csv(self.main_ctx.root_path + "/test_ts.csv", index=False)
            # table_for_ai.to_csv(self.main_ctx.root_path + "/tb_for_ai_1.csv", index=False)
            # 파일과 관련 설정
            root_path = self.main_ctx.root_path
            
            for quarter_str, quarter in [('Q1', 0.2), ('Q2', 0.4), ('Q3', 0.6), ('Q4', 0.8)]: #1Q, 2Q, 3Q, 4Q
                base_year_period = cur_year + quarter            
                file_path = os.path.join(self.main_ctx.root_path, "ml_per_year", f"rnorm_fs_{str(cur_year)}_{quarter_str}.csv")
                file2_path = os.path.join(self.main_ctx.root_path, "ml_per_year", f"rnorm_ml_{str(cur_year)}_{quarter_str}.csv")
                if os.path.isfile(file2_path):
                    print(f"*** there is csv file {str(cur_year)}_{quarter_str}")
                    print(f"*** there is csv file {str(cur_year)}_{quarter_str}")
                    continue
                
                print(base_year_period)
                filtered_data = fs_metrics[fs_metrics['year_period'] <= float(base_year_period)]
                # filtered_data.to_csv(self.main_ctx.root_path + f"/window_data1_{str(cur_year)}.csv", index=False)
                # 각 symbol 별로 최근 12개의 row만 선택하는 함수
                def get_last_12_rows(group):
                    return group.tail(12)
                # symbol 별로 groupby 한 후, 각 그룹에서 최근 12개의 row를 선택
                window_data = filtered_data.groupby('symbol').apply(get_last_12_rows).reset_index(drop=True)
                print(window_data)
                # 'symbol' 별로 row 수를 세는 코드
                symbol_counts = window_data['symbol'].value_counts()
                # row 수가 12개 미만인 'symbol'을 필터링하는 코드
                symbols_to_remove = symbol_counts[symbol_counts < 12].index
                # window_data에서 해당 'symbol'을 제거하는 코드
                window_data = window_data[~window_data['symbol'].isin(symbols_to_remove)]                
                # window_data.to_csv(self.main_ctx.root_path + f"/window_data2_{str(cur_year)}.csv", index=False)

                df_for_extract_feature = pd.DataFrame()
                for target_col in cal_timefeature_col_list:
                    if target_col in window_data.columns and not window_data[target_col].isna().any():
                        # print("***window_data***")
                        # print(window_data)
                        temp_df = pd.DataFrame({
                            'id': window_data['symbol'],
                            'kind' : target_col,
                            'time': window_data['time_for_sort'],
                            'value': window_data[target_col].values,
                            'year_period' : window_data['year_period'] 
                        })
                        df_for_extract_feature = pd.concat([df_for_extract_feature, temp_df])
                    
                if not df_for_extract_feature.empty:
                    features = extract_features(df_for_extract_feature, column_id='id', column_kind = 'kind', column_sort='time', column_value='value',
                                                default_fc_parameters=EfficientFCParameters())
                    features = features.rename(columns=lambda x: f"{x.partition('__')[0]}_ts_{x.partition('__')[2]}")

                    # 각 키워드에 대해 필터링을 수행하고, 결과 컬럼을 새로운 DataFrame에 추가합니다.
                    df_w_time_feature = pd.DataFrame()
                    # 모든 조건을 적용하여 컬럼 필터링
                    filtered_columns = self.filter_columns_by_suffixes(features)
                    # 필터링된 컬럼으로 새로운 DataFrame 생성
                    df_w_time_feature = features[filtered_columns]
                    df_w_time_feature['symbol'] = features.index
                    window_data = window_data[window_data['year_period'] == float(base_year_period)]
                    df_w_time_feature = pd.merge(window_data, df_w_time_feature, how='inner', on='symbol')
                    
                    # 절대 값 column 도 입력으로 쓰기엔 의미 없으니 제거
                    abs_col_list = list(set(meaning_col_list) - set(ratio_col_list))
                    for col in abs_col_list:
                        df_w_time_feature = df_w_time_feature.drop([col], axis=1, errors='ignore')

                    # 정규화에서 제외한 컬럼들은 별도로 보관
                    excluded_columns = ['symbol', 'rebalance_date', 'report_date', 'fillingDate_x', 'year_period']
                    excluded_df = df_w_time_feature[excluded_columns]
                    filtered_columns = [
                        col for col in df_w_time_feature.columns 
                        if ('_ts_' in col) or 
                        (col in ratio_col_list) or 
                        col.startswith('OverMC_') or 
                        col.startswith('adaptiveMC_')
                    ]
                    # filtered_columns.append("volume_mul_price")
                    filtered_df = df_w_time_feature[filtered_columns]
                    # RobustScaler를 사용하여 정규화
                    scaler = RobustScaler()
                    scaled_data = scaler.fit_transform(filtered_df)
                    scaled_df = pd.DataFrame(scaled_data, columns=filtered_df.columns)
                    # 정규화된 데이터와 제외한 컬럼들을 합쳐 최종 DataFrame 생성
                    scaled_df = pd.concat([excluded_df, scaled_df], axis=1)

                    # price 정보가 없는 최신 데이터를 위한 처리
                    symbol_industry = table_for_ai[['symbol', 'industry', 'volume_mul_price']]
                    symbol_industry = symbol_industry.drop_duplicates('symbol', keep='first')                    
                    fs_df = pd.merge(symbol_industry, scaled_df, how='inner', on=['symbol'])
                    fs_df["sector"] = fs_df["industry"].map(sector_map)
                    fs_df.to_csv(file_path)
                        
                    cur_table_for_ai = pd.merge(table_for_ai, scaled_df, how='inner', on=['symbol','rebalance_date'])                                        
                    
                    # *** 이 이후에는 시계열까지 뽑고 후처리들 
                    # *** 이 이후에는 시계열까지 뽑고 후처리들                
                    cur_table_for_ai["sector"] = cur_table_for_ai["industry"].map(sector_map)
                    
                    cur_table_for_ai['price_dev_subavg'] \
                        = cur_table_for_ai['price_dev'] - cur_table_for_ai['price_dev'].mean()                    
                    
                    sector_list = list(cur_table_for_ai['sector'].unique())
                    sector_list = [x for x in sector_list if str(x) != 'nan']
                    for sec in sector_list:
                        sec_mask = cur_table_for_ai['sector'] == sec
                        sec_mean = cur_table_for_ai.loc[sec_mask, 'price_dev'].mean()
                        cur_table_for_ai.loc[sec_mask, 'sec_price_dev_subavg'] = cur_table_for_ai.loc[sec_mask, 'price_dev'] - sec_mean
                    cur_table_for_ai.to_csv(file2_path, index=False)
                        
                    # # N% 넘게 비어있는 row drop
                    # logging.info("before fs_metric len : {}".format(len(cur_table_for_ai)))
                    # cur_table_for_ai['nan_count_per_row'] = cur_table_for_ai.isnull().sum(axis=1)
                    # filtered_row = cur_table_for_ai['nan_count_per_row'] < int(len(cur_table_for_ai.columns)*0.7)
                    # cur_table_for_ai = cur_table_for_ai.loc[filtered_row,:]
                    # logging.info("after fs_metric len : {}".format(len(cur_table_for_ai)))