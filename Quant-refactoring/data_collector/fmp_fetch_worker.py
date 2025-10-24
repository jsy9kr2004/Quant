import ray
import os
import requests
import urllib
import urllib.error
import urllib.request
from time import sleep
import json
import pandas as pd

def __flatten_json(js, expand_all=False):
    df = pd.json_normalize(json.loads(js) if type(js) == str else js)
    # get first column that contains lists
    ex = df.applymap(type).astype(str).eq("<class 'list'>").all().to_frame(name='bool')
    isin_classlist = (ex['bool'] == True).any()
    if isin_classlist == True:
        col = df.applymap(type).astype(str).eq("<class 'list'>").all().idxmax()
        # explode list and expand embedded dictionaries
        df = df.explode(col).reset_index(drop=True)
        df = df.drop(columns=[col]).join(df[col].apply(pd.Series), rsuffix=f".{col}")
        # any lists left?
        if expand_all and df.applymap(type).astype(str).eq("<class 'list'>").any(axis=1).all():
            df = __flatten_json(df.to_dict("records"))
        return df
    else:
        return df

@ray.remote
def __fmp_worker(worker_id, main_ctx, file_path, symbol, file_postfix, url):
    logger = main_ctx.get_logger(f'worker-{worker_id}')
    ret = True
    while True:
        try:
            logger.info(f'Creating File "{file_path}/{symbol+file_postfix}.csv" <- "{url}"')
            # json_data = pd.read_json(api_url)
            url_data = requests.get(url)
        except ValueError:
            logger.debug("No Data. Or Different Data Type")
            ret = False
            break
        except urllib.error.HTTPError:
            logger.warning("HTTP Error 400, API_URL : ", url)
            ret = False
            break
        # 읽어왔는데 비어 있을 수 있음. ValueError와 다름.
        # ValueError는 Format이 안맞는 경우고, 이 경우는 page=50 과 같은 extra_url 처리 때문
        json_text = url_data.text
        if "Limit Reach" in json_text or "Error Message" in json_text:
            logger.error("Limit Reach. Please upgrade your plan or visit our documentation")
            sleep(1)
            continue
        try:
            json_data = json.loads(json_text)
        except json.decoder.JSONDecodeError:
            logger.error("json.decoder.JSONDecodeError")
            ret = False
            break
        if json_data == [] or json_data == {}:
            logger.info("No Data in URL")
            # 비어있는 표시를 해주기 위해 parquet 뒤에 x를 붙인 file만 만들고 fd close
            # f = open(path + "/{}.parquetx".format(elem + file_postfix), 'w')
            f = open(f"{file_path}/{symbol+file_postfix}.csvx", 'w')
            f.close()
            ret = False
            break
        json_data = __flatten_json(json_data, expand_all=True)
        # dcf 값에 대한 별도 예외처리 로직
        if 'dcf' in json_data.columns:
            json_data['dcf'] = json_data['dcf'].astype(float)
        # marketCap 값에 대한 별도 예외처리 로직 (uint64 로 바꿔도 괜찮음)
        if 'marketCap' in json_data.columns:
            json_data['marketCap'] = json_data['marketCap'].astype(float)
        json_data.to_csv(f"{file_path}/{symbol+file_postfix}.csv", na_rep='NaN', index=False)
        if json_data.empty == True:
            logger.info("No Data in CSV")
            ret = False
            break
        break
        
    return worker_id, ret

def fetch_fmp(main_ctx, api_list):
    logger = main_ctx.get_logger('fmp_fetch')
    # Limit workers to avoid API rate limits (was: cpu_count())
    # Ray handles I/O-bound tasks efficiently, but FMP API has rate limits
    # Recommended: 4-8 workers for external API calls
    worker_num = min(8, os.cpu_count())
    ray.init(num_cpus=worker_num, local_mode=False) # True면 디버깅모드
    logger.info(f'ray init / worker num: {worker_num} (max: {os.cpu_count()})')


    # normal api
    params = []
    for api in api_list:
        if not api.page_in_condition:
            params.extend(api.make_api_list()[:10]) # for test (limit 10)
    params_idx = 0
    logger.info(f'fetching normal api start / len(api): {len(params)}')
    
    tasks = []
    for worker_id in range(worker_num):
        if params_idx < len(params):
            tasks.append(__fmp_worker.remote(worker_id, main_ctx, *params[params_idx]))
            params_idx += 1
    while tasks:
        done_ids, tasks = ray.wait(tasks, num_returns=1)
        done_worker_id, ret = ray.get(done_ids[0])
        
        if params_idx < len(params):
            tasks.append(__fmp_worker.remote(done_worker_id, main_ctx, *params[params_idx]))
            params_idx += 1
    logger.info(f'fetching normal api done')

    # page api
    logger.info(f'fetching "page" api start')
    for api in api_list:
        if api.page_in_condition:
            params = api.make_api_list()
            params_idx = 0
            stop_flag = False

            tasks = []
            for worker_id in range(worker_num):
                if params_idx < len(params):
                    tasks.append(__fmp_worker.remote(worker_id, main_ctx, *params[params_idx]))
                    params_idx += 1
            while tasks:
                done_ids, tasks = ray.wait(tasks, num_returns=1)
                done_worker_id, ret = ray.get(done_ids[0])

                if not ret:
                    stop_flag = not ret

                if params_idx < len(params):
                    tasks.append(__fmp_worker.remote(done_worker_id, main_ctx, *params[params_idx]))
                    params_idx += 1
                if params_idx == len(params) and not stop_flag:
                    params.extend(api.make_api_list())
    logger.info(f'fetching "page" api done')

    ray.shutdown()