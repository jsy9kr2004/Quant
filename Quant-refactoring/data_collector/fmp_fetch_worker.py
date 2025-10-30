"""Ray 병렬 처리를 사용한 FMP 데이터 가져오기 워커 모듈입니다.

이 모듈은 Ray 분산 컴퓨팅 프레임워크를 사용하여 Financial Modeling Prep API에서
병렬 데이터 가져오기를 제공합니다. HTTP 요청, JSON 파싱, 데이터 검증, 강력한 오류
처리를 통한 CSV 파일 생성을 처리합니다.

주요 기능:
- Ray 워커를 사용한 병렬 API 요청
- 속도 제한 오류 시 자동 재시도
- 중첩된 데이터 구조를 위한 JSON 평탄화
- 특수 필드(dcf, marketCap)에 대한 데이터 타입 변환
- .csvx 마커 파일을 사용한 빈 데이터 처리
- 멀티프로세싱 환경을 위한 통합 로깅

사용 예제:
    from data_collector.fmp_fetch_worker import fetch_fmp

    # api_list에는 URL 설정이 포함된 FMPAPI 객체들이 있습니다
    fetch_fmp(main_ctx, api_list)
"""

import ray
import os
import requests
import urllib
import urllib.error
import urllib.request
from time import sleep
import json
import pandas as pd
import logging
from typing import List, Any, Tuple

# Import unified logging
from config.logger import get_logger, setup_logger_for_multiprocessing


def __flatten_json(js: Any, expand_all: bool = False) -> pd.DataFrame:
    """중첩된 JSON 데이터를 pandas DataFrame으로 평탄화합니다.

    JSON 객체와 배열을 테이블 형식으로 재귀적으로 평탄화합니다.
    중첩된 리스트를 확장하고 임베디드 딕셔너리를 펼쳐서 처리합니다.

    Args:
        js (Any): 평탄화할 문자열 또는 dict/list 객체로 된 JSON 데이터.
        expand_all (bool, optional): True인 경우 중첩된 모든 리스트를 재귀적으로
            평탄화. 기본값은 False (첫 번째 레벨만 평탄화).

    Returns:
        pd.DataFrame: 테이블 형식으로 평탄화된 데이터.

    Example:
        >>> json_data = {"symbol": "AAPL", "data": [{"date": "2023-01-01", "price": 150}]}
        >>> df = __flatten_json(json_data, expand_all=True)
        >>> df.columns
        Index(['symbol', 'date', 'price'])

    Note:
        초기 평탄화에 pandas.json_normalize를 사용한 다음 리스트 컬럼을 확장합니다.
    """
    # Normalize JSON to DataFrame
    df = pd.json_normalize(json.loads(js) if type(js) == str else js)

    # Find columns containing lists
    ex = df.applymap(type).astype(str).eq("<class 'list'>").all().to_frame(name='bool')
    isin_classlist = (ex['bool'] == True).any()

    if isin_classlist == True:
        # Get first column with lists
        col = df.applymap(type).astype(str).eq("<class 'list'>").all().idxmax()

        # Explode list and expand embedded dictionaries
        df = df.explode(col).reset_index(drop=True)
        df = df.drop(columns=[col]).join(df[col].apply(pd.Series), rsuffix=f".{col}")

        # Recursively flatten if expand_all is True and more lists exist
        if expand_all and df.applymap(type).astype(str).eq("<class 'list'>").any(axis=1).all():
            df = __flatten_json(df.to_dict("records"))

        return df
    else:
        return df


@ray.remote
def __fmp_worker(
    worker_id: int,
    main_ctx,
    file_path: str,
    symbol: str,
    file_postfix: str,
    url: str
) -> Tuple[int, bool]:
    """Ray worker function for fetching and processing FMP API data.

    Each worker makes an HTTP request to the FMP API, processes the JSON response,
    flattens the data structure, and saves it as a CSV file. Implements automatic
    retry on rate limit errors.

    Args:
        worker_id (int): Unique identifier for this worker instance.
        main_ctx: Main context object with configuration.
        file_path (str): Directory path for output file.
        symbol (str): Stock symbol being fetched.
        file_postfix (str): Suffix for output filename (e.g., '_2023_Q1').
        url (str): Complete FMP API URL to fetch.

    Returns:
        Tuple[int, bool]: (worker_id, success_flag)
            - worker_id: The worker's ID for task management
            - success_flag: True if data was fetched and saved, False otherwise

    Note:
        - Creates .csvx file (instead of .csv) when API returns empty data
        - Retries indefinitely on "Limit Reach" errors with 1-second delay
        - Logs all operations with worker-specific logger

    Example:
        >>> result = ray.get(__fmp_worker.remote(0, ctx, '/data/profile', 'AAPL', '', url))
        >>> worker_id, success = result
    """
    # Setup multiprocessing logger for Ray worker
    setup_logger_for_multiprocessing()
    logger = get_logger(f'fmp.worker-{worker_id}')

    ret = True
    while True:
        try:
            logger.info(f'Creating File "{file_path}/{symbol+file_postfix}.csv" <- "{url}"')
            # Fetch data from API
            url_data = requests.get(url)
        except ValueError:
            logger.debug("No Data. Or Different Data Type")
            ret = False
            break
        except urllib.error.HTTPError:
            logger.warning("HTTP Error 400, API_URL : ", url)
            ret = False
            break

        # Parse JSON response
        json_text = url_data.text

        # Check for API rate limit errors
        if "Limit Reach" in json_text or "Error Message" in json_text:
            logger.error("Limit Reach. Please upgrade your plan or visit our documentation")
            sleep(1)  # Wait before retry
            continue

        try:
            json_data = json.loads(json_text)
        except json.decoder.JSONDecodeError:
            logger.error("json.decoder.JSONDecodeError")
            ret = False
            break

        # Handle empty responses
        if json_data == [] or json_data == {}:
            logger.info("No Data in URL")
            # Create .csvx marker file to indicate empty data
            # This prevents re-fetching known-empty endpoints
            f = open(f"{file_path}/{symbol+file_postfix}.csvx", 'w')
            f.close()
            ret = False
            break

        # Flatten nested JSON structure
        json_data = __flatten_json(json_data, expand_all=True)

        # Handle special data type conversions
        # dcf (Discounted Cash Flow) values need float conversion
        if 'dcf' in json_data.columns:
            json_data['dcf'] = json_data['dcf'].astype(float)

        # marketCap values can exceed uint32/int64 limits, use float
        if 'marketCap' in json_data.columns:
            json_data['marketCap'] = json_data['marketCap'].astype(float)

        # Save to CSV file
        json_data.to_csv(f"{file_path}/{symbol+file_postfix}.csv", na_rep='NaN', index=False)

        # Verify data is not empty after conversion
        if json_data.empty == True:
            logger.info("No Data in CSV")
            ret = False
            break

        break  # Success - exit retry loop

    return worker_id, ret


def fetch_fmp(main_ctx, api_list: List[Any]) -> None:
    """Fetch data from FMP APIs using parallel Ray workers.

    Orchestrates parallel data fetching for multiple API endpoints using Ray.
    Handles two types of APIs:
    1. Normal APIs: Fetch a predetermined list of URLs
    2. Paginated APIs: Continuously fetch pages until no data is returned

    The function manages a pool of Ray workers, distributing tasks as workers
    complete their assignments. This ensures maximum parallelism while respecting
    API rate limits.

    Args:
        main_ctx: Main context object with configuration.
        api_list (List[FMPAPI]): List of FMPAPI objects to fetch data from.

    Note:
        Worker count is limited to 8 (or cpu_count, whichever is smaller) to
        avoid overwhelming the FMP API with requests and hitting rate limits.

        For testing purposes, normal APIs are limited to 10 URLs (line 104).
        Remove this limit for production use.

    Example:
        >>> from data_collector.fmp_api import FMPAPI
        >>> api1 = FMPAPI(main_ctx, "https://fmp.com/api/v3/profile/AAPL?apikey=xxx")
        >>> api2 = FMPAPI(main_ctx, "https://fmp.com/api/v3/income-statement/AAPL")
        >>> fetch_fmp(main_ctx, [api1, api2])

    Ray Architecture:
        - Initializes Ray with limited workers (8 max)
        - Maintains a task queue with active worker assignments
        - Uses ray.wait() to get completed tasks and assign new ones
        - Continues until all URLs are processed
    """
    logger = get_logger('fmp.fetch')

    # Limit workers to avoid API rate limits (was: cpu_count())
    # Ray handles I/O-bound tasks efficiently, but FMP API has rate limits
    # Recommended: 4-8 workers for external API calls
    worker_num = min(8, os.cpu_count())
    ray.init(num_cpus=worker_num, local_mode=False)  # True enables debug mode
    logger.info(f'✅ Ray initialized: {worker_num} workers (max: {os.cpu_count()})')

    # Process normal (non-paginated) APIs
    params = []
    for api in api_list:
        if not api.page_in_condition:
            # TODO: Remove [:10] limit for production use
            params.extend(api.make_api_list()[:10])  # Limited to 10 for testing

    params_idx = 0
    logger.info(f'fetching normal api start / len(api): {len(params)}')

    # Initialize worker pool with initial tasks
    tasks = []
    for worker_id in range(worker_num):
        if params_idx < len(params):
            tasks.append(__fmp_worker.remote(worker_id, main_ctx, *params[params_idx]))
            params_idx += 1

    # Process tasks as workers complete
    while tasks:
        # Wait for at least one task to complete
        done_ids, tasks = ray.wait(tasks, num_returns=1)
        done_worker_id, ret = ray.get(done_ids[0])

        # Assign new task to completed worker if more work available
        if params_idx < len(params):
            tasks.append(__fmp_worker.remote(done_worker_id, main_ctx, *params[params_idx]))
            params_idx += 1

    logger.info(f'fetching normal api done')

    # Process paginated APIs
    # These APIs return data in pages and require continuous fetching
    # until an empty page is returned
    logger.info(f'fetching "page" api start')
    for api in api_list:
        if api.page_in_condition:
            # Get first batch of pages
            params = api.make_api_list()
            params_idx = 0
            stop_flag = False

            # Check if this is a year-based API
            # Year-based APIs generate all params at once (all years in range)
            # Page-only APIs (like delisted_companies) generate params in batches
            is_year_based = 'year' in api.condition.keys()

            # Initialize worker pool
            tasks = []
            for worker_id in range(worker_num):
                if params_idx < len(params):
                    tasks.append(__fmp_worker.remote(worker_id, main_ctx, *params[params_idx]))
                    params_idx += 1

            # Process pages until empty data is returned
            while tasks:
                done_ids, tasks = ray.wait(tasks, num_returns=1)
                done_worker_id, ret = ray.get(done_ids[0])

                # If worker returned False, this page was empty
                # For page-only APIs (delisted_companies): stop fetching new batches
                # For year-based APIs: continue processing all years (some years may be empty)
                if not ret and not is_year_based:
                    stop_flag = not ret

                # Assign new task to completed worker
                if params_idx < len(params):
                    tasks.append(__fmp_worker.remote(done_worker_id, main_ctx, *params[params_idx]))
                    params_idx += 1

                # If all current batch is done and we haven't hit empty page,
                # fetch next batch of pages
                # Note: Year-based APIs already generated all params, so skip this
                if params_idx == len(params) and not stop_flag and not is_year_based:
                    params.extend(api.make_api_list())

    logger.info(f'fetching "page" api done')

    # Cleanup Ray resources
    ray.shutdown()
