"""
ETF 가격 데이터 로더 (벤치마크 전용)

이 모듈은 벤치마크 ETF의 가격 데이터를 관리합니다.
주식 데이터(price.parquet)와 완전히 분리하여, ETF 데이터 오염을 방지합니다.

주요 기능:
- 캐시 우선 로딩 (etf_price.parquet)
- 캐시 미스 시 FMP API 자동 호출
- 날짜 범위 자동 검증
- ETF와 주식 데이터의 명확한 분리

설계 원칙:
- Single Responsibility: ETF 데이터만 관리
- Fail-Safe: 실수로 ETF가 ML 학습에 포함될 가능성 제거
- On-Demand: 백테스트 실행 시에만 로드

Author: Quant Trading Team
Date: 2025-12-31
"""

import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any


class ETFDataLoader:
    """
    ETF 가격 데이터 로더

    벤치마크 비교를 위한 ETF 가격 데이터를 캐시 우선으로 로드하고,
    필요 시 FMP API에서 수집하여 저장합니다.

    Parameters:
    ----------
    config : Dict[str, Any]
        전체 설정 딕셔너리 (conf.yaml)
    root_path : str
        프로젝트 루트 경로
    logger : Optional[logging.Logger]
        로거 (없으면 자동 생성)

    Examples:
    --------
    >>> loader = ETFDataLoader(config, root_path)
    >>> etf_prices = loader.load_etf_prices(
    ...     symbols=['SPY', 'QQQ'],
    ...     start_date=datetime(2020, 1, 1),
    ...     end_date=datetime(2023, 12, 31)
    ... )
    >>> print(etf_prices)
         symbol       date   open   high    low  close     volume
    0       SPY 2020-01-02  320.5  321.2  319.8  321.1  100000000
    1       SPY 2020-01-03  321.2  322.5  320.9  322.3   95000000
    ...
    """

    def __init__(
        self,
        config: Dict[str, Any],
        root_path: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        ETFDataLoader 초기화

        Parameters:
        ----------
        config : Dict[str, Any]
            전체 설정 딕셔너리
        root_path : str
            프로젝트 루트 경로
        logger : Optional[logging.Logger]
            로거 인스턴스
        """
        self.config = config
        self.root_path = root_path
        self.logger = logger or logging.getLogger('ETFDataLoader')

        # 캐시 파일 경로
        self.cache_file = Path(root_path) / 'processed' / 'views' / 'etf_price.parquet'

        # FMP API 설정
        data_config = config.get('DATA', {})
        self.fmp_api_key = data_config.get('API_KEY', '')
        self.fmp_url = data_config.get('FMP_URL', 'https://financialmodelingprep.com')

        if not self.fmp_api_key:
            self.logger.warning("⚠️  FMP API_KEY not configured!")

    def load_etf_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        ETF 가격 데이터 로드 (캐시 우선)

        로딩 전략:
        1. 캐시 파일 존재 & 날짜 범위 충분 → 캐시 사용
        2. 캐시 없음 또는 불충분 → FMP API 수집 → 캐시 저장

        Parameters:
        ----------
        symbols : List[str]
            ETF 심볼 리스트 (예: ['SPY', 'QQQ', 'AAPL'])
        start_date : datetime
            시작 날짜
        end_date : datetime
            종료 날짜
        force_refresh : bool
            True = 캐시 무시하고 API 재수집

        Returns:
        -------
        pd.DataFrame
            ETF 가격 데이터
            Columns: [symbol, date, open, high, low, close, volume]

        Raises:
        ------
        ValueError
            API 키가 없거나 심볼 리스트가 비어있을 때
        """
        if not symbols:
            self.logger.warning("⚠️  No ETF symbols provided")
            return pd.DataFrame()

        if not self.fmp_api_key:
            raise ValueError("FMP API_KEY is required but not configured in conf.yaml")

        self.logger.info(f"\n📊 Loading ETF prices for {len(symbols)} symbols...")
        self.logger.info(f"   Symbols: {symbols}")
        self.logger.info(f"   Period: {start_date.date()} ~ {end_date.date()}")

        # 1. 캐시 확인 (force_refresh=False인 경우)
        if not force_refresh:
            cached_data = self._load_from_cache(symbols, start_date, end_date)

            if cached_data is not None and not cached_data.empty:
                self.logger.info(f"✅ Loaded {len(cached_data)} rows from cache: {self.cache_file.name}")
                return cached_data

        # 2. API 수집
        self.logger.info(f"📡 Fetching ETF prices from FMP API...")
        fetched_data = self._fetch_from_api(symbols, start_date, end_date)

        if fetched_data.empty:
            self.logger.warning("⚠️  No data fetched from API")
            return pd.DataFrame()

        # 3. 캐시 저장
        self._save_to_cache(fetched_data)

        return fetched_data

    def _load_from_cache(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        캐시 파일에서 ETF 가격 로드 및 유효성 검증

        검증 항목:
        - 캐시 파일 존재 여부
        - 요청한 심볼이 모두 포함되어 있는지
        - 날짜 범위가 충분한지

        Parameters:
        ----------
        symbols : List[str]
            요청한 ETF 심볼 리스트
        start_date : datetime
            시작 날짜
        end_date : datetime
            종료 날짜

        Returns:
        -------
        Optional[pd.DataFrame]
            유효한 캐시 데이터 또는 None
        """
        if not self.cache_file.exists():
            self.logger.debug(f"   Cache file not found: {self.cache_file}")
            return None

        try:
            df = pd.read_parquet(self.cache_file)
            self.logger.debug(f"   Cache file loaded: {len(df)} rows")

            # 날짜 컬럼 타입 확인
            if 'date' not in df.columns:
                self.logger.warning("   ⚠️  'date' column not found in cache")
                return None

            df['date'] = pd.to_datetime(df['date'])

            # 심볼 필터링
            available_symbols = set(df['symbol'].unique())
            requested_symbols = set(symbols)
            missing_symbols = requested_symbols - available_symbols

            if missing_symbols:
                self.logger.debug(f"   Missing symbols in cache: {missing_symbols}")
                return None  # 일부 심볼 누락 → API 재수집

            # 요청한 심볼만 필터링
            df = df[df['symbol'].isin(symbols)]

            if df.empty:
                self.logger.debug("   No data for requested symbols")
                return None

            # 날짜 범위 확인
            cache_start = df['date'].min()
            cache_end = df['date'].max()

            if cache_start > pd.Timestamp(start_date) or cache_end < pd.Timestamp(end_date):
                self.logger.debug(
                    f"   Cache date range insufficient: "
                    f"{cache_start.date()} ~ {cache_end.date()} "
                    f"(need {start_date.date()} ~ {end_date.date()})"
                )
                return None  # 날짜 범위 부족 → API 재수집

            # 날짜 범위로 필터링
            df = df[
                (df['date'] >= pd.Timestamp(start_date)) &
                (df['date'] <= pd.Timestamp(end_date))
            ]

            self.logger.debug(f"   ✅ Cache valid: {len(df)} rows after date filtering")
            return df

        except Exception as e:
            self.logger.warning(f"   ⚠️  Cache loading failed: {e}")
            return None

    def _fetch_from_api(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        FMP API에서 ETF 가격 데이터 수집

        API Endpoint:
        /api/v3/historical-price-full/{symbol}?from=YYYY-MM-DD&to=YYYY-MM-DD

        Parameters:
        ----------
        symbols : List[str]
            ETF 심볼 리스트
        start_date : datetime
            시작 날짜
        end_date : datetime
            종료 날짜

        Returns:
        -------
        pd.DataFrame
            수집된 ETF 가격 데이터
            Columns: [symbol, date, open, high, low, close, volume]
        """
        all_data = []

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        for symbol in symbols:
            try:
                # FMP API URL 생성
                url = (
                    f"{self.fmp_url}/api/v3/historical-price-full/{symbol}"
                    f"?from={start_str}&to={end_str}&apikey={self.fmp_api_key}"
                )

                self.logger.debug(f"   Fetching {symbol}...")

                # API 호출
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                data = response.json()

                # 데이터 파싱
                if 'historical' in data and data['historical']:
                    df = pd.DataFrame(data['historical'])
                    df['symbol'] = symbol

                    # 날짜 변환
                    df['date'] = pd.to_datetime(df['date'])

                    # 컬럼 선택 (필요한 것만)
                    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
                    df = df[required_cols]

                    all_data.append(df)
                    self.logger.info(f"   ✅ {symbol}: {len(df)} days fetched")
                else:
                    self.logger.warning(f"   ⚠️  {symbol}: No data returned from API")

            except requests.exceptions.RequestException as e:
                self.logger.error(f"   ❌ {symbol}: API request failed - {e}")
                continue
            except Exception as e:
                self.logger.error(f"   ❌ {symbol}: Data parsing failed - {e}")
                continue

        if not all_data:
            self.logger.warning("⚠️  No data fetched from API for any symbol")
            return pd.DataFrame()

        # 모든 심볼 데이터 합치기
        combined = pd.concat(all_data, ignore_index=True)

        # 날짜 순 정렬
        combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)

        self.logger.info(f"✅ Total fetched: {len(combined)} rows for {len(all_data)} symbols")

        return combined

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """
        ETF 가격 데이터를 캐시 파일에 저장

        저장 방식:
        - 기존 캐시와 병합 (중복 제거)
        - Parquet 포맷 (압축, 빠른 로딩)

        Parameters:
        ----------
        df : pd.DataFrame
            저장할 ETF 가격 데이터
        """
        if df.empty:
            self.logger.warning("⚠️  No data to save to cache")
            return

        try:
            # 디렉토리 생성
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            # 기존 캐시와 병합 (있다면)
            if self.cache_file.exists():
                existing = pd.read_parquet(self.cache_file)
                existing['date'] = pd.to_datetime(existing['date'])

                # 새 데이터와 기존 데이터 합치기
                combined = pd.concat([existing, df], ignore_index=True)

                # 중복 제거 (symbol + date 기준)
                combined = combined.drop_duplicates(subset=['symbol', 'date'], keep='last')

                # 정렬
                combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)

                self.logger.debug(f"   Merged with existing cache: {len(combined)} total rows")
                df = combined

            # Parquet 저장
            df.to_parquet(self.cache_file, index=False, engine='pyarrow', compression='snappy')

            self.logger.info(f"💾 Saved to cache: {self.cache_file}")
            self.logger.info(f"   Total rows: {len(df)}")
            self.logger.info(f"   Symbols: {sorted(df['symbol'].unique())}")
            self.logger.info(f"   Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save cache: {e}")
            # 캐시 저장 실패해도 계속 진행 (데이터는 반환됨)

    def clear_cache(self) -> None:
        """
        캐시 파일 삭제

        사용 시나리오:
        - 데이터 오류 발견 시 캐시 초기화
        - 강제 재수집 필요 시
        """
        if self.cache_file.exists():
            self.cache_file.unlink()
            self.logger.info(f"🗑️  Cache cleared: {self.cache_file}")
        else:
            self.logger.info("   No cache file to clear")
