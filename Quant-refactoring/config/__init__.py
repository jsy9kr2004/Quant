"""Quant Trading System의 Configuration 모듈입니다.

이 패키지는 Quant Trading System을 위한 configuration 관리, logging 인프라, 그리고
전역 변수를 제공합니다. 애플리케이션 전체의 설정과 유틸리티를 위한 중앙 허브 역할을 합니다.

config 패키지는 다음을 포함합니다:

1. **Configuration Loading** (context_loader.py):
   - ConfigLoader: dot notation을 지원하는 현대적인 YAML configuration loader
   - MainContext: logging이 통합된 메인 application context
   - ContextLoader: 하위 호환성을 위한 레거시 configuration loader
   - load_config: configuration 파일 로딩을 위한 편의 함수

2. **Unified Logging System** (logger.py):
   - setup_logging: multiprocessing을 지원하는 중앙화된 logging 초기화
   - get_logger: 애플리케이션 전체에서 logger 인스턴스 가져오기
   - 자동 rotation이 있는 색상 console 출력
   - Thread-safe 및 multiprocessing-safe logging

3. **Global Variables** (g_variables.py):
   - Machine learning을 위한 feature column 리스트
   - 섹터 분류 매핑
   - 데이터 품질 지표
   - Financial metric 정의

주요 Exports:
    - load_config: YAML 파일에서 configuration 로드
    - MainContext: logging과 설정이 포함된 메인 application context

사용법:
    기본 application 설정::

        from config import load_config, MainContext

        # Configuration 파일 로드
        config = load_config('config/conf.yaml')

        # 메인 application context 생성
        # 이것은 자동으로 logging을 설정하고 모든 설정을 로드합니다
        context = MainContext(config)

        # 모듈을 위한 logger 가져오기
        logger = context.get_logger(__name__)
        logger.info("Application initialized")

        # Configuration 설정 접근
        print(f"Data path: {context.root_path}")
        print(f"Processing {context.start_year} to {context.end_year}")

    현대적인 config loader 사용::

        from config.context_loader import ConfigLoader

        # Configuration 로드 및 접근
        config = ConfigLoader('config/conf.yaml')

        # dot notation으로 중첩된 값 접근
        start_year = config.get('DATA.START_YEAR', 2015)
        api_key = config.get('DATA.API_KEY')

        # Configuration 섹션 가져오기
        ml_config = config.get_ml_config()
        backtest_config = config.get_backtest_config()

    Logging system 직접 사용::

        from config.logger import setup_logging, get_logger

        # Logging 설정 (애플리케이션 시작 시 한 번 호출)
        setup_logging(
            log_level='INFO',
            log_file='trading.log',
            log_dir='logs',
            use_colors=True
        )

        # 모든 모듈에서 logger 가져오기
        logger = get_logger(__name__)
        logger.info("Processing data")
        logger.error("Failed to connect", extra={'host': 'api.example.com'})

    전역 변수 사용::

        from config.g_variables import ratio_col_list, sector_map

        # Financial ratio feature 선택
        ratio_features = df[ratio_col_list]

        # Industry를 sector로 매핑
        sector = sector_map.get(industry, 'Unknown')

모듈 구조:
    config/
    ├── __init__.py           # 이 파일 - 패키지 초기화
    ├── context_loader.py     # Configuration 로딩 및 context 관리
    ├── logger.py            # Unified logging system
    ├── g_variables.py       # 전역 변수 및 상수
    └── conf.yaml            # Configuration 파일 (Python 패키지 외부)

Configuration 파일 형식 (conf.yaml):
    Configuration 파일은 다음 섹션을 포함해야 합니다::

        DATA:
          START_YEAR: 2015
          END_YEAR: 2023
          ROOT_PATH: /home/user/Quant/data
          API_KEY: your_api_key_here
          FMP_URL: https://financialmodelingprep.com

        ML:
          MODEL_TYPE: xgboost
          TRAIN_START_YEAR: 2015
          TEST_START_YEAR: 2020

        BACKTEST:
          INITIAL_CAPITAL: 1000000
          REBALANCE_PERIOD: monthly

        LOG_LVL: 20  # 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL

설계 원칙:
    - **중앙화된 Configuration**: 모든 설정을 한 곳에 (conf.yaml)
    - **Type Safety**: IDE 지원 향상을 위한 전체 type hint
    - **하위 호환성**: 기존 코드를 위한 레거시 ContextLoader 유지
    - **현대적인 패턴**: 새 코드는 ConfigLoader와 MainContext 사용
    - **Multiprocessing 지원**: Logging system이 multiprocessing-safe
    - **쉬운 마이그레이션**: 기존 코드가 점진적으로 새 패턴으로 마이그레이션 가능

마이그레이션 가이드:
    레거시 ContextLoader에서 현대적인 방식으로::

        # 기존 코드
        from config.context_loader import ContextLoader
        context = ContextLoader()
        logger = context.get_logger('MyModule')

        # 새 코드 (권장)
        from config import load_config, MainContext
        config = load_config('config/conf.yaml')
        context = MainContext(config)
        logger = context.get_logger('MyModule')

참고:
    - context_loader: 상세한 configuration 로딩 문서
    - logger: Logging system 아키텍처 및 기능
    - g_variables: 사용 가능한 feature 리스트 및 섹터 매핑

주의사항:
    - Unified logging system은 MainContext가 초기화될 때 자동으로 구성됩니다
    - Configuration 파일은 안전한 YAML 파싱을 사용하여 로드됩니다 (yaml.safe_load)
    - 모든 logger는 일관성을 위해 동일한 configuration을 공유합니다
    - 전역 변수는 상수이며 runtime에 수정하면 안 됩니다

예제:
    완전한 application 초기화::

        from config import load_config, MainContext
        from config.logger import get_logger

        def main():
            # Configuration 로드
            config = load_config('config/conf.yaml')

            # 메인 context 초기화 (자동으로 logging 설정)
            context = MainContext(config)

            # 메인 모듈을 위한 logger 가져오기
            logger = get_logger(__name__)
            logger.info("="*80)
            logger.info("Application Starting")
            logger.info("="*80)

            # 필요한 디렉토리 생성
            context.create_dir(context.root_path)

            # 애플리케이션 로직
            logger.info(f"Processing data from {context.start_year}")

            logger.info("Application completed successfully")

        if __name__ == '__main__':
            main()

    Multiprocessing application::

        from config import load_config, MainContext
        from config.logger import setup_logger_for_multiprocessing, get_logger
        import multiprocessing as mp

        def worker_function(symbol, config):
            # 자식 프로세스를 위한 logging 설정
            setup_logger_for_multiprocessing()
            logger = get_logger(__name__)

            logger.info(f"Processing {symbol}")
            # Worker 로직
            logger.info(f"Completed {symbol}")

        def main():
            # 메인 프로세스에서 설정
            config = load_config('config/conf.yaml')
            context = MainContext(config)
            logger = get_logger(__name__)

            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

            # Worker pool 생성
            with mp.Pool(4) as pool:
                pool.starmap(worker_function, [(s, config) for s in symbols])

            logger.info("All workers completed")

        if __name__ == '__main__':
            main()
"""

from .context_loader import load_config, MainContext

__all__ = ['load_config', 'MainContext']
