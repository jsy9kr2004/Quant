"""Quant Trading System을 위한 configuration loader 및 context manager입니다.

이 모듈은 YAML 파일에서 configuration 설정을 로드하고 관리하기 위한 클래스와 함수를
제공합니다. 현대적이고 레거시인 configuration 로딩 패턴을 모두 지원하며
unified logging system과 통합됩니다.

주요 컴포넌트:
    - ConfigLoader: dot notation을 지원하는 현대적인 configuration loader
    - ContextLoader: 하위 호환성을 위한 레거시 configuration loader
    - MainContext: logging이 통합된 메인 application context
    - load_config: configuration 로딩을 위한 편의 함수

사용법:
    Modern approach::

        from config.context_loader import ConfigLoader, MainContext

        # Load configuration
        config_loader = ConfigLoader('config/conf.yaml')

        # Access with dot notation
        start_year = config_loader.get('DATA.START_YEAR', 2015)

        # Get section config
        ml_config = config_loader.get_ml_config()

        # Create main context
        context = MainContext(config_loader.config)
        logger = context.get_logger('MyModule')
        logger.info('Application started')

    Legacy approach::

        from config.context_loader import ContextLoader

        # Create legacy context
        context = ContextLoader()
        logger = context.get_logger('MyModule')

Attributes:
    None (module-level)

See Also:
    - config.logger: Unified logging system
    - config/conf.yaml: Configuration file format
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Import new logging system
from config.logger import setup_logging, get_logger


class ConfigLoader:
    """향상된 기능을 가진 현대적인 configuration 파일 loader입니다.

    이 클래스는 dot notation을 사용한 중첩 키 지원과 type-safe한 기본값을 통해
    YAML configuration 파일을 로드하고 접근하는 견고한 방법을 제공합니다.

    Loader는 configuration 파일이 존재하는지 검증하고 configuration의 다른 섹션에
    접근할 수 있는 편리한 메서드를 제공합니다.

    Attributes:
        config_path (Path): Configuration 파일 경로.
        config (Dict[str, Any]): 로드된 configuration dictionary.

    사용 예시:
        loader = ConfigLoader('config/conf.yaml')
        # dot notation으로 중첩된 값 접근
        start_year = loader.get('DATA.START_YEAR', 2015)
        # 전체 섹션 접근
        ml_config = loader.get_ml_config()
        # Dictionary 스타일 접근
        api_key = loader['DATA.API_KEY']

    Raises:
        FileNotFoundError: Configuration 파일이 존재하지 않는 경우.
    """

    def __init__(self, config_path: str = "config/conf.yaml") -> None:
        """Configuration loader를 초기화합니다.

        Args:
            config_path (str, optional): YAML configuration 파일 경로.
                기본값은 "config/conf.yaml".

        Raises:
            FileNotFoundError: 지정된 경로에 configuration 파일이 없는 경우.

        사용 예시:
            loader = ConfigLoader('config/conf.yaml')
            loader = ConfigLoader()  # 기본 경로 사용
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """YAML configuration 파일을 로드하고 파싱합니다.

        YAML 파일을 읽어서 dictionary로 파싱하는 내부 메서드입니다.
        성공적인 configuration 로딩을 로깅합니다.

        Returns:
            Dict[str, Any]: YAML 파일의 모든 설정을 포함하는 파싱된 configuration dictionary.

        Raises:
            yaml.YAMLError: YAML 파일이 잘못된 형식인 경우.
            IOError: 파일 읽기에 문제가 있는 경우.
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logging.info(f"✅ Configuration loaded from: {self.config_path}")
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Dot notation을 사용하여 configuration 값을 가져옵니다.

        Dot notation을 사용한 중첩 키 접근을 지원합니다. 경로의 키가 존재하지 않으면
        오류를 발생시키는 대신 기본값을 반환합니다.

        Args:
            key (str): Configuration 키, 중첩 접근을 위한 dot notation 지원.
                예: 'ML.TRAIN_START_YEAR'는 config['ML']['TRAIN_START_YEAR']에 접근.
            default (Any, optional): 키를 찾지 못한 경우 반환할 기본값.
                기본값은 None.

        Returns:
            Any: 찾은 경우 configuration 값, 그렇지 않으면 기본값.

        사용 예시:
            loader.get('DATA.START_YEAR', 2015)
            2020
            loader.get('DATA.NONEXISTENT_KEY', 'default_value')
            'default_value'
            loader.get('ML.MODEL_TYPE')
            'xgboost'
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_data_config(self) -> Dict[str, Any]:
        """Configuration의 DATA 섹션을 가져옵니다.

        데이터 경로, 날짜 범위, API 키, 데이터 소스 설정을 포함한 모든 데이터 관련
        configuration 설정을 반환합니다.

        Returns:
            Dict[str, Any]: 모든 DATA configuration 설정을 포함하는 dictionary.
                DATA 섹션을 찾지 못하면 빈 dict 반환.

        사용 예시:
            data_config = loader.get_data_config()
            print(data_config['START_YEAR'])
            2015
            print(data_config['ROOT_PATH'])
            '/home/user/Quant/data'
        """
        return self.config.get('DATA', {})

    def get_ml_config(self) -> Dict[str, Any]:
        """Configuration의 ML (Machine Learning) 섹션을 가져옵니다.

        모델 파라미터, 훈련 설정, 특성 엔지니어링 옵션을 포함한 모든 머신러닝 관련
        configuration 설정을 반환합니다.

        Returns:
            Dict[str, Any]: 모든 ML configuration 설정을 포함하는 dictionary.
                ML 섹션을 찾지 못하면 빈 dict 반환.

        사용 예시:
            ml_config = loader.get_ml_config()
            print(ml_config['MODEL_TYPE'])
            'xgboost'
            print(ml_config['TRAIN_START_YEAR'])
            2015
        """
        return self.config.get('ML', {})

    def get_backtest_config(self) -> Dict[str, Any]:
        """Configuration의 BACKTEST 섹션을 가져옵니다.

        포트폴리오 파라미터, 거래 규칙, 성능 메트릭을 포함한 모든 백테스팅 관련
        configuration 설정을 반환합니다.

        Returns:
            Dict[str, Any]: 모든 BACKTEST configuration 설정을 포함하는 dictionary.
                BACKTEST 섹션을 찾지 못하면 빈 dict 반환.

        사용 예시:
            backtest_config = loader.get_backtest_config()
            print(backtest_config['INITIAL_CAPITAL'])
            1000000
            print(backtest_config['REBALANCE_PERIOD'])
            'monthly'
        """
        return self.config.get('BACKTEST', {})

    def get_features_config(self) -> Dict[str, Any]:
        """Configuration의 FEATURES 섹션을 가져옵니다.

        특성 선택, 변환, 전처리 옵션을 포함한 모든 특성 엔지니어링 관련
        configuration 설정을 반환합니다.

        Returns:
            Dict[str, Any]: 모든 FEATURES configuration 설정을 포함하는 dictionary.
                FEATURES 섹션을 찾지 못하면 빈 dict 반환.

        사용 예시:
            features_config = loader.get_features_config()
            print(features_config['NORMALIZE'])
            True
            print(features_config['FEATURE_SELECTION_METHOD'])
            'importance'
        """
        return self.config.get('FEATURES', {})

    def __getitem__(self, key: str) -> Any:
        """Configuration 값에 대한 dictionary 스타일 접근을 활성화합니다.

        Dictionary 접근과 유사한 bracket notation을 사용하여 configuration 값에 접근할 수
        있습니다. 이는 내부적으로 get()을 호출하는 편의 메서드입니다.

        Args:
            key (str): Configuration 키, dot notation 지원.

        Returns:
            Any: 찾은 경우 configuration 값, 그렇지 않으면 None.

        사용 예시:
            api_key = loader['DATA.API_KEY']
            start_year = loader['DATA.START_YEAR']
        """
        return self.get(key)


class ContextLoader:
    """하위 호환성을 위한 레거시 configuration loader입니다.

    이 클래스는 원래 configuration 로딩 패턴을 사용하는 오래된 코드와의 호환성을
    유지합니다. YAML configuration을 로드하고 인스턴스에 속성을 동적으로 설정합니다.

    주의:
        이 클래스는 하위 호환성을 위해서만 제공됩니다. 새 코드는
        ConfigLoader와 MainContext를 대신 사용해야 합니다.

    Attributes:
        config_file (str): Configuration 파일 경로.
        log_path (str): Log 파일 경로.
        [dynamic attributes]: Configuration 키가 인스턴스 속성으로 설정됩니다.

    경고:
        이 클래스는 bare except 절을 사용하여 오류를 숨길 수 있습니다. 가능하면
        현대적인 ConfigLoader로 마이그레이션하는 것을 권장합니다.

    사용 예시:
        context = ContextLoader()
        logger = context.get_logger('MyModule')
        logger.info('Legacy context loaded')
        # Configuration을 속성으로 접근
        start_year = context.data['START_YEAR']
    """

    def __init__(self) -> None:
        """레거시 context loader를 초기화합니다.

        Configuration 파일을 로드하고 기본 로깅을 설정합니다. Configuration
        키가 자동으로 인스턴스 속성으로 설정됩니다.

        Raises:
            Exception: Config 파일을 찾지 못하거나 중복 키가 존재하는 경우.
        """
        self.config_file = 'config/conf.yaml'
        self.log_path = 'log.txt'
        self.__load_config()

    def __load_config(self) -> None:
        """YAML 파일에서 configuration을 로드합니다 (내부 메서드).

        YAML configuration 파일을 읽고 configuration 값을 인스턴스 속성으로 동적으로
        설정합니다. 키는 소문자로 변환됩니다.

        주의:
            오류를 숨길 수 있는 bare except 절을 사용합니다. 이는 레거시 코드의
            알려진 문제입니다.

        Raises:
            Exception: Config 파일을 찾지 못하거나 중복 키가 존재하는 경우.
        """
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                # 키를 소문자로 변환하고 속성으로 설정
                for key, value in config.items():
                    key = key.lower()
                    if not hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        raise Exception('클래스 변수 내 중복 키 존재')
        except:
            raise Exception('conf.yaml 파일 없음')
        finally:
            logger = self.get_logger('contextLoader')
            logger.info(f'config loaded successfully')

    def get_logger(self, logger_name: str) -> logging.LoggerAdapter:
        """지정된 이름으로 logger 인스턴스를 가져옵니다.

        Console 및 file handler가 있는 logger를 생성하거나 가져옵니다. 이는
        unified logging system 이전의 레거시 로깅 구현입니다.

        Args:
            logger_name (str): Logger 인스턴스 이름.

        Returns:
            logging.LoggerAdapter: Extra context에 logger_name이 있는 logger adapter.

        사용 예시:
            logger = context.get_logger('DataCollector')
            logger.info('Starting data collection')

        주의:
            새 코드는 이 레거시 구현 대신 config.logger의 unified logging system을
            사용해야 합니다.
        """
        logger = logging.getLogger(logger_name)

        # 이 logger에 대한 첫 호출 시 handler 초기화
        if len(logger.handlers) == 0:
            logger.setLevel('INFO')

            formatter = logging.Formatter(
                '[%(asctime)s][%(levelname)s][%(logger_name)s] '
                '%(message)s (%(filename)s:%(lineno)d)'
            )

            # Console handler
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

            # File handler
            file_handler = logging.FileHandler(self.log_path, mode="a+")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Context에 logger name이 있는 adapter 생성
        extra = {'logger_name': logger_name}
        logger = logging.LoggerAdapter(logger, extra)

        return logger

    @staticmethod
    def create_dir(path: str) -> bool:
        """존재하지 않는 경우 디렉토리를 생성합니다.

        필요한 상위 디렉토리를 포함하여 지정된 디렉토리를 생성합니다.
        작업 상태를 로깅합니다.

        Args:
            path (str): 생성할 디렉토리 경로.

        Returns:
            bool: 디렉토리가 존재하거나 성공적으로 생성된 경우 True,
                생성에 실패한 경우 False.

        사용 예시:
            success = ContextLoader.create_dir('/path/to/new/directory')
            if success:
                print('Directory ready')

        주의:
            이것은 static 메서드이며 클래스를 인스턴스화하지 않고 호출할 수 있습니다.
        """
        if not Path(path).exists():
            logging.info('Creating Folder "{}" ...'.format(path))
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                return True
            except OSError:
                logging.error('Cannot Creating "{}" directory.'.format(path))
                return False
        return True


class MainContext:
    """Configuration과 logging이 통합된 메인 application context입니다.

    이 클래스는 Quant Trading System의 주요 context 역할을 하며,
    configuration 관리와 unified logging system을 결합합니다.
    모든 필요한 설정을 초기화하고 logger에 대한 접근을 제공합니다.

    Context는 configuration 설정을 기반으로 logging system을 자동으로 설정하고
    디렉토리 생성 및 logger 접근을 위한 메서드를 제공합니다.

    Attributes:
        start_year (int): 데이터 수집/분석 시작 년도.
        end_year (int): 데이터 수집/분석 종료 년도.
        root_path (str): 데이터 저장 루트 경로.
        fmp_url (str): Financial Modeling Prep API base URL.
        api_key (str): FMP 접근을 위한 API key.
        ex_symbol (str): URL 파싱/테스트를 위한 예제 심볼.
        target_api_list (str): Target API 리스트 CSV 파일 경로.
        log_lvl (int): 정수로 표현된 logging level (10=DEBUG, 20=INFO 등).
        log_level_name (str): 문자열로 표현된 logging level ('DEBUG', 'INFO' 등).
        log_path (str): Log 파일 경로.

    사용 예시:
        from config.context_loader import load_config, MainContext

        # Configuration 로드 및 context 생성
        config = load_config('config/conf.yaml')
        context = MainContext(config)

        # 모듈을 위한 logger 가져오기
        logger = context.get_logger('DataCollector')
        logger.info(f'Processing data from {context.start_year} to {context.end_year}')

        # 필요에 따라 디렉토리 생성
        context.create_dir(context.root_path)

    주의:
        Unified logging system은 MainContext가 초기화될 때 자동으로 구성됩니다.
        get_logger()를 통해 얻은 모든 logger는 이 중앙화된 configuration을 사용합니다.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """메인 application context를 초기화합니다.

        Configuration 설정을 추출하고 지정된 파라미터로 unified logging system을
        설정합니다. 초기화 정보를 로깅합니다.

        Args:
            config (Dict[str, Any]): YAML 파일에서 로드된 configuration dictionary.
                최소한 DATA와 LOG_LVL 섹션을 포함해야 합니다.

        사용 예시:
            config = load_config('config/conf.yaml')
            context = MainContext(config)

        주의:
            Logging system은 rotation, colored output, multiprocessing 지원과 함께
            초기화 중에 자동으로 구성됩니다.
        """
        # 데이터 configuration 설정
        data_config = config.get('DATA', {})
        self.start_year = int(data_config.get('START_YEAR', 2015))
        self.end_year = int(data_config.get('END_YEAR', 2023))
        self.root_path = data_config.get('ROOT_PATH', '/home/user/Quant/data')

        # Debug mode: Save CSV files alongside Parquet for inspection
        save_debug_csv_value = data_config.get('SAVE_DEBUG_CSV', 'N')
        self.save_debug_csv = (save_debug_csv_value == 'Y' or save_debug_csv_value == True)

        # FMP (Financial Modeling Prep) API 설정
        self.fmp_url = data_config.get('FMP_URL', 'https://financialmodelingprep.com')
        self.api_key = data_config.get('API_KEY', '')
        self.ex_symbol = data_config.get('EX_SYMBOL', 'AAPL')  # URL 파싱을 위한 예제 심볼
        self.target_api_list = data_config.get('TARGET_API_LIST', 'data_collector/target_api_list.csv')

        # Logging configuration
        # 정수 log level을 문자열 이름으로 매핑
        log_level_map = {10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}
        log_lvl_int = int(config.get('LOG_LVL', 20))
        self.log_lvl = log_lvl_int
        self.log_level_name = log_level_map.get(log_lvl_int, 'INFO')
        self.log_path = "log.txt"

        # 모든 기능과 함께 unified logging system 설정
        setup_logging(
            log_level=self.log_level_name,
            log_file=self.log_path,
            log_dir='.',
            console_output=True,
            file_output=True,
            use_colors=True,
            max_bytes=10 * 1024 * 1024,  # 파일당 10MB
            backup_count=5  # 5개의 백업 파일 유지
        )

        # 초기화 배너 로깅
        logger = get_logger('MainContext')
        logger.info("="*80)
        logger.info("Quant Trading System - Refactored")
        logger.info("="*80)
        logger.info(f"Data period: {self.start_year} - {self.end_year}")
        logger.info(f"Root path: {self.root_path}")
        logger.info("="*80)

    @staticmethod
    def create_dir(path: str) -> bool:
        """존재하지 않는 경우 디렉토리를 생성합니다.

        필요한 상위 디렉토리를 포함하여 지정된 디렉토리를 생성합니다.
        Unified logging system을 사용하여 작업 상태를 로깅합니다.

        Args:
            path (str): 생성할 디렉토리 경로.

        Returns:
            bool: 디렉토리가 존재하거나 성공적으로 생성된 경우 True,
                OSError로 인해 생성에 실패한 경우 False.

        사용 예시:
            MainContext.create_dir('/path/to/data/directory')
            True
            MainContext.create_dir('invalid\x00path')
            False

        주의:
            이것은 static 메서드이며 클래스를 인스턴스화하지 않고 호출할 수 있습니다.
        """
        logger = get_logger('MainContext')
        path_obj = Path(path)
        if not path_obj.exists():
            logger.info(f'Creating directory: {path}')
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                return True
            except OSError as e:
                logger.error(f'Cannot create directory "{path}": {e}')
                return False
        return True

    def get_logger(self, logger_name: str) -> logging.Logger:
        """Unified logging system에서 logger 인스턴스를 가져옵니다.

        Unified logging system 설정으로 구성된 logger를 반환합니다.
        이 메서드는 현대적인 로깅 인프라를 사용하면서 기존 FMP 코드와의
        호환성을 제공합니다.

        Args:
            logger_name (str): Logger 인스턴스 이름. 일반적으로 모듈 이름(__name__) 또는
                설명적인 컴포넌트 이름이어야 합니다.

        Returns:
            logging.Logger: 사용 준비가 된 구성된 logger 인스턴스.

        사용 예시:
            logger = context.get_logger('DataProcessor')
            logger.info('Processing started')
            logger.error('Error occurred', extra={'symbol': 'AAPL'})

        주의:
            이 메서드를 통해 얻은 모든 logger는 MainContext 초기화 중에 설정된
            동일한 configuration (handler, formatter, rotation 설정)을 공유합니다.
        """
        return get_logger(logger_name)


def load_config(config_path: str = "config/conf.yaml") -> Dict[str, Any]:
    """YAML 파일에서 configuration을 로드합니다 (편의 함수).

    ConfigLoader 인스턴스를 생성하고 로드된 configuration dictionary를 반환하는
    편의 함수입니다. ConfigLoader 인스턴스를 관리할 필요 없이 일회성
    configuration 로딩에 유용합니다.

    Args:
        config_path (str, optional): YAML configuration 파일 경로.
            기본값은 "config/conf.yaml".

    Returns:
        Dict[str, Any]: YAML 파일에서 파싱된 configuration dictionary.

    Raises:
        FileNotFoundError: Configuration 파일이 존재하지 않는 경우.
        yaml.YAMLError: YAML 파일이 잘못된 형식인 경우.

    사용 예시:
        config = load_config('config/conf.yaml')
        print(config['DATA']['START_YEAR'])
        2015

        # MainContext와 함께 사용
        context = MainContext(config)

    See Also:
        ConfigLoader: 더 고급 configuration 접근 패턴을 위한 클래스.
    """
    loader = ConfigLoader(config_path)
    return loader.config
