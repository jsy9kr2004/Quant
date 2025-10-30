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

    예제:
        >>> loader = ConfigLoader('config/conf.yaml')
        >>> # dot notation으로 중첩된 값 접근
        >>> start_year = loader.get('DATA.START_YEAR', 2015)
        >>> # 전체 섹션 접근
        >>> ml_config = loader.get_ml_config()
        >>> # Dictionary 스타일 접근
        >>> api_key = loader['DATA.API_KEY']

    Raises:
        FileNotFoundError: Configuration 파일이 존재하지 않는 경우.
    """

    def __init__(self, config_path: str = "config/conf.yaml") -> None:
        """Initialize the configuration loader.

        Args:
            config_path (str, optional): Path to the YAML configuration file.
                Defaults to "config/conf.yaml".

        Raises:
            FileNotFoundError: If the configuration file is not found at the
                specified path.

        Example:
            >>> loader = ConfigLoader('config/conf.yaml')
            >>> loader = ConfigLoader()  # Uses default path
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file.

        This is an internal method that reads the YAML file and parses it into
        a dictionary. It logs the successful loading of the configuration.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary containing all
                settings from the YAML file.

        Raises:
            yaml.YAMLError: If the YAML file is malformed.
            IOError: If there are issues reading the file.
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logging.info(f"✅ Configuration loaded from: {self.config_path}")
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Supports nested key access using dot notation. If any key in the path
        doesn't exist, returns the default value instead of raising an error.

        Args:
            key (str): Configuration key, supports dot notation for nested access.
                For example: 'ML.TRAIN_START_YEAR' accesses config['ML']['TRAIN_START_YEAR'].
            default (Any, optional): Default value to return if key is not found.
                Defaults to None.

        Returns:
            Any: The configuration value if found, otherwise the default value.

        Example:
            >>> loader.get('DATA.START_YEAR', 2015)
            2020
            >>> loader.get('DATA.NONEXISTENT_KEY', 'default_value')
            'default_value'
            >>> loader.get('ML.MODEL_TYPE')
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
        """Get the DATA section of the configuration.

        Returns all data-related configuration settings including data paths,
        date ranges, API keys, and data source settings.

        Returns:
            Dict[str, Any]: Dictionary containing all DATA configuration settings.
                Returns empty dict if DATA section is not found.

        Example:
            >>> data_config = loader.get_data_config()
            >>> print(data_config['START_YEAR'])
            2015
            >>> print(data_config['ROOT_PATH'])
            '/home/user/Quant/data'
        """
        return self.config.get('DATA', {})

    def get_ml_config(self) -> Dict[str, Any]:
        """Get the ML (Machine Learning) section of the configuration.

        Returns all machine learning related configuration settings including
        model parameters, training settings, and feature engineering options.

        Returns:
            Dict[str, Any]: Dictionary containing all ML configuration settings.
                Returns empty dict if ML section is not found.

        Example:
            >>> ml_config = loader.get_ml_config()
            >>> print(ml_config['MODEL_TYPE'])
            'xgboost'
            >>> print(ml_config['TRAIN_START_YEAR'])
            2015
        """
        return self.config.get('ML', {})

    def get_backtest_config(self) -> Dict[str, Any]:
        """Get the BACKTEST section of the configuration.

        Returns all backtesting related configuration settings including
        portfolio parameters, trading rules, and performance metrics.

        Returns:
            Dict[str, Any]: Dictionary containing all BACKTEST configuration settings.
                Returns empty dict if BACKTEST section is not found.

        Example:
            >>> backtest_config = loader.get_backtest_config()
            >>> print(backtest_config['INITIAL_CAPITAL'])
            1000000
            >>> print(backtest_config['REBALANCE_PERIOD'])
            'monthly'
        """
        return self.config.get('BACKTEST', {})

    def get_features_config(self) -> Dict[str, Any]:
        """Get the FEATURES section of the configuration.

        Returns all feature engineering related configuration settings including
        feature selection, transformations, and preprocessing options.

        Returns:
            Dict[str, Any]: Dictionary containing all FEATURES configuration settings.
                Returns empty dict if FEATURES section is not found.

        Example:
            >>> features_config = loader.get_features_config()
            >>> print(features_config['NORMALIZE'])
            True
            >>> print(features_config['FEATURE_SELECTION_METHOD'])
            'importance'
        """
        return self.config.get('FEATURES', {})

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to configuration values.

        Allows accessing configuration values using bracket notation, similar
        to dictionary access. This is a convenience method that calls get()
        internally.

        Args:
            key (str): Configuration key, supports dot notation.

        Returns:
            Any: The configuration value if found, otherwise None.

        Example:
            >>> api_key = loader['DATA.API_KEY']
            >>> start_year = loader['DATA.START_YEAR']
        """
        return self.get(key)


class ContextLoader:
    """Legacy configuration loader for backward compatibility.

    This class maintains compatibility with older code that uses the original
    configuration loading pattern. It loads YAML configuration and dynamically
    sets attributes on the instance.

    Note:
        This class is provided for backward compatibility only. New code should
        use ConfigLoader and MainContext instead.

    Attributes:
        config_file (str): Path to the configuration file.
        log_path (str): Path to the log file.
        [dynamic attributes]: Configuration keys are set as instance attributes.

    Warning:
        This class uses bare except clauses and may mask errors. It's recommended
        to migrate to the modern ConfigLoader when possible.

    Example:
        >>> context = ContextLoader()
        >>> logger = context.get_logger('MyModule')
        >>> logger.info('Legacy context loaded')
        >>> # Access configuration as attributes
        >>> start_year = context.data['START_YEAR']
    """

    def __init__(self) -> None:
        """Initialize the legacy context loader.

        Loads the configuration file and sets up basic logging. Configuration
        keys are automatically set as instance attributes.

        Raises:
            Exception: If config file is not found or if duplicate keys exist.
        """
        self.config_file = 'config/conf.yaml'
        self.log_path = 'log.txt'
        self.__load_config()

    def __load_config(self) -> None:
        """Load configuration from YAML file (internal method).

        Reads the YAML configuration file and dynamically sets configuration
        values as instance attributes. Keys are converted to lowercase.

        Note:
            Uses bare except clause which may mask errors. This is a known
            issue in the legacy code.

        Raises:
            Exception: If config file is not found or duplicate keys exist.
        """
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                # Convert keys to lowercase and set as attributes
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
        """Get a logger instance with the specified name.

        Creates or retrieves a logger with console and file handlers. This is
        a legacy logging implementation that predates the unified logging system.

        Args:
            logger_name (str): Name for the logger instance.

        Returns:
            logging.LoggerAdapter: Logger adapter with logger_name in extra context.

        Example:
            >>> logger = context.get_logger('DataCollector')
            >>> logger.info('Starting data collection')

        Note:
            New code should use the unified logging system from config.logger
            instead of this legacy implementation.
        """
        logger = logging.getLogger(logger_name)

        # Initialize handlers on first call for this logger
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

        # Create adapter with logger name in context
        extra = {'logger_name': logger_name}
        logger = logging.LoggerAdapter(logger, extra)

        return logger

    @staticmethod
    def create_dir(path: str) -> bool:
        """Create a directory if it doesn't exist.

        Creates the specified directory including any necessary parent directories.
        Logs the operation status.

        Args:
            path (str): Path to the directory to create.

        Returns:
            bool: True if directory exists or was created successfully,
                False if creation failed.

        Example:
            >>> success = ContextLoader.create_dir('/path/to/new/directory')
            >>> if success:
            ...     print('Directory ready')

        Note:
            This is a static method and can be called without instantiating the class.
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

    예제:
        >>> from config.context_loader import load_config, MainContext
        >>>
        >>> # Configuration 로드 및 context 생성
        >>> config = load_config('config/conf.yaml')
        >>> context = MainContext(config)
        >>>
        >>> # 모듈을 위한 logger 가져오기
        >>> logger = context.get_logger('DataCollector')
        >>> logger.info(f'Processing data from {context.start_year} to {context.end_year}')
        >>>
        >>> # 필요에 따라 디렉토리 생성
        >>> context.create_dir(context.root_path)

    주의:
        Unified logging system은 MainContext가 초기화될 때 자동으로 구성됩니다.
        get_logger()를 통해 얻은 모든 logger는 이 중앙화된 configuration을 사용합니다.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the main application context.

        Extracts configuration settings and sets up the unified logging system
        with the specified parameters. Logs initialization information.

        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from YAML file.
                Should contain DATA and LOG_LVL sections at minimum.

        Example:
            >>> config = load_config('config/conf.yaml')
            >>> context = MainContext(config)

        Note:
            The logging system is configured automatically during initialization
            with rotation, colored output, and multiprocessing support.
        """
        # Data configuration settings
        data_config = config.get('DATA', {})
        self.start_year = int(data_config.get('START_YEAR', 2015))
        self.end_year = int(data_config.get('END_YEAR', 2023))
        self.root_path = data_config.get('ROOT_PATH', '/home/user/Quant/data')

        # FMP (Financial Modeling Prep) API settings
        self.fmp_url = data_config.get('FMP_URL', 'https://financialmodelingprep.com')
        self.api_key = data_config.get('API_KEY', '')
        self.ex_symbol = data_config.get('EX_SYMBOL', 'AAPL')  # Example symbol for URL parsing
        self.target_api_list = data_config.get('TARGET_API_LIST', 'data_collector/target_api_list.csv')

        # Logging configuration
        # Map integer log levels to string names
        log_level_map = {10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}
        log_lvl_int = int(config.get('LOG_LVL', 20))
        self.log_lvl = log_lvl_int
        self.log_level_name = log_level_map.get(log_lvl_int, 'INFO')
        self.log_path = "log.txt"

        # Setup unified logging system with all features
        setup_logging(
            log_level=self.log_level_name,
            log_file=self.log_path,
            log_dir='.',
            console_output=True,
            file_output=True,
            use_colors=True,
            max_bytes=10 * 1024 * 1024,  # 10MB per log file
            backup_count=5  # Keep 5 backup files
        )

        # Log initialization banner
        logger = get_logger('MainContext')
        logger.info("="*80)
        logger.info("Quant Trading System - Refactored")
        logger.info("="*80)
        logger.info(f"Data period: {self.start_year} - {self.end_year}")
        logger.info(f"Root path: {self.root_path}")
        logger.info("="*80)

    @staticmethod
    def create_dir(path: str) -> bool:
        """Create a directory if it doesn't exist.

        Creates the specified directory including any necessary parent directories.
        Uses the unified logging system to log the operation status.

        Args:
            path (str): Path to the directory to create.

        Returns:
            bool: True if directory exists or was created successfully,
                False if creation failed due to an OSError.

        Example:
            >>> MainContext.create_dir('/path/to/data/directory')
            True
            >>> MainContext.create_dir('invalid\x00path')
            False

        Note:
            This is a static method and can be called without instantiating the class.
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
        """Get a logger instance from the unified logging system.

        Returns a logger configured with the unified logging system settings.
        This method provides compatibility with existing FMP code while using
        the modern logging infrastructure.

        Args:
            logger_name (str): Name for the logger instance. Typically should be
                the module name (__name__) or a descriptive component name.

        Returns:
            logging.Logger: Configured logger instance ready for use.

        Example:
            >>> logger = context.get_logger('DataProcessor')
            >>> logger.info('Processing started')
            >>> logger.error('Error occurred', extra={'symbol': 'AAPL'})

        Note:
            All loggers obtained through this method share the same configuration
            (handlers, formatters, rotation settings) established during MainContext
            initialization.
        """
        return get_logger(logger_name)


def load_config(config_path: str = "config/conf.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file (convenience function).

    This is a convenience function that creates a ConfigLoader instance and
    returns the loaded configuration dictionary. Useful for one-off configuration
    loading without needing to manage a ConfigLoader instance.

    Args:
        config_path (str, optional): Path to the YAML configuration file.
            Defaults to "config/conf.yaml".

    Returns:
        Dict[str, Any]: Configuration dictionary parsed from the YAML file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is malformed.

    Example:
        >>> config = load_config('config/conf.yaml')
        >>> print(config['DATA']['START_YEAR'])
        2015
        >>>
        >>> # Use with MainContext
        >>> context = MainContext(config)

    See Also:
        ConfigLoader: For more advanced configuration access patterns.
    """
    loader = ConfigLoader(config_path)
    return loader.config
