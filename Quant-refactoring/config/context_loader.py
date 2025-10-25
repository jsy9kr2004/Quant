"""
Configuration loader and context manager (Improved)
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Import new logging system
from config.logger import setup_logging, get_logger


class ConfigLoader:
    """설정 파일 로더 (개선 버전)"""

    def __init__(self, config_path: str = "config/conf.yaml"):
        """
        Initialize config loader

        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logging.info(f"✅ Configuration loaded from: {self.config_path}")
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 가져오기

        Args:
            key: 키 (점 표기법 지원, 예: 'ML.TRAIN_START_YEAR')
            default: 기본값

        Returns:
            설정 값
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_data_config(self) -> Dict:
        """데이터 설정 반환"""
        return self.config.get('DATA', {})

    def get_ml_config(self) -> Dict:
        """ML 설정 반환"""
        return self.config.get('ML', {})

    def get_backtest_config(self) -> Dict:
        """백테스트 설정 반환"""
        return self.config.get('BACKTEST', {})

    def get_features_config(self) -> Dict:
        """피처 설정 반환"""
        return self.config.get('FEATURES', {})

    def __getitem__(self, key: str) -> Any:
        """딕셔너리 스타일 접근"""
        return self.get(key)


class ContextLoader:
    """기존 호환성을 위한 레거시 클래스"""

    def __init__(self):
        self.config_file = 'config/conf.yaml'
        self.log_path = 'log.txt'
        self.__load_config()

    def __load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
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

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)

        # 최초 호출
        if len(logger.handlers) == 0:
            logger.setLevel('INFO')

            formatter = logging.Formatter(
                '[%(asctime)s][%(levelname)s][%(logger_name)s] '
                '%(message)s (%(filename)s:%(lineno)d)'
            )

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(self.log_path, mode="a+")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        extra = {'logger_name': logger_name}
        logger = logging.LoggerAdapter(logger, extra)

        return logger

    @staticmethod
    def create_dir(path):
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
    """
    메인 컨텍스트 (기존 코드와의 호환성 유지)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize main context

        Args:
            config: 설정 딕셔너리
        """
        # 데이터 설정
        data_config = config.get('DATA', {})
        self.start_year = int(data_config.get('START_YEAR', 2015))
        self.end_year = int(data_config.get('END_YEAR', 2023))
        self.root_path = data_config.get('ROOT_PATH', '/home/user/Quant/data')

        # FMP 관련 설정
        self.target_api_list = data_config.get('TARGET_API_LIST', 'data_collector/target_api_list.csv')

        # 로깅 설정 (새로운 통합 시스템 사용)
        log_level_map = {10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}
        log_lvl_int = int(config.get('LOG_LVL', 20))
        self.log_lvl = log_lvl_int
        self.log_level_name = log_level_map.get(log_lvl_int, 'INFO')
        self.log_path = "log.txt"

        # Setup unified logging system
        setup_logging(
            log_level=self.log_level_name,
            log_file=self.log_path,
            log_dir='.',
            console_output=True,
            file_output=True,
            use_colors=True,
            max_bytes=10 * 1024 * 1024,  # 10MB
            backup_count=5
        )

        logger = get_logger('MainContext')
        logger.info("="*80)
        logger.info("Quant Trading System - Refactored")
        logger.info("="*80)
        logger.info(f"Data period: {self.start_year} - {self.end_year}")
        logger.info(f"Root path: {self.root_path}")
        logger.info("="*80)

    @staticmethod
    def create_dir(path: str) -> bool:
        """디렉토리 생성"""
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

    def get_logger(self, logger_name: str):
        """
        로거 생성 (기존 FMP 호환성을 위해)
        새로운 통합 로거 시스템 사용

        Args:
            logger_name: 로거 이름

        Returns:
            로거 인스턴스
        """
        return get_logger(logger_name)


def load_config(config_path: str = "config/conf.yaml") -> Dict[str, Any]:
    """
    설정 파일 로드 (편의 함수)

    Args:
        config_path: 설정 파일 경로

    Returns:
        설정 딕셔너리
    """
    loader = ConfigLoader(config_path)
    return loader.config
