"""Multiprocessing을 지원하는 Unified Logging System입니다.

이 모듈은 Quant Trading System을 위한 thread-safe하고 multiprocessing-safe한
logging system을 제공합니다. 색상 console 출력, 자동 log rotation,
context가 있는 구조화된 logging, 기존 logging 코드로부터의 쉬운 마이그레이션 기능이 있습니다.

Logging system은 QueueHandler와 QueueListener를 사용하여 여러 프로세스의 log가
충돌 없이 안전하게 기록되도록 합니다. 이는 데이터 수집 및 병렬 처리 작업에 필수적입니다.

기능:
    - QueueHandler/QueueListener를 사용한 thread-safe 및 multiprocessing-safe
    - 더 나은 가독성을 위한 ANSI 코드를 사용한 색상 console 출력
    - 구성 가능한 크기 및 백업 개수로 자동 log 파일 rotation
    - 사용자 정의 context 지원이 있는 구조화된 logging
    - Multiprocessing 디버깅을 위한 프로세스 이름 추적
    - 표준 Python logging으로부터의 쉬운 마이그레이션
    - 애플리케이션 전체에서 일관된 logging을 위한 중앙화된 configuration

Architecture:
    Main Process:
        setup_logging() -> QueueListener -> [ConsoleHandler, FileHandler]
                              ^
                              |
    Child Processes:         Queue
        Logger -> QueueHandler -|

Usage:
    Basic setup (call once at application start)::

        from config.logger import setup_logging, get_logger

        # Setup logging system once
        setup_logging(
            log_level='INFO',
            log_file='app.log',
            log_dir='logs',
            use_colors=True
        )

        # Get logger in any module
        logger = get_logger(__name__)
        logger.info("Application started")
        logger.error("Error occurred", extra={'symbol': 'AAPL', 'price': 150})

    Multiprocessing usage::

        from config.logger import setup_logger_for_multiprocessing, get_logger
        import multiprocessing as mp

        def worker_function(symbol):
            # Setup logger in child process
            setup_logger_for_multiprocessing()
            logger = get_logger(__name__)
            logger.info(f"Processing {symbol}")

        if __name__ == '__main__':
            setup_logging()  # Setup in main process
            with mp.Pool(4) as pool:
                pool.map(worker_function, ['AAPL', 'GOOGL', 'MSFT'])

    Custom context::

        logger = get_logger(__name__, context={'module': 'DataCollector'})
        logger.info("Data collected", extra={'records': 1000})

Module Attributes:
    _log_queue (Optional[Queue]): Global queue for multiprocessing logs.
    _queue_listener (Optional[QueueListener]): Listener that processes queued logs.
    _initialized (bool): Flag indicating if logging system is initialized.

See Also:
    - config.context_loader: For MainContext that auto-configures logging
    - Python logging documentation: https://docs.python.org/3/library/logging.html
    - QueueHandler/QueueListener: For multiprocessing details
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import multiprocessing as mp
from queue import Queue
import atexit


# 콘솔 출력을 위한 ANSI 색상 코드
class LogColors:
    """터미널 출력 포맷팅을 위한 ANSI 색상 코드입니다.

    이 클래스는 터미널 출력을 색상화하기 위한 ANSI 이스케이프 시퀀스를 정의합니다.
    색상은 개발 및 디버깅 중 가독성 향상을 위해 로그 레벨과 컴포넌트를 시각적으로
    구분하는 데 사용됩니다.

    Attributes:
        RESET (str): 모든 포맷팅을 기본값으로 재설정.
        BOLD (str): 굵은 텍스트 활성화.
        DEBUG (str): DEBUG 레벨 메시지를 위한 청록색.
        INFO (str): INFO 레벨 메시지를 위한 녹색.
        WARNING (str): WARNING 레벨 메시지를 위한 노란색.
        ERROR (str): ERROR 레벨 메시지를 위한 빨간색.
        CRITICAL (str): CRITICAL 레벨 메시지를 위한 자홍색.
        TIME (str): 타임스탬프를 위한 어두운 회색.
        LEVEL (str): 로그 레벨 이름을 위한 굵은 포맷팅.
        NAME (str): Logger 이름을 위한 파란색.

    주의:
        이 코드는 ANSI 호환 터미널(Linux, macOS, Windows 10+)에서 작동합니다.
        ColoredFormatter는 색상을 적용하기 전에 터미널 호환성을 확인합니다.

    사용 예시:
        print(f"{LogColors.ERROR}Error message{LogColors.RESET}")
        print(f"{LogColors.BOLD}Bold text{LogColors.RESET}")
    """

    RESET = '\033[0m'
    BOLD = '\033[1m'

    # 로그 레벨 - 각 레벨은 고유한 색상을 가집니다
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta

    # 컴포넌트 - 로그 메시지의 다른 부분에 사용됩니다
    TIME = '\033[90m'       # Dark gray
    LEVEL = '\033[1m'       # Bold
    NAME = '\033[94m'       # Blue


class ColoredFormatter(logging.Formatter):
    """콘솔 출력에 ANSI 색상 코드를 추가하는 커스텀 로그 formatter입니다.

    이 formatter는 표준 logging.Formatter를 확장하여 로그 레벨에 따라 색상 코드를
    추가합니다. 출력이 TTY인지 자동으로 감지하고 TTY가 아닌 경우(예: 파일로 파이핑)
    색상을 비활성화합니다.

    Formatter는 LogRecord 속성을 일시적으로 수정하여 색상을 추가하고 메시지를
    포맷한 다음 원본 값을 복원하여 다른 handler에 색상이 영향을 미치는 것을 방지합니다.

    Attributes:
        LEVEL_COLORS (Dict[int, str]): 로그 레벨에서 색상 코드로의 매핑.
        use_colors (bool): 색상을 적용할지 여부 (TTY가 아니면 False).

    사용 예시:
        formatter = ColoredFormatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            use_colors=True
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

    See Also:
        LogColors: 사용 가능한 색상 정의를 위한 클래스.
    """

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True) -> None:
        """Colored formatter를 초기화합니다.

        Args:
            fmt (str, optional): 로그 메시지를 위한 포맷 문자열. None인 경우 기본
                포맷을 사용합니다. 모든 표준 logging 포맷 문자열을 지원합니다.
            use_colors (bool, optional): 색상 출력 활성화 여부.
                기본값은 True이지만 출력이 TTY가 아니면 비활성화됩니다.

        사용 예시:
            formatter = ColoredFormatter(
                fmt='[%(levelname)s] %(message)s',
                use_colors=True
            )
        """
        super().__init__(fmt)
        # stderr가 터미널(TTY)인 경우에만 색상 사용
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """활성화된 경우 색상 코드로 로그 레코드를 포맷합니다.

        이 메서드는 로그 레벨과 logger 이름에 ANSI 색상 코드를 추가하고
        완전한 메시지를 포맷한 다음 원본 값을 복원합니다.

        Args:
            record (logging.LogRecord): 포맷할 로그 레코드.

        Returns:
            str: 활성화된 경우 색상 코드가 있는 포맷된 로그 메시지,
                그렇지 않으면 일반 포맷된 메시지.

        주의:
            동일한 LogRecord 인스턴스를 공유하는 다른 handler에 영향을 주지 않도록
            원본 레코드 속성이 보존됩니다.
        """
        if self.use_colors:
            # 나중에 복원하기 위해 원본 값 저장
            orig_levelname = record.levelname
            orig_name = record.name

            # 로그 레벨에 따라 색상 적용
            level_color = self.LEVEL_COLORS.get(record.levelno, '')
            record.levelname = f"{level_color}{record.levelname}{LogColors.RESET}"
            record.name = f"{LogColors.NAME}{record.name}{LogColors.RESET}"

            # 색상과 함께 메시지 포맷
            result = super().format(record)

            # 다른 handler를 위해 원본 값 복원
            record.levelname = orig_levelname
            record.name = orig_name

            return result
        else:
            return super().format(record)


class ContextFilter(logging.Filter):
    """모든 로그 레코드에 사용자 정의 context를 추가하는 로깅 필터입니다.

    이 필터는 프로세스 이름 및 사용자 정의 키-값 쌍과 같은 추가 context 정보로
    로그 레코드를 강화합니다. 특정 컴포넌트의 모든 로그에 메타데이터를 추가하거나
    multiprocessing 시나리오를 위한 프로세스 식별을 추가하는 데 특히 유용합니다.

    Attributes:
        context (Dict[str, Any]): 모든 로그 레코드에 추가할 사용자 정의 context.

    사용 예시:
        context_filter = ContextFilter({'module': 'DataCollector', 'version': '2.0'})
        logger.addFilter(context_filter)
        # 이 logger의 모든 로그는 레코드에 module과 version을 포함합니다

    See Also:
        get_logger: Logger별 context를 위한 context 파라미터를 허용합니다.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Context 필터를 초기화합니다.

        Args:
            context (Dict[str, Any], optional): 모든 로그 레코드에 추가할
                키-값 쌍의 dictionary. 기본값은 빈 dict.

        사용 예시:
            filter = ContextFilter({'env': 'production', 'region': 'us-east-1'})
        """
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """로그 레코드에 context를 추가합니다.

        이 메서드는 모든 로그 레코드에 대해 호출됩니다. 모든 context 키-값 쌍을
        레코드의 속성으로 추가하고 multiprocessing 디버깅을 위해 process_name이
        사용 가능하도록 보장합니다.

        Args:
            record (logging.LogRecord): 강화할 로그 레코드.

        Returns:
            bool: 레코드가 로깅되어야 함을 나타내기 위해 항상 True를 반환합니다.
                레코드를 필터링하려면 False를 반환합니다.

        주의:
            명시적으로 설정되지 않은 경우에도 포맷 문자열에서 안전하게 사용할 수 있도록
            process_name 속성이 자동으로 추가됩니다.
        """
        # 레코드에 모든 사용자 정의 context 추가
        for key, value in self.context.items():
            setattr(record, key, value)

        # Multiprocessing 디버깅을 위한 프로세스 정보 추가
        if not hasattr(record, 'process_name'):
            record.process_name = mp.current_process().name

        return True


# Logging system을 위한 전역 상태
_log_queue: Optional[Queue] = None
_queue_listener: Optional[logging.handlers.QueueListener] = None
_initialized: bool = False


def setup_logging(
    log_level: str = 'INFO',
    log_file: str = 'log.txt',
    log_dir: str = '.',
    console_output: bool = True,
    file_output: bool = True,
    use_colors: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> None:
    """Multiprocessing 지원과 함께 unified logging system을 설정합니다.

    이 함수는 전체 애플리케이션을 위한 중앙화된 logging system을 초기화합니다.
    메인 프로세스에서 애플리케이션 시작 시 한 번 호출되어야 합니다. 이 함수는
    multiprocessing-safe logging을 위한 handler, formatter, queue 기반 system을 설정합니다.

    구성되는 기능:
        - 크기 기반 로그 파일 rotation
        - 색상 콘솔 출력 (터미널이 지원하는 경우)
        - 디버깅을 위한 프로세스 이름 추적
        - 자동 queue 기반 multiprocessing 지원
        - 프로그램 종료 시 우아한 종료

    Args:
        log_level (str, optional): Logging 레벨 이름. 다음 중 하나여야 합니다:
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
            기본값은 'INFO'.
        log_file (str, optional): 로그 파일 이름. 기본값은 'log.txt'.
        log_dir (str, optional): 로그 파일을 위한 디렉토리. 존재하지 않으면 생성됩니다.
            기본값은 '.' (현재 디렉토리).
        console_output (bool, optional): 콘솔/stderr 출력 활성화.
            기본값은 True.
        file_output (bool, optional): rotation이 있는 파일 출력 활성화.
            기본값은 True.
        use_colors (bool, optional): 색상 콘솔 출력 활성화.
            TTY가 아닌 경우 자동으로 비활성화됩니다. 기본값은 True.
        max_bytes (int, optional): Rotation 전 최대 로그 파일 크기(바이트).
            기본값은 10MB (10 * 1024 * 1024).
        backup_count (int, optional): 유지할 백업 로그 파일 수.
            기본값은 5 (log.txt, log.txt.1, ..., log.txt.5 유지).
        format_string (str, optional): 사용자 정의 로그 포맷 문자열. None인 경우
            타임스탬프, 레벨, 프로세스, 메시지, 위치가 있는 기본 포맷 사용.

    Raises:
        ValueError: log_level이 유효한 logging 레벨 이름이 아닌 경우.
        OSError: log_dir을 생성할 수 없는 경우.

    사용 예시:
        기본 설정::

            setup_logging(log_level='INFO', log_file='app.log')

        사용자 정의 포맷팅이 있는 고급 설정::

            setup_logging(
                log_level='DEBUG',
                log_file='debug.log',
                log_dir='logs',
                console_output=True,
                file_output=True,
                use_colors=True,
                max_bytes=50 * 1024 * 1024,  # 50MB
                backup_count=10,
                format_string='[%(asctime)s] %(levelname)s: %(message)s'
            )

    주의:
        - 이 함수는 멱등성이 있습니다; 첫 초기화 후 여러 번 호출해도 효과가 없습니다.
        - 자식 프로세스는 queue를 사용하도록 logger를 구성하기 위해
          setup_logger_for_multiprocessing()을 호출해야 합니다.
        - QueueListener는 atexit를 통해 프로그램 종료 시 자동으로 중지됩니다.

    See Also:
        get_logger: 설정 후 logger 인스턴스를 가져옵니다.
        setup_logger_for_multiprocessing: 자식 프로세스에서 logging을 구성합니다.
    """
    global _log_queue, _queue_listener, _initialized

    # 재초기화 방지
    if _initialized:
        return

    # 존재하지 않으면 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 전체 로그 파일 경로
    log_file_path = log_path / log_file

    # 기본 포맷은 타임스탬프, 레벨, 프로세스, 메시지, 위치를 포함합니다
    if format_string is None:
        format_string = (
            '[%(asctime)s][%(levelname)-8s][%(processName)s] '
            '%(message)s (%(name)s:%(lineno)d)'
        )

    # 콘솔과 파일을 위한 formatter 생성
    console_formatter = ColoredFormatter(format_string, use_colors=use_colors)
    file_formatter = logging.Formatter(format_string)

    # Handler 리스트 생성
    handlers = []

    if console_output:
        # 콘솔 handler는 색상과 함께 stderr에 작성합니다
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level))
        handlers.append(console_handler)

    if file_output:
        # Windows에서는 multiprocessing 환경에서 RotatingFileHandler가 PermissionError 발생
        # 따라서 Windows에서는 일반 FileHandler 사용
        if sys.platform == 'win32':
            # Windows: 일반 FileHandler 사용 (rotation 없음)
            file_handler = logging.FileHandler(
                str(log_file_path),
                encoding='utf-8'
            )
        else:
            # Linux/Mac: RotatingFileHandler 사용 (rotation 지원)
            file_handler = logging.handlers.RotatingFileHandler(
                str(log_file_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, log_level))
        handlers.append(file_handler)

    # Multiprocessing을 위한 queue 설정 (메인 프로세스에서만)
    _log_queue = mp.Manager().Queue(-1) if mp.current_process().name == 'MainProcess' else None

    if _log_queue is not None:
        # 메인 프로세스에서 queue에 있는 로그를 처리하는 queue listener 생성
        _queue_listener = logging.handlers.QueueListener(
            _log_queue,
            *handlers,
            respect_handler_level=True
        )
        _queue_listener.start()

        # 종료 시 listener가 우아하게 중지되도록 보장
        atexit.register(_queue_listener.stop)

    # Root logger 구성
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Root logger에 context 필터 추가
    root_logger.addFilter(ContextFilter())

    # Queue가 사용 가능하면 QueueHandler 사용; 그렇지 않으면 직접 handler 사용
    if _log_queue is not None:
        # 메인 프로세스: queue handler 사용
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)
    else:
        # 대체: handler 직접 추가 (예: MainProcess가 아닌 경우)
        for handler in handlers:
            root_logger.addHandler(handler)

    _initialized = True

    # 성공적인 초기화 로깅
    logger = logging.getLogger('logger')
    logger.info(f"✅ Logging system initialized (level={log_level}, file={log_file_path})")


def get_logger(name: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """선택적 사용자 정의 context가 있는 logger 인스턴스를 가져옵니다.

    이것은 애플리케이션 전체에서 logger 인스턴스를 얻기 위한 주요 함수입니다.
    Logging system이 초기화되도록 보장하고 선택적으로 이 logger의 모든 로그에
    사용자 정의 context를 추가합니다.

    Args:
        name (str, optional): Logger 이름, 일반적으로 호출하는 모듈의 __name__.
            None인 경우 root logger를 반환합니다. __name__ 사용을 권장하며
            코드 구조와 일치하는 계층적 logger 구조를 생성합니다.
        context (Dict[str, Any], optional): 이 logger의 모든 로그에 포함할 추가 context.
            Context 항목은 LogRecord의 속성이 되며 포맷 문자열에서 사용하거나
            필터링에 사용할 수 있습니다.

    Returns:
        logging.Logger: 사용 준비가 된 구성된 logger 인스턴스.

    사용 예시:
        기본 사용법::

            logger = get_logger(__name__)
            logger.info("Processing started")
            logger.warning("Low memory detected")
            logger.error("Failed to connect", extra={'host': 'api.example.com'})

        사용자 정의 context와 함께::

            logger = get_logger(__name__, context={'component': 'DataCollector'})
            logger.info("Started")  # 레코드에 component='DataCollector'

        클래스 내에서::

            class DataProcessor:
                def __init__(self):
                    self.logger = get_logger(__name__)

                def process(self, symbol):
                    self.logger.info(f"Processing {symbol}")

    주의:
        - setup_logging()이 호출되지 않은 경우 기본 설정으로 자동 호출됩니다.
        - Logger 인스턴스는 이름으로 캐시되므로 동일한 모듈에서 get_logger(__name__)를
          여러 번 호출하면 동일한 logger 인스턴스를 반환합니다.
        - Logging 호출에서 extra 파라미터를 사용하여 로그별로 추가 context를 추가할 수 있습니다.

    See Also:
        setup_logging: Logging system을 초기화합니다.
        ContextFilter: Context가 로그에 추가되는 방법에 대한 세부 정보.
    """
    if not _initialized:
        # 아직 초기화되지 않은 경우 기본값으로 자동 초기화
        setup_logging()

    logger = logging.getLogger(name or 'root')

    # 사용자 정의 context가 제공된 경우 context 필터 추가
    if context:
        logger.addFilter(ContextFilter(context))

    return logger


def shutdown_logging() -> None:
    """Logging system을 우아하게 종료합니다.

    이 함수는 queue listener를 중지하고 모든 handler를 닫습니다. 모든 로그 메시지가
    flush되고 handler가 제대로 닫히도록 프로그램 종료 전에 호출되어야 합니다.

    주의:
        Queue listener는 setup_logging() 중에 atexit에 자동으로 등록되므로
        명시적 호출이 일반적으로 필요하지 않습니다. 그러나 다음과 같은 특정 시나리오에서는
        명시적으로 호출할 수 있습니다:
        - 사용자 정의 종료 절차
        - 다른 설정으로 logging을 재초기화하기 전
        - 테스트 정리에서

    사용 예시:
        setup_logging()
        logger = get_logger(__name__)
        logger.info("Application running")
        # ... 애플리케이션 작업 ...
        shutdown_logging()  # 모든 로그가 작성되도록 보장

    See Also:
        setup_logging: Queue listener는 설정 중에 atexit에 등록됩니다.
    """
    global _queue_listener, _initialized

    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None

    logging.shutdown()
    _initialized = False


def setup_logger_for_multiprocessing() -> None:
    """Multiprocessing 시나리오에서 자식 프로세스를 위한 logging을 구성합니다.

    이 함수는 queue handler를 사용하도록 logger를 구성하기 위해 각 자식 프로세스에서
    호출되어야 합니다. 이는 자식 프로세스의 로그가 처리를 위해 메인 프로세스로 안전하게
    전송되도록 보장하여 race condition과 손상된 출력을 방지합니다.

    함수는 root logger에서 기존 handler를 모두 제거하고 메인 프로세스의 queue listener로
    로그를 전송하는 QueueHandler를 추가합니다.

    사용 예시:
        multiprocessing.Pool과 함께 사용::

            from config.logger import setup_logging, setup_logger_for_multiprocessing, get_logger
            import multiprocessing as mp

            def process_symbol(symbol):
                # 자식 프로세스에서 logger 구성
                setup_logger_for_multiprocessing()
                logger = get_logger(__name__)

                logger.info(f"Processing {symbol}")
                # ... 처리 로직 ...
                logger.info(f"Completed {symbol}")

            if __name__ == '__main__':
                # 메인 프로세스에서 설정
                setup_logging(log_level='INFO', log_file='processing.log')

                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
                with mp.Pool(4) as pool:
                    pool.map(process_symbol, symbols)

        multiprocessing.Process와 함께 사용::

            def worker_function(queue, symbol):
                setup_logger_for_multiprocessing()
                logger = get_logger('worker')
                logger.info(f"Worker processing {symbol}")

            if __name__ == '__main__':
                setup_logging()
                p = mp.Process(target=worker_function, args=(None, 'AAPL'))
                p.start()
                p.join()

    주의:
        - 이 함수는 queue가 사용 불가능한 경우(예: setup_logging()이 메인 프로세스에서
          호출되지 않은 경우) 아무것도 하지 않습니다.
        - 한 번이 아니라 각 자식 프로세스에서 호출되어야 합니다.
        - 메인 프로세스의 setup_logging()은 자식 프로세스를 포크하기 전에
          호출되어야 합니다.

    경고:
        자식 프로세스에서 이 함수를 호출하지 않으면 다음과 같은 문제가 발생할 수 있습니다:
        - 중복 로그 항목
        - 뒤섞인/손상된 로그 출력
        - 로그 파일의 파일 손상
        - 자식 프로세스의 로그 누락

    See Also:
        setup_logging: 자식을 생성하기 전에 메인 프로세스에서 호출되어야 합니다.
        get_logger: 자식 프로세스에서 logger 인스턴스를 가져오기 위해 설정 후 사용합니다.
    """
    if _log_queue is None:
        return

    # Root logger에서 모든 기존 handler 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 메인 프로세스로 로그를 전송하기 위한 queue handler 추가
    queue_handler = logging.handlers.QueueHandler(_log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.addFilter(ContextFilter())


def get_logger_adapter(name: str) -> logging.LoggerAdapter:
    """프로세스 이름 context가 있는 logger adapter를 가져옵니다.

    이 함수는 LoggerAdapter 인스턴스를 기대하는 코드와의 하위 호환성을 제공합니다.
    Logger를 생성하고 프로세스 이름 context가 있는 LoggerAdapter로 래핑합니다.

    Args:
        name (str): Logger 이름.

    Returns:
        logging.LoggerAdapter: 프로세스 context가 있는 logger adapter.

    사용 예시:
        logger = get_logger_adapter('legacy_module')
        logger.info('This is compatible with old code')

    주의:
        새 코드의 경우 ContextFilter를 통해 동일한 기능을 가진 get_logger()를 직접
        사용하는 것이 좋습니다.

    See Also:
        get_logger: Adapter wrapper가 없는 현대적인 접근 방식.
    """
    logger = get_logger(name)
    extra = {
        'logger_name': name,
        'process_name': mp.current_process().name
    }
    return logging.LoggerAdapter(logger, extra)


# 주요 함수 내보내기
__all__ = [
    'setup_logging',
    'get_logger',
    'shutdown_logging',
    'setup_logger_for_multiprocessing',
    'get_logger_adapter',
]
