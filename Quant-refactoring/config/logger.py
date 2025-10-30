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


# ANSI color codes for console output
class LogColors:
    """ANSI color codes for terminal output formatting.

    This class defines ANSI escape sequences for colorizing terminal output.
    Colors are used to visually distinguish log levels and components for
    improved readability during development and debugging.

    Attributes:
        RESET (str): Reset all formatting to default.
        BOLD (str): Enable bold text.
        DEBUG (str): Cyan color for DEBUG level messages.
        INFO (str): Green color for INFO level messages.
        WARNING (str): Yellow color for WARNING level messages.
        ERROR (str): Red color for ERROR level messages.
        CRITICAL (str): Magenta color for CRITICAL level messages.
        TIME (str): Dark gray color for timestamps.
        LEVEL (str): Bold formatting for log level names.
        NAME (str): Blue color for logger names.

    Note:
        These codes work on ANSI-compatible terminals (Linux, macOS, Windows 10+).
        The ColoredFormatter checks terminal compatibility before applying colors.

    Example:
        >>> print(f"{LogColors.ERROR}Error message{LogColors.RESET}")
        >>> print(f"{LogColors.BOLD}Bold text{LogColors.RESET}")
    """

    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Log levels - each level has a distinct color
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta

    # Components - used for different parts of log messages
    TIME = '\033[90m'       # Dark gray
    LEVEL = '\033[1m'       # Bold
    NAME = '\033[94m'       # Blue


class ColoredFormatter(logging.Formatter):
    """Custom log formatter that adds ANSI color codes to console output.

    This formatter extends the standard logging.Formatter to add color codes
    based on log level. It automatically detects if the output is a TTY and
    disables colors if not (e.g., when piping to a file).

    The formatter temporarily modifies the LogRecord attributes to add colors,
    formats the message, then restores the original values to prevent colors
    from affecting other handlers.

    Attributes:
        LEVEL_COLORS (Dict[int, str]): Mapping from log level to color code.
        use_colors (bool): Whether to apply colors (False if not a TTY).

    Example:
        >>> formatter = ColoredFormatter(
        ...     '[%(asctime)s][%(levelname)s] %(message)s',
        ...     use_colors=True
        ... )
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(formatter)

    See Also:
        LogColors: For available color definitions.
    """

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True) -> None:
        """Initialize the colored formatter.

        Args:
            fmt (str, optional): Format string for log messages. If None, uses
                the default format. Supports all standard logging format strings.
            use_colors (bool, optional): Whether to enable colored output.
                Defaults to True, but will be disabled if output is not a TTY.

        Example:
            >>> formatter = ColoredFormatter(
            ...     fmt='[%(levelname)s] %(message)s',
            ...     use_colors=True
            ... )
        """
        super().__init__(fmt)
        # Only use colors if stderr is a terminal (TTY)
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with color codes if enabled.

        This method adds ANSI color codes to the log level and logger name,
        formats the complete message, then restores the original values.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: Formatted log message with color codes if enabled, otherwise
                plain formatted message.

        Note:
            Original record attributes are preserved to avoid affecting other
            handlers that share the same LogRecord instance.
        """
        if self.use_colors:
            # Save original values to restore later
            orig_levelname = record.levelname
            orig_name = record.name

            # Apply colors based on log level
            level_color = self.LEVEL_COLORS.get(record.levelno, '')
            record.levelname = f"{level_color}{record.levelname}{LogColors.RESET}"
            record.name = f"{LogColors.NAME}{record.name}{LogColors.RESET}"

            # Format the message with colors
            result = super().format(record)

            # Restore original values for other handlers
            record.levelname = orig_levelname
            record.name = orig_name

            return result
        else:
            return super().format(record)


class ContextFilter(logging.Filter):
    """Logging filter that adds custom context to all log records.

    This filter enriches log records with additional context information such as
    process name and custom key-value pairs. It's particularly useful for adding
    metadata to all logs from a specific component or adding process identification
    for multiprocessing scenarios.

    Attributes:
        context (Dict[str, Any]): Custom context to add to every log record.

    Example:
        >>> context_filter = ContextFilter({'module': 'DataCollector', 'version': '2.0'})
        >>> logger.addFilter(context_filter)
        >>> # All logs from this logger will include module and version in the record

    See Also:
        get_logger: Accepts context parameter for per-logger context.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the context filter.

        Args:
            context (Dict[str, Any], optional): Dictionary of key-value pairs to
                add to every log record. Defaults to empty dict.

        Example:
            >>> filter = ContextFilter({'env': 'production', 'region': 'us-east-1'})
        """
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to a log record.

        This method is called for every log record. It adds all context key-value
        pairs as attributes on the record and ensures process_name is available
        for multiprocessing debugging.

        Args:
            record (logging.LogRecord): The log record to enhance.

        Returns:
            bool: Always returns True to indicate the record should be logged.
                Return False to filter out the record.

        Note:
            The process_name attribute is automatically added if not present,
            making it safe to use in format strings even if not explicitly set.
        """
        # Add all custom context to the record
        for key, value in self.context.items():
            setattr(record, key, value)

        # Add process info for multiprocessing debugging
        if not hasattr(record, 'process_name'):
            record.process_name = mp.current_process().name

        return True


# Global state for the logging system
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
    """Setup the unified logging system with multiprocessing support.

    This function initializes the centralized logging system for the entire
    application. It should be called once at application startup in the main
    process. The function sets up handlers, formatters, and the queue-based
    system for multiprocessing-safe logging.

    Features configured:
        - Log file rotation based on size
        - Colored console output (if terminal supports it)
        - Process name tracking for debugging
        - Automatic queue-based multiprocessing support
        - Graceful shutdown on program exit

    Args:
        log_level (str, optional): Logging level name. Must be one of:
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
            Defaults to 'INFO'.
        log_file (str, optional): Name of the log file. Defaults to 'log.txt'.
        log_dir (str, optional): Directory for log files. Will be created if
            it doesn't exist. Defaults to '.' (current directory).
        console_output (bool, optional): Enable console/stderr output.
            Defaults to True.
        file_output (bool, optional): Enable file output with rotation.
            Defaults to True.
        use_colors (bool, optional): Enable colored console output.
            Automatically disabled for non-TTY. Defaults to True.
        max_bytes (int, optional): Maximum log file size in bytes before rotation.
            Defaults to 10MB (10 * 1024 * 1024).
        backup_count (int, optional): Number of backup log files to keep.
            Defaults to 5 (keeps log.txt, log.txt.1, ..., log.txt.5).
        format_string (str, optional): Custom log format string. If None, uses
            a default format with timestamp, level, process, message, and location.

    Raises:
        ValueError: If log_level is not a valid logging level name.
        OSError: If log_dir cannot be created.

    Example:
        Basic setup::

            setup_logging(log_level='INFO', log_file='app.log')

        Advanced setup with custom formatting::

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

    Note:
        - This function is idempotent; calling it multiple times has no effect
          after the first initialization.
        - Child processes should call setup_logger_for_multiprocessing() to
          configure their loggers to use the queue.
        - The QueueListener is automatically stopped on program exit via atexit.

    See Also:
        get_logger: Get a logger instance after setup.
        setup_logger_for_multiprocessing: Configure logging in child processes.
    """
    global _log_queue, _queue_listener, _initialized

    # Prevent re-initialization
    if _initialized:
        return

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Full log file path
    log_file_path = log_path / log_file

    # Default format includes timestamp, level, process, message, and location
    if format_string is None:
        format_string = (
            '[%(asctime)s][%(levelname)-8s][%(processName)s] '
            '%(message)s (%(name)s:%(lineno)d)'
        )

    # Create formatters for console and file
    console_formatter = ColoredFormatter(format_string, use_colors=use_colors)
    file_formatter = logging.Formatter(format_string)

    # Create handlers list
    handlers = []

    if console_output:
        # Console handler writes to stderr with colors
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level))
        handlers.append(console_handler)

    if file_output:
        # File handler with automatic rotation
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, log_level))
        handlers.append(file_handler)

    # Setup queue for multiprocessing (only in main process)
    _log_queue = mp.Manager().Queue(-1) if mp.current_process().name == 'MainProcess' else None

    if _log_queue is not None:
        # Create queue listener that processes queued logs in main process
        _queue_listener = logging.handlers.QueueListener(
            _log_queue,
            *handlers,
            respect_handler_level=True
        )
        _queue_listener.start()

        # Ensure listener stops gracefully on exit
        atexit.register(_queue_listener.stop)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Add context filter to root logger
    root_logger.addFilter(ContextFilter())

    # If queue is available, use QueueHandler; otherwise use direct handlers
    if _log_queue is not None:
        # Main process: use queue handler
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)
    else:
        # Fallback: add handlers directly (e.g., if not in MainProcess)
        for handler in handlers:
            root_logger.addHandler(handler)

    _initialized = True

    # Log successful initialization
    logger = logging.getLogger('logger')
    logger.info(f"✅ Logging system initialized (level={log_level}, file={log_file_path})")


def get_logger(name: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Get a logger instance with optional custom context.

    This is the primary function for obtaining logger instances throughout the
    application. It ensures the logging system is initialized and optionally
    adds custom context to all logs from this logger.

    Args:
        name (str, optional): Logger name, typically __name__ of the calling module.
            If None, returns the root logger. Using __name__ is recommended as it
            creates a hierarchical logger structure matching your code structure.
        context (Dict[str, Any], optional): Additional context to include in all
            logs from this logger. Context items become attributes on LogRecords
            and can be used in format strings or for filtering.

    Returns:
        logging.Logger: Configured logger instance ready for use.

    Example:
        Basic usage::

            logger = get_logger(__name__)
            logger.info("Processing started")
            logger.warning("Low memory detected")
            logger.error("Failed to connect", extra={'host': 'api.example.com'})

        With custom context::

            logger = get_logger(__name__, context={'component': 'DataCollector'})
            logger.info("Started")  # component='DataCollector' in record

        In a class::

            class DataProcessor:
                def __init__(self):
                    self.logger = get_logger(__name__)

                def process(self, symbol):
                    self.logger.info(f"Processing {symbol}")

    Note:
        - If setup_logging() hasn't been called, it will be called automatically
          with default settings.
        - Logger instances are cached by name, so calling get_logger(__name__)
          multiple times in the same module returns the same logger instance.
        - Extra context can be added per-log using the extra parameter in
          logging calls.

    See Also:
        setup_logging: Initialize the logging system.
        ContextFilter: For details on how context is added to logs.
    """
    if not _initialized:
        # Auto-initialize with defaults if not already done
        setup_logging()

    logger = logging.getLogger(name or 'root')

    # Add context filter if custom context provided
    if context:
        logger.addFilter(ContextFilter(context))

    return logger


def shutdown_logging() -> None:
    """Shutdown the logging system gracefully.

    This function stops the queue listener and closes all handlers. It should
    be called before program exit to ensure all log messages are flushed and
    handlers are properly closed.

    Note:
        The queue listener is automatically registered with atexit during
        setup_logging(), so explicit calling is usually not necessary. However,
        you may want to call this explicitly in certain scenarios like:
        - Custom shutdown procedures
        - Before reinitializing logging with different settings
        - In test teardown

    Example:
        >>> setup_logging()
        >>> logger = get_logger(__name__)
        >>> logger.info("Application running")
        >>> # ... application work ...
        >>> shutdown_logging()  # Ensure all logs are written

    See Also:
        setup_logging: The queue listener is registered with atexit during setup.
    """
    global _queue_listener, _initialized

    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None

    logging.shutdown()
    _initialized = False


def setup_logger_for_multiprocessing() -> None:
    """Configure logging for child processes in multiprocessing scenarios.

    This function should be called in each child process to configure its logger
    to use the queue handler. This ensures logs from child processes are safely
    sent to the main process for handling, preventing race conditions and
    garbled output.

    The function removes any existing handlers from the root logger and adds a
    QueueHandler that sends logs to the main process's queue listener.

    Example:
        Using with multiprocessing.Pool::

            from config.logger import setup_logging, setup_logger_for_multiprocessing, get_logger
            import multiprocessing as mp

            def process_symbol(symbol):
                # Configure logger in child process
                setup_logger_for_multiprocessing()
                logger = get_logger(__name__)

                logger.info(f"Processing {symbol}")
                # ... processing logic ...
                logger.info(f"Completed {symbol}")

            if __name__ == '__main__':
                # Setup in main process
                setup_logging(log_level='INFO', log_file='processing.log')

                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
                with mp.Pool(4) as pool:
                    pool.map(process_symbol, symbols)

        Using with multiprocessing.Process::

            def worker_function(queue, symbol):
                setup_logger_for_multiprocessing()
                logger = get_logger('worker')
                logger.info(f"Worker processing {symbol}")

            if __name__ == '__main__':
                setup_logging()
                p = mp.Process(target=worker_function, args=(None, 'AAPL'))
                p.start()
                p.join()

    Note:
        - This function does nothing if the queue is not available (e.g., if
          setup_logging() was not called in the main process).
        - Must be called in each child process, not just once.
        - The main process's setup_logging() must be called before forking
          child processes.

    Warning:
        Failing to call this function in child processes can lead to:
        - Duplicate log entries
        - Interleaved/garbled log output
        - File corruption in log files
        - Missing logs from child processes

    See Also:
        setup_logging: Must be called in main process before spawning children.
        get_logger: Use after setup to get logger instances in child processes.
    """
    if _log_queue is None:
        return

    # Remove all existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add queue handler to send logs to main process
    queue_handler = logging.handlers.QueueHandler(_log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.addFilter(ContextFilter())


def get_logger_adapter(name: str) -> logging.LoggerAdapter:
    """Get a logger adapter with process name context.

    This function provides backward compatibility with code that expects
    LoggerAdapter instances. It creates a logger and wraps it in a LoggerAdapter
    with process name context.

    Args:
        name (str): Logger name.

    Returns:
        logging.LoggerAdapter: Logger adapter with process context.

    Example:
        >>> logger = get_logger_adapter('legacy_module')
        >>> logger.info('This is compatible with old code')

    Note:
        For new code, prefer using get_logger() directly as it has the same
        functionality through the ContextFilter.

    See Also:
        get_logger: Modern approach without adapter wrapper.
    """
    logger = get_logger(name)
    extra = {
        'logger_name': name,
        'process_name': mp.current_process().name
    }
    return logging.LoggerAdapter(logger, extra)


# Export main functions
__all__ = [
    'setup_logging',
    'get_logger',
    'shutdown_logging',
    'setup_logger_for_multiprocessing',
    'get_logger_adapter',
]
