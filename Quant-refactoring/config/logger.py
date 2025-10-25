"""
Unified Logging System with Multiprocessing Support

Features:
- Thread-safe and multiprocessing-safe using QueueHandler
- Colored console output for better readability
- Automatic log file rotation
- Structured logging with context
- Easy migration from existing logging code

Usage:
    from config.logger import get_logger, setup_logging

    # Setup once at application start
    setup_logging(log_level='INFO', log_file='app.log')

    # Get logger anywhere
    logger = get_logger(__name__)
    logger.info("This is a log message")
    logger.error("Error with context", extra={'symbol': 'AAPL', 'price': 150})
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
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Log levels
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta

    # Components
    TIME = '\033[90m'       # Dark gray
    LEVEL = '\033[1m'       # Bold
    NAME = '\033[94m'       # Blue


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def __init__(self, fmt: str = None, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            # Save original values
            orig_levelname = record.levelname
            orig_name = record.name

            # Colorize
            level_color = self.LEVEL_COLORS.get(record.levelno, '')
            record.levelname = f"{level_color}{record.levelname}{LogColors.RESET}"
            record.name = f"{LogColors.NAME}{record.name}{LogColors.RESET}"

            # Format
            result = super().format(record)

            # Restore
            record.levelname = orig_levelname
            record.name = orig_name

            return result
        else:
            return super().format(record)


class ContextFilter(logging.Filter):
    """Add custom context to log records"""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        # Add context to record
        for key, value in self.context.items():
            setattr(record, key, value)

        # Add process info for multiprocessing
        if not hasattr(record, 'process_name'):
            record.process_name = mp.current_process().name

        return True


# Global queue for multiprocessing
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
    """
    Setup unified logging system

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name
        log_dir: Log file directory
        console_output: Enable console output
        file_output: Enable file output
        use_colors: Use colored output in console
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        format_string: Custom format string
    """
    global _log_queue, _queue_listener, _initialized

    if _initialized:
        return

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Full log file path
    log_file_path = log_path / log_file

    # Default format
    if format_string is None:
        format_string = (
            '[%(asctime)s][%(levelname)-8s][%(process_name)s] '
            '%(message)s (%(name)s:%(lineno)d)'
        )

    # Create formatters
    console_formatter = ColoredFormatter(format_string, use_colors=use_colors)
    file_formatter = logging.Formatter(format_string)

    # Create handlers
    handlers = []

    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level))
        handlers.append(console_handler)

    if file_output:
        # Use RotatingFileHandler for automatic rotation
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, log_level))
        handlers.append(file_handler)

    # Setup queue for multiprocessing
    _log_queue = mp.Manager().Queue(-1) if mp.current_process().name == 'MainProcess' else None

    if _log_queue is not None:
        # Create queue listener (runs in main process)
        _queue_listener = logging.handlers.QueueListener(
            _log_queue,
            *handlers,
            respect_handler_level=True
        )
        _queue_listener.start()

        # Ensure listener stops on exit
        atexit.register(_queue_listener.stop)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Add context filter
    root_logger.addFilter(ContextFilter())

    # If queue is available, use QueueHandler; otherwise use direct handlers
    if _log_queue is not None:
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)
    else:
        for handler in handlers:
            root_logger.addHandler(handler)

    _initialized = True

    # Log initialization
    logger = logging.getLogger('logger')
    logger.info(f"✅ Logging system initialized (level={log_level}, file={log_file_path})")


def get_logger(name: str = None, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger instance

    Args:
        name: Logger name (typically __name__)
        context: Additional context to include in logs

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("Failed to process", extra={'symbol': 'AAPL'})
    """
    if not _initialized:
        # Auto-initialize with defaults
        setup_logging()

    logger = logging.getLogger(name or 'root')

    # Add context filter if provided
    if context:
        logger.addFilter(ContextFilter(context))

    return logger


def shutdown_logging():
    """Shutdown logging system gracefully"""
    global _queue_listener, _initialized

    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None

    logging.shutdown()
    _initialized = False


# Convenience functions for backward compatibility
def setup_logger_for_multiprocessing():
    """
    Setup logger for child processes
    Should be called in each child process
    """
    if _log_queue is None:
        return

    # Remove all handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add queue handler
    queue_handler = logging.handlers.QueueHandler(_log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.addFilter(ContextFilter())


# Alias for backward compatibility
def get_logger_adapter(name: str) -> logging.LoggerAdapter:
    """
    Get logger adapter with process name context
    For backward compatibility with existing code
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
