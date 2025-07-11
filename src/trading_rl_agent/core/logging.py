"""
Centralized logging configuration for the trading system.
"""

import logging
import logging.config
from pathlib import Path
from typing import Any, cast

import structlog

# Default logging configuration
DEFAULT_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/trading_system.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "error_file": {
            "level": "ERROR",
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/trading_errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default", "file", "error_file"],
            "level": "INFO",
            "propagate": False,
        },
        "trading_rl_agent": {
            "handlers": ["default", "file", "error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "ray": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        },
        "torch": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}


def setup_logging(
    config: dict[str, Any] | None = None,
    log_level: str = "INFO",
    log_dir: Path | None = None,
    structured: bool = True,
) -> None:
    """
    Set up comprehensive logging for the trading system.

    Args:
        config: Custom logging configuration dict
        log_level: Default log level
        log_dir: Directory for log files
        structured: Whether to use structured logging
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Use provided config or default
    if config is None:
        config = DEFAULT_LOG_CONFIG.copy()

    if "handlers" in config:
        # Update file paths with log_dir
        for handler_config in cast(dict[str, Any], config["handlers"]).values():
            if "filename" in handler_config:
                filename = handler_config["filename"]
                handler_config["filename"] = str(log_dir / Path(filename).name)

    # Configure standard logging
    logging.config.dictConfig(config)

    # Configure structured logging if requested
    if structured:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def get_structured_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)
