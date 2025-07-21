"""
Comprehensive logging configuration for Trading RL Agent.

Provides dictConfig-based logging setup with verbose level support:
- -v → DEBUG in project modules
- -vv → DEBUG + include external libs
- -vvv → very verbose (trace loops)

Integrates with Typer callback for automatic setup.
"""

import logging
import logging.config
from pathlib import Path
from typing import Any

import structlog


def get_logging_config(verbose_level: int = 0, log_dir: Path | None = None) -> dict[str, Any]:
    """
    Generate logging configuration based on verbose level.

    Args:
        verbose_level: Number of -v flags (0-3)
        log_dir: Directory for log files

    Returns:
        dictConfig configuration dictionary
    """
    if log_dir is None:
        log_dir = Path("logs")

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine log levels based on verbose count
    if verbose_level >= 3:
        # -vvv: Very verbose - trace everything including loops
        root_level = "DEBUG"
        project_level = "DEBUG"
        external_level = "DEBUG"
        trace_loops = True
    elif verbose_level == 2:
        # -vv: Debug + external libs
        root_level = "DEBUG"
        project_level = "DEBUG"
        external_level = "DEBUG"
        trace_loops = False
    elif verbose_level == 1:
        # -v: Debug in project modules only
        root_level = "INFO"
        project_level = "DEBUG"
        external_level = "WARNING"
        trace_loops = False
    else:
        # Default: INFO level
        root_level = "INFO"
        project_level = "INFO"
        external_level = "WARNING"
        trace_loops = False

    # Base formatters
    formatters = {
        "standard": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "trace": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(funcName)s() [%(threadName)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
        },
    }

    # Handlers
    handlers = {
        "console": {
            "level": "DEBUG" if verbose_level > 0 else "INFO",
            "formatter": ("trace" if trace_loops else "detailed" if verbose_level > 0 else "standard"),
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_dir / "trading_system.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "error_file": {
            "level": "ERROR",
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_dir / "trading_errors.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    }

    # Add debug file handler for verbose levels
    if verbose_level > 0:
        handlers["debug_file"] = {
            "level": "DEBUG",
            "formatter": "trace" if trace_loops else "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_dir / "trading_debug.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3,
        }

    # Logger configurations
    loggers = {
        "": {  # root logger
            "handlers": ["console", "file", "error_file"],
            "level": root_level,
            "propagate": False,
        },
        # Project modules - always detailed logging
        "trading_rl_agent": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "src.trading_rl_agent": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        # Core components
        "trading_rl_agent.core": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.data": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.models": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.agents": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.training": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.envs": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.execution": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.risk": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.portfolio": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.monitoring": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.features": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
        "trading_rl_agent.utils": {
            "handlers": ["console", "file", "error_file"],
            "level": project_level,
            "propagate": False,
        },
    }

    # Add debug file handler to project loggers if verbose
    if verbose_level > 0:
        for logger_name, logger_config in loggers.items():
            if logger_name and logger_name != "":
                if "handlers" not in logger_config or not isinstance(logger_config["handlers"], list):
                    logger_config["handlers"] = []
                if isinstance(logger_config["handlers"], list):
                    logger_config["handlers"].append("debug_file")

    # External library configurations
    external_libs = {
        # ML/DL frameworks
        "torch": {"level": external_level, "propagate": False},
        "torch.nn": {"level": external_level, "propagate": False},
        "torch.optim": {"level": external_level, "propagate": False},
        "tensorflow": {"level": external_level, "propagate": False},
        "keras": {"level": external_level, "propagate": False},
        # RL frameworks
        "gymnasium": {"level": external_level, "propagate": False},
        "stable_baselines3": {"level": external_level, "propagate": False},
        "ray": {"level": external_level, "propagate": False},
        "ray.rllib": {"level": external_level, "propagate": False},
        "ray.tune": {"level": external_level, "propagate": False},
        # Data processing
        "pandas": {"level": external_level, "propagate": False},
        "numpy": {"level": external_level, "propagate": False},
        "sklearn": {"level": external_level, "propagate": False},
        "scipy": {"level": external_level, "propagate": False},
        # Data sources
        "yfinance": {"level": external_level, "propagate": False},
        "alpaca": {"level": external_level, "propagate": False},
        "ccxt": {"level": external_level, "propagate": False},
        "requests": {"level": external_level, "propagate": False},
        "urllib3": {"level": external_level, "propagate": False},
        # Monitoring and tracking
        "mlflow": {"level": external_level, "propagate": False},
        "wandb": {"level": external_level, "propagate": False},
        "tensorboard": {"level": external_level, "propagate": False},
        # Utilities
        "tqdm": {"level": external_level, "propagate": False},
        "rich": {"level": external_level, "propagate": False},
        "structlog": {"level": external_level, "propagate": False},
    }

    # Add external libs to loggers if verbose level >= 2
    if verbose_level >= 2:
        loggers.update(external_libs)

    # Very verbose: add trace logging for loops and iterations
    if trace_loops:
        # Add specific loggers for loop tracing
        loop_loggers = {
            "trading_rl_agent.loops": {"level": "DEBUG", "propagate": False},
            "trading_rl_agent.iterations": {"level": "DEBUG", "propagate": False},
            "trading_rl_agent.training.loops": {"level": "DEBUG", "propagate": False},
            "trading_rl_agent.data.loops": {"level": "DEBUG", "propagate": False},
        }
        loggers.update(loop_loggers)

    # Complete configuration
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": loggers,
    }


def setup_logging_from_verbose(verbose_level: int = 0, log_dir: Path | None = None, structured: bool = True) -> None:
    """
    Setup logging based on verbose level from Typer.

    Args:
        verbose_level: Number of -v flags (0-3)
        log_dir: Directory for log files
        structured: Whether to use structured logging
    """
    # Get configuration
    config = get_logging_config(verbose_level, log_dir)

    # Apply configuration
    logging.config.dictConfig(config)

    # Setup structured logging if requested
    if structured:
        setup_structured_logging(verbose_level)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with verbose level {verbose_level}")
    logger.debug(f"Log directory: {log_dir}")
    logger.debug(f"Structured logging: {structured}")


def setup_structured_logging(verbose_level: int = 0) -> None:
    """
    Setup structured logging with appropriate processors.

    Args:
        verbose_level: Verbose level for processor configuration
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add more detailed processors for higher verbose levels
    if verbose_level >= 2:
        processors.extend(
            [
                structlog.stdlib.add_log_level_number,
                structlog.processors.CallsiteParameterAdder(
                    parameters={
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    },
                ),
            ],
        )

    if verbose_level >= 3:
        # Add thread and process info for very verbose logging
        processors.extend(
            [
                structlog.stdlib.add_log_level_number,
                structlog.stdlib.add_log_level_number,
            ],
        )

    # Add formatter wrapper
    processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)

    # Configure structlog
    structlog.configure(
        processors=processors,
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


def log_verbose_info(verbose_level: int) -> None:
    """
    Log verbose information about the current configuration.

    Args:
        verbose_level: Current verbose level
    """
    logger = get_logger(__name__)

    logger.info("=== Logging Configuration ===")
    logger.info(f"Verbose Level: {verbose_level}")

    if verbose_level == 0:
        logger.info("Log Level: INFO (default)")
        logger.info("Scope: Project modules only")
    elif verbose_level == 1:
        logger.info("Log Level: DEBUG")
        logger.info("Scope: Project modules")
        logger.info("Format: Detailed with line numbers")
    elif verbose_level == 2:
        logger.info("Log Level: DEBUG")
        logger.info("Scope: Project modules + external libraries")
        logger.info("Format: Detailed with line numbers")
        logger.info("External libs: torch, ray, pandas, etc.")
    elif verbose_level >= 3:
        logger.info("Log Level: DEBUG")
        logger.info("Scope: Everything (very verbose)")
        logger.info("Format: Trace with thread info")
        logger.info("Features: Loop tracing, thread tracking")

    logger.info("=== End Logging Configuration ===")


# Convenience function for Typer integration
def setup_logging_for_typer(verbose: int, log_dir: Path | None = None) -> None:
    """
    Convenience function for Typer callback integration.

    Args:
        verbose: Verbose count from Typer
        log_dir: Optional log directory
    """
    setup_logging_from_verbose(verbose, log_dir)
    log_verbose_info(verbose)


if __name__ == "__main__":
    # Demo the logging configuration

    # Test different verbose levels
    for level in range(4):
        print(f"\n=== Testing Verbose Level {level} ===")
        setup_logging_for_typer(level)

        # Test different loggers
        logger = get_logger("trading_rl_agent.test")
        logger.info("This is an info message")
        logger.debug("This is a debug message")
        logger.warning("This is a warning message")

        # Test external logger
        if level >= 2:
            ext_logger = get_logger("torch")
            ext_logger.info("External library logging enabled")

        print("=== End Test ===\n")
