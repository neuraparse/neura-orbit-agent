"""Logging configuration and utilities for Neura-Orbit-Agent."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_enabled: bool = True,
    file_enabled: bool = True,
    colorize: bool = True,
    max_file_size: str = "10 MB",
    backup_count: int = 5,
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_enabled: Enable console logging
        file_enabled: Enable file logging
        colorize: Enable colored console output
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Remove default logger
    logger.remove()
    
    # Console logging
    if console_enabled:
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        if not colorize:
            console_format = (
                "{time:HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} - "
                "{message}"
            )
        
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=colorize,
        )
    
    # File logging
    if file_enabled and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
        
        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation=max_file_size,
            retention=backup_count,
            compression="zip",
            enqueue=True,  # Thread-safe logging
        )


def get_logger(name: str) -> "logger":
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Component-specific loggers
def get_screen_logger():
    """Get logger for screen capture components."""
    return get_logger("neura_orbit.screen")


def get_llm_logger():
    """Get logger for LLM components."""
    return get_logger("neura_orbit.llm")


def get_automation_logger():
    """Get logger for automation components."""
    return get_logger("neura_orbit.automation")


def get_security_logger():
    """Get logger for security components."""
    return get_logger("neura_orbit.security")


def get_api_logger():
    """Get logger for API components."""
    return get_logger("neura_orbit.api")


def get_cli_logger():
    """Get logger for CLI components."""
    return get_logger("neura_orbit.cli")
