"""This module serves as a wrapper for the logging module."""

import logging

import colorama


class CustomFormatter(logging.Formatter):
    """Custom formatter class that adds color to the logger."""

    message = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    COLORED_FORMAT = {
        logging.DEBUG: message,
        logging.INFO: message,
        logging.WARNING: colorama.Fore.YELLOW + message + colorama.Style.RESET_ALL,
        logging.ERROR: colorama.Fore.LIGHTRED_EX + message + colorama.Style.RESET_ALL,
        logging.CRITICAL: colorama.Fore.RED + message + colorama.Style.RESET_ALL,
    }

    def format(self, record: logging.LogRecord) -> str:
        record_level = record.levelno
        log_fmt = self.COLORED_FORMAT[record_level]
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Returns a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    return logger


class LoggerMixin:
    """Mixin class that provides a logger."""

    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)
