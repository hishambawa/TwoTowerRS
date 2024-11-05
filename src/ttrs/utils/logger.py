import structlog
import logging

LOG_LEVELS = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warn": logging.WARN,
            "error": logging.ERROR,
            "fatal": logging.FATAL
        }

class BasicLogger:
    def __init__(self, log_level):
        # set the log level.
        # if the passed level is not found, default value is set to INFO
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVELS.get(log_level, logging.INFO))
        )

        self.logger = structlog.get_logger()

    def log_debug(self, message, **kwargs):
        self.logger.debug(message, **kwargs)

    def log_info(self, message, **kwargs):
        self.logger.info(message, **kwargs)

    def log_warn(self, message, **kwargs):
        self.logger.warn(message, **kwargs)

    def log_error(self, message, error, **kwargs):
        self.logger.exception(message, exc_info=error, **kwargs)

    def log_fatal(self, message, error, **kwargs):
        self.logger.fatal(message, exc_info=error, **kwargs)
        exit(1)