import logging
import sys

def init_logger(module, log_level=logging.NOTSET, filename=None, force_stdout=False):
    """Initializes the framework logger with a specific filter level, and optional file output."""
    if getattr(module, "LOGGER_INITIALIZED", None) is None:
        logging.getLogger().setLevel(logging.NOTSET)
        module.logger.propagate = 0
        logger_format = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s : %(message)s")
        if filename is not None:
            logger_fh = logging.FileHandler(filename)
            logger_fh.setLevel(logging.NOTSET)
            logger_fh.setFormatter(logger_format)
            module.logger.addHandler(logger_fh)
        stream = sys.stdout if force_stdout else None
        logger_ch = logging.StreamHandler(stream=stream)
        logger_ch.setLevel(log_level)
        logger_ch.setFormatter(logger_format)
        module.logger.addHandler(logger_ch)
        setattr(module, "LOGGER_INITIALIZED", True)