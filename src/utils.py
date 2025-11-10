import logging
import sys
import os

def setup_logger(name, log_file=None, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger