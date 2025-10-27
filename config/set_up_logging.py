import logging

def set_up_logging(verbosity_level:int):
    verbosity_levels = {0: logging.CRITICAL, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=verbosity_levels[verbosity_level], force=True)