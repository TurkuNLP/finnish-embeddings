from config.init_argument_parser import init_argument_parser
from config.Config import Config
import logging
from .run_pipeline import run_pipeline

def set_up_logger(verbosity_level:int):
    verbosity_levels = {0: logging.CRITICAL, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=verbosity_levels[verbosity_level], force=True)
    return logging.getLogger(__name__)

def log_args(config):
    logger.info(config)

def main(config):

    log_args(config)
    run_pipeline(config)

if __name__ == "__main__":

    parser = init_argument_parser()
    args = parser.parse_args()
    config = Config.parse_config(args)
    logger = set_up_logger(config.verbosity)

    main(config)