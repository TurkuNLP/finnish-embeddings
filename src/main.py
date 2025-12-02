from config.set_up_logging import set_up_logging
from config.init_argument_parser import init_argument_parser
from config.Config import Config
import logging
from src.run_pipeline import run_pipeline
from src.run_final_evaluation import run_final_evaluation

logger = logging.getLogger(__name__)

def log_args(config):
    logger.info(config)

def main(config):

    log_args(config)

    if config.final:
        run_final_evaluation(config)
    else:
        run_pipeline(config)

if __name__ == "__main__":

    parser = init_argument_parser()
    args = parser.parse_args()
    config = Config.parse_config(args)
    set_up_logging(config.verbosity_level)

    main(config)