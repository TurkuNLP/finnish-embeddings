import argparse
import logging
from config.Config import Config
from .run_pipeline import run_pipeline

def set_up_logger(verbosity_level:int):
    verbosity_levels = {0: logging.CRITICAL, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=verbosity_levels[verbosity_level], force=True)
    return logging.getLogger(__name__)

def log_args(args):
    logger.info(args)

def main(args):

    log_args(args)
    run_pipeline(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        help="Which model to use. If using SentenceTransformer, refer to \
                            https://sbert.net/docs/sentence_transformer/pretrained_models.html for an overview of available models.")
    args = parser.parse_args()

    config = Config(args.model)
    logger = set_up_logger(config.verbosity)
    main(args)