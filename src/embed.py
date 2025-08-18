import argparse
import logging
from utils.get_data import yield_from_jsonl
from sentence_transformers import SentenceTransformer
import numpy as np
from os import path


def set_up_logger(verbosity_level:int):
    verbosity_levels = {0: logging.CRITICAL, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=verbosity_levels[verbosity_level], force=True)
    return logging.getLogger(__name__)

def log_args(args):
    logger.info(args)

def main(args):

    log_args(args)

    model_name = args.model
    data_files = args.data_files
    key = args.key
    k_documents = args.k
    dest_folder = args.save_to

    embedding_file = path.join(dest_folder, "mock_embeddings.npy")

    documents = list(yield_from_jsonl(data_files, key, k_documents))

    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents)
    logger.info(f"Embeddings array has the following shape: {embeddings.shape}")
    np.save(embedding_file, embeddings)
    logger.info(f"Embeddings saved into {embedding_file}")

    # Test loading
    loaded_embeddings = np.load(embedding_file)
    logger.info(f"Loaded embeddings array with the following shape: {loaded_embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Which Sentence Transformer model to use. Refer to https://sbert.net/docs/sentence_transformer/pretrained_models.html for an overview of available models.")
    parser.add_argument("data_files",
                        help="Path to the JSONL file.")
    parser.add_argument("save_to",
                        help="Path to the directory where the resulting embedding array should be saved.")
    parser.add_argument("--key",
                        default="text_end",
                        help="The key to the field that should be extracted from each row of JSONL file defined with parameter 'data_files'.")
    parser.add_argument("--k",
                        help="Pick first k documents from the JSONL file defined with parameter 'data_files'.")
    parser.add_argument("--verbosity",
                        default=3,
                        choices=[0, 1, 2, 3],
                        type=int,
                        help="Logging levels given as an int (the higher the number the more detailed the output): 0 (only critical); 1 (warning); 2 (info); 3 (debug)"
                        )
    args = parser.parse_args()

    logger = set_up_logger(args.verbosity)

    main(args)