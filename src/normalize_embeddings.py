import faiss
import numpy as np
import argparse
from config.set_up_logging import set_up_logging
import logging

logger = logging.getLogger(__name__)

def main(args):

    new_filename = args.embeddings_filename.replace(".npy", "_normalized.npy")
    logger.info(f"Reading embeddings from {args.embeddings_filename} and writing them normalized to {new_filename}")

    with open(args.embeddings_filename, "rb") as read_from, open(new_filename, "wb") as write_to:
        embeddings = np.load(read_from)
        logger.info(f"Loaded embeddings with shape {embeddings.shape}")
        faiss.normalize_L2(embeddings)
        logger.info(f"Sum of first row: {np.linalg.norm(embeddings[0, :])} (expected 1)")
        logger.info(f"Sum of first column: {np.linalg.norm(embeddings[:, 0])} (not expected to be 1)")
        np.save(write_to, embeddings)

if __name__ == "__main__":
    
    set_up_logging(3)

    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_filename")
    args = parser.parse_args()

    main(args)