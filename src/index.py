import logging
import argparse
import faiss
import numpy as np
from os import path

logger = logging.getLogger(__name__)

def save_index(embeddings:np.array, save_to:str, return_index:bool=True):

    logger.debug(f"Going to build an index of shape {embeddings.shape}")
    
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)   # build the index
    index.add(embeddings)            # add embedding vectors to the index
    logger.info(f"{index.ntotal} vectors in index")

    faiss.write_index(index, save_to)
    logger.info(f"Index saved to {save_to}")

    if return_index:
        return index

def main(args):

    embeddings = np.load(args.read_from)
    save_index(embeddings, args.save_to, return_index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("read_from",
                        help="Path to the .npy file.")
    parser.add_argument("save_to",
                        help="Path to the directory where the resulting index should be saved.")
    args = parser.parse_args()

    main(args)