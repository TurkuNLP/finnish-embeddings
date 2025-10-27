import logging
from config.set_up_logging import set_up_logging
import argparse
import faiss
import numpy as np

logger = logging.getLogger(__name__)

def save_index(read_embeddings_from:str, save_to:str, batch_size:int=1000, return_index:bool=True):

    # Load embeddings
    with open(read_embeddings_from, "rb") as file:
        embeddings = np.load(file)
        logger.debug(f"Going to build an index of shape {embeddings.shape}")
    
    # Build the index (IndexFlatIP, inner product)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Add embedding vectors to the index one batch at a time
    for batch_num, i in enumerate(range(0, embeddings.shape[0], batch_size), 1):
        batch = embeddings[i : i + batch_size]
        faiss.normalize_L2(batch) # Important to normalize the vectors
        index.add(batch)
        logger.debug(f"Added batch {batch_num}, index size is now {index.ntotal}")
    logger.info(f"{index.ntotal} vectors in index")

    faiss.write_index(index, save_to)
    logger.info(f"Index saved to {save_to}")

    if return_index:
        return index

def load_index(read_index_from:str):
    return faiss.read_index(read_index_from)

def main(args):

    save_index(args.read_from, args.save_to, return_index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("read_from",
                        help="Path to the .npy file.")
    parser.add_argument("save_to",
                        help="Path to the directory where the resulting index should be saved.")
    args = parser.parse_args()

    set_up_logging(3)
    main(args)