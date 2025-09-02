import logging
import argparse
import faiss
import numpy as np

logger = logging.getLogger(__name__)

def save_index(read_embeddings_from:str, save_to:str, batch_size:int=1000, return_index:bool=True):

    # Load as a memory-mapped file
    embeddings = np.load(read_embeddings_from, mmap_mode="r")
    logger.debug(f"Going to build an index of shape {embeddings.shape}")
    
    # Build the index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    # Add embedding vectors to the index one batch at a time
    for i in range(0, dim, batch_size):
        batch = embeddings[i : i + batch_size]
        index.add(batch)
    logger.info(f"{index.ntotal} vectors in index")

    faiss.write_index(index, save_to)
    logger.info(f"Index saved to {save_to}")

    if return_index:
        return index

def main(args):

    save_index(args.read_from, args.save_to, return_index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("read_from",
                        help="Path to the .npy file.")
    parser.add_argument("save_to",
                        help="Path to the directory where the resulting index should be saved.")
    args = parser.parse_args()

    main(args)