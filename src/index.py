import argparse
import faiss
import numpy as np
from os import path


def main(args):

    embeddings = np.load(args.read_from)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)   # build the index
    index.add(embeddings)            # add embedding vectors to the index
    print(f"{index.ntotal} vectors in index")

    index_file = path.join(args.save_to, "index.faiss")
    faiss.write_index(index, index_file)
    print(f"Index saved to {index_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("read_from",
                        help="Path to the .npy file.")
    parser.add_argument("save_to",
                        help="Path to the directory where the resulting index should be saved.")
    args = parser.parse_args()

    main(args)