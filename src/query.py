import argparse
import faiss
from sentence_transformers import SentenceTransformer


def main(args):

    # Log used arguments
    print(args)

    index = faiss.read_index(args.index_path)
    model = SentenceTransformer(args.model)
    query = model.encode(query)

    # Get the search results, D correponding to distances, I corresponding to the ordinal ids of the retrieved documents
    D, I = index.search(query, args.k_nearest)

    print(I)
    print(D)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Which Sentence Transformer model to use.")
    parser.add_argument("index_path",
                        help="Path to the stored index on disk.")
    parser.add_argument("query",
                        help="Query (a single string) to be encoded.")
    parser.add_argument("--k_nearest",
                        type=int,
                        default=3,
                        help="How many documents to retrieve.")
    args = parser.parse_args()

    main(args)