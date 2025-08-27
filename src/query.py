import argparse
from utils.helpers import yield_from_jsonl, yield_values_from_text_file, get_data_as_dict
import faiss
from sentence_transformers import SentenceTransformer

def parse_query_args(query:str, dict_key:str, indices:list|set=None):

    # If the given query argument is a path to or a name of a JSONL file
    if query.endswith(".jsonl"):
        return list(yield_from_jsonl(query, dict_key, indices=indices))
    return [query]

def show_textual_evaluation(filename, queries, result_indices):

    retrieved_documents = get_data_as_dict(filename, "text_end", set(result_indices.reshape(-1)))

    for query, top_k in zip(queries, result_indices):
        print(f"Query: {query}")
        print(f"Top-{len(top_k)} results:")
        for i, article_index in enumerate(top_k, 1):
            print(f"Result {i}:")
            print(retrieved_documents[article_index])
            print("\n")
        print("\n")

def query(index, query_embeddings, k_nearest):
    return index.search(query_embeddings, k_nearest)

def main(args):

    # Log used arguments
    print(args)

    index = faiss.read_index(args.index_path)
    model = SentenceTransformer(args.model)
    queries = parse_query_args(args.query, args.dict_key, args.indices)
    query_embeddings = model.encode(queries)

    # Get the search results, D correponding to distances, I corresponding to the ordinal ids of the retrieved documents
    D, I = index.search(query_embeddings, args.k_nearest)

    # Show the indices of the top-k retrieved documents
    print(I)

    # Show the squared Euclidian (L2) distances (if the index is a IndexFlatL2) of the the top-k retrieved documents 
    print(D)

    if args.show:
        show_textual_evaluation(args.query, queries, I)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Which Sentence Transformer model to use.")
    parser.add_argument("index_path",
                        help="Path to the stored index on disk.")
    parser.add_argument("query",
                        help="Query (a single string) to be encoded, or a filename to a JSONL file where strings can be extracted from.")
    parser.add_argument("--dict_key",
                        default="title",
                        help="The key to the field that should be extracted if a filename ending with .jsonl is given as the 'query' argument.")
    parser.add_argument("--indices",
                        type=int,
                        nargs='+',
                        help="Which rows of the JSONL file specified with parameter 'query' to get. (Indexing starting at 0.)")
    parser.add_argument("--k_nearest",
                        type=int,
                        default=3,
                        help="How many documents to retrieve.")
    parser.add_argument("--show",
                        action="store_true",
                        help="If the retrieval results should be shown as texts. Should only be used with a small number of queries and a low 'k-nearest' value.")
    args = parser.parse_args()

    main(args)