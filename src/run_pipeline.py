import argparse
import logging
from utils.helpers import yield_from_jsonl, yield_values_from_text_file, get_line_count
from .embed import embed
from .index import save_index
from .query import query


def get_query_indices(filename):
    return set(int(value) for value in yield_values_from_text_file(filename))

def run_pipeline(args):

    num_documents = get_line_count(args.data_files)
    data_generator = yield_from_jsonl(args.data_files, args.passage_key)

    embeddings = embed(model_name = args.model_name,
                       documents=data_generator,
                       num_documents=num_documents,
                       batch_size = args.batch_size,
                       return_embeddings=True)
    
    index = save_index(embeddings, args.save_index_to)

    query_indices = get_query_indices(args.query_indices)
    query_generator = yield_from_jsonl(args.data_files, args.query_key, indices=query_indices)

    query_embeddings = embed(model_name=args.model_name,
                             documents=query_generator,
                             num_documents=len(query_indices),
                             return_embeddings=True)

    D, I = query(index, query_embeddings, args.k_nearest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_files",
                        help="Path to data stored as a JSONL file.")
    parser.add_argument("model_name",
                        help="Which model to use. If using SentenceTransformer, refer to \
                            https://sbert.net/docs/sentence_transformer/pretrained_models.html for an overview of available models.")
    parser.add_argument("save_embeddings_to",
                        help="Path to the directory where the resulting embedding array should be saved.")
    parser.add_argument("save_index_to",
                        help="Path to the directory where the resulting index should be saved.")
    parser.add_argument("--passage_key",
                        default="text_end",
                        help="The key to the field that should be extracted from each row of JSONL file defined with parameter 'data_files'.")
    parser.add_argument("--query_key",
                        default="title",
                        help="The key to the field that should be extracted and used as a query from specified rows of JSONL file defined with parameter 'data_files'.\
                            If argument 'indices' is specified, only uses the data from those rows.")
    parser.add_argument("--k_first",
                        type=int,
                        help="Pick first k documents from the JSONL file defined with parameter 'data_files'.")
    parser.add_argument("--query_indices",
                        help="Which rows of the JSONL file specified with parameter 'data_files' to get. (Indexing starting at 0.)\
                            Assumes the indices to be stored in a text file, one index per row."
                            )
    parser.add_argument("--k_nearest",
                        type=int,
                        default=3,
                        help="How many documents to retrieve.") 
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="How many documents should be encoded simultaneously."
                        )
    parser.add_argument("--verbosity",
                        default=3,
                        choices=[0, 1, 2, 3],
                        type=int,
                        help="Logging levels given as an int (the higher the number the more detailed the output): 0 (only critical); 1 (warning); 2 (info); 3 (debug)"
                        )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    run_pipeline(args)