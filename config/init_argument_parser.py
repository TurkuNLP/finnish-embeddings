import argparse

def init_argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        help="Which model to use. If using SentenceTransformer, refer to \
                            https://sbert.net/docs/sentence_transformer/pretrained_models.html for an overview of available models."
                            )
    parser.add_argument("--news_data_path",
                        help="Path to data stored as a JSONL file."
                        )
    parser.add_argument("--read_query_indices_from",
                        help="Which rows of the JSONL file specified with parameter 'data_files' to get. (Indexing starting at 0.)\
                            Assumes the indices to be stored in a text file, one index per row."
                        )
    parser.add_argument("--save_embeddings_to",
                        help="Path to the directory where the resulting embedding array should be saved."
                        )
    parser.add_argument("--save_index_to",
                        help="Path to the directory where the resulting index should be saved."
                        )
    parser.add_argument("--save_results_to",
                        help="Path to the directory where the evaluation results should be saved."
                        )
    parser.add_argument("--passage_key",
                        type=str,
                        help="The key to the field that should be extracted from each row of JSONL file defined with parameter 'data_files'."
                        )
    parser.add_argument("--query_key",
                        type=str,
                        help="The key to the field that should be extracted and used as a query from specified rows of JSONL file defined with parameter 'data_files'.\
                            If argument 'read_query_indices_from' is specified, only uses the data from those rows."
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        help="How many documents should be encoded simultaneously."
                        )
    parser.add_argument("--top_k",
                        type=int,
                        help="How many documents to retrieve."
                        ) 
    parser.add_argument("--verbosity_level",
                        choices=[0, 1, 2, 3],
                        type=int,
                        help="Logging levels given as an int (the higher the number the more detailed the output): 0 (only critical); 1 (warning); 2 (info); 3 (debug)"
                        )
    parser.add_argument("--test",
                        action="store_true",
                        help="If specified, loads test related variables from configuration."
                        )
    return parser
