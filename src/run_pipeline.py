from config.init_argument_parser import init_argument_parser
from config.Config import Config
import logging
from utils.helpers import yield_values_from_jsonl, get_line_count, get_query_indices, get_data_in_order
from .run_bm25s import run_bm25s
from .embed import BatchEmbedder
from .index import save_index
from .query import query
from .evaluate import save_evaluation, show_textual_evaluation


def run_pipeline(config:Config):

    # Initialize generators for queries and passages
    num_documents = get_line_count(config.news_data_path)
    data_generator = yield_values_from_jsonl(config.news_data_path, config.passage_key)

    query_indices = get_query_indices(config.read_query_indices_from)
    query_generator = get_data_in_order(config.news_data_path, config.query_key, query_indices)
    
    if "bm25" in config.model_name:
        result_matrix, bm25scores = run_bm25s(
            passages=data_generator,
            corpus_len=num_documents,
            queries=query_generator,
            queries_len=len(query_indices),
            save_index_to=config.save_index_to,
            top_k_list=config.top_k,
            language=config.language
            )

    else:
        batch_embedder = BatchEmbedder(
            model_name=config.model_name,
            batch_size=config.batch_size,
            max_tokens_per_batch=config.max_tokens_per_batch,
            test=config.test
            )

        batch_embedder.encode(
            documents=data_generator,
            num_documents=num_documents,
            save_to=config.save_embeddings_to,
            )
        
        index = save_index(
            read_embeddings_from=config.save_embeddings_to,
            save_to=config.save_index_to)

        query_embeddings = batch_embedder.encode_queries(
            documents=query_generator,
            num_documents=len(query_indices),
            return_embeddings=True
            )

        max_top_k = max(config.top_k)
        _, result_matrix = query(index, query_embeddings, max_top_k)
    
    save_evaluation(
        result_matrix=result_matrix,
        top_k_list=config.top_k,
        query_indices=query_indices,
        save_to=config.save_results_to
        )
    
    # Show the queries and retrieved articles for the k first queries
    eval_queries = list(get_data_in_order(config.news_data_path, config.query_key, query_indices[:config.first_k]))
    subset_for_textual_evaluation = result_matrix[:config.first_k]
    show_textual_evaluation(config.news_data_path,
                            eval_queries,
                            subset_for_textual_evaluation)


if __name__ == "__main__":
    parser = init_argument_parser()
    args = parser.parse_args()
    config = Config.parse_config()
    logger = logging.getLogger(__name__)

    run_pipeline(config)