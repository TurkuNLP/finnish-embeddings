from config.set_up_logging import set_up_logging
from config.init_argument_parser import init_argument_parser
from config.Config import Config
import logging
from utils.helpers import yield_values_from_jsonl, get_line_count, get_query_indices, get_data_in_order
from src.run_bm25s import run_bm25s
from src.embed import BatchEmbedder
from src.index import load_index
from src.query import query
from src.evaluate import save_matrices, save_evaluation, show_textual_evaluation

logger = logging.getLogger(__name__)

def run_final_evaluation(config:Config): 

    query_indices = get_query_indices(config.read_query_indices_from)
    query_generator = yield_values_from_jsonl(config.news_data_path, config.query_key, indices=query_indices) # final query indices are expected to be in order

    index = load_index(config.save_index_to)
    logger.info(f"Index loaded from {config.save_index_to} with shape ({index.ntotal}, {index.d})")
    
    if "bm25" in config.model_name:

        num_documents = get_line_count(config.news_data_path)
        data_generator = yield_values_from_jsonl(config.news_data_path, config.passage_key)
        
        result_matrix, _ = run_bm25s(
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

        query_embeddings = batch_embedder.encode_queries(
            documents=query_generator,
            num_documents=len(query_indices),
            return_embeddings=True
            )

        max_top_k = max(config.top_k)
        similarities, result_matrix = query(index, query_embeddings, max_top_k)
    
    save_matrices(
        results_dir=config.save_matrices_to,
        model_name=config.model_name,
        similarities_matrix = similarities,
        result_matrix=result_matrix
        )
    
    save_evaluation(
        result_matrix=result_matrix,
        top_k_list=config.top_k,
        query_indices=query_indices,
        save_to=config.save_results_to
        )
    
    # Show the queries and retrieved articles for the k first queries
    eval_queries = list(get_data_in_order(config.news_data_path, config.query_key, query_indices[:config.first_k]))
    subset_for_textual_evaluation = result_matrix[:config.first_k, :config.first_k]
    show_textual_evaluation(config.news_data_path,
                            eval_queries,
                            subset_for_textual_evaluation)


if __name__ == "__main__":
    set_up_logging(3)
    parser = init_argument_parser()
    args = parser.parse_args()
    config = Config.parse_config()

    run_final_evaluation(config)