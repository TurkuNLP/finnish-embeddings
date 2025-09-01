from config.init_argument_parser import init_argument_parser
from config.Config import Config
import logging
from utils.helpers import yield_values_from_jsonl, yield_values_from_text_file, get_line_count
from .embed import embed
from .index import save_index
from .query import query, show_textual_evaluation


def get_query_indices(filename):
    return set(int(value) for value in yield_values_from_text_file(filename))

def run_pipeline(config):

    num_documents = get_line_count(config.news_data_path)
    data_generator = yield_values_from_jsonl(config.news_data_path, config.passage_key)

    embeddings = embed(model_name=config.model_name,
                       documents=data_generator,
                       num_documents=num_documents,
                       batch_size=config.batch_size,
                       save_to=config.save_embeddings_to,
                       return_embeddings=True)
    
    index = save_index(embeddings, config.save_index_to)

    query_indices = get_query_indices(config.read_query_indices_from)
    query_generator = yield_values_from_jsonl(config.news_data_path, config.query_key, indices=query_indices)

    query_embeddings = embed(model_name=config.model_name,
                             documents=query_generator,
                             num_documents=len(query_indices),
                             batch_size=config.batch_size,
                             return_embeddings=True)

    max_top_k = max(config.top_k)
    D, I = query(index, query_embeddings, max_top_k)

    if config.test:
        show_textual_evaluation(config.news_data_path,
                                yield_values_from_jsonl(config.news_data_path, config.query_key, indices=query_indices),
                                I)


if __name__ == "__main__":
    parser = init_argument_parser()
    args = parser.parse_args()
    config = Config.parse_config()
    logger = logging.getLogger(__name__)

    run_pipeline(config)