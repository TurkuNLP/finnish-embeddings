import os
import logging
from dotenv import load_dotenv
from utils.helpers import yield_values_from_jsonl, get_line_count, yield_indices_by_split, yield_values_by_split
from embed import BatchEmbedder
from index import load_index
from query import query
from run_bm25s import run_bm25s
import numpy as np

def set_up_logger(verbosity_level:int):
    verbosity_levels = {0: logging.CRITICAL, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=verbosity_levels[verbosity_level], force=True)
    return logging.getLogger(__name__)

class QwenConfig:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-Embedding-8B"
        self.max_tokens_per_batch = 65000
        self.batch_size = -1 # stands for undefined

class E5Config:
    def __init__(self):
        self.model_name = "intfloat/multilingual-e5-large-instruct"
        self.max_tokens_per_batch = 500000
        self.batch_size = -1 # stands for undefined

class BertConfig:
    def __init__(self):
        self.model_name = "TurkuNLP/bert-base-finnish-cased-v1"
        self.max_tokens_per_batch = -1 # stands for undefined
        self.batch_size = 1024

class XlmConfig:
    def __init__(self):
        self.model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
        self.max_tokens_per_batch = -1 # stands for undefined
        self.batch_size = 512

class BM25Config:
    def __init__(self):
        self.model_name = "bm25s"

def get_index_path(index_dir, model_name):
    return os.path.join(index_dir, f"{model_name.replace("/", "__")}_indexIP.faiss") # get FlatIP index, not FlatL2

def get_results_paths(results_dir, model_name):
    distances_path = os.path.join(results_dir, f"{model_name.replace("/", "__")}_distancesIP.npy")
    indices_path = os.path.join(results_dir, f"{model_name.replace("/", "__")}_indicesIP.npy")
    return distances_path, indices_path

def main():

    logger = set_up_logger(3) # debug level

    load_dotenv()
    DATA_PATH = os.getenv("NEWS_DATA_DEVTEST")
    INDEX_DIR = os.getenv("INDEX_DIR")
    RESULTS_DIR = os.getenv("DEV_TOP_100_DIR")

    k_nearest = 100

    query_indices = list(yield_indices_by_split(DATA_PATH))

    configs = (
        QwenConfig(),
        E5Config(),
        BertConfig(),
        XlmConfig(),
        BM25Config(),
        )

    for config in configs:

        logger.info(f"Starting the run with {config.model_name}")

        if config.model_name == "bm25s":
            result_indices, distances = run_bm25s(
                passages=yield_values_from_jsonl(DATA_PATH, key="text_end"),
                corpus_len=get_line_count(DATA_PATH),
                queries=yield_values_by_split(DATA_PATH, value_key="title", split="dev"),
                queries_len=len(query_indices),
                save_index_to="",
                top_k_list=[k_nearest],
                language="finnish"
            )

        else:
            read_index_from = get_index_path(INDEX_DIR, config.model_name)
            index = load_index(read_index_from)
            logger.info(f"Index loaded from {read_index_from} with shape ({index.ntotal}, {index.d})")

            batch_embedder = BatchEmbedder(
                model_name=config.model_name,
                batch_size=config.batch_size,
                max_tokens_per_batch=config.max_tokens_per_batch)
            
            encoded_queries = batch_embedder.encode_queries(
                documents=yield_values_by_split(DATA_PATH),
                num_documents=len(query_indices),
                return_embeddings=True)
            
            distances, result_indices = query(index, encoded_queries, k_nearest)

            del batch_embedder # clear memory before using another model

        distances_path, indices_path = get_results_paths(RESULTS_DIR, config.model_name) 
        with open(distances_path, "bw") as d, open(indices_path, "bw") as i:
            np.save(d, distances)
            np.save(i, result_indices)
        logger.info(f"Distance matrix saved to {distances_path}")
        logger.info(f"Result indices matrix saved to {indices_path}")


if __name__ == "__main__":
    main()
    
