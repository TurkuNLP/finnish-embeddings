from dotenv import load_dotenv
import os
from config.ModelConfig import QwenConfig, E5Config
from config.task_prompts import TASK_PROMPTS, CUSTOM_PROMPTS
from embed import BatchEmbedder
from utils.helpers import yield_values_by_split, yield_indices_by_split, yield_titles_with_instructions, get_detailed_instruct, get_results_paths, save_to_jsonl
from config.set_up_logging import set_up_logging
import faiss
from query import query
from evaluate import save_evaluation
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)

def main():

    load_dotenv()

    DATA_PATH = os.getenv("NEWS_DATA_DEVTEST")
    INDEX_DIR = os.getenv("INDEX_DIR")
    PROMPT_EVAL_DIR = os.getenv("PROMPT_EVAL_DIR")
    RESULTS_DIR = os.getenv("DEV_TOP_100_ALL_PROMPTS_DIR")
    ALL_PROMPT_RESULTS_JSONL = os.getenv("ALL_PROMPT_RESULTS_JSONL")

    titles = list(yield_values_by_split(DATA_PATH, value_key="title", split="dev"))
    title_indices = list(yield_indices_by_split(DATA_PATH, split="dev"))
    model_configs = (E5Config(), QwenConfig())
    top_k_list = [1, 5]

    for config in model_configs:

        logger.info(f"Starting the run with {config.model_name}")

        read_index_from = os.path.join(INDEX_DIR, f"{config.model_name.replace("/", "__")}_indexIP.faiss") # make sure to use IndexFlatIP
        index = faiss.read_index(read_index_from)
        logger.info(f"Index loaded from {read_index_from} with shape ({index.ntotal}, {index.d})")

        batch_embedder = BatchEmbedder(config.model_name, config.batch_size, config.max_tokens_per_batch)

        task_configs = TASK_PROMPTS | CUSTOM_PROMPTS # combine prompt dictionaries
        logger.info(f"Going to run evaluation for {len(task_configs)} unique task descriptions.")
        
        for task_name, task_description in task_configs.items():

            logger.info(f"Embedding the titles with configuration {task_name}")
            logger.info(f"Example query with the specified instructions:")
            logger.info(f"\n{get_detailed_instruct(task_description=task_description, query=titles[random.randint(0,4999)])}")

            embedded_titles = batch_embedder.encode_queries(
                documents = yield_titles_with_instructions(titles, task_description=task_description),
                num_documents = len(titles),
                return_embeddings = True,
                prompt=task_description
                )
            logger.info(f"Shape of the embedded titles: {embedded_titles.shape}")

            similarities, result_matrix = query(index, embedded_titles, max(top_k_list))

            # Save the similarity and retrieved article index matrices
            similarities_path, indices_path = get_results_paths(RESULTS_DIR, config.model_name, task=task_name) 
            with open(similarities_path, "bw") as d, open(indices_path, "bw") as i:
                np.save(d, similarities)
                np.save(i, result_matrix)
            logger.info(f"Similarity matrix saved to {similarities_path}")
            logger.info(f"Result indices matrix saved to {indices_path}")

            save_to = os.path.join(PROMPT_EVAL_DIR, f"{config.model_name.replace("/", "__")}_{task_name}_resultsIP.json")
            # Save the evaluation in a separate file
            evaluation = save_evaluation(result_matrix, top_k_list, title_indices, save_to)
            # Save the evaluation by appending to a JSONL file with other results
            save_to_jsonl(ALL_PROMPT_RESULTS_JSONL, {task_description: evaluation})
            logger.info(f"Results for {config.model_name} with prompt {task_name} saved to {save_to} and {ALL_PROMPT_RESULTS_JSONL}")
        
        del batch_embedder # clear memory before using another model

if __name__ == "__main__":

    set_up_logging(2) # info level
    main()