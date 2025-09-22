from dotenv import load_dotenv
import os
import json
from embed import BatchEmbedder
import faiss
from query import query
from evaluate import save_evaluation
import logging
import random

def set_up_logger(verbosity_level:int):
    verbosity_levels = {0: logging.CRITICAL, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=verbosity_levels[verbosity_level], force=True)
    return logging.getLogger(__name__)

def get_detailed_instruct(*, task_description: str, query: str, use_fin: bool = False) -> str:
    if task_description == None:
        return query
    if use_fin:
        return f"Ohje: {task_description}\nUutisotsikko: {query}"
    return f"Instruct: {task_description}\nQuery: {query}"

def yield_titles_with_instructions(titles: list[str], task_description: str = None, use_fin: bool = False):
    for title in titles:
        yield get_detailed_instruct(task_description=task_description, query=title, use_fin=use_fin)

def yield_dev_titles(filename, query_key="title"):
    with open(filename) as file:
        for line in file:
            obj = json.loads(line)
            if obj["split"] == "dev":
                yield obj[query_key]

def yield_dev_indices(filename):
    with open(filename) as file:
        for i, line in enumerate(file):
            obj = json.loads(line)
            if obj["split"] == "dev":
                yield i

class QwenConfig:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-Embedding-8B"
        self.max_tokens_per_batch = 65000
        self.batch_size = -1 # stands for undefined
        self.default_prompt = "Retrieval the relevant passage for the given query" # from https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json
        self.custom_prompts = task_descriptions

class E5Config:
    def __init__(self):
        self.model_name = "intfloat/multilingual-e5-large-instruct"
        self.max_tokens_per_batch = 500000
        self.batch_size = -1 # stands for undefined
        self.default_prompt = "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper" # closest to the task at hand, from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
        self.custom_prompts = task_descriptions


if __name__ == "__main__":

    logger = set_up_logger(3) # debug level

    load_dotenv()

    DATA_PATH = os.getenv("NEWS_DATA_DEVTEST")
    INDEX_DIR = os.getenv("INDEX_DIR")
    PROMPT_EVAL_DIR = os.getenv("PROMPT_EVAL_DIR")

    task_descriptions = {
        "eng": "Given a news title, retrieve the article corresponding to it",
        "fin": "Hae uutisotsikkoa vastaava artikkeli"
    }

    titles = list(yield_dev_titles(DATA_PATH))
    title_indices = list(yield_dev_indices(DATA_PATH))
    model_configs = (E5Config(), QwenConfig())
    top_k_list = [1, 5]

    for config in model_configs:

        logger.info(f"Starting the run with {config.model_name}")

        read_index_from = os.path.join(INDEX_DIR, f"{config.model_name.replace("/", "__")}_index.faiss")
        index = faiss.read_index(read_index_from)
        logger.info(f"Index loaded from {read_index_from} with shape ({index.ntotal}, {index.d})")

        batch_embedder = BatchEmbedder(config.model_name, config.batch_size, config.max_tokens_per_batch)

        # Encode titles with different setups

        task_configs = {
            "no_instruction": {"task_description": None, "use_fin": False},
            "default_eng": {"task_description": config.default_prompt, "use_fin": False},
            "custom_eng": {"task_description": config.custom_prompts["eng"], "use_fin": False},
            "custom_fin": {"task_description": config.custom_prompts["fin"], "use_fin": True}
        }
        
        for task_config_name, config_values in task_configs.items():

            logger.info(f"Embedding the titles with configuration {task_config_name}")
            logger.info(f"Example query with the specified instructions:")
            logger.info(f"\n{get_detailed_instruct(task_description=config_values["task_description"], query=titles[random.randint(0,4999)], use_fin=config_values["use_fin"])}")

            embedded_titles = batch_embedder.encode(
                documents = yield_titles_with_instructions(titles, task_description=config_values["task_description"], use_fin=config_values["use_fin"]),
                num_documents = len(titles),
                return_embeddings = True
                )
            logger.info(f"Shape of the embedded titles: {embedded_titles.shape}")

            _, result_matrix = query(index, embedded_titles, max(top_k_list))

            logger.info(f"Getting the results for {task_config_name}")
            save_to = os.path.join(PROMPT_EVAL_DIR, f"{config.model_name.replace("/", "__")}_{task_config_name}_results.json")
            save_evaluation(result_matrix, top_k_list, title_indices, save_to)
            logger.info(f"Results saved to {save_to}")
        
        del batch_embedder # clear memory before using another model





    
