from dotenv import load_dotenv
import os
import json
from embed import BatchEmbedder
from utils.helpers import yield_values_by_split, yield_indices_by_split, yield_titles_with_instructions, get_detailed_instruct
from config.set_up_logging import set_up_logging
import faiss
from query import query
from evaluate import save_evaluation
import logging
import random

logger = logging.getLogger(__name__)

def get_task_configs():
    pass

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

    set_up_logging(3) # debug level

    load_dotenv()

    DATA_PATH = os.getenv("NEWS_DATA_DEVTEST")
    INDEX_DIR = os.getenv("INDEX_DIR")
    PROMPT_EVAL_DIR = os.getenv("PROMPT_EVAL_DIR")

    task_descriptions = {
        "eng_web_search": "Given a web search query, retrieve relevant passages that answer the query",
        "eng_sts": "Retrieve semantically similar text",
        "eng_1": "Given a news title, retrieve the article corresponding to it",
        "eng_2": "Given a news title, retrieve the article that is the correct pair for the given title",
        "eng_3": "Given a news title, retrieve the article that best corresponds to the given title",
        "eng_4": "Retrieve the relevant article for the given news title",
        "fin_1": "Hae uutisotsikkoa vastaava artikkeli",
        "fin_2": "Hae oikea artikkeli, joka kuuluu seuraavalle uutisotsikolle",
        "fin_3": "Löydä seuraavalle uutisotsikolle kuuluva artikkeli"
    }

    titles = list(yield_values_by_split(DATA_PATH, value_key="title", split="dev"))
    title_indices = list(yield_indices_by_split(DATA_PATH, split="dev"))
    model_configs = (E5Config(), QwenConfig())
    top_k_list = [1, 5]

    for config in model_configs:

        logger.info(f"Starting the run with {config.model_name}")

        read_index_from = os.path.join(INDEX_DIR, f"{config.model_name.replace("/", "__")}_indexIP.faiss")
        index = faiss.read_index(read_index_from)
        logger.info(f"Index loaded from {read_index_from} with shape ({index.ntotal}, {index.d})")

        batch_embedder = BatchEmbedder(config.model_name, config.batch_size, config.max_tokens_per_batch)

        # Encode titles with different setups

        task_configs = {
            "no_instruction": {"task_description": None, "use_fin": False},
            "default_eng": {"task_description": config.default_prompt, "use_fin": False},
            "eng_web_search": {"task_description": config.custom_prompts["eng_web_search"], "use_fin": False},
            "eng_sts": {"task_description": config.custom_prompts["eng_sts"], "use_fin": False},
            "custom_eng_1": {"task_description": config.custom_prompts["eng_1"], "use_fin": False},
            "custom_eng_2": {"task_description": config.custom_prompts["eng_2"], "use_fin": False},
            "custom_eng_3": {"task_description": config.custom_prompts["eng_3"], "use_fin": False},
            "custom_eng_4": {"task_description": config.custom_prompts["eng_4"], "use_fin": False},
            "fin_prefix": {"task_description": config.custom_prompts["fin_1"], "use_fin": True},
            "custom_fin_1": {"task_description": config.custom_prompts["fin_1"], "use_fin": False},
            "custom_fin_2": {"task_description": config.custom_prompts["fin_2"], "use_fin": False},
            "custom_fin_3": {"task_description": config.custom_prompts["fin_3"], "use_fin": False}
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

            similarities, result_matrix = query(index, embedded_titles, max(top_k_list))

            save_to = os.path.join(PROMPT_EVAL_DIR, f"{config.model_name.replace("/", "__")}_{task_config_name}_resultsIP.json")
            save_evaluation(result_matrix, top_k_list, title_indices, save_to)
            logger.info(f"Results for {config.model_name} with prompt {task_config_name} saved to {save_to}")
        
        del batch_embedder # clear memory before using another model





    
