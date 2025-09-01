import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Literal

load_dotenv()

@dataclass
class Config:

    # Model name
    model_name: str

    # TODO: Add model-appropriate default values

    # Filenames
    news_data_path: str = os.getenv("NEWS_DATA")
    read_query_indices_from = os.getenv("QUERY_INDICES")

    # Can only be initialized after the model name is set
    save_embeddings_to: str = None
    save_index_to: str = None
    save_results_to: str = None

    # Data extraction
    passage_key: str = "text_end"
    query_key: str = "title"

    # Processing
    batch_size: int = 32

    # Language for stemming (bm25)
    language: str = "finnish"

    # How many articles to retrieve in evaluation
    top_k: List[int] = field(default_factory=lambda: [1, 5])

    # Logging
    verbosity_level: Literal[0, 1, 2, 3] = 3

    def replace_slashes_in_model_name(self):
        return self.model_name.replace("/", "__")

    # Initialize filenames for saving embeddings array and index
    def __post_init__(self):
        self.save_embeddings_to: str = f"{os.getenv("EMBEDDING_DIR")}/{self.replace_slashes_in_model_name()}_embeddings.npy"
        self.save_index_to: str = f"{os.getenv("INDEX_DIR")}/{self.replace_slashes_in_model_name()}_index.faiss"
        self.save_results_to: str = f"{os.getenv("EVAL_DIR")}/{self.replace_slashes_in_model_name()}_results.json" # TODO: Modify once the format is decided

    @classmethod
    def parse_config(cls, args):
        # Convert args namespace to dict, then filter out None values
        filtered_args = {k: v for k, v in vars(args).items() if v is not None}

        # Create instance with provided values, falling back to defaults
        return cls(**filtered_args)