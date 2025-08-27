import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Literal

load_dotenv()

# Path to the news data
NEWS_DATA = os.getenv("NEWS_DATA")

# Path to the directory where embeddings should be saved to
EMBEDDING_DIR = os.getenv("EMBEDDING_DIR")

# Path to the directory where Faiss Indices should be saved to
INDEX_DIR = os.getenv("INDEX_DIR")

# Path to the directory where evaluation should be saved to
EVAL_DIR = os.getenv("EVAL_DIR")

# Path to the file that stores the indices that should be used as queries (indexing starting at 0)
QUERY_INDICES = os.getenv("QUERY_INDICES")

@dataclass
class Config:

    def replace_slashes(self):
        return self.model_name.replace("/", "__")

    # Model name
    model_name: str

    # TODO: Add model-appropriate default values

    # Filenames
    news_data_path: str = NEWS_DATA
    save_embeddings_to: str = f"{EMBEDDING_DIR}/{replace_slashes(model_name)}_embeddings.npy"
    save_index_to: str = f"{INDEX_DIR}/{replace_slashes(model_name)}_index.faiss"
    read_query_indices_from = QUERY_INDICES

    # Data extraction
    passage_key: str = "text_end"
    query_key: str = "title"

    # Processing
    batch_size: 32

    # Language for stemming (bm25)
    language: str = "finnish"

    # How many articles to retrieve in evaluation
    top_k: List[int] = field(default_factory=lambda: [1, 5])

    # Logging
    verbosity_level: Literal[0, 1, 2, 3] = 3

