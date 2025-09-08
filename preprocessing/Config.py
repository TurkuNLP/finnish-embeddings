import os
from dotenv import load_dotenv
from datetime import date
from dataclasses import dataclass, field
from typing import List

load_dotenv()

# Where to find the unfiltered corpus and where to save the filtered one
NEWS_DATA_ROOT = os.getenv("NEWS_DATA_ROOT")

# Name of the unfiltered corpus
UNFILTERED_CORPUS = os.getenv("UNFILTERED_CORPUS")

# Name of the corpus with deduplicated URLs
UNIQUE_URL_CORPUS = os.getenv("UNIQUE_URL_CORPUS")

#Path to the file that contains titles that are not present in the unfiltered corpus, but can be found in texts
EXTRA_TITLES = os.getenv("EXTRA_TITLES")

# Path to the file that contains the prefixes that should be filtered out
JUNK_PREFIXES = os.getenv("JUNK_PREFIXES")

# Path to the file that contains the junk lines that should be filtered out
JUNK_LINES = os.getenv("JUNK_LINES")

# Filename for the resulting filtered file
DATE = date.today()
FILTERED_CORPUS = f"yle-news-filtered-{DATE}.jsonl"

# Filename for writing the removed lines into
REMOVED_DATA_ROOT = os.getenv("REMOVED_DATA_ROOT")
REMOVED_LINES = f"yle-news-removed-lines-{DATE}.jsonl"

# Filename for the final file with the bm25s scores
FINAL_SCORES = f"yle-news-final-scores-{DATE}.jsonl"

@dataclass
class Config:

    # Filenames
    unfiltered_corpus_filename: str = os.path.join(NEWS_DATA_ROOT, UNFILTERED_CORPUS)
    unique_url_corpus_filename: str = os.path.join(NEWS_DATA_ROOT, UNIQUE_URL_CORPUS)
    filtered_corpus_filename: str = os.path.join(NEWS_DATA_ROOT, FILTERED_CORPUS)
    extra_titles_filename: str = EXTRA_TITLES
    junk_prefix_filename: str = JUNK_PREFIXES
    junk_lines_filename: str = JUNK_LINES
    removed_lines_filename: str = os.path.join(REMOVED_DATA_ROOT, REMOVED_LINES)
    final_scores_filename: str = os.path.join(NEWS_DATA_ROOT, FINAL_SCORES)

    # To run or not to run the initial deduplication
    unique_url_corpus_available = True

    # Language for stemming
    language: str = "finnish"

    # If extra inspection files should be written
    write_inspection_files: bool = False

    # Keywords for filtering
    filtering_keywords_title: List[str] = field(default_factory=lambda: ["katso lähetys tästä", "vaaratiedote", "herätys:"])
    filtering_keywords_tags: List[str] = field(default_factory=lambda: ["kooste", "tietokilpailut", "vaaratiedotteet"])

    # Values for length-based filtering
    min_sents: int = 3
    min_len: int = 150
    max_len: int = 15000

    # How many articles to retrieve with bm25s
    top_k: int = 2