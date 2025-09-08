import pandas as pd
from nltk.tokenize import sent_tokenize
from utils.helper import print_announcement, update_article_count

def get_sentence_count(text:str, language="finnish"):
    sentences = 0
    # Split text by line breaks, otherwise will be counted as one sentence if no punctuation is found
    for line in text.split("\n"):
        # Use NLTK sentence tokenizator
        sentences += len(sent_tokenize(line, language=language))
    return sentences

def print_length_statistics(news: pd.DataFrame):

    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    print_announcement("Text length statistics (characters)")
    print(news["text_len"].describe(percentiles))

    print_announcement("Text length statistics (sentences)")
    print(news["sents"].describe(percentiles))

def filter_based_on_text_len(news:pd.DataFrame, min_sents=3, min_len=150, max_len=15000, write_inspection_files=False):
    """
    Parameters
    ----------
    min_len: int
        Drop articles that are shorter than {min_len} characters.
    min_sents: int
        Drop articles that are shorter tha {min_sents} sentences.
    max_len: int
        Drop articles that are longer than {max_len} characters.
    """
    
    print_announcement(f"Filtering out the texts that are shorter than {min_len:,} characters and {min_sents:,} sentences, and longer than {max_len:,} characters.")

    num_articles = news.shape[0]

    # Create a new column for storing the number of sentences
    news["sents"] = news["text_end"].apply(get_sentence_count)
    
    # Filter out the articles that are shorter than {min_len} characters
    news = news.loc[news["text_len"] >= min_len]
    num_articles = update_article_count(num_articles, news.shape[0], f"Remove articles shorter than {min_len:,} characters")

    # Filter out the articles that are shorter than {min_sents} sentences
    news = news.loc[news["sents"] >= min_sents]
    num_articles = update_article_count(num_articles, news.shape[0], f"Remove articles shorter than {min_sents:,} sentences")

    # Filter out the articles that are longer than {max_len} characters
    news = news.loc[news["text_len"] <= max_len]
    num_articles = update_article_count(num_articles, news.shape[0], f"Remove articles longer than {max_len:,} characters")

    if write_inspection_files:
        with open("under-200-chars-25-06-25.txt", "w") as file:
            short = news.query("text_len < 200")
            for s in short.text_end.to_list():
                print(repr(s), file=file)

    print_length_statistics(news)
    
    return news