import pandas as pd
from utils.inspector import inspect_duplicates
from utils.helper import print_announcement, update_article_count

def filter_duplicates(news:pd.DataFrame, keep="last", write_inspection_files=False):

    if write_inspection_files:
        inspect_duplicates(news)

    # Turn timestamp into a datetime object
    news["timestamp"] = pd.to_datetime(news["timestamp"])
    news["date"] = news["timestamp"].dt.date
    news["orig_idx"] = news.index # needed for removing the actual first instances of duplicates

    # Sort data in chronological order, based on the timestamp and the original index
    news = news.sort_values(["timestamp", "orig_idx"])

    # Create a column for storing the text lengths
    news["text_len"] = news["text_end"].str.len()

    num_articles = news.shape[0]
    
    # Drop any articles that do not contain text
    news = news.loc[news["text_len"] > 0]
    num_articles = update_article_count(num_articles, news.shape[0], "Remove empty articles (first paragraph ignored)")
    
    print_announcement(f"Starting deduplication. Keeping the {keep} occurrence of the chronologically ordered data.")

    news = news.drop_duplicates(["title", "summary", "text_end"], keep=keep)
    num_articles = update_article_count(num_articles, news.shape[0], "Remove title-summary-text duplicates")
    
    news = news.drop_duplicates("text_end", keep=keep)
    num_articles = update_article_count(num_articles, news.shape[0], "Remove articles with duplicate texts")
    
    news = news.drop_duplicates(["title", "date"], keep=keep)
    num_articles = update_article_count(num_articles, news.shape[0], "Remove articles with duplicate titles from the same day")

    # Drop the date and orig_idx columns
    news = news.drop(columns=["date", "orig_idx"])

    return news