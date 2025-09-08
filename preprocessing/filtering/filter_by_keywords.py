from collections import Counter
from tqdm import tqdm
from utils.file_operator import yield_corpus, get_line_count
from utils.helper import print_announcement
import pandas as pd

class FilteringReasons:
    def __init__(self,
                 keywords_title: list,
                 keywords_tags: list
                 ):
        self.keywords_title = keywords_title
        self.keywords_tags = keywords_tags
    
        self.encountered = Counter()

def filter_based_on_keywords(article:dict, filtering_reasons_mapping:FilteringReasons):
    for keyword in filtering_reasons_mapping.keywords_title:
        if keyword in article["title"].lower():
            filtering_reasons_mapping.encountered.update({keyword: 1})
            return False
    for keyword in filtering_reasons_mapping.keywords_tags:
        for tag in [tag.lower() for tag in article["tags"]]:
            if keyword in tag:
                filtering_reasons_mapping.encountered.update({keyword: 1})
                return False
    return True

def filter_by_keywords(corpus_filename: str,
                       filtering_keywords_title: list,
                       filtering_keywords_tags: list):

    filtering_reasons = FilteringReasons(keywords_title=filtering_keywords_title,
                                         keywords_tags=filtering_keywords_tags)

    articles_initially = get_line_count(corpus_filename)
    print_announcement(f"Number of articles initially: {articles_initially:,}")

    filtered_articles = (
        article for article in tqdm(
            yield_corpus(corpus_filename),
            desc="Filtering articles based on keywords",
            total=articles_initially,
            leave=False
        ) if filter_based_on_keywords(article, filtering_reasons))
    
    news = pd.DataFrame(filtered_articles)

    # Statistics can be generated only after the 'filtered_articles' generator has been exhausted
    num_removed = filtering_reasons.encountered.total()
    print_announcement(f"Filtering on keywords removed {num_removed:,} articles. Remaining: {articles_initially-num_removed:,} articles.")
    print("Reasons for filtering:")
    for reason, count in filtering_reasons.encountered.most_common():
        print(f"'{reason}': {count}")

    return news