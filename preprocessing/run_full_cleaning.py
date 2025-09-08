from datetime import datetime
from Config import Config
from deduplicate_by_url import deduplicate_by_url
from clean_up_line_junk import clean_up
from filtering.filter_by_keywords import filter_by_keywords
from filtering.filter_duplicates import filter_duplicates
from filtering.filter_by_length import filter_based_on_text_len
from add_bm25s_scores import add_bm25s_scores

def main():

    start_time = datetime.now()
    print("Started running at", start_time.time())

    config = Config()
    
    if not config.unique_url_corpus_available:
        deduplicate_by_url(unfiltered_corpus_filename=config.unfiltered_corpus_filename,
                           unique_url_corpus_filename=config.unique_url_corpus_filename)
    
    clean_up(unfiltered_corpus_filename=config.unfiltered_corpus_filename,
             unique_url_corpus_filename=config.unique_url_corpus_filename,
             extra_titles_filename=config.extra_titles_filename,
             junk_prefix_filename=config.junk_prefix_filename,
             junk_lines_filename=config.junk_lines_filename,
             filtered_corpus_filename=config.filtered_corpus_filename,
             removed_lines_filename=config.removed_lines_filename)
    
    news = filter_by_keywords(corpus_filename=config.filtered_corpus_filename,
                              filtering_keywords_title=config.filtering_keywords_title,
                              filtering_keywords_tags=config.filtering_keywords_tags)
    
    news = filter_duplicates(news,
                             write_inspection_files=config.write_inspection_files)

    news = filter_based_on_text_len(news,
                                    min_sents=config.min_sents,
                                    min_len=config.min_len,
                                    max_len=config.max_len)

    add_bm25s_scores(news,
                     save_as=config.final_scores_filename,
                     language=config.language,
                     top_k=config.top_k)
    
    end_time = datetime.now()
    print(f"Ready at {end_time.time()} | Runtime: {end_time - start_time}")
    

if __name__ == "__main__":
    main()