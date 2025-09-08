# Preprocessing

This directory contains code for preprocessing the news data. The main functionality is summarized here:

* The original data in JSON and HTML formats is turned into a JSONL file, where each line corresponds to a news article and its metadata. `combine_all_news.py`
* Articles with duplicate URLs are deduplicated. The latest instance (according to the timestamp) is kept. If the timestamps are identical, the latest version in the JSON/HTML data is kept. `deduplicate_by_url.py`
* Further clean-up steps are wrapped in `run_full_cleaning.py`.
* Three external files are used for removing extra noise from the text. Through different heuristics, any lines/sentences that are considered "junk" (non-essential content) are filtered out from the data. `clean_up_line_junk.py`
* Additional filtering is done based on keywords, duplicates, and length of the articles. `filtering/*`
* Two variations of bm25 model are used for calculating a score of how well a title of an article corresponds to the text of the same article. Additionally, a score for a best match (if the correct text is not the best match) or a second best match (if the correct text is the best match) for the title is reported with its URL. `add_bm25_scores.py`
