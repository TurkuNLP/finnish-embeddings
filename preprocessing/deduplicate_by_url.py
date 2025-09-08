import json
from datetime import datetime
from collections import Counter
from tqdm import tqdm

def deduplicate_by_url(unfiltered_corpus_filename: str,
                       unique_url_corpus_filename: str,
                      ):
    
    stats = Counter()
    corpus_by_url = {}

    def is_update(new, existing):
        stats.update({"timestamps_compared": 1})
        if existing > new:
            stats.update({"not_in_chronological_order": 1})
            return False
        if new > existing:
            stats.update({"in_chronological_order": 1})
            return True
        else:
            stats.update({"identical_timestamp": 1})
            return True

    with open(unfiltered_corpus_filename) as file:
        for line in tqdm(file, desc="Filtering out duplicate URLs", leave=False):
            obj = json.loads(line)

            # Turn timestamp into a datetime object for easy comparison
            obj["timestamp"] = datetime.fromisoformat(obj["timestamp"])

            # Remove the potential RSS suffix from the url
            obj["url"] = obj["url"].replace("?origin=rss", "")
            url = obj["url"]

            if url in corpus_by_url:
                # If the current object has an earlier timestamp, do not overwrite the existing article
                if not is_update(obj["timestamp"], corpus_by_url[url]["timestamp"]):
                    continue
                
            corpus_by_url[obj["url"]] = obj # overwrites an existing url

    corpus_as_list = sorted([article for article in corpus_by_url.values()], key=lambda x: x["timestamp"])
    assert isinstance(corpus_as_list[0], dict), f"List doesn't include dicts but {type(corpus_as_list[0])}s"

    with open(unique_url_corpus_filename, "w") as file:
        for article in tqdm(corpus_as_list, desc="Writing deduplicated articles into file", leave=False):
            article["timestamp"] = article["timestamp"].isoformat()
            print(json.dumps(article, ensure_ascii=False), file=file)

    print("--- STATISTICS ---")
    print(f"Articles after URL deduplication: {len(corpus_as_list)}")
    print(stats)