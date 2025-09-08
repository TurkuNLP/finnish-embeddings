from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from itertools import pairwise
from difflib import Differ

def get_tag_counts_as_counter(tag_lists):
    return Counter((tag.lower() for tag_list in tag_lists for tag in tag_list))

def get_tag_counts(tag_lists, n_most_common=None, return_counts=True):
    """Returns a list of tuples with the tags and optionally their counts, sorted by count and then alphabetically."""
    tags_and_counts = sorted(get_tag_counts_as_counter(tag_lists).most_common(n_most_common), key=lambda x: (x[1], x[0]), reverse=True)
    if return_counts:
        return tags_and_counts
    else:
        return [tag for tag, count in tags_and_counts]

def write_tag_counts(tag_lists, filename="tag_counts.txt"):
    tags = get_tag_counts(tag_lists)
    with open(filename, "w") as file:
        for tag, count in tags:
            print(f"{tag} | {count}", file=file)

def show_word_cloud(tag_lists):
    tag_counts = get_tag_counts_as_counter(tag_lists)
    print("Number of unique tags (lowercased):", len(tag_counts))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tag_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Tag Frequency Word Cloud")
    plt.show()

def write_readable_sample(sample: pd.DataFrame, columns=["title", "tags", "text", "url"], filename="sample.txt"):
    """Writes a sample of the corpus to a file in an easy-to-read format."""
    with open(filename, "w") as file:
        for i, row in sample.iterrows():
            for column in columns:
                print(f"{column.upper()}: {row[column]}", file=file)
            print(file=file)

def write_readable_sample_from_list(sample: list[dict], dict_keys=["title", "text_beginning", "text_end", "url"], indices: list[int]=None, filename="sample.txt"):
    """Writes a sample of the corpus (stored as a list of dicts) to a file in an easy-to-read format.
    If parameter indices is omitted, writes everything from parameter sample to the file. If indices is given, then only writes <sample[i] for i in indices> to file."""
    with open(filename, "w") as file:
        if indices:
            for i in indices:
                for key in dict_keys:
                    print(f"{key.upper()}: {sample[i][key]}", file=file)
                print(file=file)
        else:
            for item in sample:
                for key in dict_keys:
                    print(f"{key.upper()}: {item[key]}", file=file)
                print(file=file)

def inspect_dict_structure(dictionary: dict, indent=0, file=None):
    def inspect_item(item, indent):
        if isinstance(item, dict):
            for key, value in item.items():
                print(f"{'  ' * indent}{key}: {type(value)}", file=file)
                inspect_item(value, indent + 1)
        elif isinstance(item, list) and len(item) > 0:
            inspect_item(item[0], indent)

    inspect_item(dictionary, indent)

def contains_tag(tags:list, search_pattern):
    return any(re.search(search_pattern, tag) for tag in tags)

def inspect_pairwise(a:dict, b:dict, differ:Differ, file):
    print(a["title"], file=file)
    print(f"Index: {a["orig_idx"]} | {b["orig_idx"]}", file=file)
    print(f"Length (characters): {len(a["text_end"])} | {len(b["text_end"])}", file=file)
    print(f"{a["timestamp"]} | {b["timestamp"]}", file=file)
    print(f"{a["url"]} | {b["url"]}", file=file, end="\n\n")
    x = a["text_end"].splitlines()
    y = b["text_end"].splitlines()

    for line in differ.compare(x, y):
        print(line, file=file)
    print("="*20, file=file)

def inspect_duplicates(news:pd.DataFrame):
    news["orig_idx"] = news.index
    news = news.drop(columns=["tags"])
    news = news.drop_duplicates()
    news = news.drop_duplicates("text_end")
    duplicate_mask = news.duplicated(["title", "timestamp"], keep=False)
    duplicates = news[duplicate_mask].sort_values(["title", "orig_idx"])
    to_inspect = duplicates.to_dict(orient="records")
    print(f"{len(to_inspect)} title duplicates found")

    with open("title-inspection.txt", "w") as file:
        d = Differ()
        for a, b in pairwise(to_inspect):
            if a["title"] == b["title"]:
                inspect_pairwise(a, b, d, file)

    with open("title-counts.txt", "w") as file:
        print(duplicates.title.value_counts().to_string(), file=file)