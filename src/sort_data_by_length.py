import os
import json
from dotenv import load_dotenv
from utils.helpers import get_corpus_as_dict, yield_values_from_jsonl
from collections import defaultdict

load_dotenv()

original_data = get_corpus_as_dict(os.getenv("NEWS_DATA"))
length_sorted_data_filename = os.getenv("NEWS_DATA_SORTED_BY_LEN")
new_index_filename_full_data = os.path.join(os.getenv("INDEX_MAPPING_DIR"), "new_index_mapping.txt")
new_index_filename_queries = os.path.join(os.getenv("QUERY_INDICES_DIR"), "1000_query_indices_mapped_to_data_sorted_by_length.txt")

length_map = defaultdict(set)
for url in original_data:
    length = len(original_data[url]["text_end"].split()) # use naive whitespace splitting for counting "words"
    length_map[length].add(url)

with open(length_sorted_data_filename, "w") as file, open("lengths.txt", "w") as inspection_file:
    for l, urls in sorted(length_map.items(), reverse=True):
        for url in urls:
            print(json.dumps(original_data[url], ensure_ascii=False), file=file)
        print(l, file=inspection_file)

length_sorted_data = yield_values_from_jsonl(length_sorted_data_filename, "url")

new_mapping = {}
for i, url in enumerate(length_sorted_data):
    new_mapping[url] = i

index_mapping = {}
for i, url in enumerate(original_data):
    orig_index = i
    new_index = new_mapping[url]
    index_mapping[orig_index] = new_index

with open(new_index_filename_full_data, "w") as file:
    for n in index_mapping.values():
        print(n, file=file)

def map_old_indices_to_new(read_index_from:str, write_new_index_to:str):
    """Given a text file that contains an index per row, write a new file
    where the indices correspond to the same articles in the newly ordered data."""

    with open(read_index_from) as orig_index, open(write_new_index_to, "w") as new_index:
        for i in orig_index:
            to_map = int(i.strip())
            print(index_mapping[to_map], file=new_index)

map_old_indices_to_new(os.getenv("QUERY_INDICES"), new_index_filename_queries)





