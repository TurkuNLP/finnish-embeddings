import json
from itertools import islice
from typing import Iterable

def yield_values_from_jsonl(filename, key:str=None, first_k:int=None, indices:list[int]|set[int]=None):
    if indices and isinstance(indices, list):
        indices = set(indices)
    with open(filename) as file:
        for i, line in enumerate(file):
            if first_k and i == first_k:
                return
            if indices and i not in indices:
                    continue
            if key:
                yield json.loads(line)[key]
            else:
                yield json.loads(line)

def get_data_as_dict(filename:str, key:str=None, indices:list[int]|set[int]=None):
    data_dict = {}
    if indices and not isinstance(indices, set):
        indices = set(indices)
    with open(filename) as file:
        for i, line in enumerate(file):
            if indices and i not in indices:
                continue
            if key:
                data_dict[i] = json.loads(line)[key]
            else:
                data_dict[i] = json.loads(line)
    return data_dict

def get_corpus_as_dict(filename, id_key="url") -> dict:
    """Returns the news articles as a dict, where the keys are the ids of the articles. The id key defaults to 'url'.
    Expects a jsonl file, where each line corresponds to an article and its metadata."""
    news = yield_values_from_jsonl(filename)
    return {item[id_key]: item for item in news}

def get_query_indices(filename):
    return [int(value) for value in yield_values_from_text_file(filename)]

def get_query_data_in_order(read_data_from:str, key:str, indices:Iterable[int]):

    data_dict = get_data_as_dict(read_data_from, key, indices)
    
    for index in indices:
        yield data_dict[index]

def yield_values_from_text_file(filename):
    with open(filename) as file:
        for line in file:
            if line.strip():
                yield line.strip()

def do_batching(iterable, batch_size, *, strict=False):
    """Yields batches of given size as lists. If strict=True, raises an error if the length of the iterable is not divisible by batch_size.
    Modified from https://docs.python.org/3/library/itertools.html#itertools.batched"""
    # batched('ABCDEFG', 2) → AB CD EF G
    if batch_size < 1:
        raise ValueError('batch_size must be at least one')
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        if strict and len(batch) != batch_size:
            raise ValueError('do_batching(): incomplete batch')
        yield batch

def get_line_count(filename):
    """Counts the number of lines in a file, skipping lines that only contain whitespace."""
    with open(filename) as file:
        return sum(1 for line in file if line.strip())
    
def save_to_json(filename, obj:dict):
    with open(filename, "w") as file:
        json.dump(obj, file)