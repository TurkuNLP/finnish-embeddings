import json
from itertools import islice

def yield_from_jsonl(filename, key:str=None, first_k:int=None):
    with open(filename) as file:
        for i, line in enumerate(file):
            if first_k and i == first_k:
                return
            if key:
                yield json.loads(line)[key]
            else:
                yield json.loads(line)

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