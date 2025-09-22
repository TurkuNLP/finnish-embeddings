from dotenv import load_dotenv
import os
import json
import numpy as np
from tqdm import tqdm

load_dotenv()

DATA_PATH = os.getenv("NEWS_DATA_SORTED_BY_LEN")
DEVTEST_PATH = DATA_PATH.replace(".json", "-dev-test.json")

def get_indices_to_exclude(filename, keyword_to_exclude="kolumni"):
    exclude = set()
    with open(filename) as file:
        for i, line in enumerate(file):
            obj = json.loads(line)
            if any(tag.lower() == keyword_to_exclude for tag in obj["tags"]):
                exclude.add(i)
    return exclude, i

def get_random_indices(range_stop:int, n=5000, seed=333, exclude:list|set=None):
    """Returns a list of integers of size n, sampled from a range(0, range_stop) without replacement.
    If 'exclude' parameter is given, removes those integers from the original range before sampling.
    The integers to exclude should be smaller than the given 'range_stop'.
    Parameter 'n' is expected to be smaller than len(list(range(0, range_stop)) - len(exclude)."""
    
    rng = np.random.default_rng(seed=seed)

    indices = np.array(range(range_stop))

    if exclude:
        exclude = np.array(exclude)
        mask = ~np.isin(indices, exclude)
        indices = indices[mask]

    return set(rng.choice(indices, size=n, replace=False))

indices_to_exclude, last_index = get_indices_to_exclude(DATA_PATH)
dev_indices = get_random_indices(last_index, exclude=indices_to_exclude)

with open(DATA_PATH) as read_from, open(DEVTEST_PATH, "w") as write_to:
    for i, line in tqdm(enumerate(read_from)):
        obj = json.loads(line)
        obj["split"] = "dev" if i in dev_indices else "test"
        print(json.dumps(obj, ensure_ascii=False), file=write_to)