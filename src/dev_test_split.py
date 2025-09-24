from dotenv import load_dotenv
import os
import json
import numpy as np
from tqdm import tqdm
from utils.helpers import get_line_count

load_dotenv()

DATA_PATH = os.getenv("NEWS_DATA_SORTED_BY_LEN")
DEVTEST_PATH = DATA_PATH.replace(".json", "-dev-test.json")

path_start, devtest_filename = DEVTEST_PATH.rsplit("/", 1)
dev_indices_filename = "5000_dev_indices_mapped_to_"+devtest_filename
test_indices_filename = "134313_test_indices_mapped_to_"+devtest_filename

DEV_INDICES_PATH = os.path.join(path_start, dev_indices_filename)
TEST_INDICES_PATH = os.path.join(path_start, test_indices_filename)

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

range_stop = get_line_count(DATA_PATH)
dev_indices = get_random_indices(range_stop)

with open(DATA_PATH) as read_from, open(DEVTEST_PATH, "w") as write_to, open(DEV_INDICES_PATH, "w") as write_dev_i, open(TEST_INDICES_PATH, "w") as write_test_i:
    for i, line in tqdm(enumerate(read_from)):
        obj = json.loads(line)
        if i in dev_indices:
            obj["split"] = "dev"
            print(i, file=write_dev_i)
        else:
            obj["split"] = "test"
            print(i, file=write_test_i)
        print(json.dumps(obj, ensure_ascii=False), file=write_to)