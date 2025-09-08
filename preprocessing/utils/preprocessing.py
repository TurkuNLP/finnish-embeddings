import numpy as np

def get_beginning(text:str):
    """Returns the first paragraph of a text."""
    return text.split("\n")[0]

def remove_beginning(text:str):
    """Removes the first paragraph of a text and returns the rest."""
    return "\n".join(text.split("\n")[1:])

def shuffle_indices(num_items:int, seed=333):
    """Shuffles a range(0, num_items) and returns the resulting integers as a list."""

    np.random.seed(seed)
    indices = np.array(range(num_items))
    np.random.shuffle(indices)

    return indices.tolist()

def get_random_indices(range_stop:int, n=1000, seed=333, exclude:list=None):
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

    return rng.choice(indices, size=n, replace=False).tolist()