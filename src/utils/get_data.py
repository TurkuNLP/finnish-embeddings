import json

def yield_from_jsonl(filename, key:str=None, first_k:int=None):
    with open(filename) as file:
        for i, line in enumerate(file):
            if first_k and i == first_k:
                return
            if key:
                yield json.loads(line)[key]
            else:
                yield json.loads(line)
