import json
import pandas as pd
import datasets
import os
import re

def get_corpus_as_list(filename, num_items=None) -> list:
    """Returns the news articles as a list. Expects a jsonl file, where each line corresponds to an article and its metadata.
    If num_items is given, only returns max that amount of items.
    If the given number is larger than there are lines in the file, returns all items."""
    with open(filename) as file:
        if num_items:
            corpus = []
            for i, line in enumerate(file):
                if i == num_items:
                    break
                corpus.append(json.loads(line))   
            return corpus
        return [json.loads(line) for line in file]

def get_corpus_as_dict(filename, id_key="url") -> dict:
    """Returns the news articles as a dict, where the keys are the ids of the articles. The id key defaults to 'url'.
    Expects a jsonl file, where each line corresponds to an article and its metadata."""
    news = get_corpus_as_list(filename)
    return {item[id_key]: item for item in news}

def get_corpus_as_df(filename) -> pd.DataFrame:
    """Returns the news articles as a pandas DataFrame. Expects a jsonl file, where each line corresponds to an article and its metadata."""
    return pd.DataFrame(get_corpus_as_list(filename))

def get_corpus_as_ds(filename) -> datasets.Dataset:
    """Returns the news articles as a HuggingFace Dataset. Expects a jsonl file, where each line corresponds to an article and its metadata."""
    return datasets.load_dataset("json", data_files=filename)

def yield_corpus(filename):
    with open(filename) as file:
        for line in file:
            yield json.loads(line)

def yield_values_from_jsonl_file(filename, key="text_end"):
    with open(filename) as file:
        for line in file:
            yield json.loads(line)[key]

def yield_values_from_text_file(filename):
    with open(filename) as file:
        for line in file:
            if line.strip():
                yield line.strip()

def get_line_count(filename):
    """Counts number of lines in a file. Lines containing only whitespace are not included in the count."""
    with open(filename) as file:
        return sum(1 for line in file if line.strip())

def write_df_to_jsonl(df:pd.DataFrame, filename="dataset_new.jsonl"):
    """Writes a DataFrame into jsonl file, where each line corresponds to one row in the DataFrame."""

    # If datetime objects detected, turn them to isoformat
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
    
    records = df.to_dict(orient="records")

    with open(filename, "w") as file:
        for record in records:
            #record["timestamp"] = record["timestamp"].isoformat()
            print(json.dumps(record, ensure_ascii=False), file=file)

def rename(filepath:str):
    """Check if a filepath already exists, and if yes, return a new string from the given one
    by adding a trailing number enclosed in parenthesis (starting from '(1)'),
    or by incrementing an existing trailing number in the string.
    Will modify the last filepath component (filename).
    
    For example, rename('file.txt') would return 'file(1).txt'"""

    head, tail = os.path.split(filepath)
    name, extension = os.path.splitext(tail)
    end_pattern = r"\(\d{1,}\)"

    def increment_name(name):
        match = re.findall(end_pattern, name)
        if match:
            if name.endswith(match[-1]):
                num = int(match[-1][1:-1]) + 1 # remove parenthesis, increment by one
                name = name[:-len(match[-1])] + "(" + str(num) + ")"
        else:
            name = f"{name}(1)"
        return name
    
    new_filepath = os.path.join(head, f"{name}{extension}")

    while os.path.exists(new_filepath):
        name = increment_name(name)
        new_filepath = os.path.join(head, f"{name}{extension}")
    
    if tail != name+extension:
        print(f"Renamed {tail} to {name}{extension} to avoid overwriting")
    
    return new_filepath