from dotenv import load_dotenv
from os import getenv
from utils.file_operator import get_corpus_as_list
from utils.inspector import contains_tag
from utils.preprocessing import get_random_indices
import json

load_dotenv()

def write_sample_to_file(corpus, sample_indices, filename="random_sample_1000.jsonl"):
    i_filename = filename.replace(".jsonl", "_indices.txt")
    with open(filename, "w") as file, open(i_filename, "w") as i_file:
        for i in sample_indices:
            print(json.dumps(corpus[i], ensure_ascii=False), file=file)
            print(i, file=i_file)
        
    print(f"{len(sample_indices)} articles written to file '{filename}'")
        
def main():

    corpus = get_corpus_as_list(getenv("FINAL_DATA"))

    seed = 123
    n = 1000 # number of queries
    keyword_to_exclude = "kolumni"

    column_indices = [i for i, article in enumerate(corpus) if contains_tag(article["tags"], keyword_to_exclude)]
    print(f"{len(column_indices)} articles tagged with '{keyword_to_exclude}' found. These will be excluded from the population to be sampled.")

    sample_indices = get_random_indices(len(corpus), n, seed, column_indices)
    write_sample_to_file(corpus, sample_indices)


if __name__ == "__main__":
    main()