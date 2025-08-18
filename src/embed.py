import argparse
from utils.get_data import yield_from_jsonl
from sentence_transformers import SentenceTransformer

def main(args):

    model_name = args.model
    data_files = args.data_files
    key = args.key

    documents = yield_from_jsonl(data_files, key)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents)
    print(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Which Sentence Transformer model to use. Refer to https://sbert.net/docs/sentence_transformer/pretrained_models.html for an overview of available models.")
    parser.add_argument("data_files",
                        help="Path to the JSONL file.")
    parser.add_argument("--key",
                        default="text_end",
                        help="The key to the field that should be extracted from each row of JSONL file defined with parameter 'data_files'.")
    args = parser.parse_args()

    main(args)