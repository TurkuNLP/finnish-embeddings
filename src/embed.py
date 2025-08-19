import argparse
import logging
from utils.helpers import yield_from_jsonl, do_batching, get_line_count
from sentence_transformers import SentenceTransformer
import numpy as np
from os import path


def set_up_logger(verbosity_level:int):
    verbosity_levels = {0: logging.CRITICAL, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=verbosity_levels[verbosity_level], force=True)
    return logging.getLogger(__name__)

def log_args(args):
    logger.info(args)

def encode_in_batches(documents, num_documents:int, model:SentenceTransformer, output_file:str, batch_size:int):
    
    batched_documents = do_batching(documents, batch_size)
    
    # Get embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    
    # Create memory-mapped file
    memmap_file = np.lib.format.open_memmap(
        output_file, 
        dtype=np.float32, 
        mode='w+',
        shape=(num_documents, embedding_dim)
    )

    logger.debug(f"Created a memory-mapped file with the following shape: ({num_documents}, {embedding_dim})")
    
    # Process in batches
    for i, batch in enumerate(batched_documents):
        logger.debug(f"Processing batch {i+1}/{num_documents//batch_size + 1}")
        start_idx = i * batch_size
        end_idx = start_idx + len(batch)
        
        # Encode batch and write directly to memmap
        batch_embeddings = model.encode(batch)
        memmap_file[start_idx:end_idx] = batch_embeddings
    
    # Flush to disk
    memmap_file.flush()
    
    return

def load_data(filename):
    loaded_embeddings = np.load(filename)
    logger.debug(f"Loaded embeddings array with the following shape: {loaded_embeddings.shape}")

def main(args):

    log_args(args)

    model_name = args.model
    data_files = args.data_files
    dict_key = args.key
    k_documents = args.k
    batch_size = args.batch_size
    dest_folder = args.save_to

    embedding_file = path.join(dest_folder, "mock_embeddings.npy")

    model = SentenceTransformer(model_name)

    num_documents = get_line_count(data_files)
    documents = yield_from_jsonl(data_files, dict_key, k_documents)
    encode_in_batches(documents, num_documents, model, embedding_file, batch_size)

    # Test loading
    load_data()    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Which Sentence Transformer model to use. Refer to https://sbert.net/docs/sentence_transformer/pretrained_models.html for an overview of available models.")
    parser.add_argument("data_files",
                        help="Path to the JSONL file.")
    parser.add_argument("save_to",
                        help="Path to the directory where the resulting embedding array should be saved.")
    parser.add_argument("--key",
                        default="text_end",
                        help="The key to the field that should be extracted from each row of JSONL file defined with parameter 'data_files'.")
    parser.add_argument("--k",
                        help="Pick first k documents from the JSONL file defined with parameter 'data_files'.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="How many documents should be encoded simultaneously."
                        )
    parser.add_argument("--verbosity",
                        default=3,
                        choices=[0, 1, 2, 3],
                        type=int,
                        help="Logging levels given as an int (the higher the number the more detailed the output): 0 (only critical); 1 (warning); 2 (info); 3 (debug)"
                        )
    args = parser.parse_args()

    logger = set_up_logger(args.verbosity)

    main(args)