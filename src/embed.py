import argparse
import logging
import os
from utils.helpers import yield_values_from_jsonl, do_batching, get_line_count
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Generator, Optional

logger = logging.getLogger(__name__)

class BatchEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, **kwargs):
        self.model_name = model_name
        self.batch_size = batch_size
        self.kwargs = kwargs

        self.model = self.get_model(model_name, **self.kwargs)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()  # only for SentenceTransformer

    def get_model(model_name: str, kwargs=None):
        if model_name.lower() == "bm25" or model_name.lower() == "bm25s":
            raise NotImplementedError("Currently only supports SentenceTransformer models")
        else:
            return SentenceTransformer(model_name, **kwargs)

    def encode(
            self,
            documents: Generator[str, None, None],
            num_documents: int,
            save_to: Optional[str] = None,
            return_embeddings: bool = False,
            **kwargs
        ) -> Optional[np.ndarray]:
            """
            Encode documents in batches with optional saving to disk.

            Args:
                documents: Generator yielding text documents to encode
                num_documents: Total number of documents to encode
                save_to: Path to save embeddings (None to skip saving)
                return_embeddings: Whether to return the embeddings array

            Returns:
                Numpy array of embeddings if return_embeddings=True, else None
            """
            # Initialize memory-mapped file if saving is enabled
            memmap_file = None
            if save_to is not None:
                memmap_file = self._initialize_memmap_file(save_to, num_documents)

            # Initialize array to store embeddings if requested
            embeddings = None
            if return_embeddings:
                embeddings = np.empty((num_documents, self.embedding_dim), dtype=np.float32)

            # Process in batches
            num_batches = num_documents // self.batch_size + 1 if num_documents % self.batch_size != 0 else num_documents // self.batch_size
            for i, batch in enumerate(self._batch_documents(documents, self.batch_size)):
                logger.debug(f"Processing batch {i+1}/{num_batches}")

                # Calculate indices for this batch
                start_idx = i * self.batch_size
                end_idx = start_idx + len(batch)

                # Encode batch
                batch_embeddings = self.model.encode(batch, batch_size=self.batch_size, **kwargs)

                # Save to disk if enabled
                if memmap_file is not None:
                    memmap_file[start_idx:end_idx] = batch_embeddings

                # Store in memory if requested
                if embeddings is not None:
                    embeddings[start_idx:end_idx] = batch_embeddings

                # Flush to disk periodically
                if i % 10 == 0:  # Flush every 10 batches
                    if memmap_file is not None:
                        memmap_file.flush()

            # Final flush if saving is enabled
            if memmap_file is not None:
                memmap_file.flush()
                logger.info(f"Embeddings saved to {save_to}")

            return embeddings if return_embeddings else None

    def _initialize_memmap_file(self, output_file: str, num_documents: int) -> np.memmap:
        """Initialize memory-mapped file for embeddings"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Create memory-mapped file
        memmap_file = np.lib.format.open_memmap(
            output_file,
            dtype=np.float32,
            mode='w+',
            shape=(num_documents, self.embedding_dim)
        )

        logger.debug(f"Created memory-mapped file with shape: ({num_documents}, {self.embedding_dim})")
        return memmap_file
    
    def _batch_documents(documents, batch_size):
        do_batching(documents, batch_size)


def load_embeddings(filename):
    loaded_embeddings = np.load(filename)
    logger.debug(f"Loaded embeddings array with the following shape: {loaded_embeddings.shape}")


def embed(model_name: str,
          documents: Generator[str, None, None] | list[str],
          num_documents: int,
          batch_size: int,
          save_to: Optional[str] = None,
          return_embeddings: bool = False,
          **kwargs):
    
    batch_embedder = BatchEmbedder(model_name, batch_size, **kwargs)
    return batch_embedder.encode(documents=documents,
                                 num_documents=num_documents,
                                 save_to=save_to,
                                 return_embeddings=return_embeddings)

def main(args):

    model_name = args.model
    data_files = args.data_files
    dict_key = args.dict_key
    k_first = args.k_first
    batch_size = args.batch_size
    dest_folder = args.save_to

    embedding_file = os.path.join(dest_folder, f"{model_name.replace('/', '__')}_embeddings.npy")

    model = SentenceTransformer(model_name)
    
    num_documents = get_line_count(data_files) if not k_first else k_first
    documents = yield_values_from_jsonl(data_files, dict_key, k_first)
    encode_in_batches(documents, num_documents, model, embedding_file, batch_size)    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Which Sentence Transformer model to use. Refer to https://sbert.net/docs/sentence_transformer/pretrained_models.html for an overview of available models.")
    parser.add_argument("data_files",
                        help="Path to the JSONL file.")
    parser.add_argument("save_to",
                        help="Path to the directory where the resulting embedding array should be saved.")
    parser.add_argument("--dict_key",
                        default="text_end",
                        help="The key to the field that should be extracted from each row of JSONL file defined with parameter 'data_files'.")
    parser.add_argument("--k_first",
                        type=int,
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
    return parser.parse_args()

if __name__ == "__main__":

    main(parse_arguments())


def encode_in_batches(documents, num_documents:int, model:SentenceTransformer, output_file:str, batch_size:int, return_array:bool=True):
    
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
    num_batches = num_documents // batch_size + 1 if num_documents % batch_size != 0 else num_documents // batch_size
    for i, batch in enumerate(batched_documents):
        logger.debug(f"Processing batch {i+1}/{num_batches}")
        start_idx = i * batch_size
        end_idx = start_idx + len(batch) # use len() instead of batch_size as last batch might be smaller
        
        # Encode batch and write directly to memmap
        batch_embeddings = model.encode(batch, batch_size=batch_size)
        memmap_file[start_idx:end_idx] = batch_embeddings
    
    # Flush to disk
    memmap_file.flush()
    
    if return_array:
        return batch_embeddings
    else: return