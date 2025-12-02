import os
import numpy as np
import logging
from typing import Iterable
import transformers
from transformers import BitsAndBytesConfig
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
import tiktoken
from src.utils.helpers import do_batching, yield_titles_with_instructions
from config.task_prompts import BEST_PROMPTS

logger = logging.getLogger(__name__)

def get_tiktoken_encoding():
    return tiktoken.get_encoding("cl100k_base") # used for OpenAI embedding models

class BatchEmbedder:
    def __init__(self, model_name: str, batch_size: int, max_tokens_per_batch: int, **kwargs):
        self.model_name = model_name.lower()
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.gpu = kwargs.get("gpu", torch.cuda.is_available())
        self.test = kwargs.get("test", False)
        self.tokenizer, self.model, self.embedding_dim, self.max_length = self.get_model_and_tokenizer()
        self.encoding = get_tiktoken_encoding()
        self.prompt = kwargs.get("prompt", self.get_prompt()) # accept a custom prompt
        logger.info(f"Registered task description at initialization time: {self.prompt}")

        # Print model placement and memory statistics
        self.check_devices()
        self.report_memory_usage("after model initialization")

    def get_model_and_tokenizer(self):
        if "bert" in self.model_name:
            tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
            model = transformers.BertModel.from_pretrained(self.model_name)
            if self.gpu:
                model.cuda()
            model.eval()
            embedding_dim = model.config.hidden_size
            return tokenizer, model, embedding_dim, None #TODO: max_length?
        
        elif "qwen" in self.model_name:
            
            # Use 8-bit quantization as the full model doesn't leave room for processing the data on a single GPU
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            model = transformers.AutoModel.from_pretrained(self.model_name, quantization_config=quantization_config, device_map=self._set_device_map())
            model.eval()
            embedding_dim = model.config.hidden_size # expected: 4096
            max_length = 8192
            return tokenizer, model, embedding_dim, max_length
        
        elif "multilingual-e5" in self.model_name:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            model = transformers.AutoModel.from_pretrained(self.model_name, device_map=self._set_device_map())
            model.eval()
            embedding_dim = model.config.hidden_size # expected: 1024
            max_length = 512
            return tokenizer, model, embedding_dim, max_length
        
        else:
            model = SentenceTransformer(self.model_name)
            return None, model, model.get_sentence_embedding_dimension(), None #TODO: max_length?

    def get_prompt(self):
        if "qwen" in self.model_name:
            return BEST_PROMPTS["BestQwen"]
        elif "multilingual-e5" in self.model_name:
            return BEST_PROMPTS["BestE5"]
        else:
            return ""
        
    def batch_based_on_token_count(self, documents: Iterable[str], num_documents: int, max_tokens:int):

        current_max_len = 0
        current_batch = []

        num_processed = 0
        estimated_batches_remaining = num_documents

        for i, document in enumerate(documents):

            # Use tiktoken for token count estimation; much faster than self.tokenizer and gives a better estimate than
            # just str.split() * arbitrary int for error marginal
            num_estimated_tokens = int(len(self.encoding.encode(document)) * 1.3) # still use some error marginal here

            # If a single document exceeds the max_token limit
            if num_estimated_tokens > max_tokens:
                raise Exception(f"Estimated token count ({num_estimated_tokens}) over max token limit ({max_tokens}) at index {i}")

            # Check if the current document can be added to the batch by "padding" documents to the longest one
            if (len(current_batch) + 1) * max(current_max_len, num_estimated_tokens) > max_tokens:
                
                if current_batch:
                    num_processed += len(current_batch)
                    estimated_batches_remaining = int((num_documents - num_processed) / len(current_batch))
                    logger.info(f"{num_processed}/{num_documents} documents iterated")
                    logger.debug(f"Latest batch of size {len(current_batch)} documents and {len(current_batch) * current_max_len} estimated tokens. Estimated number of batches remaining: {estimated_batches_remaining}")

                    yield current_batch

                current_batch = []
                current_max_len = 0

            current_batch.append(document)
            current_max_len = max(current_max_len, num_estimated_tokens)
        
        if current_batch:
            yield current_batch

    def _do_batching(self, documents, num_documents):
        if "qwen" in self.model_name or "multilingual-e5" in self.model_name:
            logger.info(f"Going to do dynamic batching with max estimated token count per batch {self.max_tokens_per_batch}")
            return self.batch_based_on_token_count(documents, num_documents, self.max_tokens_per_batch)
        else:
            logger.info(f"Going to use static batch size of {self.batch_size}")
            return do_batching(documents, self.batch_size)

    def encode_queries(self, documents, num_documents, save_to=None, return_embeddings=False, task_description=None):
        """Encode queries by optionally augmenting with a task description.
        If task_description is not specified, will use self.prompt."""
        task_description = task_description if task_description is not None else self.prompt
        if task_description != self.prompt:
            logger.info(f"Overriding defult model prompt ('{self.prompt}') with user-passed prompt '{task_description}'")
        documents_with_instruction = yield_titles_with_instructions(documents, task_description=task_description)
        return self.encode(documents_with_instruction, num_documents, save_to, return_embeddings)

    def encode(self, documents, num_documents, save_to=None, return_embeddings=False):
        logger.debug(f"Arguments passed for encode function: save_to={save_to}, return_embeddings={return_embeddings}")
        
        memmap_file = self.initialize_memmap_file(save_to, num_documents) if save_to else None
        embeddings = np.empty((num_documents, self.embedding_dim), dtype=np.float32) if return_embeddings else None

        #TODO: Error prone, could specify this already at initialization time
        if "bert" in self.model_name:
            batch_processor = self.process_bert_batch
        elif "qwen" in self.model_name:
            batch_processor = self.process_qwen_batch
        elif "multilingual-e5" in self.model_name:
            batch_processor = self.process_e5_batch
        else:
            # Wrap with SentenceTransformer
            batch_processor = self.process_st_batch

        # TODO: This should also be defined based on if dynamic or static batching is used
        if self.batch_size <= 0 or "qwen" in self.model_name or "multilingual-e5" in self.model_name:
            num_batches = "unknown"
        else:
            num_batches = num_documents // self.batch_size + 1 if num_documents % self.batch_size != 0 else num_documents // self.batch_size

        # As the batch sizes may vary, keep track of the current indices
        start_idx = 0
        end_idx = 0
        for i, batch in enumerate(self._do_batching(documents, num_documents)):

            logger.debug(f"Processing batch {i+1}/{num_batches}")
            logger.debug(f"Received a batch of {len(batch)} documents")

            # Calculate indices for this batch
            start_idx = end_idx
            end_idx = start_idx + len(batch)
            logger.debug(f"Indices for memmap file / embeddings array: start {start_idx}, end: {end_idx}")
            assert len(batch) == end_idx - start_idx, f"Received a batch of {len(batch)} documents, allocated for {end_idx - start_idx}"

            embedding_batch = batch_processor(batch)
            logger.debug(f"Shape of received embeddings: {embedding_batch.shape}")
            assert embedding_batch.shape[0] == len(batch), f"Number of embeddings {embedding_batch.shape[0]} != {len(batch)}"

            # Save to disk if enabled
            if memmap_file is not None:
                memmap_file[start_idx:end_idx] = embedding_batch
                memmap_file.flush()

            # Store in memory if requested
            if embeddings is not None:
                embeddings[start_idx:end_idx] = embedding_batch

            del embedding_batch # Remove reference to batch before a new batch is processed

        if memmap_file is not None:
            memmap_file.flush()
        logger.info(f"Embeddings saved to {save_to}")
        
        return embeddings

    def _set_device_map(self):
        if self.gpu:
            num_gpus = torch.cuda.device_count()
            logger.debug(f"Number of GPUs available: {num_gpus}")
            if num_gpus == 1:
                return "cuda"
            return "auto"
        
    def report_memory_usage(self, message):
        logger.info(f'max memory allocation {message}:')
        total = 0
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.max_memory_allocated(i)
            logger.info(f'  cuda:{i}: {mem/2**30:.1f}G')
            total += mem
        logger.info(f'  TOTAL: {total/2**30:.1f}G')

    def check_devices(self):
        logger.debug(f'devices:')
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                logger.debug(f'  {name}.{param_name}:{param.device}')
                if param.device.type != 'cuda':
                    logger.warning(f'{name}.{param_name} on device {param.device}')

    def get_available_memory(self, debug_message=""):

        # Total GPU memory
        total_memory = torch.cuda.get_device_properties().total_memory  # in bytes

        # Memory currently used by PyTorch
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()

        # Available memory
        available = total_memory - reserved

        memory_info = {
            "total_memory": total_memory,
            "allocated": allocated,
            "reserved": reserved,
            "available": available
        }

        def to_string():
            return "\n".join((f"{key}: {value / (1024**3):.2f} GB" for key, value in memory_info.items()))
        
        logger.debug(f"{debug_message}:\n{to_string()}")
        
        return memory_info

    def estimate_memory_per_token(self, initial_mem_use, num_tokens):

        # Calculate memory used by the forward pass
        memory_used = torch.cuda.max_memory_allocated() - initial_mem_use

        logger.debug(f"Memory used (max_allocated - initial): {memory_used} ({memory_used / (1024**3):.2f} GB)")
        logger.debug(f"Number of tokens in the sample: {num_tokens}")

        # Estimate memory per token
        memory_per_token = memory_used / num_tokens
        logger.debug(f"Estimated memory per token: {memory_per_token} bytes ({memory_per_token / (1024**2):.2f} MiB)")

        return memory_per_token, memory_used
    
    def calculate_max_tokens(self, memory_info, initial_mem_use, num_tokens):
        # Estimate memory per token
        memory_per_token, memory_used = self.estimate_memory_per_token(initial_mem_use, num_tokens)

        # Calculate maximum tokens based on available memory
        try:
            max_tokens = int((memory_info["available"] +  memory_used) / memory_per_token)

            # Only use 90% of the estimated max capacity
            max_tokens = int(max_tokens * 0.9)

            logger.debug(f"Estimated max tokens (with 10% memory buffer): {max_tokens}")
        
        except ZeroDivisionError:
            logger.warning(f"ZeroDivisionError when calculating ({memory_info["available"]} + {memory_used}) / {memory_per_token} (memory used divided with memory_per_token)")

    def process_st_batch(self, batch):
        return self.model.encode(batch, batch_size=self.batch_size)

    # The implementation is from https://github.com/TurkuNLP/paraphrase-span-detection/blob/master/baselines/bert_baseline.py
    def process_bert_batch(self, batch):
    
        data = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt",
                              return_special_tokens_mask=True)
        if self.gpu:
            input_ids=data["input_ids"].cuda()
            token_type_ids=data["token_type_ids"].cuda()
            attention_mask=data["attention_mask"].cuda()
            spec_mask=data["special_tokens_mask"].cuda()
        else:
            input_ids=data["input_ids"]
            token_type_ids=data["token_type_ids"]
            attention_mask=data["attention_mask"]
            spec_mask=data["special_tokens_mask"]

        with torch.no_grad():
            emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            last_hidden=emb.last_hidden_state
            attention_mask=attention_mask*(spec_mask*-1+1)
            attention_mask_sum=torch.sum(attention_mask,dim=-1) 
            last_hidden_masked=last_hidden.mul(attention_mask.unsqueeze(-1))
            last_hidden_masked_sum=torch.sum(last_hidden_masked,dim=1)
            last_hidden_mean=torch.div(last_hidden_masked_sum,attention_mask_sum.unsqueeze(-1))
            if self.gpu:
                return last_hidden_mean.cpu().numpy()
            else:
                return last_hidden_mean
            
    def _tokenize_with_qwen(self, batch):
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to("cuda") # assumes a single GPU is being used

    def _last_token_pool(self, last_hidden_states: Tensor,
                attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
    def get_embeddings(self, inputs, initial_memory_info: dict=None):
        with torch.no_grad():
            outputs = self.model(**inputs)

            if self.test:
                memory_info = self.get_available_memory("Memory use after embedding")
                self.calculate_max_tokens(memory_info, initial_memory_info["allocated"], inputs.input_ids.numel())

            embeddings = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            return embeddings.cpu().numpy()

    # The implementation is based on https://huggingface.co/Qwen/Qwen3-Embedding-8B Transfromers Usage (now split into multiple functions)
    def process_qwen_batch(self, batch):

        # Clear cache and reset memory stats
        # TODO: are these okay to be here for each iteration?
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
        data = self._tokenize_with_qwen(batch)

        if self.test:
            initial_memory_info = self.get_available_memory("Memory use after tokenization")
        else:
            initial_memory_info = None

        # Check the total number of tokens after tokenization and split the batch to half
        if data.input_ids.numel() > self.max_tokens_per_batch:
            logger.warning(f"Number of tokens ({data.input_ids.numel()}) exceeds the given max token limit ({self.max_tokens_per_batch}). Attempting to split the current batch and re-tokenizing before moving tensors to GPU.")
            del data
            half = len(batch) // 2
            first_embeddings = self.get_embeddings(self._tokenize_with_qwen(batch[:half]), initial_memory_info)
            second_embeddings = self.get_embeddings(self._tokenize_with_qwen(batch[half:]), initial_memory_info)
            return np.vstack([first_embeddings, second_embeddings])

        # Otherwise, encode the full batch
        return self.get_embeddings(data, initial_memory_info)
     
    def _tokenize_with_e5(self, batch):
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.model.device) # assumes a single GPU is being used

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def process_e5_batch(self, batch):

        torch.cuda.empty_cache()

        data = self._tokenize_with_e5(batch)

        if self.test:
            initial_memory_info = self.get_available_memory("Memory use after tokenization")
        else:
            initial_memory_info = None

        # Check the total number of tokens after tokenization and split the batch to half
        # TODO: Could be one function instead of a copy from process_qwen_batch
        if data.input_ids.numel() > self.max_tokens_per_batch:
            logger.warning(f"Number of tokens ({data.input_ids.numel()}) exceeds the given max token limit ({self.max_tokens_per_batch}). Attempting to split the current batch and re-tokenizing before moving tensors to GPU.")
            del data
            half = len(batch) // 2
            first_embeddings = self.get_embeddings(self._tokenize_with_e5(batch[:half]), initial_memory_info)
            second_embeddings = self.get_embeddings(self._tokenize_with_e5(batch[half:]), initial_memory_info)
            return np.vstack([first_embeddings, second_embeddings])

        with torch.no_grad():
            outputs = self.model(**data)

            if self.test:
                memory_info = self.get_available_memory("Memory use after embedding")
                self.calculate_max_tokens(memory_info, initial_memory_info["allocated"], data.input_ids.numel())

            embeddings = self._average_pool(outputs.last_hidden_state, data['attention_mask'])
            
            return embeddings.cpu().numpy()

    def initialize_memmap_file(self, output_file, num_documents):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.debug(f"Initializing a memmap file of shape ({num_documents}, {self.embedding_dim})")
        return np.lib.format.open_memmap(output_file, mode='w+', dtype=np.float32,
                                         shape=(num_documents, self.embedding_dim))

    def __del__(self):
        if hasattr(self, 'model') and self.gpu:
            del self.model  # Properly clear GPU memory