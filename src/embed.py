import os
import numpy as np
import logging
import transformers
import torch
from sentence_transformers import SentenceTransformer
from src.utils.helpers import do_batching

logger = logging.getLogger(__name__)

class BatchEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, **kwargs):
        self.model_name = model_name.lower()
        self.batch_size = batch_size
        self.gpu = kwargs.get('gpu', torch.cuda.is_available())
        self.tokenizer, self.model, self.embedding_dim = self.get_model_and_tokenizer()

    def get_model_and_tokenizer(self):
        if "bert" in self.model_name:
            tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
            model = transformers.BertModel.from_pretrained(self.model_name)
            if self.gpu:
                model.cuda()
            model.eval()
            embedding_dim = model.config.hidden_size
            return tokenizer, model, embedding_dim
        else:
            model = SentenceTransformer(self.model_name)
            return None, model, model.get_sentence_embedding_dimension()

    def encode(self, documents, num_documents, save_to=None, return_embeddings=False):
        logger.debug(f"Model type: {'BERT' if 'bert' in self.model_name else 'SentenceTransformer'}")
        
        memmap_file = self.initialize_memmap_file(save_to, num_documents) if save_to else None
        embeddings = np.empty((num_documents, self.embedding_dim), dtype=np.float32) if return_embeddings else None

        batch_processor = self.process_bert_batch if "bert" in self.model_name else self.process_st_batch

        num_batches = num_documents // self.batch_size + 1 if num_documents % self.batch_size != 0 else num_documents // self.batch_size
        for i, batch in enumerate(do_batching(documents, self.batch_size)):

            logger.debug(f"Processing batch {i+1}/{num_batches}")

            # Calculate indices for this batch
            start_idx = i * self.batch_size
            end_idx = start_idx + len(batch)

            embedding_batch = batch_processor(batch)

            # Save to disk if enabled
            if memmap_file is not None:
                memmap_file[start_idx:end_idx] = embedding_batch

            # Store in memory if requested
            if embeddings is not None:
                embeddings[start_idx:end_idx] = embedding_batch

            if i % 10 == 0 and memmap_file is not None:
                memmap_file.flush()

        if memmap_file is not None:
            memmap_file.flush()
            logger.info(f"Embeddings saved to {save_to}")
        
        return embeddings

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

    def initialize_memmap_file(self, output_file, num_documents):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        return np.lib.format.open_memmap(output_file, mode='w+', dtype=np.float32,
                                         shape=(num_documents, self.embedding_dim))

    def __del__(self):
        if hasattr(self, 'model') and self.gpu:
            del self.model  # Properly clear GPU memory