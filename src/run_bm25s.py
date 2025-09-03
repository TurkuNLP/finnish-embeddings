from src.utils.helpers import yield_values_from_jsonl, get_line_count, get_query_indices, get_query_data_in_order
from src.evaluate import save_evaluation
import pandas as pd
import nltk
import Stemmer
import bm25s

def run_bm25s(read_from:str,
              read_query_indices_from: str,
              save_index_to: str,
              save_results_to: str,
              passage_key: str = "text_end",
              query_key: str = "title",
              language: str = "finnish",
              top_k: list[int] = [1, 5]):

    query_indices = get_query_indices(read_query_indices_from)
    corpus_len = get_line_count(read_from)

    def prepare_scores():
        
        stemmer = Stemmer.Stemmer(language)
        stopwords = nltk.corpus.stopwords.words(language)

        tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer, stopwords=stopwords)
        # Create a retriever with default settings
        retriever = bm25s.BM25()

        # Tokenize the corpus
        data_generator = yield_values_from_jsonl(read_from, passage_key)
        corpus_tokenized = tokenizer.tokenize(data_generator,
                                              return_as="tuple", # prevents the ids getting mixed up
                                              length=corpus_len, # needs to be passed explicitly with generator
                                              update_vocab=True)  # explicitly make sure that vocabulary is updated

        # Tokenize the queries
        query_generator = get_query_data_in_order(read_from, query_key, query_indices)
        query_tokenized = tokenizer.tokenize(query_generator,
                                             return_as="tuple",
                                             length=corpus_len,
                                             update_vocab=False) # explicitly make sure that vocabulary is not updated

        # Index the corpus
        retriever.index(corpus_tokenized)

        # Save the index
        retriever.save(save_index_to)

        return query_tokenized.ids, retriever

    # Tokenize and index the articles
    queries, bm25s_retriever = prepare_scores()

    # Retrieve the top-k results with the non-naive retriever for recording the closest incorrect match
    retrieved_documents = bm25s_retriever.retrieve(queries, k=top_k, return_as="documents")

    save_evaluation(result_matrix=retrieved_documents,
                    top_k_list=top_k,
                    query_indices=query_indices,
                    save_to=save_results_to)