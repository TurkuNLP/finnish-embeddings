from typing import Iterable
import nltk
import Stemmer
import bm25s

def run_bm25s(passages:Iterable[str],
              corpus_len: int,
              queries: Iterable[str],
              save_index_to: str,
              top_k_list: list[int],
              language: str):
    
    top_k = max(top_k_list)

    def prepare_scores():
        
        stemmer = Stemmer.Stemmer(language)
        stopwords = nltk.corpus.stopwords.words(language)

        tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer, stopwords=stopwords)
        # Create a retriever with default settings
        retriever = bm25s.BM25()

        # Tokenize the corpus
        corpus_tokenized = tokenizer.tokenize(passages,
                                              return_as="tuple", # prevents the ids getting mixed up
                                              length=corpus_len, # needs to be passed explicitly with generator
                                              update_vocab=True)  # explicitly make sure that vocabulary is updated

        # Tokenize the queries
        query_tokenized = tokenizer.tokenize(queries,
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

    # Retrieve the top-k results with the non-naive retriever, return value being a matrix of document indices
    retrieved_documents = bm25s_retriever.retrieve(queries, k=top_k, return_as="documents")

    return retrieved_documents