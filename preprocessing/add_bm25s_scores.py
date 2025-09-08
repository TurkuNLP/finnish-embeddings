import pandas as pd
import nltk
import Stemmer
import bm25s
from tqdm import tqdm
import json

def add_bm25s_scores(news: pd.DataFrame,
                     save_as: str,
                     text_col: str = "text_end",
                     query_col: str = "title",
                     id_col: str = "url",
                     language: str = "finnish",
                     top_k: int = 2):

    corpus_len = len(news)

    def prepare_scores(naive=False):

        if naive:
            splitter = str.split
            stemmer = None
            stopwords = None
            k1 = 1.2 # to match the other implementation (by Jenna)

            tokenizer = bm25s.tokenization.Tokenizer(splitter=splitter, stemmer=stemmer, stopwords=stopwords)
            # Create a retrieve with modified k1 value
            retriever = bm25s.BM25(k1=k1)
        
        else:
            stemmer = Stemmer.Stemmer(language)
            stopwords = nltk.corpus.stopwords.words(language)

            tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer, stopwords=stopwords)
            # Create a retrieve with default settings
            retriever = bm25s.BM25()

        # Tokenize the corpus
        corpus_tokenized = tokenizer.tokenize((text for text in news[text_col]),
                                               return_as="tuple", # prevents the ids getting mixed up
                                               length=corpus_len, # needs to be passed explicitly with generator
                                               update_vocab=True)  # explicitly make sure that vocabulary is updated

        # Tokenize the queries
        query_tokenized = tokenizer.tokenize((query for query in news[query_col]),
                                              return_as="tuple",
                                              length=corpus_len,
                                              update_vocab=False) # explicitly make sure that vocabulary is not updated

        # Index the corpus
        retriever.index(corpus_tokenized)

        return query_tokenized.ids, retriever

    # Tokenize and index the articles
    queries, bm25s_retriever = prepare_scores()
    queries_naive, bm25s_retriever_naive = prepare_scores(naive=True)

    # Retrieve the top-k results with the non-naive retriever for recording the closest incorrect match
    documents, scores = bm25s_retriever.retrieve(queries, k=top_k)

    # Turn timestamp values back to string
    news["timestamp"] = news["timestamp"].apply(lambda x: x.isoformat() if pd.notna(x) else None)

    # Drop extra columns
    news = news.drop(columns=["text_len", "sents"])

    # Convert the DataFrame to a dictionary for writing to file
    news = news.to_dict(orient="records")

    # Record the number of correct retrievals
    correct_top_1 = 0

    # Extend the articles with a bm25s similarity score, and add ids and scores for closest incorrect matches
    with open(save_as, "w") as file:
        for i, (query, query_naive, document, score, article) in tqdm(enumerate(zip(queries, queries_naive, documents, scores, news)), desc="Adding the bm25s scores", total=corpus_len):
            
            # Add scores for title-text match within the current article
            article["score"] = float(bm25s_retriever.get_scores(query)[i]) # force to float to avoid 'TypeError: Object of type float32 is not JSON serializable'
            article["naive_score"] = float(bm25s_retriever_naive.get_scores(query_naive)[i])

            # Add scores for the closest incorrect match with the highest score
            retrieved_index, position = (document[1], 1) if i == document[0] else (document[0], 0)
            if position == 1:
                correct_top_1 += 1
            article["closest_match_url"] = news[retrieved_index][id_col] if score[position] != 0 else None
            article["closest_match_score"] = float(score[position])

            # Print the article into file with bm25s info
            print(json.dumps(article, ensure_ascii=False), file=file)

    print(f"There were {correct_top_1} ({correct_top_1/corpus_len:.1%}) articles where the correct pair had the highest bm25 score.")