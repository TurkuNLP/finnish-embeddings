import faiss

def query(index:faiss.IndexFlatIP, query_embeddings, k_nearest:int):
    """Normalize embeddings and query an IndexFlat IP.

    Parameters
    ----------
        index: faiss.IndexFlatIP
            The index to search.
        query_embeddings: ndarray
            The embeddings produced by an embedding model, not normalized.
        k_nearest: int
            How many nearest results to return per query.
    
    Returns
    -------
        similarities: ndarray
            A matrix with shape (n_queries, k_nearest) of cosine similarities of the returned documents.
        document_indices: ndarray
            A matrix with shape (n_queries, k_nearest) of indices of the returned documents.
    """
    faiss.normalize_L2(query_embeddings) # important to normalize the vectors
    return index.search(query_embeddings, k_nearest)