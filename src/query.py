import faiss

def query(index:faiss.IndexFlatL2, query_embeddings, k_nearest:int):
    faiss.normalize_L2(query_embeddings) # important to normalize the vectors
    return index.search(query_embeddings, k_nearest)