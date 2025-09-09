import faiss

def query(index:faiss.IndexFlatL2, query_embeddings, k_nearest:int):
    return index.search(query_embeddings, k_nearest)