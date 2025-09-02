import logging
from src.utils.helpers import save_to_json

logger = logging.getLogger(__name__)

def map_query_indices(query_indices:list):
    return {i: query_index for i, query_index in enumerate(query_indices)}

def count_recall(hits:int, total:int):
    return (hits / total) * 100
        
# Note! Half-hard-coded to count recall at 1 and 5
def evaluate(result_matrix, top_k_list:list[int], query_indices:list):

    index_map = map_query_indices(query_indices)

    assert result_matrix.shape[0] == len(index_map), f"The number of rows in result matrix ({result_matrix.shape[0]} is different than the number of indices in index_map ({len(index_map)}))"
    assert result_matrix.shape[1] <= max(top_k_list), f"Cannot get top-{max(top_k_list)} from top-{result_matrix.shape[1]} results"
    
    recall_evaluation_counts = {k: 0 for k in top_k_list}

    for i in range(len(index_map)):

        # Chek recall @ 1
        if result_matrix[i][0] == index_map[i]:
            recall_evaluation_counts[1] += 1 # hard-coded!
        
        # Check recall @ 5
        if index_map[i] in result_matrix[i][:5]:
            recall_evaluation_counts[5] += 1 # hard-coded!
    
    return {f"recall_at_{k}": count_recall(v, len(index_map)) for k, v in recall_evaluation_counts.items()}

def save_evaluation(result_matrix, top_k_list:list[int], query_indices:list, save_to:str):
    evaluation = evaluate(result_matrix, top_k_list, query_indices)
    save_to_json(save_to, evaluation)