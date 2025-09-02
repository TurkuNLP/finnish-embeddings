from src.evaluate import evaluate
import numpy as np

def test_evaluate_with_perfect_recall():

    test_result_matrix = np.array([[0, 1, 2, 3, 4],
                                   [1, 2, 3, 4, 0],
                                   [2, 3, 4, 0, 1],
                                   ])
    
    top_k_list = [1, 5]

    query_indices = [0, 1, 2]

    evaluation = evaluate(test_result_matrix, top_k_list, query_indices)
    assert evaluation == {"recall_at_1": 100, "recall_at_5": 100}

def test_evaluate_with_not_perfect_recall():
    test_result_matrix = np.array([[0, 1, 2, 3, 4],
                                   [1, 2, 3, 4, 0],
                                   [2, 3, 4, 0, 5],
                                   ])
    
    top_k_list = [1, 5]

    query_indices = [0, 2, 1] # if the matrix had perfect recall, the first column should be equal to this list

    expected_recall_at_1 = (1/3)*100 # only index at [0][0] in test_result_matrix is correct
    expected_recall_at_5 = (2/3)*100 # only rows 0 and 1 include the correct index

    evaluation = evaluate(test_result_matrix, top_k_list, query_indices)
    assert evaluation == {"recall_at_1": expected_recall_at_1, "recall_at_5": expected_recall_at_5}
