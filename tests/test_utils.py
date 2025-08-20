from src.utils.helpers import yield_from_jsonl, get_data_as_dict, do_batching
from dotenv import load_dotenv
import os
import pytest

load_dotenv("tests/.env.test")

path_to_mock_data = os.getenv("MOCK_DATA")

def test_yield_from_jsonl():
    
    documents = list(yield_from_jsonl(path_to_mock_data, "text_end"))
    assert len(documents) == 10

    titles = list(yield_from_jsonl(path_to_mock_data, "title"))
    assert len(titles) == 10

    k_values = [1, 3, 5]

    for k in k_values:
        k_titles = list(yield_from_jsonl(path_to_mock_data, "title", k))
        assert len(k_titles) == k
    
    i_values = [0, 9]

    i_titles = list(yield_from_jsonl(path_to_mock_data, "title", indices=i_values))
    assert len(i_titles) == 2

def test_get_data_as_dict():

    indices = [0, 2, 4]
    data_dict = get_data_as_dict(path_to_mock_data, indices=indices)

    assert len(data_dict) == len(indices)
    assert 2 in data_dict
    assert 1 not in data_dict

def test_batch_process():

    documents = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

    assert len(list(do_batching(documents, 5))) == 2
    assert len(list(do_batching(documents, 5))[0]) == 5
    assert len(list(do_batching(documents, 11))) == 1
    with pytest.raises(ValueError):
        list(do_batching(documents, 3, strict=True))
    with pytest.raises(ValueError):
        list(do_batching(documents, 0))

    def yield_documents():
        for doc in documents:
            yield doc
    
    assert len(list(do_batching(yield_documents(), 2))) == 5
    assert list(do_batching(yield_documents(), 2))[0] == ["one", "two"]

def test_batch_counter():
    def batch_counter(num_documents:int, batch_size:int):
        return num_documents // batch_size + 1 if num_documents % batch_size != 0 else num_documents // batch_size
    
    assert batch_counter(10, 2) == 5
    assert batch_counter(11, 2) == 6
    assert batch_counter(128, 32) == 4