from src.utils.helpers import yield_from_jsonl, do_batching
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