from src.utils.get_data import yield_from_jsonl
from dotenv import load_dotenv
import os

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