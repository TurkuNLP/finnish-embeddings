from config.Config import Config
from config.init_argument_parser import init_argument_parser
from argparse import Namespace
import pytest

def test_config_fails_without_model_name():
    with pytest.raises(TypeError):
        Config()

def test_model_name_renaming(monkeypatch):

    monkeypatch.setenv("EMBEDDING_DIR", "path/to/dir")
    model_name = "some/awesome/model"
    config = Config(model_name)

    assert config.model_name == model_name
    assert config.save_embeddings_to == "path/to/dir/some__awesome__model_embeddings.npy"

def test_parse_config():

    default_query_key = "title"
    custom_passage_key = "text_beginning"

    args = Namespace(
        model_name="some-awesome-model",
        passage_key=custom_passage_key
    )

    config = Config.parse_config(args)
    
    assert config.model_name == "some-awesome-model"
    assert config.passage_key == custom_passage_key
    assert config.query_key == default_query_key

def test_parse_config_fails_without_model_name():

    args = Namespace()
    with pytest.raises(TypeError):
        Config.parse_config(args)

def test_config_with_test(monkeypatch):

    monkeypatch.setenv("TEST_DIR", "path/to/test/dir")
    model_name = "some/awesome/model"

    parser = init_argument_parser()
    args = parser.parse_args([
        model_name,
        "--test"
    ])
    config = Config.parse_config(args)

    assert config.model_name == model_name
    assert config.save_embeddings_to == "path/to/test/dir/some__awesome__model_embeddings_test.npy"