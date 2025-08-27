from config.Config import Config
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
