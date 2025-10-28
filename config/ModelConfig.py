from abc import ABC

class ModelConfig(ABC):
    def __init__(self,
                 model_name: str,
                 batch_size: int = -1,
                 max_tokens_per_batch: int = -1,
                 default_prompt: str = "",
                 ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.default_prompt = default_prompt

    # Return the attributes of the object as a string
    def __str__(self):
        attributes = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"

class QwenConfig(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "Qwen/Qwen3-Embedding-8B",
            max_tokens_per_batch = 65000,
            default_prompt = "Retrieval the relevant passage for the given query", # from https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json
        )

class E5Config(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "intfloat/multilingual-e5-large-instruct",
            max_tokens_per_batch = 500000,
            default_prompt = "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper", # closest to the task at hand, from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
        )

class BertConfig(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "TurkuNLP/bert-base-finnish-cased-v1",
            batch_size = 1024
        )

class XlmConfig(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
            batch_size = 512
        )

class BM25Config(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "bm25s"
        )
        