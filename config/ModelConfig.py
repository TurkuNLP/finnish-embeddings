from abc import ABC

class ModelConfig(ABC):
    def __init__(self,
                 model_name: str,
                 batch_size: int = -1,
                 max_tokens_per_batch: int = -1,
                 default_prompt: str = "",
                 best_prompt: str = "",
                 short_name: str = None
                 ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.default_prompt = default_prompt
        self.best_prompt = best_prompt
        self.short_name = short_name

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
            best_prompt = "Hae oikea artikkeli, joka kuuluu seuraavalle uutisotsikolle",
            short_name = "qwen"
        )

class E5Config(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "intfloat/multilingual-e5-large-instruct",
            max_tokens_per_batch = 500000,
            default_prompt = "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper", # closest to the task at hand, from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
            best_prompt = "Retrieve text based on user query.",
            short_name = "e5"
        )

class BertConfig(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "TurkuNLP/bert-base-finnish-cased-v1",
            batch_size = 1024,
            short_name = "bert"
        )

class XlmConfig(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
            batch_size = 512,
            short_name = "xlm"
        )

class BM25Config(ModelConfig):
    def __init__(self):
        super().__init__(
            model_name = "bm25s",
            short_name = self.model_name
        )
        