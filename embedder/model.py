import os
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel, FlagLLMReranker

CACHE_DIR = Path(os.getenv("HF_HOME", "./model_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

_model = None
_reranker = None

def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        print("Загружаем bge-m3 на GPU...")
        _model = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=True,
            device='cuda',
        )
        print("Модель готова")
    return _model

def get_reranker() -> FlagLLMReranker:
    global _reranker
    if _reranker is None:
        print("Загружаем bge-reranker-v2-gemma на GPU...")
        _reranker = FlagLLMReranker(
            'BAAI/bge-reranker-v2-gemma',
            use_fp16=True,
            max_length=1024,  # код длиннее текста — увеличили с дефолтных 512
        )
        print("Реранкер готов")
    return _reranker