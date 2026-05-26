import os
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel, FlagReranker

CACHE_DIR = Path("./model_cache")
CACHE_DIR.mkdir(exist_ok=True)
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

def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        print("Загружаем bge-reranker-v2-m3 на GPU...")
        _reranker = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=True,
            device='cuda',
        )
        print("Реранкер готов")
    return _reranker