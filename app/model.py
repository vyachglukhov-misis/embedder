import os
import torch
from sentence_transformers import SentenceTransformer

# Кэш HF — направляем на смонтированный том /data
CACHE_DIR = os.getenv("HF_HOME", "/data")
os.environ["HF_HOME"] = CACHE_DIR

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Ленивая загрузка модели bge-multilingual-gemma2.

    Загружаем в дефолтной precision, потом .half() — это обходит баг
    transformers с передачей dtype в __init__ старых моделей (мы на нём
    уже горели с FlagEmbedding/GemmaReranker).
    """
    global _model
    if _model is None:
        print("Загружаем bge-multilingual-gemma2 на GPU...")
        _model = SentenceTransformer(
            "BAAI/bge-multilingual-gemma2",
            device="cuda",
        )
        _model.half()  # fp16 после загрузки — не через torch_dtype в __init__
        dim = _model.get_sentence_embedding_dimension()
        print(f"Модель готова: dim={dim}")
    return _model
