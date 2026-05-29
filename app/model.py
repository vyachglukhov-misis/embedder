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

    Загружаем сразу в fp16 через model_kwargs — это критично для VRAM:
    fp32 потребовал бы ~36 ГБ при загрузке, fp16 — ~18 ГБ.
    Для Gemma2 передача torch_dtype безопасна (баг с dtype был только
    у XLM-RoBERTa в реранкере).
    """
    global _model
    if _model is None:
        print("Загружаем bge-multilingual-gemma2 на GPU (fp16)...")
        _model = SentenceTransformer(
            "BAAI/bge-multilingual-gemma2",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        dim = _model.get_sentence_embedding_dimension()
        print(f"Модель готова: dim={dim}")
    return _model
