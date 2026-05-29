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

    torch_dtype=float16  — загружаем сразу в fp16, не в fp32 (иначе нужно ~36 ГБ).
    low_cpu_mem_usage=True — загружает слои сразу на GPU без промежуточного
                             CPU-копирования. Критично для больших моделей на
                             GPU с малым запасом VRAM (~22 ГБ из 24 ГБ).
    """
    global _model
    if _model is None:
        print("Загружаем bge-multilingual-gemma2 на GPU (fp16, low_cpu_mem_usage)...")
        _model = SentenceTransformer(
            "BAAI/bge-multilingual-gemma2",
            device="cuda",
            model_kwargs={
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            },
        )
        dim = _model.get_sentence_embedding_dimension()
        print(f"Модель готова: dim={dim}")
    return _model
