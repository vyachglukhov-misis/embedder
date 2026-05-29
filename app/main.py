from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model import get_model, _model as _loaded_model
from app.schemas import EmbedRequest, EmbedResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Стартап: прогреваем модель, чтобы первый запрос не ждал загрузки весов.
    get_model()
    yield
    # Шатдаун: ничего особенного — PyTorch освобождает VRAM при завершении процесса.


app = FastAPI(
    title="Embedder",
    description="FastAPI-сервис для получения эмбеддингов через BAAI/bge-multilingual-gemma2",
    version="2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    from app import model as m
    return {"status": "ok", "model_loaded": m._model is not None}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    model = get_model()

    encode_kwargs: dict = {
        "normalize_embeddings": True,  # L2-нормализация для cosine similarity
        "batch_size": 32,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }

    # prompt_name='query' активирует query-режим bge-gemma2 (добавляет prefix-промпт).
    # Для индексации чанков prompt_name=None — passages эмбеддятся без промпта.
    if req.prompt_name:
        encode_kwargs["prompt_name"] = req.prompt_name

    embeddings = model.encode(req.texts, **encode_kwargs)

    return EmbedResponse(
        dense=embeddings.tolist(),
        dim=embeddings.shape[1],
    )
