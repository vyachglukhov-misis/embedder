from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., description="Список текстов для эмбеддинга")
    # bge-multilingual-gemma2 поддерживает встроенные prompts:
    #   "query"   — для пользовательских поисковых запросов (query-режим)
    #   None      — для passage/чанков (без дополнительного prefix-промпта)
    prompt_name: str | None = Field(
        default=None,
        description="Имя prompt из конфига модели. 'query' для запросов, None для чанков.",
    )


class EmbedResponse(BaseModel):
    dense: list[list[float]] = Field(..., description="Массив векторов размерности dim")
    dim: int = Field(..., description="Размерность вектора (3584 для bge-multilingual-gemma2)")
