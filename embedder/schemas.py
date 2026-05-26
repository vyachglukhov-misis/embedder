from pydantic import BaseModel

class EmbedRequest(BaseModel):
    texts: list[str]
    return_sparse: bool = True

class EmbedResponse(BaseModel):
    dense: list[list[float]]
    sparse: list[dict[str, float]] | None = None

# --- Reranker ---

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    normalize: bool = True   # scores в диапазоне [0, 1]

class RerankResponse(BaseModel):
    scores: list[float]