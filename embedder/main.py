from fastapi import FastAPI
from embedder.schemas import EmbedRequest, EmbedResponse, RerankRequest, RerankResponse
from embedder.model import get_model, get_reranker

app = FastAPI()

@app.on_event("startup")
async def startup():
    get_model()
    get_reranker()

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    model = get_model()
    output = model.encode(
        request.texts,
        return_dense=True,
        return_sparse=request.return_sparse,
        batch_size=64,
    )
    sparse = None
    if request.return_sparse:
        sparse = [
            {k: float(v) for k, v in weights.items()}
            for weights in output['lexical_weights']
        ]
    return EmbedResponse(dense=output['dense_vecs'].tolist(), sparse=sparse)

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    reranker = get_reranker()
    pairs = [[request.query, doc] for doc in request.documents]
    scores = reranker.compute_score(pairs, normalize=request.normalize)
    # compute_score возвращает float если одна пара, иначе list
    if isinstance(scores, float):
        scores = [scores]
    return RerankResponse(scores=scores)

@app.get("/health")
async def health():
    return {"status": "ok"}