# Embedder (FastAPI · bge-multilingual-gemma2)

Сервис эмбеддингов для гибридного RAG-поиска. Реализован на **FastAPI** +
**sentence-transformers**, инференс через PyTorch + CUDA. Заменил предыдущую
реализацию на TEI (Text Embeddings Inference) — возвращаем кастомный
Python-сервис для полного контроля над API и нарративом главы 4 диплома.

## Модель

**BAAI/bge-multilingual-gemma2** — 9B параметров, бэкбон Gemma2, мультиязычный.

| Параметр            | Значение                        |
| ------------------- | ------------------------------- |
| Размерность вектора | **3584**                        |
| Контекст            | 8192 токенов                    |
| VRAM в fp16         | ~18 ГБ                          |
| Параметров          | 9B                              |
| Целевая видеокарта  | NVIDIA Tesla A10 (24 ГБ, sm_86) |

## Запуск на VM

```bash
cd ~/embedder

# Первый запуск — собрать образ и скачать веса (~18 ГБ в ~/models-cache)
docker compose up -d --build

# Последующие запуски (веса из кэша, ~1-2 мин)
docker compose up -d

# Логи
docker compose logs -f
```

Сервис готов, когда `/health` возвращает `{"status":"ok","model_loaded":true}`.

## Доступ с Mac

`~/.ssh/config` (без изменений):

```
Host 2diplom
    HostName 185.182.108.158
    User root
    Port 22
    IdentityFile ~/.ssh/diplomas2/id_ed25519
    LocalForward 8080 127.0.0.1:8080
    LocalForward 8081 127.0.0.1:8081
    ServerAliveInterval 30
    ServerAliveCountMax 10
```

Туннель: `autossh -M 0 -N 2diplom`

## Проверка с Mac

```bash
# Healthcheck
curl http://localhost:8080/health
# → {"status":"ok","model_loaded":true}

# Эмбеддинг passage (без prompt_name)
curl -sX POST http://localhost:8080/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["function pLimit(concurrency) { /* ... */ }"]}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print(f"dim={r[\"dim\"]}, first 3: {r[\"dense\"][0][:3]}")'
# → dim=3584, first 3: [...]

# Эмбеддинг поискового запроса (query-режим)
curl -sX POST http://localhost:8080/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["limit concurrent async operations"], "prompt_name":"query"}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print(f"dim={r[\"dim\"]}")'
```

## API

```
POST /embed
  Body: {
    "texts": ["text1", "text2", ...],
    "prompt_name": null | "query"
  }
  → {
    "dense": [[3584 float], ...],
    "dim": 3584
  }

GET /health
  → {"status": "ok", "model_loaded": true|false}
```

`prompt_name="query"` активирует query-режим bge-multilingual-gemma2 (prefix-промпт
для retrieval). Для индексации чанков `prompt_name` не передаётся.

## Структура

```
embedder/
├── app/
│   ├── __init__.py
│   ├── main.py       # FastAPI-приложение, /embed, /health, lifespan
│   ├── model.py      # ленивая загрузка SentenceTransformer + .half()
│   └── schemas.py    # EmbedRequest / EmbedResponse (Pydantic)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── legacy/
    └── python_service/   # старая реализация на FlagEmbedding (архив)
```

## Известные проблемы

- **`dtype` kwarg в XLM-RoBERTa** — обойдён загрузкой в дефолтной precision с
  последующим `.half()` (не через `torch_dtype` в конструкторе).
- **`@app.on_event` deprecated** — используем `lifespan` контекст-менеджер.
- **Долгая загрузка при первом старте** — gemma2 ~18 ГБ, healthcheck настроен
  с `start_period: 180s`.
