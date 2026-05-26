# Базовый образ: Python 3.11 + PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9 (runtime).
# torch с поддержкой GPU уже внутри — не переустанавливаем (иначе CPU-версия с PyPI всё сломает).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# git нужен FlagEmbedding/huggingface_hub для подтягивания весов с HF.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — слой кэшируется, пока requirements.txt не меняется.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем код сервиса. Структура: <context>/embedder/{main,model,schemas}.py
COPY embedder/ ./embedder/

# Кэш моделей HuggingFace направляем на ПРИМОНТИРОВАННЫЙ том,
# а не внутрь образа — иначе образ распухнет до ~10 ГБ.
# model.py уважает эту переменную после правки.
ENV HF_HOME=/data/hf

EXPOSE 8000

CMD ["uvicorn", "embedder.main:app", "--host", "0.0.0.0", "--port", "8000"]
