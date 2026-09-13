FROM python:3.11-slim

WORKDIR /app

# Слой с зависимостями отдельно, чтобы не пересобирать его на каждое изменение кода.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag ./rag
COPY scripts ./scripts
COPY knowledge_base ./knowledge_base
COPY terms_map.json ./

# Кеш моделей выносим в отдельную папку, её монтируем томом.
ENV HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
