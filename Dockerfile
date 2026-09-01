FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_NO_TF=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    CHROMA_DIR=/app/chroma_db \
    OLLAMA_CLOUD_HOST=https://ollama.com \
    OLLAMA_MODEL=gemma4:31b-cloud

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app.py rag.py index.html ./

RUN mkdir -p /app/chroma_db && chmod -R 777 /app/chroma_db

EXPOSE 8000

CMD ["sh", "-c", "mkdir -p /app/chroma_db && chmod -R 777 /app/chroma_db 2>/dev/null || true; uvicorn app:app --host 0.0.0.0 --port 8000"]
