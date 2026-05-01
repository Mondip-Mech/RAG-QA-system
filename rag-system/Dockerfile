FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

# System deps for PyMuPDF + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download embedding + reranker models so first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('BAAI/bge-reranker-base')"

COPY . .

# HuggingFace Spaces convention is port 7860
EXPOSE 7860 8000

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
