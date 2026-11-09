FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/app/data CACHE_DIR=/app/cache
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY legal_rag ./legal_rag
COPY static ./static
COPY data/knowledge-base ./data/knowledge-base
COPY server.py solution.py ./
COPY cache ./cache

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
