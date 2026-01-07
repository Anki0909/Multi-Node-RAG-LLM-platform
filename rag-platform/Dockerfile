FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser
WORKDIR /app
RUN chown -R appuser:appuser /app

COPY rag-platform/docker-requirement.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install \
    --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt

COPY rag-platform/src /app/src

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/models/model.gguf
ENV HOST=0.0.0.0
ENV PORT=8000

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]