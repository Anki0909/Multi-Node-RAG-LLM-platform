#!/bin/sh
set -e

echo "Starting RAG service.."

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found at $MODEL_PATH"
    exit 1
fi

exec uvicorn api.main:app \
    --host ${HOST} \
    --port ${PORT} \
    --workers 1