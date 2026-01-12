from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

app = FastAPI(
    title="RAG Platform – API",
)

from api.routes import ingest, query, health
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)