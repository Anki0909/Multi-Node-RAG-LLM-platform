from fastapi import FastAPI
from contextlib import asynccontextmanager
from embeddings.embedder import TextEmbedder
from state.app_state import state

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    state.embedder = TextEmbedder()
    print("Embedder loaded")
    yield

app = FastAPI(
    title="RAG Ingestion Service",
    lifespan=lifespan
)

from api.routes import ingest, health
app.include_router(health.router)
app.include_router(ingest.router)
