from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.routes import ingest, query, health
from state.app_state import state
from llm.model import LLM
from embeddings.embedder import TextEmbedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.llm is None:
        print("Loading LLM model...")
        state.llm = LLM().llm_model
        print("LLM model loaded")

    if state.embedder is None:
        print("Loading embedding model...")
        state.embedder = TextEmbedder()
        print("Embedding model loaded")

    yield

    print("Embedding model loaded")

app = FastAPI(
    title="RAG Platform",
    lifespan=lifespan
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)