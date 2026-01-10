from fastapi import FastAPI
from contextlib import asynccontextmanager
from state.app_state import state
from llm.model import LLM
from embeddings.embedder import TextEmbedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    # state.llm = LLM().llm_model
    # state.embedder = TextEmbedder()
    yield
    # optional cleanup

app = FastAPI(lifespan=lifespan)

from api.routes import ingest, query, health
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
