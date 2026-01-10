from fastapi import FastAPI
from contextlib import asynccontextmanager
from llm.model import LLM
from state.app_state import state

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading LLM model...")
    state.llm = LLM().llm_model
    print("LLM loaded")
    yield

app = FastAPI(
    title="RAG Inference Service",
    lifespan=lifespan
)

from api.routes import query, health
app.include_router(health.router)
app.include_router(query.router)
