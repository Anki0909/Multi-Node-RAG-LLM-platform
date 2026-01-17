from fastapi import FastAPI
from llm.model import LLM
from api.routes import inference, health

app = FastAPI(title="RAG Inference Service")

llm = LLM().llm_model
inference.set_llm(llm)

app.include_router(health.router)
app.include_router(inference.router)
