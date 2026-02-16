from fastapi import FastAPI
import requests
import os

from rag.store import VectorStore
from rag.loader import load_documents
from rag.retriever import hybrid_retrieve
from rag.prompt import build_prompt

GPU_LLM_ENDPOINT = os.getenv("GPU_LLM_ENDPOINT")

app = FastAPI()
store = VectorStore()

@app.on_event("startup")
def startup():
    docs = load_documents()
    store.build(docs)

from pydantic import BaseModel

class InferRequest(BaseModel):
    query: str

@app.post("/infer")
def infer(req: InferRequest):
    query = req.query

    contexts = hybrid_retrieve(store, query)

    context_block = "\n---\n".join(contexts)

    prompt = build_prompt(context_block, query)

    payload = {
        "model":"qwen2.5-3b-instruct-q4_k_m.gguf",
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9
    }

    print("PROMPT SIZE =", len(prompt))

    print("DEBUG PAYLOAD =", payload)

    resp = requests.post(GPU_LLM_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()

    return resp.json()


@app.get("/health")
def health():
    return {"status": "ok"}
