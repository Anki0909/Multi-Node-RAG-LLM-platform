from fastapi import FastAPI
import requests
import os

from rag.store import VectorStore
from rag.loader import load_documents
from rag.chunker import TextChunker
from rag.adaptive_retriever import adaptive_retrieve
from rag.prompt import build_prompt
from rag.query_expander import QueryExpander
from rag.compressor import ContextCompressor

GPU_LLM_ENDPOINT = os.getenv("GPU_LLM_ENDPOINT")

chunker = TextChunker()
app = FastAPI()
store = VectorStore()
expander = QueryExpander()
compressor = ContextCompressor()

@app.on_event("startup")
def startup():
    docs = load_documents()

    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk(d))

    store.build(chunks)

from pydantic import BaseModel

class InferRequest(BaseModel):
    query: str

@app.post("/infer")
def infer(req: InferRequest):
    query = req.query

    expanded_queries = expander.expand(query)

    all_contexts = []

    for q in expanded_queries:
        all_contexts.extend(adaptive_retrieve(store, q))

    compressed_context = compressor.compress(query, all_contexts)

    prompt = build_prompt(compressed_context, query)

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
