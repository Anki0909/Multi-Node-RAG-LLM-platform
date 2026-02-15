from fastapi import FastAPI
import requests
import os
import re

from rag.store import VectorStore
from rag.loader import load_documents

GPU_LLM_ENDPOINT = os.getenv("GPU_LLM_ENDPOINT")

app = FastAPI()
store = VectorStore()

def compress_context(contexts, query, max_sentences=5):
    """
    Simple keyword-based context compression.
    Keeps only most relevant sentences.
    """
    query_words = set(query.lower().split())
    scored_sentences = []

    for ctx in contexts:
        sentences = re.split(r'(?<=[.!?]) +', ctx)

        for s in sentences:
            score = sum(1 for w in query_words if w in s.lower())
            if score > 0:
                scored_sentences.append((s, score))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    return "\n".join([s for s, _ in scored_sentences[:max_sentences]])

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

    # contexts = "\n\n".join(store.search(query))
    semantic = store.search(query, k=3)
    keyword = store.keyword_search(query, k=3)

    retrieved = list(dict.fromkeys(semantic + keyword))
    raw_contexts = compress_context(retrieved, query)

    if not isinstance(raw_contexts, list):
        raw_contexts = [str(raw_contexts)]

    contexts = "\n---\n".join([
        str(c).replace("\x00", " ").strip()
        for c in raw_contexts
    ])[:4000]


    augmented_prompt = f"""
        You are a precise technical assistant.

        Answer the question ONLY using the context.
        If possible, answer in 2-4 concise sentences.

        Context:
        {contexts}

        Question:
        {query}

        Answer:
        """.strip()

    payload = {
        "model":"qwen2.5-3b-instruct-q4_k_m.gguf",
        "prompt": augmented_prompt,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9
    }

    print("PROMPT SIZE =", len(augmented_prompt))

    print("DEBUG PAYLOAD =", payload)

    resp = requests.post(GPU_LLM_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()

    return resp.json()


@app.get("/health")
def health():
    return {"status": "ok"}
