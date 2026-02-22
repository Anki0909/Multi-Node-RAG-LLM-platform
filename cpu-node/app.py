from fastapi import FastAPI
import os

from rag.store import VectorStore
from rag.loader import load_documents
from rag.chunker import TextChunker
from rag.adaptive_retriever import adaptive_retrieve
from rag.query_expander import QueryExpander
from rag.compressor import ContextCompressor
from rag.pipeline import RAGPipeline

GPU_LLM_ENDPOINT = os.getenv("GPU_LLM_ENDPOINT")

app = FastAPI()
chunker = TextChunker()
store = VectorStore()
expander = QueryExpander()
compressor = ContextCompressor()

pipeline = RAGPipeline(
                store=store, 
                retriever=adaptive_retrieve, 
                expander=expander,
                compressor=compressor)

@app.on_event("startup")
def startup():
    try:
        docs = load_documents()

        chunks = []
        for d in docs:
            chunks.extend(chunker.chunk(d))

        store.build(chunks)
        print(f"[INFO] RAG store initialized with {len(chunks)} chunks")
    except Exception as e:
        print(f"[WARN] Startup indexing skipped: {e}")

from pydantic import BaseModel

class InferRequest(BaseModel):
    query: str

@app.post("/infer")
def infer(req: InferRequest):
    return pipeline.run(req.query)


@app.get("/health")
def health():
    return {"status": "ok"}
