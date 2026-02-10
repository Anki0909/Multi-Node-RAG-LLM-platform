import time
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from schemas import InferenceRequest, InferenceResponse
from router import run_inference#, build_prompt
from gpu_client import generate_completion

from rag.loader import load_documents
from rag.chunker import TextChunker
from rag.vector_store import VectorStore
from rag.embedder import embed_texts, Embedder
from rag.retriever import Retriever

EMBEDDING_DIM = 384
MAX_PROMPT_CHARS = 6000

app = FastAPI(
    title="CPU Inference Router",
    version="2.1",
    description="Phase-3 CPU → GPU RAG Orchestrator",
)

chunker = TextChunker(chunk_size=512, overlap=64)
embedder = Embedder(dim=EMBEDDING_DIM)
vector_store = VectorStore(dim=EMBEDDING_DIM)
retriever = Retriever(embedder, vector_store)
@app.get("/health")
def health():
    return {"status": "ok", "device": "cpu-router"}

@app.post("/documents/load")
def load_docs():
    docs = load_documents(
        "/home/ankur/Documents/Multi-Node-RAG-LLM-platform/data/documents"
    )

    if not docs:
        raise HTTPException(status_code=400, detail="No documents found")

    chunks = []
    for doc in docs:
        chunks.append(chunker.chunk(doc))

    embeddings = embedder.embed(chunks)
    
    vector_store.add(embeddings, docs)

    return {
        "status": "ok",
        "documents_loaded": len(docs),
        "number_of_chunks": len(chunks)
    }

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):

    retrieved_chunks = []

    final_prompt = req.prompt

    if req.use_rag:
        retrieved_chunks = retriever.retrieve(req.prompt, top_k=req.top_k)
        context_block = "\n".join(retrieved_chunks)

        final_prompt = f"""
            You are an assistant.

            Use the context below to answer the question.

            Context:
            {context_block}

            Question:
            {req.prompt}
            """.strip()

    if len(final_prompt) > MAX_PROMPT_CHARS:
        final_prompt = final_prompt[:MAX_PROMPT_CHARS]

    gpu_result = generate_completion(final_prompt, max_tokens=256)

    response_text = gpu_result["text"]
    model_name = gpu_result["model"]

    return {
        "prompt": req.prompt,
        "response": response_text,
        "model": model_name,
        "device": "gpu",
        "latency_ms": gpu_result["latency_ms"],
        "exit_code": gpu_result["exit_code"],
        "use_rag": req.use_rag,
        "retrieved_chunks": retrieved_chunks,
        "error": gpu_result["error"],
    }
