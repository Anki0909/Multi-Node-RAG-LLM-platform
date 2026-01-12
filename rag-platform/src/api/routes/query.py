from fastapi import APIRouter, HTTPException
import os, requests

from embeddings.embedder import TextEmbedder
from llm.prompt import PromptSetUp

router = APIRouter(prefix="/query")

RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://rag-inference")

@router.post("")
def query(question: str):
    embedder = TextEmbedder()
    vector_db = embedder.get_vector_store()

    docs = vector_db.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = PromptSetUp().generate_prompt()

    resp = requests.post(
        f"{RAG_BACKEND_URL}/generate",
        json={
            "prompt": prompt,
            "context": context,
            "question": question
        },
        timeout=120
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Inference failed")

    return resp.json()
