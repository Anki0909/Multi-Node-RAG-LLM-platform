from fastapi import APIRouter, HTTPException
import os

from ingestion.loader import FileLoader
from ingestion.chunker import TextChunker
# from llm.prompt import PromptSetUp
# from retrieval.manual_rag import ManualRAG
from state.app_state import state

router = APIRouter(prefix="/ingest")

@router.post("")
def ingest(file_path: str):
    # if state.llm is None or state.embedder is None:
        # raise HTTPException(status_code=500, detail="Models not initialized")

    PDF_BASE_PATH = os.getenv("PDF_BASE_PATH", "/data/pdfs")
    pdf_full_path = os.path.join(PDF_BASE_PATH, file_path)

    file_parser = FileLoader(pdf_full_path)
    texts = file_parser.read_file()["file_content"]

    text_chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    text_chunks = text_chunker.chunk(texts)

    vector_db = state.embedder.create_embedding(text_chunks)

    vector_db.save_local("/data/vector-db")
    # prompter = PromptSetUp()
    # prompt = prompter.generate_prompt()

    # state.vector_db = vector_db
    # state.rag = ManualRAG(state.llm, vector_db, prompt)

    return {"status": "ingested", "chunks": len(text_chunks)}
