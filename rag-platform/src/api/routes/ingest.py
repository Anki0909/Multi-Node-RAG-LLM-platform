from fastapi import APIRouter, HTTPException
import os

from ingestion.loader import FileLoader
from ingestion.chunker import TextChunker
from embeddings.embedder import TextEmbedder
from state.app_state import state

router = APIRouter(prefix="/ingest")

@router.post("")
def ingest(file_path: str):
    PDF_BASE_PATH = os.getenv("PDF_BASE_PATH", "/data/pdfs")
    pdf_full_path = os.path.join(PDF_BASE_PATH, file_path)

    file_parser = FileLoader(pdf_full_path)
    texts = file_parser.read_file()["file_content"]

    text_chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    text_chunks = text_chunker.chunk(texts)

    embedder = TextEmbedder()
    vector_db = embedder.get_vector_store()

    vector_db.add_texts(text_chunks)
    vector_db.persist()

    return {"status": "ingested", "chunks": len(text_chunks)}
