from fastapi import APIRouter, HTTPException
import os

from ingestion.loader import FileLoader
from ingestion.chunker import TextChunker
from embeddings.embedder import TextEmbedder

router = APIRouter(prefix="/ingest")

@router.post("")
def ingest(file_path: str):
    PDF_BASE_PATH = os.getenv("PDF_BASE_PATH", "/data/pdfs")
    pdf_full_path = os.path.join(PDF_BASE_PATH, file_path)

    if not os.path.exists(pdf_full_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    loader = FileLoader(pdf_full_path)
    texts = loader.read_file()["file_content"]

    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk(texts)

    embedder = TextEmbedder()
    embedder.add_documents(chunks)

    return {
        "status": "ingested",
        "chunks": len(chunks)
    }