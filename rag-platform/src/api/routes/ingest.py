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

    file_parser = FileLoader(pdf_full_path)
    texts = file_parser.read_file()["file_content"]

    text_chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    text_chunks = text_chunker.chunk(texts)

    embedder = TextEmbedder()
    embedder.ingest(text_chunks)

    return {"status": "ingested", "chunks": len(text_chunks)}
