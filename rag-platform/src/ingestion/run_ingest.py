import os
from ingestion.loader import FileLoader
from ingestion.chunker import TextChunker
from embeddings.embedder import TextEmbedder

PDF = os.path.join(os.getenv("PDF_BASE_PATH"), "test.pdf")

print(f"📄 Loading PDF: {PDF}")
text = FileLoader(PDF).read_file()["file_content"]

print("✂️ Chunking text")
chunks = TextChunker(500, 100).chunk(text)

print(f"🧠 Creating embeddings for {len(chunks)} chunks")
embedder = TextEmbedder(
    model_path=os.getenv("HF_MODEL_PATH"),
    persist_dir=os.getenv("VECTOR_DB_PATH"),
)

embedder.add_documents(chunks)

print("✅ Ingestion complete")
