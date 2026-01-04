from fastapi import APIRouter

from ingestion.loader import FileLoader
from ingestion.chunker import TextChunker
from embeddings.embedder import TextEmbedder
from llm.model import LLM
from llm.prompt import PromptSetUp
from retrieval.manual_rag import ManualRAG
from state.app_state import state

if state.llm is None:
    state.llm = LLM().llm_model

router = APIRouter(prefix="/ingest")

@router.post("")
def ingest(file_path: str):
    file_pharser = FileLoader(file_path)
    texts = file_pharser.read_file()['file_content']

    text_chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    text_chunks = text_chunker.chunk(texts)

    embedder = TextEmbedder()
    vector_db = embedder.create_embedding(text_chunks)

    # llm = LLM('/home/ankur/Documents/Multi-Node-RAG-LLM-platform/rag-platform/qwen2.5-3b-instruct-q4_k_m.gguf')
    # llm_model = llm.llm_model
    
    prompter = PromptSetUp()
    prompt = prompter.generate_prompt()

    state.vector_db = vector_db
    state.rag = ManualRAG(state.llm, vector_db, prompt)

    return {"status": "ingested", "chunks": len(text_chunks)}