from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse

from llm.model import LLM
from llm.prompt import PromptSetUp
from retrieval.manual_rag import ManualRAG
from embeddings.embedder import TextEmbedder
from state.app_state import state
from vectorstore.loader import load_vector_db

router = APIRouter(prefix="/query")



@router.post("", response_model=QueryResponse)
def query(req: QueryRequest):

    embedder = TextEmbedder()
    vector_db = embedder.get_vector_store()

    prompter = PromptSetUp()
    prompt = prompter.generate_prompt()

    llm = LLM().llm_model

    rag = ManualRAG(llm, vector_db, prompt)

    answer = rag.generate(req.query)

    return {"query": req.query, "answer": answer}
