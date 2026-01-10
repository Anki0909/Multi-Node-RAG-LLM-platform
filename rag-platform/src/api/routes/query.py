from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse

from llm.model import LLM
from llm.prompt import PromptSetUp
from retrieval.manual_rag import ManualRAG
from state.app_state import state
from vectorstore.loader import load_vector_db

router = APIRouter(prefix="/query")

@router.post("", response_model=QueryResponse)
def query(req: QueryRequest):
    if state.llm is None:
        state.llm = LLM().llm_model

    vector_db = load_vector_db()
    if vector_db is None:
        raise HTTPException(status_code=400, detail="No document ingested")
    
    prompter = PromptSetUp()
    prompt = prompter.generate_prompt()

    rag = ManualRAG(
        llm = state.llm, 
        vector_db = vector_db,
        prompt = prompt
    )

    answer = rag.generate(req.query)
    return {"query": req.query, "answer": answer}