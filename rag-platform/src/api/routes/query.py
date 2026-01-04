from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse
from state.app_state import state

router = APIRouter(prefix="/query")

@router.post("", response_model=QueryResponse)
def query(req: QueryRequest):
    if not state.rag:
        raise HTTPException(status_code=400, detail="No document ingested")
    
    answer = state.rag.generate(req.query)
    return {"query": req.query, "answer": answer}