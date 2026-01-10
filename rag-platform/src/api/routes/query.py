from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse

from langchain_chroma import Chroma
from vectorstore.chroma_client import get_chroma_client
from llm.prompt import PromptSetUp
from state.app_state import state

router = APIRouter(prefix="/query")

@router.post("", response_model=QueryResponse)
def query(req: QueryRequest):

    if state.llm is None:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    chroma_client = get_chroma_client()

    vector_db = Chroma(
        client=chroma_client,
        collection_name="documents",
        embedding_function=None
    )

    docs = vector_db.similarity_search(req.query, k=4)
    context = "\n".join(d.page_content for d in docs)

    prompter = PromptSetUp()
    prompt = prompter.generate_prompt(context=context, question=req.query)

    answer = state.llm(prompt)

    return {
        "query": req.query,
        "answer": answer
    }
