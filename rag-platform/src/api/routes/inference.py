from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/generate")

_llm = None

def set_llm(llm):
    global _llm
    _llm = llm

class GenerateRequest(BaseModel):
    prompt: str
    context: str
    question: str

class GenerateResponse(BaseModel):
    answer: str

@router.post("", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    full_prompt = f"""
{req.prompt}

Context:
{req.context}

Question:
{req.question}

Answer:
"""
    answer = _llm(full_prompt)
    return {"answer": answer}
