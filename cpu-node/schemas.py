from pydantic import BaseModel
from typing import List, Optional

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128
    use_rag: bool = False
    top_k: int = 3


class InferenceResponse(BaseModel):
    prompt: str
    response: str
    model: str
    device: str
    latency_ms: int
    exit_code: int
    use_rag: bool
    retrieved_chunks: List[str]
    error: Optional[str] = None
