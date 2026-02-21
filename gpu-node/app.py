import subprocess
import time
from fastapi import FastAPI
from pydantic import BaseModel

LLAMA_BIN = "/app/llama.cpp/build/bin/llama"
MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"

app = FastAPI(title="GPU Inference Service")

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    text: str
    tokens: int
    latency_ms: int

@app.get("/health")
def health():
    return {'status': 'ok'}

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    start = time.time()

    cmd = [
        LLAMA_BIN,
        "--model", MODEL_PATH,
        "-n", str(req.max_tokens),
        "--n-gpu-layers", "24",
        "--ctx-size", "2048",
        "--batch-size", "256",
        "--temp", str(req.temperature),
        "--no-interactive"
    ]

    proc = subprocess.run(
        cmd,
        input=req.prompt,
        text=True,
        capture_output=True
    )

    latency_ms = int((time.time() - start) * 1000)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    return InferenceResponse(
        text=proc.stdout.strip(),
        tokens=req.max_tokens,
        latency_ms=latency_ms
    )
