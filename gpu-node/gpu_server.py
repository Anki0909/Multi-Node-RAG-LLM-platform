import subprocess
import tempfile
import time
from fastapi import FastAPI
from pydantic import BaseModel

LLAMA_BIN = "/usr/local/bin/llama"
MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/infer")
def infer(req: InferenceRequest):
    start = time.time()

    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        f.write(req.prompt)
        f.flush()

        cmd = [
            LLAMA_BIN,
            "-m", MODEL_PATH,
            "--file", f.name,
            "-n", str(req.max_tokens),
            "--temp", str(req.temperature),
            "--n-gpu-layers", "24",
            "--ctx-size", "2048",
            "--batch-size", "128",
            "--n-gpu-layers", "20"
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    return {
        "text": proc.stdout,
        "latency_ms": int((time.time() - start) * 1000),
        "exit_code": proc.returncode,
        "stderr": proc.stderr
    }
