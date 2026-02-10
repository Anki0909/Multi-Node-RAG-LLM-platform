import requests
import time

GPU_COMPLETION_URL = "http://localhost:8081/completion"

class GPUInferenceError(Exception):
    pass


def generate_completion(prompt: str, max_tokens: int = 128):
    start = time.time()

    try:
        resp = requests.post(
            GPU_COMPLETION_URL,
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "ok": True,
            "text": data.get("content", ""),
            "model": data.get("model", "unknown"),
            "latency_ms": int((time.time() - start) * 1000),
            "exit_code": 0,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "text": "[GPU unavailable — serving degraded response]",
            "model": "gpu-unavailable",
            "latency_ms": int((time.time() - start) * 1000),
            "exit_code": 1,
            "error": str(e),
        }