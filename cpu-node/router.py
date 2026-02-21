import re
import requests

def _extract_model_response(raw_output: str) -> str:
    """
    Extract only the generated model text from llama.cpp output.
    Removes CUDA banners, ASCII art, spinners, prompts, stats.
    """

    # Remove ANSI control characters & spinner artifacts
    cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw_output)
    cleaned = re.sub(r"[|/\\\-]\b", "", cleaned)

    # Split on prompt echo ("> prompt")
    if ">" in cleaned:
        cleaned = cleaned.split(">", 1)[1]

    # Remove performance footer
    cleaned = re.sub(r"\[ Prompt:.*?\]", "", cleaned, flags=re.DOTALL)

    return cleaned.strip()

MODEL = "qwen2.5-3b-instruct-q4_k_m.gguf"

def run_inference(prompt: str):
    resp = requests.post(
        "http://gpu-node:9000/infer",
        json={
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7
        },
        timeout=120
    )
    return resp.json()
