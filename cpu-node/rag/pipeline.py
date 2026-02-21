import os
import requests
from fastapi import HTTPException
from rag.prompt import build_prompt
from shared import cache

GPU_LLM_ENDPOINT = os.getenv("GPU_LLM_ENDPOINT")

class RAGPipeline:
    def __init__(self, store, retriever, expander, compressor):
        self.store = store
        self.retriever = retriever
        self.expander = expander
        self.compressor = compressor


    def run(self, query):
        cached = cache.get(query)
        if cached is not None:
            return cached
        
        expanded_queries = self.expander.expand(query)

        all_context = []
        for q in expanded_queries:
            all_context.extend(self.retriever(self.store, q))

        compressed_context = self.compressor.compress(query, all_context)

        prompt = build_prompt(compressed_context, query)

        payload = {
            "model" : "qwen2.5-3b-instruct-q4_k_m.gguf",
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9
        }

        print("PROMPT SIZE = ", len(prompt))

        print("DEBUG PAYLOAD = ")
        for key, value in payload.items():
            print(key, ": ", value)

        try:
            resp = requests.post(GPU_LLM_ENDPOINT, json=payload, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            return {
                "error": "LLM unavailable",
                "details": str(e)
            }
        
        resp_json = resp.json()

        if "choices" not in resp_json:
            raise HTTPException(500, "Invalid LLM response")
        
        cache.set(query, resp_json)

        return resp_json
