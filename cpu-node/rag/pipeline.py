import os
import requests
from rag.prompt import build_prompt

GPU_LLM_ENDPOINT = os.getenv("GPU_LLM_ENDPOINT")

class RAGPipeline:
    def __init__(self, store, retriever, expander, compressor):
        self.store = store
        self.retriever = retriever
        self.expander = expander
        self.compressor = compressor


    def run(self, query):
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

        resp = requests.post(GPU_LLM_ENDPOINT, json=payload, timeout=60)
        resp.raise_for_status()

        return resp.json()