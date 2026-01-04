# Load and configure the LLM runtime
import os
from time import time
from langchain_community.llms import LlamaCpp

class LLM:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            raise RuntimeError("MODEL_PATH env variable not set")
        
        self.llm_model = self.load_llm_model(model_path)

    def load_llm_model(self, model_path):
        print("INFO: Loading LLM model...")
        start = time()
        llm_model = LlamaCpp(
            model_path = model_path,
            n_gpu_layers = -1,
            max_tokens = 150,
            n_ctx = 2048,
            temperature=0.0,
            verbose = False,
            stop=["\n\n", "Question:", "Context:"],
        )
        print(f"INFO: LLM model loaded in {time() - start} seconds")

        return llm_model
