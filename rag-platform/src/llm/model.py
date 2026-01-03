# Load and configure the LLM runtime

from time import time
from langchain_community.llms import LlamaCpp

class LLM:
    def __init__(self, model_path):
        self.llm_model = self.load_llm_model(model_path)

    def load_llm_model(self, model_path):
        print("INFO: Loading LLM model...")
        start = time()
        llm_model = LlamaCpp(
            model_path = model_path,
            n_gpu_layers = -1,
            max_tokens = 200,
            n_ctx = 2048,
            seed = 42,
            verbose = False,
            stop=["\n\n", "Question:", "Context:"],
        )
        print(f"INFO: LLM model loaded in {time() - start} seconds")

        return llm_model
