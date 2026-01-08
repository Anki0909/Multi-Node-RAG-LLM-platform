# Converts text chunks into embedding
import os
from time import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

HF_MODEL_PATH = "/data/hf-models/gte-small"

class TextEmbedder:
    def __init__(self):
        if not os.path.exists(HF_MODEL_PATH):
            raise RuntimeError(f"HF model not found at {HF_MODEL_PATH}")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=HF_MODEL_PATH,
            cache_folder=HF_MODEL_PATH
        )

    def create_embedding(self, texts):
        print("INFO: Creating embeddings...")
        start = time()
        db = FAISS.from_documents(texts, self.embedding_model)
        print(f"INFO: Text embedding completed in {time() - start} seconds.")
        return db