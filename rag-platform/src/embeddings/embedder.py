# Converts text chunks into embedding

from time import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class TextEmbedder:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = 'thenlper/gte-small'
        )

    def create_embedding(self, texts):
        print("INFO: Creating embeddings...")
        start = time()
        db = FAISS.from_documents(texts, self.embedding_model)
        print(f"INFO: Text embedding completed in {time() - start} seconds.")
        return db