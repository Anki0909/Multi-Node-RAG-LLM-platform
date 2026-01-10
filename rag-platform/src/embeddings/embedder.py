from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

class TextEmbedder:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("HF_MODEL_PATH", "thenlper/gte-small")
        )

        self.persist_dir = os.getenv("VECTOR_DB_PATH", "/data/vector-db")

    def get_vector_store(self):
        return Chroma(
            collection_name="rag",
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir
        )