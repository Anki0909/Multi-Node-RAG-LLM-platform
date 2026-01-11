from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class TextEmbedder:
    def __init__(self):
        print("Initializing embedder")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small"
        )

        self.vector_db = Chroma(
            persist_directory="/data/vector-db",
            embedding_function=self.embedding_model
        )

        print("Embedder ready")

    def add_documents(self, documents):
        self.vector_db.add_documents(documents)
        return self.vector_db