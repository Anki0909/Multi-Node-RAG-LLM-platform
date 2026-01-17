from langchain_huggingface import HuggingFaceEmbeddings
import os

class TextEmbedder:
    def __init__(self):
        model_path = os.getenv(
            "HF_MODEL_PATH",
            "/data/hf-models/thenlper/gte-small"
        )

        print(f"🔹 Loading embedding model from: {model_path}")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"local_files_only": True}
        )

        print("Embedder ready")

    def add_documents(self, documents):
        from langchain_chroma import Chroma

        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=os.getenv("VECTOR_DB_PATH", "/data/vector-db"),
            collection_name="documents",
        )
        return vector_db
