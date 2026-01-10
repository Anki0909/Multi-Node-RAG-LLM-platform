from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from vectorstore.chroma_client import get_chroma_client
import os

class TextEmbedder:
    def __init__(self):
        model_path = os.getenv("HF_MODEL_PATH")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_path
        )

    def ingest(self, texts, collection_name="documents"):
        chroma_client = get_chroma_client()

        vectordb = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=self.embedding_model
        )

        vectordb.add_texts(texts)
        vectordb.persist()

        return vectordb