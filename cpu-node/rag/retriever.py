from .embedder import embed_texts
from .vector_store import VectorStore

class Retriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k=3):
        query_emb = self.embedder.embed([query])
        return self.vector_store.search(query_emb, top_k)