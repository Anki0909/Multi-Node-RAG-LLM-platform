from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return np.array(_model.encode(texts, normalize_embeddings=True))

class Embedder:
    def __init__(self, dim=384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = np.random.rand(len(texts), self.dim)
        return np.asarray(embeddings, dtype="float32")