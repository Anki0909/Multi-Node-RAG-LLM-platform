from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def build(self, documents):
        documents = [d for d in documents if isinstance(d, str) and len(d.strip()) > 0]

        if len(documents) == 0:
            raise ValueError("No valid documents found for embeddings")

        embeddings = model.encode(documents)

        if len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.texts = documents

    def search(self, query, k=5, rerank_top=2):
        # FAISS retrieval
        q_emb = model.encode([query])
        distances, indices = self.index.search(q_emb, k)

        candidates = [self.texts[i] for i in indices[0]]

        # Cross-encoder reranking
        pairs = [[query, c] for c in candidates]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # return BEST chunks
        return [x[0] for x in ranked[:rerank_top]]

    def keyword_search(self, query, k=3):
        query_words = set(query.lower().split())
        scored = []

        for text in self.texts:
            score = sum(1 for w in query_words if w in text.lower())
            if score > 0:
                scored.append((text, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [t for t, _ in scored[:k]]