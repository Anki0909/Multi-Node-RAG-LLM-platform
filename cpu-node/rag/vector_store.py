import faiss
import os
import pickle
import numpy as np

STORE_PATH = "data/vectorstore"
INDEX_FILE = f"{STORE_PATH}/index.faiss"
META_FILE = f"{STORE_PATH}/meta.pkl"

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        os.makedirs(STORE_PATH, exist_ok=True)

        if os.path.exists(INDEX_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            self.metadata = pickle.load(open(META_FILE, "rb"))
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []

    def add(self, embeddings, texts):
        self.index.add(embeddings)
        self.metadata.extend(texts)
        self.persist()

    def search(self, query_embeddings, k=3):
        if self.index.ntotal == 0:
            return []

        query_embeddings = np.asarray(query_embeddings, dtype="float32").reshape(1, -1)
        scores, idxs = self.index.search(query_embeddings, k)

        results = []
        for i in idxs[0]:
            if 0 <= i < len(self.metadata):
                results.append(self.metadata[i])
        return results
    
    def persist(self):
        faiss.write_index(self.index, INDEX_FILE)
        pickle.dump(self.metadata, open(META_FILE, "wb"))