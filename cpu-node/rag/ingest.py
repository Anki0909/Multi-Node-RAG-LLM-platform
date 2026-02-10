# from loader import load_documents
# from embedder import embed_texts
# from vector_store import VectorStore

# docs = load_documents("data/documents")
# embeddings = embed_texts(docs)

# store = VectorStore(dim=embeddings.shape[1])
# store.add(embeddings, docs)

# print(f"Ingested {len(docs)} documents")

def ingest(docs, chunker, embedder, store, reset=False):
    if reset:
        store.index.reset()
        store.metadata = []

    for doc in docs:
        chunks = chunker.chunk(doc["text"])
        embeddings = embedder.embed(chunks)
        store.add(embeddings, chunks)
