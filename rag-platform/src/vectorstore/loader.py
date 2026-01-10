import os
from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "/data/vector-db")

def load_vector_db():
    if not os.path.exists(VECTOR_DB_PATH):
        return None
    
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings=None,
        allow_dangerous_deserialization=True
    )