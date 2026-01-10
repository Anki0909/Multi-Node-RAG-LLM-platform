import os
from chromadb import Client
from chromadb.config import Settings

def get_chroma_client():
    persist_dir = os.getenv("VECTOR_DB_PATH", "/data/vector-db")

    return Client(
        Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
    )