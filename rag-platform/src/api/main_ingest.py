from fastapi import FastAPI

app = FastAPI(title="RAG Ingest API")

from api.routes import ingest, health
app.include_router(health.router)
app.include_router(ingest.router)
