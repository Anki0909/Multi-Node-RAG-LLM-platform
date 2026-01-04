from fastapi import FastAPI
from api.routes import ingest, query, health

app = FastAPI(title="RAG Platform")

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)