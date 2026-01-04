# Avoid reloading model per request

class AppState:
    llm = None
    vector_db = None
    rag = None

state = AppState()