from rag.query_analyzer import QueryAnalyzer
from rag.retriever import hybrid_retrieve

analyzer = QueryAnalyzer()

def adaptive_retrieve(store, query):

    q_type = analyzer.classify(query)

    if q_type == "definition":
        k = 2

    elif q_type == "explanatory":
        k = 5

    elif q_type == "short":
        k = 4

    else:
        k = 3

    contexts = hybrid_retrieve(store, query)

    return contexts[:k]