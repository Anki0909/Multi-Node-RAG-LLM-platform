# orchestration of hybrid retrieval
from rag.reranker import Reranker

reranker = Reranker()

def hybrid_retrieve(store, query):

    semantic = store.search_semantic(query, k=5)
    keyword = store.keyword_search(query, k=3)

    merged = list(dict.fromkeys(semantic + keyword))

    reranked = reranker.rerank(query, merged, top_k=3)

    return reranked