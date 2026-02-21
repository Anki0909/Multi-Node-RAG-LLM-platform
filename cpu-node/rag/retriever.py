# orchestration of hybrid retrieval
from rag.reranker import Reranker

reranker = Reranker()

def merge_results(semantic, keyword):
    merged = {}
    
    for r in semantic + keyword:
        text = r["text"]

        # keep BEST score if duplicate
        if text not in merged or r["score"] < merged[text]["score"]:
            merged[text] = r

    return list(merged.values())

def hybrid_retrieve(store, query, semantic_k=5, keyword_k=3, rerank_top_k=3):

    semantic = store.search_semantic(query, k=semantic_k)
    keyword = store.keyword_search(query, k=keyword_k)

    merged = merge_results(semantic, keyword)

    reranked = reranker.rerank(query, merged, top_k=rerank_top_k)

    confidence = compute_confidence(reranked)

    return {
        "results": reranked,
        "confidence": confidence
    }


def compute_confidence(results):
    if not results:
        return 0.0

    avg_score = sum(r["score"] for r in results) / len(results)

    # FAISS L2 distance:
    # lower = better
    confidence = 1 / (1 + avg_score)

    return confidence
