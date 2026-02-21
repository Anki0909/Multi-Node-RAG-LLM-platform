from rag.query_analyzer import QueryAnalyzer
from rag.retriever import hybrid_retrieve, compute_confidence
from rag.query_expander import QueryExpander
from rag.reranker import Reranker

reranker = Reranker()
analyzer = QueryAnalyzer()
expander = QueryExpander()



def adaptive_retrieve(store, query, confidence_threshold=0.7):

    q_type = analyzer.classify(query)

    # Phase 1: base config
    if q_type == "definition":
        semantic_k = 3
        keyword_k = 2
        rerank_top_k = 2
        confidence_threshold = 0.65

    elif q_type == "explanatory":
        semantic_k = 8
        keyword_k = 5
        rerank_top_k = 4
        confidence_threshold = 0.75

    elif q_type == "short":
        semantic_k = 5
        keyword_k = 3
        rerank_top_k = 3
        confidence_threshold = 0.6

    else:
        semantic_k = 4
        keyword_k = 3
        rerank_top_k = 3

    # Attempt 1
    result = hybrid_retrieve(
        store,
        query,
        semantic_k=semantic_k,
        keyword_k=keyword_k,
        rerank_top_k=rerank_top_k,
    )

    if result["confidence"] > confidence_threshold:
        return result["results"]

    # Attempt 2 (wider recall)
    result = hybrid_retrieve(
        store,
        query,
        semantic_k=semantic_k + 6,
        keyword_k=keyword_k + 4,
        rerank_top_k=rerank_top_k + 2,
    )

    if result["confidence"] > confidence_threshold:
        return result["results"]

    # Attempt 3 (query expansion)
    expanded_queries = expander.expand(query)

    all_results = []

    for q in expanded_queries:
        result = hybrid_retrieve(
            store,
            q,
            semantic_k=semantic_k + 6,
            keyword_k=keyword_k + 4,
            rerank_top_k=rerank_top_k + 2,
        )

        all_results.extend(result["results"])

    final = reranker.rerank(
        query,
        all_results,
        top_k=rerank_top_k + 2
    )

    confidence = compute_confidence(final)

    if confidence > confidence_threshold:
        return final

    # Attempt 4 (semantic only strong)
    result = hybrid_retrieve(
        store,
        query,
        semantic_k=15,
        keyword_k=2,
        rerank_top_k=5,
    )

    # Final fallback
    return result["results"]