class QueryAnalyzer:

    def classify(self, query: str):

        q = query.lower()

        if any(x in q for x in ["what is", "define", "meaning"]):
            return "definition"

        if any(x in q for x in ["how", "explain", "why"]):
            return "explanatory"

        if len(q.split()) <= 3:
            return "short"

        return "general"