import re

class ContextCompressor:

    def compress(self, query, contexts, max_sentences=12):

        keywords = query.lower().split()

        sentences = []

        for ctx in contexts:
            text = ctx["text"] if isinstance(ctx, dict) else ctx
            parts = re.split(r'[.!?]\s+', text)

            for s in parts:
                score = sum(k in s.lower() for k in keywords)

                if score > 0:
                    sentences.append((score, s))

        sentences.sort(reverse=True, key=lambda x: x[0])

        return "\n".join([s for _, s in sentences[:max_sentences]])