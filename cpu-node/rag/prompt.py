# prompt construction
def build_prompt(context, query):

    return f"""
        You are a precise technical assistant.

        Answer the question ONLY using the context.
        If possible, answer in 2-4 concise sentences.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """.strip()
