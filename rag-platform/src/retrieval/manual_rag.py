from time import time

class ManualRAG:
    def __init__(self, llm, vector_db, prompt_template, top_k=2):
        self.llm = llm
        self.vector_db = vector_db
        self.prompt_template = prompt_template
        self.top_k = top_k

    def retrieve(self, query):
        start = time()
        docs = self.vector_db.similarity_search(query, k=self.top_k)
        print(f"\n[Retrieval] Retrieved {len(docs)} chunks in {time()-start:.2f}s")
        return docs
    
    def inspect_retrieval(self, docs):
        print("\n[Retrieval Inspection]")
        for i, doc in enumerate(docs):
            preview = doc.page_content[:300].replace("\n", " ")
            print(f"\n--- Chunk {i+1} ---")
            print(preview)

    def build_context(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate(self, query):
        start = time()

        docs = self.retrieve(query)
        self.inspect_retrieval(docs)
        context = self.build_context(docs)
        prompt = self.prompt_template.format(context=context, question=query)
        print("\n[LLM] Generating answer...")
        answer = self.llm.invoke(prompt)
        answer = answer.strip().split("\n")[0]

        print(f"\n[Result]")
        print(f"Answer: {answer}")
        print(f"Total time: {time() - start:.2f}s")

        return answer