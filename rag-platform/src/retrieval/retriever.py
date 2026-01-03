# Perform semantic search over stored vectors

from time import time
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

class Retriever:
    def __init__(self, llm_model, vector_db, prompt):
        self.llm_model = llm_model
        self.vector_db = vector_db
        self.prompt = prompt
        self.set_retrievalQA()

    def set_retrievalQA(self):
        self.rag = RetrievalQA.from_chain_type(
            llm = self.llm_model,
            chain_type = 'stuff',
            retriever = self.vector_db.as_retriever(search_kwargs={"k":2}),
            chain_type_kwargs = {
                "prompt": self.prompt
            }
        )

    def get_similarity(self, query, num_doc):
        similarity_result = self.vector_db.similarity_search(query, k=num_doc)
        for idx, chunk in enumerate(similarity_result):
            print(f'{idx}: {chunk}')

    def invoke_retievalQA(self, query):
        print("INFO: Invoking RAG...")
        start = time()
        response = self.rag.invoke(query)
        end = time()
        print(f"Query: {query}")
        reply = response['result'].strip()
        reply = reply.split("\n")[0]
        print(f"Answer: {reply}")
        print(f"It took {end - start} seconds to get the answer")