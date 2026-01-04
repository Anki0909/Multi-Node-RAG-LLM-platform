# Define all prompt template

from time import time
from langchain_core.prompts import PromptTemplate

class PromptSetUp:
    def __init__(self):
        self.template = (
            "Answer the question using only the context below.\n"
            "If the answer is not present, respond exactly with: I don't know.\n"
            "Respond with a single concise answer.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
        
    def generate_prompt(self):
        return self.template