# Define all prompt template

from time import time
from langchain_core.prompts import PromptTemplate

class PromptSetUp:
    def __init__(self):
        self.template = """Answer the question using only the context below.
            Respond with a single, concise answer.
            If the answer is not present, respond exactly with: I don't know.

            Context:
            {context}

            Question:
            {question}

            Answer:"""
        
    def generate_prompt(self):
        self.prompt = PromptTemplate(
            template = self.template,
            input_variables = ["context", "question"]
        )
        return self.prompt