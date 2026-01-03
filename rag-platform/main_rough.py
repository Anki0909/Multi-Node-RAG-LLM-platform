# This code is basically a placeholder to create a basic end to end script to understand the flow and the components
'''
Basic first idea of the flow
1. Read the document (pdf, txt etc)
2. Perform chunking on the document
3. Convert the chunks to vector embeddings
4. Read a query from the user
5. Feed the query as prompt to an LLM 
6. Get a relavant answer from the document provided
'''
import pymupdf
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

print("Reading the document")
doc = pymupdf.open('/home/ankur/Documents/posted_ila_2017-09-13.pdf')

pdf = ''
for page in doc:
    text = page.get_text()
    pdf += text

texts = pdf.split('.')

print("Pharsing the document")
texts = [t.strip(' \n') for t in texts]

print("Creating embeddings")
embedding_model = HuggingFaceEmbeddings(
    model_name = "thenlper/gte-small"
)


db = FAISS.from_texts(texts, embedding_model)


template = """<|user|>
Relevant information:
{context}

You must answer using only the information in the context. If the answer is not present, say "I don't know".:
{question}<|end|>
<|assistant|>"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
print("Prompt template set!")

print("Loading LLM model")
llm = LlamaCpp(
    model_path = "/home/ankur/Documents/Multi-Node-RAG-LLM-platform/rag-platform/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1,
    max_tokens = 500,
    n_ctx=2048,
    seed=42,
    verbose=False
)
print("LLM model loaded!")

print('Setting up RAG model')
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt
    }
)

results = db.similarity_search("What year was this document published?", k=3)
for idx, chunks in enumerate(results):
    print(f'{idx}: {chunks}')

print("Invoking RAG model")
response = rag.invoke('What year was this document published?')
print('Question: What year was this document published?')
reply = response['result']
print(f'Answer: {reply}')
