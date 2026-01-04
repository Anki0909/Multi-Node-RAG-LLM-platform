# Orchestrate the full RAG flow

from embeddings.embedder import TextEmbedder
from ingestion.loader import FileLoader
from ingestion.chunker import TextChunker
from llm.model import LLM
from llm.prompt import PromptSetUp
from retrieval.retriever import Retriever
from retrieval.manual_rag import ManualRAG

llm_model_path = '/home/ankur/Documents/Multi-Node-RAG-LLM-platform/rag-platform/qwen2.5-3b-instruct-q4_k_m.gguf'
document_path = '/home/ankur/Documents/posted_ila_2017-09-13.pdf'

llm = LLM(llm_model_path)

def main():
    print("INFO: Pharsing the document")
    file_pharser = FileLoader(document_path)

    file_metadata = file_pharser.read_file()
    texts = file_metadata['file_content']
    if texts == None:
        print("ERROR: No text found or file format not compatible")
        return
    
    print("INFO: Creating chunks....")
    text_chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    text_chunks = text_chunker.chunk(texts)

    print("INFO: Creating embedding...")
    embedder = TextEmbedder()
    vector_db = embedder.create_embedding(text_chunks)

    prompter = PromptSetUp()
    prompt = prompter.generate_prompt()

    llm_model = llm.llm_model

    # rag_setup = Retriever(llm_model, vector_db, prompt)

    rag = ManualRAG(llm=llm_model,
                    vector_db=vector_db,
                    prompt_template=prompt,
                    top_k=2)

    query = input("Enter the query to be asked: \n")

    # rag_setup.invoke_retievalQA(query)

    # rag_setup.get_similarity(query, 5)

    rag.generate(query)

if __name__ == "__main__":
    main()