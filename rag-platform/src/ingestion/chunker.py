# Convert raw document text into semantically meaningful chunks.
# Reference
# https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

# Starting with just simple chunking for now

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size = 100, chunk_overlap = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def chunk(self, document):
        document_chunks = self.text_splitter.create_documents([document])
        return document_chunks