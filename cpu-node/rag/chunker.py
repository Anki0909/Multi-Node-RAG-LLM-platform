class TextChunker:
    def __init__(self, chunk_size=512, overlap=64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk = words[i:i + self.chunk_size]
            chunk.append(" ".join(chunk))
            i += self.chunk_size - self.overlap

        return chunks[:1000]