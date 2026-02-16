# document loading only
from pathlib import Path
import pdfplumber

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def load_documents(path="data/documents"):
    docs = []
    base = Path(path)

    for file in base.glob("*"):
        try:
            if file.suffix in {".txt", ".md"}:
                docs.append(file.read_text(encoding="utf-8", errors="ignore"))

            elif file.suffix == ".pdf":
                with pdfplumber.open(file) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

                    # will need to generalize it
                    bad_words = [
                        "bibliography",
                        "all rights reserved",
                        "copyright"
                    ]

                    clean_text = []

                    for line in text.split("\n"):
                        if not any(w in line.lower() for w in bad_words):
                            clean_text.append(line)

                    text = "\n".join(clean_text)

                    if len(text.strip()) < 200:
                        continue

                    chunks = split_text(text)

                    for c in chunks:
                        if len(c.strip()) > 200:
                            docs.append(c)

        except Exception as e:
            print(f"[WARN] Failed to load {file.name}: {e}")

    return docs
