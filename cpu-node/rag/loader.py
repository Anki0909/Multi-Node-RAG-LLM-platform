# document loading only
from pathlib import Path
import pdfplumber

from rag.cleaner import TextCleaner

def load_documents(path="data/documents"):
    docs = []
    base = Path(path)

    for file in base.glob("*"):
        try:
            if file.suffix in {".txt", ".md"}:
                cleaned = TextCleaner.clean(file.read_text(encoding="utf-8", errors="ignore"))
                if cleaned:
                    docs.append(cleaned)

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

                    cleaned = TextCleaner.clean(text)

                    if cleaned:
                        docs.append(cleaned)

        except Exception as e:
            print(f"[WARN] Failed to load {file.name}: {e}")

    return docs