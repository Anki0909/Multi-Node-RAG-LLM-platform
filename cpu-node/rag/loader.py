from pathlib import Path
from typing import List
import pdfplumber

def load_documents(path: str):
	docs = []
	base = Path(path)

	for file in base.glob("*"):
		if file.suffix in {".txt", ".md"}:
			raw_doc = file.read_text(encoding='utf-8')
			docs.append(raw_doc)

		elif file.suffix == ".pdf":
			with pdfplumber.open(file) as pdf:
				text = "\n".join(page.extract_text() or "" for page in pdf.pages)
				docs.append(text)

	return docs