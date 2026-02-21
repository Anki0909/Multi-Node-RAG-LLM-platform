import re

class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # remove bibliography/reference sections
        text = re.sub(r"\b(Bibliography|References)\b.*", "", text, flags=re.IGNORECASE)

        # remove page numbers isolated on lines
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # remove very short noisy lines
        lines = []
        for line in text.split("."):
            line = line.strip()
            if len(line) > 25:
                lines.append(line)

        text = ". ".join(lines)

        return text.strip()
