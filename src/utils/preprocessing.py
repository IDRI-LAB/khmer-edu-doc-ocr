import re


def clean_text_pipeline(text):
    """Shared pre-cleaning applied to both ground truth and prediction."""
    if not text:
        return ""
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub(r"\.{2,}", "", text)
    return text.strip()
