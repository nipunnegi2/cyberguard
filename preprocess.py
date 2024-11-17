import re

def clean_text(text):
    """Clean and normalize text."""
    if isinstance(text, str):
        text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        text = text.lower()  # Convert to lowercase
    return text
