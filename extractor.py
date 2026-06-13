import pdfplumber
import re
from typing import List


def clean_text(text: str) -> str:
    """
    Remove extra whitespace, newlines, and non-printable characters.
    """
    text = re.sub(r'\s+', ' ', text)        # collapse multiple spaces/newlines
    text = re.sub(r'[^\x20-\x7E]', '', text) # remove non-ASCII
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using pdfplumber.
    Returns a single cleaned string.
    """
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"[Page {page_num + 1}]\n{text}")

    return clean_text(" ".join(full_text))


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text: Full document text
        chunk_size: Number of words per chunk (default 500)
        overlap: Number of words to overlap between chunks (default 50)

    Returns:
        List of dicts: { chunk_id, text, word_count, start_word, end_word }
    """
    words = text.split()
    chunks = []
    chunk_id = 0
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_str,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": min(end, len(words)) - 1,
        })

        chunk_id += 1
        start += chunk_size - overlap  # move forward with overlap

        # avoid tiny trailing chunks (less than 10% of chunk_size)
        if start < len(words) and len(words) - start < chunk_size * 0.1:
            break

    return chunks


def extract_and_chunk(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """
    Main function: extract text from PDF and return chunked list.
    This is what main.py calls.
    """
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text:
        raise ValueError("Could not extract any text from the PDF. It may be scanned/image-based.")

    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    return chunks