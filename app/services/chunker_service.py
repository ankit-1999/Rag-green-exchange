"""
chunker_service.py
------------------
Responsible for splitting raw document text into overlapping chunks suitable
for embedding and retrieval.

Strategy: character-based sliding window (simple, dependency-free).
Token-based chunking can be swapped in later by replacing _estimate_tokens().
"""

import logging
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate: 1 token ≈ 4 characters (English prose).
    Replace with tiktoken or a Bedrock tokenizer for production accuracy.
    """
    return len(text) // 4


def _tokens_to_chars(token_count: int) -> int:
    return token_count * 4


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE_TOKENS,
    overlap: int = settings.CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """
    Split *text* into overlapping chunks.

    Parameters
    ----------
    text        : raw document text (already extracted from PDF/TXT/etc.)
    chunk_size  : target size of each chunk in tokens
    overlap     : number of tokens to repeat at the start of the next chunk

    Returns
    -------
    List of non-empty string chunks.
    """
    if not text or not text.strip():
        logger.warning("chunk_text received empty or blank text")
        return []

    chunk_chars = _tokens_to_chars(chunk_size)
    overlap_chars = _tokens_to_chars(overlap)
    step = chunk_chars - overlap_chars

    if step <= 0:
        raise ValueError(
            f"overlap ({overlap} tokens) must be smaller than chunk_size ({chunk_size} tokens)"
        )

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    logger.info(
        "Chunked text: total_chars=%d → %d chunks (size=%d tok, overlap=%d tok)",
        len(text),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks
