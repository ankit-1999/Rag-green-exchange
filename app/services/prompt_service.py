from typing import Dict, List


def build_rag_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build a deterministic prompt with context and source identifiers.
    """
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            (
                f"[SOURCE {idx}]\n"
                f"document_name: {chunk.get('document_name', 'unknown')}\n"
                f"chunk_id: {chunk.get('chunk_id', 'unknown')}\n"
                f"chunk_index: {chunk.get('chunk_index', 'unknown')}\n"
                f"content:\n{chunk.get('text', '')}"
            )
        )

    context_text = "\n\n".join(context_blocks)

    return (
        "You are a precise assistant for GreenGrid Exchange.\n"
        "Use only the provided CONTEXT to answer.\n"
        "If context is insufficient, say that explicitly and ask for more data.\n"
        "Do not fabricate policy or numeric values.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "Return a concise answer in plain text."
    )
