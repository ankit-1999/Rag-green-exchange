import json
from typing import Dict, List, Optional


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict],
    api_context: Optional[Dict] = None,
) -> str:
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
    api_context_text = (
        json.dumps(api_context, ensure_ascii=True, indent=2)
        if api_context is not None
        else "None"
    )

    return (
        "You are a precise assistant for GreenGrid Exchange.\n"
        "Use only the provided API_CONTEXT and CONTEXT to answer.\n"
        "When API_CONTEXT includes numeric facts (counts/statistics), treat it as the source of truth.\n"
        "Use CONTEXT chunks for explanation and citations.\n"
        "If context is insufficient, say that explicitly and ask for more data.\n"
        "Do not fabricate policy or numeric values.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"API_CONTEXT:\n{api_context_text}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "Return a concise answer in plain text."
    )
