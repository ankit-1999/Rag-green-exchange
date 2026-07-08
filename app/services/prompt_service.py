import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _needs_elaborate_response(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in ("explain", "elaborate", "detail", "detailed", "why", "how"))


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
        json.dumps(api_context, ensure_ascii=True, indent=2, default=_json_default)
        if api_context is not None
        else "None"
    )
    elaborate = _needs_elaborate_response(question)
    response_style = (
        "Provide an elaborate answer with clear sections and examples."
        if elaborate
        else "Provide a concise but meaningful answer in 2-4 sentences."
    )
    timeline_hint = (
        "If API_CONTEXT contains audit_records, format them as a simple timeline with date/time and action."
    )

    return (
        "You are a precise assistant for GreenGrid Exchange.\n"
        "Use only the provided API_CONTEXT and CONTEXT to answer.\n"
        "When API_CONTEXT includes numeric facts (counts/statistics), treat it as the source of truth.\n"
        "Use CONTEXT chunks for explanation and citations.\n"
        "If context is insufficient, say that explicitly and ask for more data.\n"
        "Do not fabricate policy or numeric values.\n\n"
        f"RESPONSE_STYLE:\n{response_style}\n\n"
        f"FORMAT_HINT:\n{timeline_hint}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"API_CONTEXT:\n{api_context_text}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "Return a concise answer in plain text."
    )
