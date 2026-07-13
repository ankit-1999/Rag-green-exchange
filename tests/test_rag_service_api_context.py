from app.schemas.query_schema import QueryRequest
from app.services import prompt_service
from app.services import rag_service
from datetime import datetime, timezone


def _sample_hit():
    return {
        "chunk_id": "doc_1_chunk_0",
        "document_id": "doc_1",
        "document_name": "sample-upload-doc.txt",
        "document_type": "GUIDE",
        "chunk_index": 0,
        "s3_uri": "s3://bucket/key",
        "score": 0.99,
        "text": "GreenGrid is a clean energy exchange platform.",
    }


def test_answer_question_uses_api_context_for_document_count_question(monkeypatch):
    captured = {"api_context": None}

    monkeypatch.setattr("app.services.bedrock_service.embed_text", lambda _q: [0.1, 0.2])
    monkeypatch.setattr(
        "app.services.opensearch_service.search_similar_chunks",
        lambda _emb, top_k: [_sample_hit()],
    )

    def fake_prompt_builder(question, chunks, api_context=None):
        captured["api_context"] = api_context
        return "prompt"

    monkeypatch.setattr("app.services.prompt_service.build_rag_prompt", fake_prompt_builder)
    monkeypatch.setattr("app.services.bedrock_service.generate_answer", lambda _p: "There are 2 docs")
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {
            "requires_api_data": True,
            "reason": "needs live count",
            "tool_calls": [{"tool": "get_documents_summary", "arguments": {}}],
        },
    )
    monkeypatch.setattr(
        "app.services.document_service.get_documents_summary",
        lambda: {
            "total_documents": 2,
            "by_type": {"GUIDE": 2},
            "sample_document_names": ["a.txt", "b.txt"],
        },
    )

    res = rag_service.answer_question(
        QueryRequest(question="How many documents are available right now?", top_k=1)
    )

    assert res.api_facts_used is True
    assert res.answer_mode == "retrieval_plus_api"
    assert res.api_summary is not None
    assert res.api_summary.total_documents == 2
    assert captured["api_context"] is not None
    assert captured["api_context"]["total_documents"] == 2


def test_answer_question_skips_api_context_for_general_question(monkeypatch):
    captured = {"api_context": "unset"}

    monkeypatch.setattr("app.services.bedrock_service.embed_text", lambda _q: [0.1, 0.2])
    monkeypatch.setattr(
        "app.services.opensearch_service.search_similar_chunks",
        lambda _emb, top_k: [_sample_hit()],
    )

    def fake_prompt_builder(question, chunks, api_context=None):
        captured["api_context"] = api_context # type: ignore
        return "prompt"

    monkeypatch.setattr("app.services.prompt_service.build_rag_prompt", fake_prompt_builder)
    monkeypatch.setattr("app.services.bedrock_service.generate_answer", lambda _p: "GreenGrid does X")
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {"requires_api_data": False, "reason": "not needed", "tool_calls": []},
    )

    res = rag_service.answer_question(QueryRequest(question="Explain this application", top_k=1))

    assert res.api_facts_used is False
    assert res.answer_mode == "retrieval_only"
    assert res.api_summary is None
    assert captured["api_context"] is None


def test_answer_question_ignores_unknown_planner_tool(monkeypatch):
    captured = {"api_context": "unset"}

    monkeypatch.setattr("app.services.bedrock_service.embed_text", lambda _q: [0.1, 0.2])
    monkeypatch.setattr(
        "app.services.opensearch_service.search_similar_chunks",
        lambda _emb, top_k: [_sample_hit()],
    )

    def fake_prompt_builder(question, chunks, api_context=None):
        captured["api_context"] = api_context # type: ignore
        return "prompt"

    monkeypatch.setattr("app.services.prompt_service.build_rag_prompt", fake_prompt_builder)
    monkeypatch.setattr("app.services.bedrock_service.generate_answer", lambda _p: "safe answer")
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {
            "requires_api_data": True,
            "reason": "wants unsupported tool",
            "tool_calls": [{"tool": "delete_everything", "arguments": {}}],
        },
    )

    res = rag_service.answer_question(QueryRequest(question="random question", top_k=1))

    assert res.api_facts_used is False
    assert res.answer_mode == "retrieval_only"
    assert res.api_summary is None
    assert captured["api_context"] is None


def test_answer_question_uses_api_data_when_no_vector_hits(monkeypatch):
    monkeypatch.setattr("app.services.bedrock_service.embed_text", lambda _q: [0.1, 0.2])
    monkeypatch.setattr(
        "app.services.opensearch_service.search_similar_chunks",
        lambda _emb, top_k: [],
    )
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {
            "requires_api_data": True,
            "reason": "count question",
            "tool_calls": [{"tool": "get_documents_summary", "arguments": {}}],
        },
    )
    monkeypatch.setattr(
        "app.services.document_service.get_documents_summary",
        lambda: {"total_documents": 0, "by_type": {}, "sample_document_names": []},
    )
    monkeypatch.setattr(
        "app.services.bedrock_service.generate_answer",
        lambda _p: "There are currently 0 documents.",
    )

    res = rag_service.answer_question(QueryRequest(question="how many documents are there", top_k=1))

    assert res.answer == "There are currently 0 documents."
    assert res.source_count == 0
    assert res.api_facts_used is True
    assert res.answer_mode == "retrieval_plus_api"


def test_build_rag_prompt_serializes_datetime_in_api_context():
    prompt = prompt_service.build_rag_prompt(
        question="tell me something about the document",
        retrieved_chunks=[],
        api_context={
            "document_name": "policy.txt",
            "indexed_at": datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc),
        },
    )

    assert "API_CONTEXT:" in prompt
    assert "2026-07-01T12:30:00+00:00" in prompt
