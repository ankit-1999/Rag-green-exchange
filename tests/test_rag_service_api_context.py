from datetime import datetime, timezone

from app.schemas.query_schema import QueryRequest
from app.services import rag_service


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
        captured["api_context"] = api_context
        return "prompt"

    monkeypatch.setattr("app.services.prompt_service.build_rag_prompt", fake_prompt_builder)
    monkeypatch.setattr("app.services.bedrock_service.generate_answer", lambda _p: "GreenGrid does X")
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {
            "requires_api_data": False,
            "reason": "not needed",
            "tool_calls": [],
        },
    )

    res = rag_service.answer_question(QueryRequest(question="Explain this application", top_k=1))

    assert res.api_facts_used is False
    assert res.answer_mode == "retrieval_only"
    assert res.api_summary is None
    assert captured["api_context"] is None


def test_answer_question_uses_credit_api_context_for_credit_reference(monkeypatch):
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
    monkeypatch.setattr(
        "app.services.bedrock_service.generate_answer",
        lambda _p: "EC-101 was created on 2026-07-01 and is owned by user_abc",
    )
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {
            "requires_api_data": True,
            "reason": "needs credit ownership and created timestamp",
            "tool_calls": [
                {
                    "tool": "get_credit_details",
                    "arguments": {"credit_reference": "EC-101"},
                }
            ],
        },
    )

    class _Credit:
        credit_code = "EC-101"
        user_id = "user_abc"

        class _Type:
            value = "solar"

        credit_type = _Type()
        price = 100.0
        created_at = datetime(2026, 7, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(
        "app.services.credit_service.get_credit_by_reference",
        lambda reference: _Credit(),
    )

    res = rag_service.answer_question(
        QueryRequest(question="Who owns EC-101 credit and when was it created?", top_k=1)
    )

    assert res.api_facts_used is True
    assert res.answer_mode == "retrieval_plus_api"
    assert res.api_summary is not None
    assert res.api_summary.context_type == "get_credit_details"
    assert res.api_summary.credit_reference == "EC-101"
    assert captured["api_context"] is not None
    assert captured["api_context"]["context_type"] == "get_credit_details"
    assert captured["api_context"]["owner_user_id"] == "user_abc"


def test_answer_question_ignores_unknown_planner_tool(monkeypatch):
    captured = {"api_context": "unset"}

    monkeypatch.setattr("app.services.bedrock_service.embed_text", lambda _q: [0.1, 0.2])
    monkeypatch.setattr(
        "app.services.opensearch_service.search_similar_chunks",
        lambda _emb, top_k: [_sample_hit()],
    )

    def fake_prompt_builder(question, chunks, api_context=None):
        captured["api_context"] = api_context
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


def test_answer_question_executes_list_users_tool(monkeypatch):
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
    monkeypatch.setattr("app.services.bedrock_service.generate_answer", lambda _p: "Listed users")
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {
            "requires_api_data": True,
            "reason": "need user inventory",
            "tool_calls": [{"tool": "list_users", "arguments": {}}],
        },
    )
    monkeypatch.setattr(
        "app.services.user_service.list_users",
        lambda: [],
    )

    res = rag_service.answer_question(QueryRequest(question="list users", top_k=1))

    assert res.api_facts_used is True
    assert res.answer_mode == "retrieval_plus_api"
    assert res.api_summary is not None
    assert captured["api_context"] is not None
    assert captured["api_context"]["tool_results"][0]["tool"] == "list_users"


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
            "tool_calls": [{"tool": "list_users", "arguments": {}}],
        },
    )
    monkeypatch.setattr("app.services.user_service.list_users", lambda: [])
    monkeypatch.setattr(
        "app.services.bedrock_service.generate_answer",
        lambda _p: "There are currently 0 users.",
    )

    res = rag_service.answer_question(QueryRequest(question="how many users are there", top_k=1))

    assert res.answer == "There are currently 0 users."
    assert res.source_count == 0
    assert res.api_facts_used is True
    assert res.answer_mode == "retrieval_plus_api"


def test_answer_question_executes_credit_audit_by_id_tool(monkeypatch):
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
    monkeypatch.setattr("app.services.bedrock_service.generate_answer", lambda _p: "timeline")
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {
            "requires_api_data": True,
            "reason": "history requested",
            "tool_calls": [
                {
                    "tool": "list_credit_audit_by_credit_id",
                    "arguments": {"credit_id": "credit_abc12345"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        "app.services.credit_service.list_credit_audit_by_credit_id",
        lambda _credit_id: [],
    )

    res = rag_service.answer_question(QueryRequest(question="show history", top_k=1))

    assert res.api_facts_used is True
    assert captured["api_context"] is not None
    assert captured["api_context"]["tool_results"][0]["tool"] == "list_credit_audit_by_credit_id"


def test_answer_question_executes_credit_history_timeline_tool(monkeypatch):
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
    monkeypatch.setattr("app.services.bedrock_service.generate_answer", lambda _p: "history")
    monkeypatch.setattr(
        "app.services.bedrock_service.plan_api_calls",
        lambda _q: {
            "requires_api_data": True,
            "reason": "history requested",
            "tool_calls": [
                {
                    "tool": "get_credit_history_timeline",
                    "arguments": {"credit_reference": "EC-101"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        "app.services.credit_service.get_credit_history_timeline",
        lambda _ref: {
            "credit_id": "credit_ab12cd34",
            "credit_code": "EC-101",
            "current_owner_user_id": "user_2",
            "timeline": [
                {
                    "timestamp": "2026-07-07T05:30:00+00:00",
                    "operation": "create",
                    "action": "Credit created by user_1",
                }
            ],
            "timeline_text": "- 2026-07-07T05:30:00+00:00: Credit created by user_1",
        },
    )

    res = rag_service.answer_question(QueryRequest(question="tell me history of EC-101", top_k=1))

    assert res.api_facts_used is True
    assert captured["api_context"] is not None
    assert captured["api_context"]["tool_results"][0]["tool"] == "get_credit_history_timeline"
