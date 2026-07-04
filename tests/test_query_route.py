from fastapi.testclient import TestClient

from app.main import app


def test_query_success(monkeypatch):
    from app.services import rag_service
    from app.schemas.query_schema import QueryResponse, QuerySource

    def fake_answer_question(_request):
        return QueryResponse(
            answer="sample answer",
            source_count=1,
            sources=[
                QuerySource(
                    chunk_id="doc_1_chunk_0",
                    document_id="doc_1",
                    document_name="sample-upload-doc.txt",
                    document_type="GUIDE",
                    chunk_index=0,
                    s3_uri="s3://bucket/key",
                    score=0.99,
                    snippet="sample context",
                )
            ],
        )

    monkeypatch.setattr(rag_service, "answer_question", fake_answer_question)

    client = TestClient(app)
    res = client.post(
        "/query",
        json={"question": "What is GreenGrid Exchange?", "top_k": 3},
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["answer"] == "sample answer"
    assert payload["source_count"] == 1
    assert len(payload["sources"]) == 1


def test_query_empty_question_validation():
    client = TestClient(app)
    res = client.post("/query", json={"question": ""})
    assert res.status_code == 422
