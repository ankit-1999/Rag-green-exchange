from fastapi.testclient import TestClient

from app.main import app


def test_clear_index_success(monkeypatch):
    from app.services import document_service
    from app.schemas.document_schema import DocumentClearIndexResponse

    monkeypatch.setattr(
        document_service,
        "clear_indexed_chunks",
        lambda: DocumentClearIndexResponse(
            deleted_chunks=14,
            cleared_documents=3,
            message="Indexed chunks and document metadata cleared successfully.",
        ),
    )

    client = TestClient(app)
    res = client.post("/documents/clear-index")

    assert res.status_code == 200
    payload = res.json()
    assert payload["deleted_chunks"] == 14
    assert payload["cleared_documents"] == 3


def test_clear_index_upstream_failure(monkeypatch):
    from app.services import document_service

    def _raise_error():
        raise RuntimeError("OpenSearch unavailable")

    monkeypatch.setattr(document_service, "clear_indexed_chunks", _raise_error)

    client = TestClient(app)
    res = client.post("/documents/clear-index")

    assert res.status_code == 502
    assert "Upstream service error" in res.json()["detail"]
