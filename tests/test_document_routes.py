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


def test_upload_document_batch_payload_success(monkeypatch):
    from app.services import document_service
    from app.schemas.document_schema import DocumentUploadResponse

    def _fake_ingest(request):
        return DocumentUploadResponse(
            document_id=f"doc_for_{request.document_name}",
            document_name=request.document_name,
            status="indexed",
            chunk_count=2,
            message="ok",
        )

    monkeypatch.setattr(document_service, "ingest_document", _fake_ingest)

    client = TestClient(app)
    res = client.post(
        "/documents/upload",
        json=[
            {
                "document_name": "doc-a.txt",
                "document_type": "GUIDE",
                "s3_uri": "s3://bucket/doc-a.txt",
            },
            {
                "document_name": "doc-b.txt",
                "document_type": "RULE",
                "s3_uri": "s3://bucket/doc-b.txt",
            },
        ],
    )

    assert res.status_code == 201
    payload = res.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["document_name"] == "doc-a.txt"
    assert payload[1]["document_name"] == "doc-b.txt"
