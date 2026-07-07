from fastapi.testclient import TestClient

from app.main import app


def _create_user(client: TestClient, first_name: str, role: str = "seller") -> str:
    res = client.post(
        "/createuser",
        json={
            "first_name": first_name,
            "last_name": "Test",
            "age": 30,
            "city": "Noida",
            "role": role,
        },
    )
    assert res.status_code == 201
    return res.json()["user_id"]


def test_createcredit_and_listings():
    client = TestClient(app)
    owner_id = _create_user(client, "OwnerOne", role="seller")

    create_res = client.post(
        "/createcredit",
        json={
            "user_id": owner_id,
            "credit_type": "solar",
            "price": 120.5,
        },
    )
    assert create_res.status_code == 201
    credit = create_res.json()
    assert credit["user_id"] == owner_id
    assert credit["credit_type"] == "solar"

    list_res = client.get("/listings")
    assert list_res.status_code == 200
    listings = list_res.json()
    assert any(item["credit_id"] == credit["credit_id"] for item in listings)


def test_credit_transfer_and_audit():
    client = TestClient(app)
    source_user_id = _create_user(client, "SourceUser", role="seller")
    destination_user_id = _create_user(client, "DestUser", role="buyer")

    create_credit_res = client.post(
        "/createcredit",
        json={
            "user_id": source_user_id,
            "credit_type": "wind",
            "price": 85.0,
        },
    )
    assert create_credit_res.status_code == 201
    credit_id = create_credit_res.json()["credit_id"]

    transfer_res = client.post(
        "/credit/transfer",
        json={
            "credit_id": credit_id,
            "source_user_id": source_user_id,
            "destination_user_id": destination_user_id,
        },
    )
    assert transfer_res.status_code == 200
    transferred_credit = transfer_res.json()
    assert transferred_credit["user_id"] == destination_user_id

    audit_res = client.get("/audit")
    assert audit_res.status_code == 200
    audit_records = audit_res.json()

    assert any(
        rec["operation"] == "create" and rec["credit_id"] == credit_id
        for rec in audit_records
    )
    assert any(
        rec["operation"] == "transfer"
        and rec["credit_id"] == credit_id
        and rec["source_user_id"] == source_user_id
        and rec["destination_user_id"] == destination_user_id
        for rec in audit_records
    )


def test_createcredit_user_not_found():
    client = TestClient(app)
    res = client.post(
        "/createcredit",
        json={
            "user_id": "user_missing",
            "credit_type": "coal",
            "price": 42,
        },
    )
    assert res.status_code == 400


def test_credits_created_by_user_endpoint():
    client = TestClient(app)
    creator_id = _create_user(client, "Creator", role="seller")
    _ = _create_user(client, "Other", role="buyer")

    create_credit_res = client.post(
        "/createcredit",
        json={
            "user_id": creator_id,
            "credit_type": "solar",
            "price": 50.0,
        },
    )
    assert create_credit_res.status_code == 201

    res = client.get(f"/credits/created-by/{creator_id}")
    assert res.status_code == 200
    payload = res.json()
    assert isinstance(payload, list)
    assert len(payload) >= 1
    assert payload[0]["credit_id"].startswith("credit_")


def test_credit_audit_by_credit_id_endpoint():
    client = TestClient(app)
    source_user_id = _create_user(client, "Src", role="seller")
    destination_user_id = _create_user(client, "Dst", role="buyer")

    create_credit_res = client.post(
        "/createcredit",
        json={
            "user_id": source_user_id,
            "credit_type": "wind",
            "price": 77.0,
        },
    )
    assert create_credit_res.status_code == 201
    credit_id = create_credit_res.json()["credit_id"]

    transfer_res = client.post(
        "/credit/transfer",
        json={
            "credit_id": credit_id,
            "source_user_id": source_user_id,
            "destination_user_id": destination_user_id,
        },
    )
    assert transfer_res.status_code == 200

    audit_res = client.get(f"/audit/{credit_id}")
    assert audit_res.status_code == 200
    records = audit_res.json()
    assert len(records) >= 2
    assert any(rec["operation"] == "create" for rec in records)
    assert any(rec["operation"] == "transfer" for rec in records)
