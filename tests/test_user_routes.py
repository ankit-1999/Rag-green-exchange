from fastapi.testclient import TestClient

from app.main import app


def test_create_user_and_get_user_success():
    client = TestClient(app)

    create_res = client.post(
        "/createuser",
        json={
            "first_name": "Ankit",
            "last_name": "Singh",
            "age": 27,
            "city": "Noida",
            "role": "seller",
        },
    )

    assert create_res.status_code == 201
    created = create_res.json()
    assert created["user_id"].startswith("user_")
    assert created["role"] == "seller"

    get_res = client.get("/getuser", params={"user_id": created["user_id"]})
    assert get_res.status_code == 200
    fetched = get_res.json()
    assert fetched["user_id"] == created["user_id"]
    assert fetched["first_name"] == "Ankit"


def test_get_user_not_found():
    client = TestClient(app)
    res = client.get("/getuser", params={"user_id": "user_unknown"})
    assert res.status_code == 404


def test_get_users_returns_list():
    client = TestClient(app)
    client.post(
        "/createuser",
        json={
            "first_name": "Riya",
            "last_name": "Sharma",
            "age": 30,
            "city": "Delhi",
            "role": "buyer",
        },
    )

    res = client.get("/getusers")
    assert res.status_code == 200
    payload = res.json()
    assert isinstance(payload, list)
    assert len(payload) >= 1


def test_create_user_batch_payload_success():
    client = TestClient(app)

    res = client.post(
        "/createuser",
        json=[
            {
                "first_name": "BatchA",
                "last_name": "One",
                "age": 25,
                "city": "Noida",
                "role": "seller",
            },
            {
                "first_name": "BatchB",
                "last_name": "Two",
                "age": 32,
                "city": "Delhi",
                "role": "buyer",
            },
        ],
    )

    assert res.status_code == 201
    payload = res.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["user_id"].startswith("user_")
    assert payload[1]["user_id"].startswith("user_")
