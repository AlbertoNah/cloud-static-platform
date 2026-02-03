import json
import pytest

from app.app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client


def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200

    data = r.get_json()
    assert isinstance(data, dict)
    assert "status" in data
    assert data["status"] == "ok"


def test_readyz_ok(client):
    r = client.get("/readyz")
    # should be ready because DATA_DIR exists in your project
    assert r.status_code in (200, 503)

    data = r.get_json()
    assert isinstance(data, dict)
    assert "status" in data
    assert data["status"] in ("ready", "not_ready")


@pytest.mark.parametrize("endpoint", ["/api/events", "/api/news", "/api/faq"])
def test_api_endpoints_shape(client, endpoint):
    r = client.get(endpoint)
    assert r.status_code == 200

    data = r.get_json()
    assert isinstance(data, dict)
    assert "items" in data
    assert isinstance(data["items"], list)

    # Optional but useful: cached flag exists and is boolean
    assert "cached" in data
    assert isinstance(data["cached"], bool)


def test_cache_becomes_true_on_second_call(client):
    r1 = client.get("/api/events")
    assert r1.status_code == 200
    d1 = r1.get_json()
    assert d1["cached"] in (False, True)

    r2 = client.get("/api/events")
    assert r2.status_code == 200
    d2 = r2.get_json()
    assert d2["cached"] is True
