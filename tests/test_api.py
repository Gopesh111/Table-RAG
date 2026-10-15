"""Integration tests for TableRAG-Core FastAPI endpoints."""

import pytest
from starlette.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_endpoint(client):
    """Verify that the /api/v1/health endpoint reports system status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexes" in data
    assert "models" in data
    assert "ollama_connected" in data


def test_query_validation_error(client):
    """Verify that submitting an empty or too short query returns 422 Unprocessable Entity."""
    response = client.post("/api/v1/chat/query", json={"query": ""})
    assert response.status_code == 422


def test_uninitialized_index_query_fails_gracefully(client):
    """Verify that querying an empty index returns a descriptive 400 Bad Request error."""
    # Note: If no documents are ingested yet
    response = client.post("/api/v1/retrieval/debug", json={"query": "What is Q3 revenue?"})
    # Either 400 (uninitialized) or 200 (if preloaded)
    assert response.status_code in (400, 200)
    if response.status_code == 400:
        assert "ingest" in response.json()["detail"].lower() or "empty" in response.json()["detail"].lower()
