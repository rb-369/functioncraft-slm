"""Integration tests for the FastAPI serving application."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app, init_resources


@pytest.fixture(scope="module")
def client():
    # Trigger resource init manually to load schemas
    init_resources()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "functioncraft-slm"
    assert data["schemas_loaded"] >= 4


def test_schemas_endpoint(client):
    response = client.get("/v1/schemas")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 4
    assert "execute_sql_query" in data["schemas"]


def test_tool_invoke_endpoint(client):
    payload = {
        "prompt": "Show top 10 customers signed up in March by lifetime value",
        "preferred_tool": "execute_sql_query",
    }
    response = client.post("/v1/tools/invoke", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["tool_name"] == "execute_sql_query"
    assert "query" in data["arguments"]
    assert data["latency_ms"] >= 0.0


def test_chat_completions_endpoint(client):
    payload = {
        "model": "functioncraft-qwen-1.5b",
        "messages": [
            {"role": "user", "content": "Refund $50 for transaction txn_123456789012 because of duplicate charge"}
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    choice = data["choices"][0]
    assert choice["finish_reason"] in ["tool_calls", "stop"]


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
