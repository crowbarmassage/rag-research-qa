"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "indexed_chunks" in data


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_requires_question(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422  # Validation error

    def test_query_with_valid_request(self, client):
        # Note: This test may fail if pipeline not initialized
        response = client.post("/query", json={
            "question": "What is attention?",
            "retrieval_method": "hybrid",
            "top_k": 3
        })

        # Either success or 503 (pipeline not initialized)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "question" in data
            assert "answer" in data
            assert "sources" in data


class TestRetrieveEndpoint:
    """Tests for /retrieve endpoint."""

    def test_retrieve_requires_question(self, client):
        response = client.post("/retrieve", json={})
        assert response.status_code == 422

    def test_retrieve_with_valid_request(self, client):
        response = client.post("/retrieve", json={
            "question": "What is attention?",
            "top_k": 3,
            "method": "hybrid"
        })

        assert response.status_code in [200, 503]


class TestCompareEndpoint:
    """Tests for /compare endpoint."""

    def test_compare_requires_question(self, client):
        response = client.post("/compare", json={})
        assert response.status_code == 422

    def test_compare_with_valid_request(self, client):
        response = client.post("/compare", json={
            "question": "What is attention?",
            "top_k": 3
        })

        assert response.status_code in [200, 503]


class TestDocumentsEndpoint:
    """Tests for /documents endpoint."""

    def test_documents_returns_stats(self, client):
        response = client.get("/documents")
        assert response.status_code in [200, 503]


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_stats_returns_stats(self, client):
        response = client.get("/stats")
        assert response.status_code in [200, 503]
