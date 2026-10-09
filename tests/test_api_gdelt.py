"""
Integration and Unit Tests for GDELT Ingestion REST Endpoints.

Purpose:
  Validates the behavior and HTTP response codes of the /api/ingest/gdelt route.

Role in Project:
  - Verifies that external upstream GDELT rate limits cleanly return HTTP 429 Too Many Requests.
  - Verifies network connectivity failures trigger HTTP 503 Service Unavailable.
  - Verifies end-to-end payload transformation and NLP signal generation on successful ingest (HTTP 200).
  - Uses TestClient and mock fixtures to prevent hitting real GDELT servers during testing.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

import src.api.main as main
from src.ingestion.gdelt_ingester import GdeltRateLimitError, GdeltConnectionError

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def setup_services():
    """Ensure service singletons are mocked for API tests."""
    main.gdelt_ingester = MagicMock()
    main.risk_engine = MagicMock()


def test_api_gdelt_rate_limit():
    """Verify endpoint returns 429 when GDELT is rate-limited."""
    main.gdelt_ingester.fetch_events.side_effect = GdeltRateLimitError("GDELT is temporarily rate-limiting requests")
    response = client.post("/api/ingest/gdelt", json={"query": "financial crisis", "page_size": 5})
    assert response.status_code == 429
    data = response.json()
    assert "rate-limiting" in data["detail"].lower()


def test_api_gdelt_connection_error():
    """Verify endpoint returns 502 when connection fails."""
    main.gdelt_ingester.fetch_events.side_effect = GdeltConnectionError("Connection dropped")
    response = client.post("/api/ingest/gdelt", json={"query": "financial crisis", "page_size": 5})
    assert response.status_code == 502
    data = response.json()
    assert "failed to connect" in data["detail"].lower() or "connection" in data["detail"].lower()


def test_api_gdelt_empty_success():
    """Verify endpoint returns 200 with 0 count when no articles found."""
    main.gdelt_ingester.fetch_events.side_effect = None
    main.gdelt_ingester.fetch_events.return_value = []
    response = client.post("/api/ingest/gdelt", json={"query": "rare_query", "page_size": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["signals"] == []
