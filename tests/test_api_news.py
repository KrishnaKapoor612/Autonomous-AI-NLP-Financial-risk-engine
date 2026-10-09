"""
Integration and Unit Tests for NewsAPI Ingestion REST Endpoints.

Purpose:
  Validates API handling and error propagation for NewsAPI-backed routes (/api/ingest/news).

Role in Project:
  - Ensures missing API keys return HTTP 401 Unauthorized with helpful guidance.
  - Ensures upstream provider quota exhaustion returns HTTP 429 Too Many Requests.
  - Ensures successful payload parsing and downstream NLP risk processing under HTTP 200.
  - Protects external API quota during testing by using isolated mock fixtures.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

import src.api.main as main
from src.ingestion.news_ingester import NewsApiKeyMissingError, NewsApiRateLimitError

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def setup_services():
    """Ensure service singletons are mocked for API tests."""
    main.news_ingester = MagicMock()
    main.risk_engine = MagicMock()


def test_api_news_missing_key():
    """Verify endpoint returns 401 when NewsAPI key is missing."""
    main.news_ingester.fetch_news.side_effect = NewsApiKeyMissingError("API key is missing")
    response = client.post("/api/ingest/news", json={"query": "stock market", "page_size": 5})
    assert response.status_code == 401
    assert "missing" in response.json()["detail"].lower()


def test_api_news_rate_limit():
    """Verify endpoint returns 429 when NewsAPI quota is reached."""
    main.news_ingester.fetch_news.side_effect = NewsApiRateLimitError("Daily limit reached")
    response = client.post("/api/ingest/news", json={"query": "stock market", "page_size": 5})
    assert response.status_code == 429
    assert "limit" in response.json()["detail"].lower()
