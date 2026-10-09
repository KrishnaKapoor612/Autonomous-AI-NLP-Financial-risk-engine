"""
Unit Tests for GDELT Data Ingester and Rate Limiter.

Purpose:
  Validates the internal mechanics of the GDELT 2.0 Doc API ingestion pipeline.

Role in Project:
  - Tests thread-safe pacing logic in GdeltRateLimiter to guarantee adherence to GDELT policy.
  - Verifies DataFrame-to-dictionary normalization and field formatting for downstream NLP consumption.
  - Tests error handling for network connection timeouts, empty search results, and API exceptions.
"""

import time
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.gdelt_ingester import (
    GdeltIngester,
    GdeltRateLimiter,
    GdeltError,
    GdeltRateLimitError,
    GdeltConnectionError,
)
from gdeltdoc.errors import RateLimitError


def test_rate_limiter_pacing():
    """Verify that GdeltRateLimiter delays calls that occur under min_interval."""
    limiter = GdeltRateLimiter(min_interval_seconds=0.2)
    start = time.monotonic()
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.18


def test_fetch_events_success():
    """Verify successful fetching and normalization of articles."""
    ingester = GdeltIngester()
    fake_df = pd.DataFrame([
        {
            "title": "Fed announces policy shift amid market tension",
            "url": "https://example.com/fed-news",
            "seendate": "20261006T060000Z",
            "domain": "reuters.com",
            "sourcecountry": "US",
        }
    ])

    with patch.object(ingester.limiter, "wait_if_needed"):
        with patch.object(ingester.client, "article_search", return_value=fake_df) as mock_search:
            articles = ingester.fetch_events(keyword="fed", max_records=5)
            assert len(articles) == 1
            assert articles[0]["text"] == "Fed announces policy shift amid market tension"
            assert articles[0]["source"] == "gdelt"
            assert articles[0]["url"] == "https://example.com/fed-news"
            # Ensure num_records was passed to Filters
            mock_search.assert_called_once()
            filter_arg = mock_search.call_args[0][0]
            assert "&maxrecords=5" in filter_arg.query_string


def test_fetch_events_rate_limit_error():
    """Verify that HTTP 429 raises GdeltRateLimitError after retrying."""
    ingester = GdeltIngester()
    with patch.object(ingester.limiter, "wait_if_needed"):
        with patch.object(ingester.client, "article_search", side_effect=RateLimitError()):
            with patch("time.sleep", return_value=None):
                with pytest.raises(GdeltRateLimitError):
                    ingester.fetch_events(keyword="financial crisis", max_records=5)


def test_fetch_events_ssl_error():
    """Verify that SSL/Connection failures raise GdeltConnectionError."""
    import requests
    ingester = GdeltIngester()
    with patch.object(ingester.limiter, "wait_if_needed"):
        with patch.object(ingester.client, "article_search", side_effect=requests.exceptions.SSLError("EOF protocol violation")):
            with pytest.raises(GdeltConnectionError):
                ingester.fetch_events(keyword="market", max_records=5)


def test_fetch_events_empty_result():
    """Verify that legitimate zero-result queries return an empty list without error."""
    ingester = GdeltIngester()
    empty_df = pd.DataFrame()
    with patch.object(ingester.limiter, "wait_if_needed"):
        with patch.object(ingester.client, "article_search", return_value=empty_df):
            articles = ingester.fetch_events(keyword="nonexistentqueryxyz", max_records=5)
            assert articles == []
