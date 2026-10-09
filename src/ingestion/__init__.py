"""
Data Ingestion Package Initialization.

Purpose:
  Provides live external news and event retrieval connectors for the platform.

Role in Project:
  - Exports NewsIngester for fetching financial media from NewsAPI.org.
  - Exports GdeltIngester for monitoring global macroeconomic and geopolitical developments via GDELT 2.0.
  - Exports domain-specific error classes for rate limiting and missing API credentials.
"""

from src.ingestion.news_ingester import (
    NewsIngester,
    NewsApiError,
    NewsApiKeyMissingError,
    NewsApiRateLimitError,
)
from src.ingestion.gdelt_ingester import (
    GdeltIngester,
    GdeltError,
    GdeltRateLimitError,
    GdeltConnectionError,
)

__all__ = [
    "NewsIngester",
    "NewsApiError",
    "NewsApiKeyMissingError",
    "NewsApiRateLimitError",
    "GdeltIngester",
    "GdeltError",
    "GdeltRateLimitError",
    "GdeltConnectionError",
]