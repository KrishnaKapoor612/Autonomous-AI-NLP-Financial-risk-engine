"""
NewsAPI Data Ingester.

Fetches financial news articles from NewsAPI.org and formats them
for consumption by the NLP Risk Engine.

Free tier: 100 requests/day, up to 100 articles per request.
Docs: https://newsapi.org/docs
"""

import logging
import os
import requests
from datetime import datetime, timedelta
from typing import Optional

import sys
sys.path.insert(0, ".")
from config.settings import NEWSAPI_KEY, TICKER_TO_NAME

logger = logging.getLogger("news_ingester")


# ─── Custom Exceptions ───────────────────────────────────────────────────────

class NewsApiError(Exception):
    """Base exception for NewsAPI errors."""
    pass


class NewsApiKeyMissingError(NewsApiError):
    """Raised when NewsAPI key is missing or placeholder."""
    pass


class NewsApiRateLimitError(NewsApiError):
    """Raised when NewsAPI quota (100 req/day) is exceeded."""
    pass


# ─── Ingester Class ──────────────────────────────────────────────────────────

class NewsIngester:
    """Fetch financial news from NewsAPI.org."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY") or NEWSAPI_KEY
        if self._is_key_missing():
            logger.warning("[NewsIngester] No valid NewsAPI key configured. Set NEWSAPI_KEY in .env")

    def _is_key_missing(self) -> bool:
        return not self.api_key or self.api_key.strip() in ("", "your_newsapi_key_here")

    def fetch_news(
        self,
        query: str,
        page_size: int = 10,
        days_back: int = 7,
        language: str = "en",
        sort_by: str = "publishedAt",
    ) -> list[dict]:
        """
        Fetch news articles matching a query.

        Args:
            query: Search query (e.g., "Apple stock", "Federal Reserve").
            page_size: Number of articles to return (max 100 on free tier).
            days_back: How many days back to search.
            language: Language filter.
            sort_by: "publishedAt", "relevancy", or "popularity".

        Returns:
            List of dicts formatted for the NLP Risk Engine.

        Raises:
            NewsApiKeyMissingError: If API key is missing.
            NewsApiRateLimitError: If daily quota is reached.
            NewsApiError: If query fails.
        """
        # Always reload in case .env was updated
        current_key = os.getenv("NEWSAPI_KEY") or self.api_key
        if not current_key or current_key.strip() in ("", "your_newsapi_key_here"):
            raise NewsApiKeyMissingError(
                "NewsAPI key is missing or unconfigured. Please add your free API key in .env (NEWSAPI_KEY=your_key) from https://newsapi.org/register."
            )

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "language": language,
            "sortBy": sort_by,
            "pageSize": max(1, min(page_size, 100)),
            "apiKey": current_key,
        }

        try:
            response = requests.get(f"{self.BASE_URL}/everything", params=params, timeout=12)
            data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}

            if response.status_code == 401:
                msg = data.get("message", "API key invalid or unauthorized.")
                raise NewsApiKeyMissingError(f"NewsAPI authentication failed: {msg}")

            if response.status_code == 429:
                msg = data.get("message", "Daily rate limit reached.")
                raise NewsApiRateLimitError(f"NewsAPI rate limit: {msg}")

            response.raise_for_status()

            if data.get("status") != "ok":
                raise NewsApiError(data.get("message", "Unknown NewsAPI error"))

            articles = data.get("articles", [])
            return self._format_articles(articles, query)

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                raise NewsApiKeyMissingError("Invalid NewsAPI key. Please check your NEWSAPI_KEY in .env.") from e
            if e.response is not None and e.response.status_code == 429:
                raise NewsApiRateLimitError("NewsAPI request limit reached (100 requests/day).") from e
            raise NewsApiError(f"NewsAPI HTTP error: {e}") from e

        except requests.RequestException as e:
            raise NewsApiError(f"NewsAPI connection error: {e}") from e

    def fetch_headlines(
        self,
        category: str = "business",
        country: str = "us",
        page_size: int = 10,
    ) -> list[dict]:
        """Fetch top headlines in a category."""
        current_key = os.getenv("NEWSAPI_KEY") or self.api_key
        if not current_key or current_key.strip() in ("", "your_newsapi_key_here"):
            raise NewsApiKeyMissingError("NewsAPI key is missing in .env.")

        params = {
            "category": category,
            "country": country,
            "pageSize": max(1, min(page_size, 100)),
            "apiKey": current_key,
        }

        try:
            response = requests.get(f"{self.BASE_URL}/top-headlines", params=params, timeout=12)
            data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            if response.status_code == 401:
                msg = data.get("message", "API key invalid or unauthorized.")
                raise NewsApiKeyMissingError(f"NewsAPI authentication failed: {msg}")
            if response.status_code == 429:
                msg = data.get("message", "Daily rate limit reached.")
                raise NewsApiRateLimitError(f"NewsAPI rate limit: {msg}")
            response.raise_for_status()
            if data.get("status") != "ok":
                raise NewsApiError(data.get("message", "Unknown NewsAPI error"))
            return self._format_articles(data.get("articles", []))
        except NewsApiError:
            raise
        except requests.RequestException as e:
            raise NewsApiError(f"Failed to fetch headlines: {e}") from e

    def fetch_for_tickers(self, tickers: list[str], page_size: int = 3) -> list[dict]:
        """Fetch news for multiple stock tickers."""
        all_articles = []
        for ticker in tickers:
            company_name = TICKER_TO_NAME.get(ticker, ticker)
            query = f'"{company_name}" OR "{ticker}" stock'
            articles = self.fetch_news(query, page_size=page_size)
            for article in articles:
                article["ticker"] = ticker
            all_articles.extend(articles)
        return all_articles

    def _format_articles(self, articles: list[dict], query: str = "") -> list[dict]:
        """Convert raw NewsAPI articles into the format expected by the Risk Engine."""
        formatted = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")

            if not title or title == "[Removed]":
                continue

            text = title
            if description and description != "[Removed]":
                text = f"{title}. {description}"

            ticker = self._detect_ticker(query, text=text)

            formatted.append({
                "text": text,
                "source": "newsapi",
                "ticker": ticker,
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "source_name": article.get("source", {}).get("name", "Unknown"),
            })

        return formatted

    def _detect_ticker(self, query: str, text: str = "") -> Optional[str]:
        """Try to detect a stock ticker from the search query or article text."""
        import re
        from config.settings import NAME_TO_TICKER, TICKER_TO_NAME

        # 1. Match from search query
        query_upper = query.upper().strip()
        if query_upper in TICKER_TO_NAME:
            return query_upper

        query_lower = query.lower().strip()
        for name, ticker in NAME_TO_TICKER.items():
            if name in query_lower:
                return ticker

        # 2. Match from article title & text if query is generic
        if text:
            # Check uppercase ticker symbols as standalone words (e.g. "NVDA", "AAPL")
            for ticker in TICKER_TO_NAME.keys():
                if re.search(r'\b' + re.escape(ticker) + r'\b', text):
                    return ticker

            # Check lowercase company names as standalone words (e.g. "nvidia", "apple")
            text_lower = text.lower()
            for name, ticker in NAME_TO_TICKER.items():
                if re.search(r'\b' + re.escape(name) + r'\b', text_lower):
                    return ticker

        return None
