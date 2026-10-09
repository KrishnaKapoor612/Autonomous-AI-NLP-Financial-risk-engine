"""
GDELT Data Ingester.

Fetches global event data from the GDELT 2.0 Doc API.
GDELT monitors global news media and categorizes events/themes.

No API key required — completely free.
Rate Limit Policy: GDELT limits requests to one every 5 seconds.
Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("gdelt_ingester")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    from gdeltdoc import GdeltDoc, Filters
    from gdeltdoc.errors import RateLimitError, ClientRequestError, ServerError
    GDELT_AVAILABLE = True
except ImportError:
    GDELT_AVAILABLE = False
    RateLimitError = Exception
    ClientRequestError = Exception
    ServerError = Exception
    logger.warning("gdeltdoc not installed. Run: pip install gdeltdoc")


# ─── Custom Exceptions ───────────────────────────────────────────────────────

class GdeltError(Exception):
    """Base exception for GDELT API errors."""
    pass


class GdeltRateLimitError(GdeltError):
    """Raised when GDELT returns HTTP 429 (rate-limited)."""
    def __init__(
        self,
        message: str = "GDELT is temporarily rate-limiting requests. Please wait a few seconds and try again.",
        retry_after: Optional[int] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


class GdeltConnectionError(GdeltError):
    """Raised when network / SSL connection to GDELT fails."""
    pass


# ─── Centralized Request Pacer / Rate Limiter ────────────────────────────────

class GdeltRateLimiter:
    """
    Centralized rate limiter ensuring safe request pacing for GDELT DOC API.
    GDELT mandates a strict minimum 5-second interval between requests.
    """

    def __init__(self, min_interval_seconds: float = 5.5):
        self.min_interval = min_interval_seconds
        self.last_request_time: float = 0.0
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Pause execution if necessary to respect the minimum interval between requests."""
        with self._lock:
            now = time.monotonic()
            if self.last_request_time > 0:
                elapsed = now - self.last_request_time
                if elapsed < self.min_interval:
                    wait_seconds = self.min_interval - elapsed
                    logger.info(
                        f"Pacing GDELT request: waiting {wait_seconds:.2f}s "
                        f"to respect rate limit (minimum {self.min_interval}s interval)..."
                    )
                    time.sleep(wait_seconds)
            self.last_request_time = time.monotonic()

    def record_request(self):
        """Update timestamp of last request."""
        with self._lock:
            self.last_request_time = time.monotonic()


# Singleton rate limiter instance for the process
_rate_limiter = GdeltRateLimiter(min_interval_seconds=5.5)


# ─── Ingester Class ──────────────────────────────────────────────────────────

class GdeltIngester:
    """Fetch global event data from GDELT 2.0 Doc API with rate-limiting and error handling."""

    def __init__(self):
        if GDELT_AVAILABLE:
            self.client = GdeltDoc()
            logger.info("GDELT client initialized.")
        else:
            self.client = None
            logger.warning("gdeltdoc not available.")
        self.limiter = _rate_limiter

    def fetch_events(
        self,
        keyword: str,
        timespan: str = "1d",
        max_records: int = 10,
        country: Optional[str] = None,
        language: Optional[str] = "English",
    ) -> list[dict]:
        """
        Fetch event articles from GDELT matching a keyword.

        Args:
            keyword: Search keyword (e.g., "financial crisis", "interest rate").
            timespan: Timespan to search (e.g., "1d", "3d", "1w").
            max_records: Maximum number of articles to return (1-250).
            country: Optional 2-letter country code (e.g., "US").
            language: Language filter (defaults to "English" for FinBERT).

        Returns:
            List of normalized article dictionaries.

        Raises:
            GdeltRateLimitError: If GDELT throttles/rate-limits (HTTP 429).
            GdeltConnectionError: If network or SSL connection fails.
            GdeltError: For other API failures or invalid queries.
        """
        if not GDELT_AVAILABLE or not self.client:
            raise GdeltError("gdeltdoc library is not available.")

        # Clean and validate parameters
        clean_keyword = keyword.strip()
        if not clean_keyword:
            return []

        # Auto-expand recognized ticker to company name to avoid GDELT "phrase is too short" constraint
        from config.settings import TICKER_TO_NAME
        search_query = TICKER_TO_NAME.get(clean_keyword.upper(), clean_keyword)
        if len(search_query) <= 4 and " " not in search_query:
            search_query = f"{search_query} stock"

        # Bound records between 1 and 250 (GDELT API maximum)
        clamped_records = max(1, min(max_records, 250))

        logger.info(
            f"GDELT request started: query='{search_query}' (input='{clean_keyword}'), "
            f"timespan='{timespan}', max_records={clamped_records}"
        )

        # Build filter kwargs with explicit num_records and English language
        filter_kwargs = {
            "keyword": search_query,
            "timespan": timespan,
            "num_records": clamped_records,
        }
        if language:
            filter_kwargs["language"] = language
        if country:
            filter_kwargs["country"] = country

        # Execute query with rate limit pacing and conservative retry
        return self._execute_search_with_retry(filter_kwargs, search_query, clamped_records)

    def _execute_search_with_retry(
        self,
        filter_kwargs: dict,
        keyword: str,
        max_records: int,
        max_retries: int = 1,
    ) -> list[dict]:
        """
        Execute article_search with pacing, explicit error classification, and at most 1 retry.
        """
        attempts = 0
        while attempts <= max_retries:
            attempts += 1

            # 1. Respect minimum delay between requests
            self.limiter.wait_if_needed()

            try:
                f = Filters(**filter_kwargs)
                df = self.client.article_search(f)

                if df is None or df.empty:
                    logger.info(f"GDELT returned 0 articles for keyword='{keyword}'")
                    return []

                articles = self._format_articles(df.head(max_records))
                logger.info(f"GDELT returned {len(articles)} articles for keyword='{keyword}'")
                return articles

            except RateLimitError as e:
                logger.warning(
                    f"GDELT returned HTTP 429 (Rate limited) on attempt {attempts}/{max_retries + 1} "
                    f"for keyword='{keyword}'"
                )
                if attempts <= max_retries:
                    # Conservative backoff: wait 6 seconds and retry once
                    wait_time = 6.0
                    logger.info(f"Retrying GDELT request in {wait_time}s...")
                    time.sleep(wait_time)
                    self.limiter.record_request()
                    continue
                else:
                    raise GdeltRateLimitError(
                        "GDELT is temporarily rate-limiting requests. Please wait a few seconds and try again."
                    ) from e

            except (ClientRequestError, ServerError) as e:
                # Check if the error text indicates a rate limit
                err_text = str(e).lower()
                if "429" in err_text or "limit requests" in err_text:
                    logger.warning(f"GDELT rate limit detected in HTTP error: {e}")
                    raise GdeltRateLimitError(
                        "GDELT is temporarily rate-limiting requests. Please wait a few seconds and try again."
                    ) from e
                logger.error(f"GDELT HTTP error: {e}")
                raise GdeltError(f"GDELT API error: {e}") from e

            except ValueError as e:
                # gdeltdoc raises ValueError when GDELT responds with HTML error messages
                err_text = str(e).lower()
                if "limit requests" in err_text or "one every 5 seconds" in err_text or "429" in err_text:
                    logger.warning(f"GDELT rate limit detected in API response text: {e}")
                    if attempts <= max_retries:
                        logger.info("Retrying GDELT request after 6s backoff...")
                        time.sleep(6.0)
                        self.limiter.record_request()
                        continue
                    raise GdeltRateLimitError(
                        "GDELT is temporarily rate-limiting requests. Please wait a few seconds and try again."
                    ) from e
                logger.error(f"GDELT query validation error: {e}")
                raise GdeltError(f"Invalid GDELT query: {e}") from e

            except Exception as e:
                # Check for SSL / Network connection drops
                err_name = type(e).__name__.lower()
                err_str = str(e).lower()

                # GDELT triggers TLS renegotiation on rate-limiting, which causes SSLEOFError in Python
                if "eof occurred in violation of protocol" in err_str or "ssleoferror" in err_name or "ssleoferror" in err_str:
                    logger.warning(
                        f"GDELT TLS renegotiation/cooldown detected on attempt {attempts}/{max_retries + 1}: {e}"
                    )
                    if attempts <= max_retries:
                        logger.info("Retrying GDELT request after 6s backoff...")
                        time.sleep(6.0)
                        self.limiter.record_request()
                        continue
                    raise GdeltRateLimitError(
                        "GDELT is temporarily rate-limiting requests. Please wait a few seconds and try again."
                    ) from e

                if "ssl" in err_name or "connection" in err_name or "timeout" in err_name or "ssl" in err_str or "connection" in err_str or "timeout" in err_str:
                    logger.error(f"GDELT network/SSL connection failure ({type(e).__name__}): {e}")
                    raise GdeltConnectionError(
                        "Failed to connect to GDELT API. Please check your network or try again later."
                    ) from e

                logger.error(f"Unexpected error in GDELT search ({type(e).__name__}): {e}")
                raise GdeltError(f"GDELT request failed: {e}") from e

        return []

    def fetch_financial_events(
        self,
        timespan: str = "1d",
        max_records: int = 20,
    ) -> list[dict]:
        """
        Fetch general financial/economic events from GDELT across key topics.
        Safely paces queries to avoid rate limits and stops as soon as max_records is reached.
        """
        keywords = [
            "financial crisis",
            "stock market crash",
            "interest rate hike",
            "economic recession",
            "bond yields",
            "central bank policy",
        ]

        all_articles = []
        for kw in keywords:
            remaining = max_records - len(all_articles)
            if remaining <= 0:
                break

            try:
                articles = self.fetch_events(
                    keyword=kw,
                    timespan=timespan,
                    max_records=min(remaining, 5),
                )
                all_articles.extend(articles)
            except GdeltRateLimitError:
                logger.warning(f"Rate limited while fetching keyword '{kw}' in multi-query batch. Returning gathered results.")
                break
            except GdeltError as e:
                logger.warning(f"Skipping keyword '{kw}' due to GDELT error: {e}")
                continue

        return all_articles[:max_records]

    def _detect_ticker(self, title: str) -> Optional[str]:
        """Detect ticker from article title."""
        if not title:
            return None
        import re
        from config.settings import NAME_TO_TICKER, TICKER_TO_NAME

        title_lower = title.lower()
        for name, ticker in NAME_TO_TICKER.items():
            if re.search(r'\b' + re.escape(name) + r'\b', title_lower):
                return ticker

        for ticker in TICKER_TO_NAME.keys():
            if re.search(r'\b' + re.escape(ticker) + r'\b', title):
                return ticker

        return None

    def _format_articles(self, df) -> list[dict]:
        """Convert GDELT DataFrame to normalized article dicts for the Risk Engine."""
        formatted = []

        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            if not title or title.lower() == "nan":
                continue

            ticker = self._detect_ticker(title)

            formatted.append({
                "text": title,
                "source": "gdelt",
                "ticker": ticker,
                "url": str(row.get("url", "")),
                "published_at": str(row.get("seendate", "")),
                "source_name": str(row.get("domain", "Unknown")),
                "source_country": str(row.get("sourcecountry", "")),
            })

        return formatted
