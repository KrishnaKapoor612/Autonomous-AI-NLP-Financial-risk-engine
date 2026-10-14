"""
FastAPI Application — Main Entry Point.

Exposes the NLP Risk Engine, Index Rebalancer, and Stress Tester
as REST API endpoints with auto-generated Swagger docs.

Run: uvicorn src.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

from src.nlp_engine.risk_engine import RiskEngine
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
from src.module_a.rebalancer import IndexRebalancer
from src.module_b.stress_tester import StressTester
from src.api.models.schemas import (
    AnalyzeRequest,
    IngestRequest,
    IngestTickersRequest,
    RebalanceRequest,
    StressTestRequest,
    RiskSignalResponse,
    PortfolioWeightsResponse,
    RebalanceResponse,
    StressTestResponse,
)

# ─── Initialize App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="AI/NLP Financial Risk Engine",
    description="Real-time financial risk intelligence powered by FinBERT and Zero-Shot NLP",
    version="1.0.0",
)

# Allow CORS for Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Initialize Services (loaded once at startup) ───────────────────────────
risk_engine: RiskEngine | None = None
news_ingester: NewsIngester | None = None
gdelt_ingester: GdeltIngester | None = None
rebalancer: IndexRebalancer | None = None
stress_tester: StressTester | None = None


@app.on_event("startup")
async def startup():
    """Load all models and services at startup."""
    global risk_engine, news_ingester, gdelt_ingester, rebalancer, stress_tester
    risk_engine = RiskEngine()
    news_ingester = NewsIngester()
    gdelt_ingester = GdeltIngester()
    rebalancer = IndexRebalancer()
    stress_tester = StressTester()


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AI/NLP Financial Risk Engine",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "engine_loaded": risk_engine is not None}


# ─── NLP Risk Engine Endpoints ──────────────────────────────────────────────

@app.post("/api/analyze", tags=["Risk Engine"])
def analyze_text(request: AnalyzeRequest):
    """Analyze a single piece of text and return a complete risk signal."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    signal = risk_engine.analyze(
        text=request.text,
        source=request.source,
        ticker=request.ticker,
        url=request.url,
    )
    return signal


@app.post("/api/analyze/batch", tags=["Risk Engine"])
def analyze_batch(items: list[AnalyzeRequest]):
    """Analyze multiple texts in a batch."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    signals = risk_engine.analyze_batch([item.model_dump() for item in items])
    return {"signals": signals, "count": len(signals)}


@app.get("/api/signals", tags=["Risk Engine"])
def get_signals(limit: int = Query(default=50, ge=1, le=500)):
    """Get all recent risk signals from history."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    signals = risk_engine.get_signals_history()
    return {"signals": signals[-limit:], "total": len(signals)}


@app.get("/api/signals/high-impact", tags=["Risk Engine"])
def get_high_impact_signals(threshold: int = Query(default=7, ge=1, le=10)):
    """Get signals with impact score above threshold."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    signals = risk_engine.get_high_impact_signals(threshold)
    return {"threshold": threshold, "signals": signals, "count": len(signals)}


@app.get("/api/signals/{ticker}", tags=["Risk Engine"])
def get_signals_by_ticker(ticker: str):
    """Get risk signals for a specific stock ticker."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    signals = risk_engine.get_signals_by_ticker(ticker.upper())
    return {"ticker": ticker.upper(), "signals": signals, "count": len(signals)}


# ─── Data Ingestion Endpoints ───────────────────────────────────────────────

@app.post("/api/ingest/news", tags=["Ingestion"])
def ingest_news(request: IngestRequest):
    """Fetch news from NewsAPI, analyze with engine, return signals."""
    if not risk_engine or not news_ingester:
        raise HTTPException(status_code=503, detail="Services not loaded")

    try:
        articles = news_ingester.fetch_news(
            query=request.query,
            page_size=request.page_size,
            days_back=request.days_back,
        )
    except NewsApiKeyMissingError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e) or "NewsAPI key is missing or invalid. Set NEWSAPI_KEY in .env",
        )
    except NewsApiRateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e) or "NewsAPI daily rate limit reached.",
        )
    except NewsApiError as e:
        raise HTTPException(
            status_code=500,
            detail=f"NewsAPI error: {e}",
        )

    if not articles:
        return {"message": "No articles found matching the query", "signals": [], "count": 0, "source": "newsapi"}

    signals = risk_engine.analyze_batch(articles)
    return {"signals": signals, "count": len(signals), "source": "newsapi"}


@app.post("/api/ingest/gdelt", tags=["Ingestion"])
def ingest_gdelt(request: IngestRequest):
    """Fetch events from GDELT, analyze with engine, return signals."""
    if not risk_engine or not gdelt_ingester:
        raise HTTPException(status_code=503, detail="Services not loaded")

    timespan = f"{request.days_back}d" if request.days_back else "7d"
    try:
        articles = gdelt_ingester.fetch_events(
            keyword=request.query,
            timespan=timespan,
            max_records=request.page_size,
        )
    except GdeltRateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e) or "GDELT is temporarily rate-limiting requests. Please wait a few seconds and try again.",
        )
    except GdeltConnectionError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e) or "Failed to connect to GDELT API. Please check your network or try again.",
        )
    except GdeltError as e:
        raise HTTPException(
            status_code=500,
            detail=f"GDELT ingestion failed: {e}",
        )

    if not articles:
        return {"message": "No articles found matching the query", "signals": [], "count": 0, "source": "gdelt"}

    signals = risk_engine.analyze_batch(articles)
    return {"signals": signals, "count": len(signals), "source": "gdelt"}


@app.post("/api/ingest/tickers", tags=["Ingestion"])
def ingest_for_tickers(request: IngestTickersRequest):
    """Fetch news for specific tickers and analyze."""
    if not risk_engine or not news_ingester:
        raise HTTPException(status_code=503, detail="Services not loaded")

    from config.settings import MOCK_INDEX_TICKERS
    tickers = request.tickers or list(MOCK_INDEX_TICKERS.keys())

    try:
        articles = news_ingester.fetch_for_tickers(tickers, page_size=request.page_size)
    except NewsApiKeyMissingError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e) or "NewsAPI key is missing or invalid. Set NEWSAPI_KEY in .env",
        )
    except NewsApiRateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e) or "NewsAPI daily rate limit reached.",
        )
    except NewsApiError as e:
        raise HTTPException(
            status_code=500,
            detail=f"NewsAPI error: {e}",
        )

    if not articles:
        return {"message": "No articles found", "signals": [], "count": 0, "tickers_searched": tickers}

    signals = risk_engine.analyze_batch(articles)
    return {"signals": signals, "count": len(signals), "tickers_searched": tickers}


# ─── Module A: Index Rebalancer Endpoints ───────────────────────────────────

@app.get("/api/portfolio/weights", tags=["Module A: Rebalancer"])
def get_portfolio_weights():
    """Get current portfolio weights."""
    if not rebalancer:
        raise HTTPException(status_code=503, detail="Rebalancer not loaded")

    weights = rebalancer.get_weights()
    return {
        "weights": weights,
        "last_rebalanced": rebalancer.last_rebalanced,
        "total_weight": round(sum(weights.values()), 4),
    }


@app.post("/api/portfolio/rebalance", tags=["Module A: Rebalancer"])
def rebalance_portfolio(request: RebalanceRequest = RebalanceRequest()):
    """Rebalance portfolio based on current risk signals."""
    if not rebalancer or not risk_engine:
        raise HTTPException(status_code=503, detail="Services not loaded")

    signals = risk_engine.get_signals_history()
    if not signals:
        return {"message": "No signals available. Ingest data first.", "changes": {}}

    result = rebalancer.rebalance(
        signals,
        adjustment_factor=request.adjustment_factor,
        sentiment_threshold=request.sentiment_threshold,
    )
    return result


@app.get("/api/portfolio/summary", tags=["Module A: Rebalancer"])
def get_portfolio_summary():
    """Get portfolio summary with top/bottom holdings."""
    if not rebalancer:
        raise HTTPException(status_code=503, detail="Rebalancer not loaded")
    return rebalancer.get_summary()


@app.get("/api/portfolio/history", tags=["Module A: Rebalancer"])
def get_rebalance_history():
    """Get rebalancing history."""
    if not rebalancer:
        raise HTTPException(status_code=503, detail="Rebalancer not loaded")
    return {"history": rebalancer.get_history()}


@app.post("/api/portfolio/reset", tags=["Module A: Rebalancer"])
def reset_portfolio():
    """Reset portfolio to initial weights."""
    if not rebalancer:
        raise HTTPException(status_code=503, detail="Rebalancer not loaded")
    rebalancer.reset_weights()
    return {"message": "Portfolio reset to initial weights", "weights": rebalancer.get_weights()}


# ─── Module B: Stress Test Endpoints ────────────────────────────────────────

@app.get("/api/stress-test/portfolio", tags=["Module B: Stress Test"])
def get_stress_portfolio():
    """Get the synthetic portfolio used for stress testing."""
    if not stress_tester:
        raise HTTPException(status_code=503, detail="Stress tester not loaded")
    return {
        "portfolio": stress_tester.get_portfolio(),
        "total_value": stress_tester.get_portfolio_value(),
    }


@app.post("/api/stress-test/run", tags=["Module B: Stress Test"])
def run_stress_test(request: StressTestRequest):
    """Run a stress test with a specific event type and impact score."""
    if not stress_tester:
        raise HTTPException(status_code=503, detail="Stress tester not loaded")

    result = stress_tester.run_stress_test(
        event_type=request.event_type,
        impact_score=request.impact_score,
    )
    return result


@app.post("/api/stress-test/auto", tags=["Module B: Stress Test"])
def auto_stress_test():
    """Automatically run stress test if any high-impact signals exist."""
    if not stress_tester or not risk_engine:
        raise HTTPException(status_code=503, detail="Services not loaded")

    high_impact = risk_engine.get_high_impact_signals(threshold=7)
    if not high_impact:
        return {"message": "No high-impact signals detected. No stress test triggered."}

    # Use the highest impact signal
    worst = max(high_impact, key=lambda s: s["impact_score"])

    result = stress_tester.run_stress_test(
        event_type=worst["event_classification"],
        impact_score=worst["impact_score"],
    )

    return {
        "triggered_by": {
            "text": worst["text"][:200],
            "impact_score": worst["impact_score"],
            "event_type": worst["event_classification"],
        },
        "stress_test_result": result,
    }


@app.get("/api/stress-test/history", tags=["Module B: Stress Test"])
def get_stress_test_history():
    """Get stress test history."""
    if not stress_tester:
        raise HTTPException(status_code=503, detail="Stress tester not loaded")
    return {"history": stress_tester.get_history()}
