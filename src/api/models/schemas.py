"""
Pydantic Data Schemas and Validation Models.

Purpose:
  Defines strongly-typed request bodies, query payloads, and response structures
  for all REST endpoints in the FastAPI application.

Role in Project:
  - Validates inbound requests for text analysis, news ingestion, rebalancing, and stress testing.
  - Serializes outbound structured Risk Signals, event classifications, and portfolio updates.
  - Powers automatic OpenAPI (Swagger) schema generation and interactive documentation.
  - Ensures clean data contracts between backend computation engines and the Streamlit frontend.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─── Request Models ─────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request body for /api/analyze endpoint."""
    text: str = Field(..., description="Financial text to analyze")
    source: str = Field(default="manual", description="Data source identifier")
    ticker: Optional[str] = Field(default=None, description="Stock ticker")
    url: Optional[str] = Field(default=None, description="Source URL")

    model_config = {"json_schema_extra": {
        "examples": [{
            "text": "Apple reports record quarterly revenue of $124 billion",
            "source": "manual",
            "ticker": "AAPL",
        }]
    }}


class IngestRequest(BaseModel):
    """Request body for /api/ingest endpoints."""
    query: str = Field(..., description="Search query")
    page_size: int = Field(default=10, ge=1, le=100)
    days_back: int = Field(default=7, ge=1, le=30)


class IngestTickersRequest(BaseModel):
    """Request body for ingesting news for multiple tickers."""
    tickers: list[str] = Field(
        default=None,
        description="List of tickers. If empty, uses all mock index tickers."
    )
    page_size: int = Field(default=3, ge=1, le=20)


class RebalanceRequest(BaseModel):
    """Request body for /api/portfolio/rebalance."""
    adjustment_factor: Optional[float] = Field(
        default=None, description="Override default adjustment factor"
    )
    sentiment_threshold: Optional[float] = Field(
        default=None, description="Minimum absolute sentiment to trigger adjustment"
    )


class StressTestRequest(BaseModel):
    """Request body for /api/stress-test."""
    event_type: str = Field(
        ..., description="Event type to simulate (e.g., 'Geopolitical')"
    )
    impact_score: int = Field(
        default=8, ge=1, le=10, description="Impact severity (scales the shocks)"
    )


# ─── Response Models ────────────────────────────────────────────────────────

class SentimentResult(BaseModel):
    sentiment_label: str
    sentiment_score: float
    confidence: float


class ClassificationResult(BaseModel):
    event_classification: str
    event_confidence: float


class RiskSignalResponse(BaseModel):
    """Complete risk signal returned by the engine."""
    text: str
    source: str
    ticker: Optional[str]
    url: Optional[str]
    timestamp: str
    sentiment_score: float
    sentiment_label: str
    sentiment_confidence: float
    event_classification: str
    event_confidence: float
    impact_score: int


class PortfolioWeightsResponse(BaseModel):
    """Current portfolio weights."""
    weights: dict[str, float]
    last_rebalanced: Optional[str]
    total_weight: float


class RebalanceResponse(BaseModel):
    """Result of a rebalance operation."""
    previous_weights: dict[str, float]
    new_weights: dict[str, float]
    changes: dict[str, float]
    signals_used: int
    timestamp: str


class StressTestAssetResult(BaseModel):
    """Stress test result for a single asset."""
    name: str
    asset_type: str
    pre_stress_value: float
    post_stress_value: float
    loss: float
    loss_pct: float


class StressTestResponse(BaseModel):
    """Complete stress test result."""
    event_type: str
    scenario_name: str
    impact_score: int
    total_pre_stress: float
    total_post_stress: float
    total_loss: float
    total_loss_pct: float
    asset_results: list[StressTestAssetResult]
    timestamp: str
