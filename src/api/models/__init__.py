"""
API Models Package Initialization.

Purpose:
  Exports all Pydantic request and response schemas for FastAPI endpoints.

Role in Project:
  Provides a clean, centralized namespace for importing data transfer objects (DTOs),
  ensuring consistent type checking and payload serialization across tests,
  routes, and client integrations.
"""

from src.api.models.schemas import (
    AnalyzeRequest,
    IngestRequest,
    IngestTickersRequest,
    RebalanceRequest,
    StressTestRequest,
    SentimentResult,
    ClassificationResult,
    RiskSignalResponse,
    PortfolioWeightsResponse,
    RebalanceResponse,
    StressTestAssetResult,
    StressTestResponse,
)

__all__ = [
    "AnalyzeRequest",
    "IngestRequest",
    "IngestTickersRequest",
    "RebalanceRequest",
    "StressTestRequest",
    "SentimentResult",
    "ClassificationResult",
    "RiskSignalResponse",
    "PortfolioWeightsResponse",
    "RebalanceResponse",
    "StressTestAssetResult",
    "StressTestResponse",
]