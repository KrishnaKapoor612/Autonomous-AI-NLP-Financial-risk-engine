"""
Module A Package Initialization: Tactical Index Rebalancer.

Purpose:
  Provides automated equity index weight rebalancing driven by sentiment signals.

Role in Project:
  - Exports IndexRebalancer for tactical shifts in a 15-stock S&P benchmark basket.
  - Dynamically rewards positive sentiment momentum and trims negative exposure.
  - Enforces portfolio normalization (sum of weights = 1.0) and allocation guardrails.
"""

from src.module_a.rebalancer import IndexRebalancer

__all__ = [
    "IndexRebalancer",
]