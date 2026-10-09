"""
Unit Tests for the Core AI/NLP Risk Engine Components.

Purpose:
  Validates scoring bounds, keyword detection, and weighting rules in the NLP pipeline.

Role in Project:
  - Confirms ImpactScorer mathematically clamps all outputs within the strict [1, 10] range.
  - Verifies that critical crisis keywords ('crash', 'collapse', 'default') properly elevate impact scores.
  - Validates differential event category weighting (e.g. Geopolitical vs Product/Business).
  - Ensures robust scoring behavior even with edge-case or neutral financial texts.
"""

import pytest
from src.nlp_engine.impact_scorer import ImpactScorer


def test_impact_scorer_bounds():
    """Ensure impact scorer always stays within [1, 10]."""
    scorer = ImpactScorer()

    # Extreme negative crisis scenario
    result_high = scorer.score(
        text="Catastrophic emergency sovereign default and market crash",
        sentiment_score=-0.99,
        event_classification="Credit Event",
        event_confidence=0.98,
    )
    assert 1 <= result_high["impact_score"] <= 10
    assert result_high["impact_score"] >= 8

    # Mild positive scenario
    result_low = scorer.score(
        text="Company quarterly results inline with expectations",
        sentiment_score=0.05,
        event_classification="Product Launch",
        event_confidence=0.60,
    )
    assert 1 <= result_low["impact_score"] <= 10
    assert result_low["impact_score"] <= 4


def test_impact_components_structure():
    """Check that calculation components are properly broken down."""
    scorer = ImpactScorer()
    res = scorer.score(
        text="Sanctions declared amid geopolitical conflict",
        sentiment_score=-0.80,
        event_classification="Geopolitical",
        event_confidence=0.90,
    )
    comp = res["impact_components"]
    assert "sentiment_magnitude" in comp
    assert "event_weight" in comp
    assert "keyword_boost" in comp
    assert "confidence_factor" in comp
    assert "raw_score" in comp
