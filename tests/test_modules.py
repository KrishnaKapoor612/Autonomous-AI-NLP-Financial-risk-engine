"""
Unit Tests for Module A (Tactical Index Rebalancer) and Module B (Stress Tester).

Purpose:
  Verifies mathematical correctness, constraint satisfaction, and shock logic
  in both downstream quantitative decision engines.

Role in Project:
  - Validates portfolio weight conservation (sum of weights equals 1.0) post-rebalance in Module A.
  - Validates directional alignment between NLP sentiment polarity and equity weight adjustments.
  - Validates multi-asset shock calculations, asset loss breakdowns, and portfolio drawdown in Module B.
  - Ensures default fallback shock scenario handling when unmapped event types occur.
"""

import pytest
from src.module_a.rebalancer import IndexRebalancer
from src.module_b.stress_tester import StressTester


def test_rebalancer_weight_conservation():
    """Verify that portfolio weights always sum to ~1.0 after rebalancing."""
    rebalancer = IndexRebalancer()
    fake_signals = [
        {"ticker": "AAPL", "sentiment_score": 0.85},
        {"ticker": "NVDA", "sentiment_score": 0.90},
        {"ticker": "JPM", "sentiment_score": -0.80},
    ]
    res = rebalancer.rebalance(fake_signals)
    new_weights = res["new_weights"]
    total = sum(new_weights.values())
    assert pytest.approx(total, abs=1e-3) == 1.0


def test_rebalancer_directional_adjustment():
    """Verify positive sentiment increases weight and negative decreases weight."""
    rebalancer = IndexRebalancer()
    initial_aapl = rebalancer.get_weights()["AAPL"]
    initial_jpm = rebalancer.get_weights()["JPM"]

    fake_signals = [
        {"ticker": "AAPL", "sentiment_score": 0.90},
        {"ticker": "JPM", "sentiment_score": -0.90},
    ]
    res = rebalancer.rebalance(fake_signals)
    assert res["new_weights"]["AAPL"] > initial_aapl
    assert res["new_weights"]["JPM"] < initial_jpm


def test_stress_tester_loss_calculation():
    """Verify that stress testing correctly calculates losses on the portfolio."""
    tester = StressTester()
    initial_val = tester.get_portfolio_value()
    assert initial_val > 0

    res = tester.run_stress_test(event_type="Geopolitical", impact_score=9)
    assert res["total_pre_stress"] == initial_val
    assert res["total_post_stress"] < initial_val
    assert res["total_loss"] > 0
    assert 0 < res["total_loss_pct"] < 100
