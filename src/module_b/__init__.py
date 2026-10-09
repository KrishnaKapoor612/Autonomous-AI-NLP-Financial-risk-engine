"""
Module B Package Initialization: Strategic Portfolio Stress Tester.

Purpose:
  Simulates extreme financial macro scenarios on an institutional wholesale banking book.

Role in Project:
  - Exports StressTester for simulating asset shocks across a $40.3M multi-asset portfolio.
  - Automatically activates when high-impact events (Impact Score >= 7) are detected.
  - Calculates asset-by-asset losses and dollar drawdown to stress capital adequacy.
"""

from src.module_b.stress_tester import StressTester

__all__ = [
    "StressTester",
]