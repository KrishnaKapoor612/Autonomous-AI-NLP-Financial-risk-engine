"""
Module A: Tactical Index Rebalancer.

Dynamically adjusts the weights of stocks in a mock S&P index
based on real-time sentiment from the NLP Risk Engine.

Logic:
  - Positive sentiment → increase stock weight
  - Negative sentiment → decrease stock weight
  - Weights are normalized to sum to 1.0
  - Min/Max constraints prevent extreme allocations
"""

from datetime import datetime, timezone
from copy import deepcopy
import math

import sys
sys.path.insert(0, ".")
from config.settings import (
    MOCK_INDEX_TICKERS,
    REBALANCE_ADJUSTMENT_FACTOR,
    REBALANCE_ADJUSTMENT_MIN,
    REBALANCE_ADJUSTMENT_MAX,
    REBALANCE_SENTIMENT_THRESHOLD,
    REBALANCE_SENTIMENT_THRESHOLD_MIN,
    REBALANCE_SENTIMENT_THRESHOLD_MAX,
    REBALANCE_MIN_WEIGHT,
    REBALANCE_MAX_WEIGHT,
    TICKER_TO_NAME,
)


class IndexRebalancer:
    """Manages a mock stock index and rebalances based on sentiment."""

    def __init__(self):
        """Initialize with default equal-ish weights."""
        self.weights = deepcopy(MOCK_INDEX_TICKERS)
        self.initial_weights = deepcopy(MOCK_INDEX_TICKERS)
        self.rebalance_history: list[dict] = []
        self.last_rebalanced: str | None = None
        print(f"[IndexRebalancer] Initialized with {len(self.weights)} stocks.")

    def get_weights(self) -> dict[str, float]:
        """Return current portfolio weights."""
        return deepcopy(self.weights)

    def rebalance(
        self,
        signals: list[dict],
        adjustment_factor: float | None = None,
        sentiment_threshold: float | None = None,
    ) -> dict:
        """
        Rebalance the index based on sentiment signals.

        Args:
            signals: List of risk signals from the NLP engine.
                     Each should have 'ticker' and 'sentiment_score'.
            adjustment_factor: Override the default adjustment strength.
            sentiment_threshold: Override the minimum absolute sentiment threshold.

        Returns:
            dict with previous_weights, new_weights, changes, signals_used.
        """
        factor = adjustment_factor if adjustment_factor is not None else REBALANCE_ADJUSTMENT_FACTOR
        threshold = sentiment_threshold if sentiment_threshold is not None else REBALANCE_SENTIMENT_THRESHOLD
        if not math.isfinite(factor) or not REBALANCE_ADJUSTMENT_MIN <= factor <= REBALANCE_ADJUSTMENT_MAX:
            raise ValueError(
                f"adjustment_factor must be between {REBALANCE_ADJUSTMENT_MIN} and {REBALANCE_ADJUSTMENT_MAX}."
            )
        if not math.isfinite(threshold) or not REBALANCE_SENTIMENT_THRESHOLD_MIN <= threshold <= REBALANCE_SENTIMENT_THRESHOLD_MAX:
            raise ValueError(
                "sentiment_threshold must be between "
                f"{REBALANCE_SENTIMENT_THRESHOLD_MIN} and {REBALANCE_SENTIMENT_THRESHOLD_MAX}."
            )
        previous_weights = deepcopy(self.weights)

        # Aggregate sentiment per ticker
        ticker_sentiments: dict[str, list[float]] = {}
        for signal in signals:
            ticker = signal.get("ticker")
            try:
                sentiment = float(signal.get("sentiment_score", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("Each sentiment_score must be a finite number from -1 to 1.") from exc
            if not math.isfinite(sentiment) or not -1 <= sentiment <= 1:
                raise ValueError("Each sentiment_score must be a finite number from -1 to 1.")
            if ticker and ticker in self.weights:
                if ticker not in ticker_sentiments:
                    ticker_sentiments[ticker] = []
                ticker_sentiments[ticker].append(sentiment)

        # Calculate average sentiment per ticker
        avg_sentiments: dict[str, float] = {}
        for ticker, scores in ticker_sentiments.items():
            avg_sentiments[ticker] = round(sum(scores) / len(scores), 4)

        # Adjust weights based on sentiment
        tickers_adjusted = []
        for ticker, avg_sentiment in avg_sentiments.items():
            if abs(avg_sentiment) >= threshold:
                # Positive sentiment → increase weight
                # Negative sentiment → decrease weight
                adjustment = avg_sentiment * factor
                self.weights[ticker] += adjustment
                tickers_adjusted.append(ticker)

        # Project onto the bounded simplex. Clamping and then normalizing can
        # violate the bounds, so solve for a shared shift before rounding.
        self.weights = self._project_bounded_simplex(self.weights)

        # Calculate changes
        changes = {}
        for ticker in self.weights:
            change = round(self.weights[ticker] - previous_weights[ticker], 6)
            if change != 0:
                changes[ticker] = change

        timestamp = datetime.now(timezone.utc).isoformat()
        self.last_rebalanced = timestamp

        # Informative status message
        if changes:
            msg = f"Successfully rebalanced {len(changes)} stocks based on sentiment signals."
        elif not avg_sentiments:
            msg = "No signals found for portfolio stocks."
        else:
            details = ", ".join([f"{t}: {s:+.3f}" for t, s in avg_sentiments.items()])
            msg = f"No changes made: Net sentiment ({details}) did not exceed the |threshold| of {threshold:.2f}."

        # Store in history
        record = {
            "timestamp": timestamp,
            "previous_weights": previous_weights,
            "new_weights": deepcopy(self.weights),
            "changes": changes,
            "signals_used": len(signals),
            "avg_sentiments": avg_sentiments,
            "threshold_used": threshold,
            "message": msg,
        }
        self.rebalance_history.append(record)

        return record

    @staticmethod
    def _project_bounded_simplex(weights: dict[str, float]) -> dict[str, float]:
        """Return weights within configured bounds that sum exactly to 1.0."""
        tickers = list(weights)
        minimum = REBALANCE_MIN_WEIGHT
        maximum = REBALANCE_MAX_WEIGHT
        if not tickers or len(tickers) * minimum > 1 or len(tickers) * maximum < 1:
            raise ValueError("Portfolio size and weight bounds cannot sum to 1.0.")

        raw = [float(weights[ticker]) for ticker in tickers]
        low = min(value - maximum for value in raw)
        high = max(value - minimum for value in raw)
        for _ in range(80):
            shift = (low + high) / 2
            projected_sum = sum(
                min(maximum, max(minimum, value - shift)) for value in raw
            )
            if projected_sum > 1.0:
                low = shift
            else:
                high = shift

        rounded = {
            ticker: round(min(maximum, max(minimum, value - high)), 6)
            for ticker, value in zip(tickers, raw)
        }

        # Distribute the rounding remainder in 1e-6 units without crossing a
        # bound, so public weights conserve the portfolio exactly.
        remainder = round((1.0 - sum(rounded.values())) * 1_000_000)
        for ticker in tickers:
            if remainder == 0:
                break
            if remainder > 0:
                capacity = round((maximum - rounded[ticker]) * 1_000_000)
                adjustment = min(remainder, capacity)
            else:
                capacity = round((rounded[ticker] - minimum) * 1_000_000)
                adjustment = -min(-remainder, capacity)
            rounded[ticker] = round(rounded[ticker] + adjustment / 1_000_000, 6)
            remainder -= adjustment

        if remainder != 0:
            raise ArithmeticError("Could not reconcile rounded portfolio weights to 1.0.")
        return rounded

    def reset_weights(self):
        """Reset weights to initial values."""
        self.weights = deepcopy(self.initial_weights)
        self.last_rebalanced = None

    def get_history(self) -> list[dict]:
        """Return rebalancing history."""
        return self.rebalance_history

    def get_summary(self) -> dict:
        """Get a summary of the current index state."""
        sorted_weights = sorted(
            self.weights.items(), key=lambda x: x[1], reverse=True
        )
        return {
            "total_stocks": len(self.weights),
            "total_weight": round(sum(self.weights.values()), 4),
            "top_3": [
                {"ticker": t, "name": TICKER_TO_NAME.get(t, t), "weight": round(w, 4)}
                for t, w in sorted_weights[:3]
            ],
            "bottom_3": [
                {"ticker": t, "name": TICKER_TO_NAME.get(t, t), "weight": round(w, 4)}
                for t, w in sorted_weights[-3:]
            ],
            "last_rebalanced": self.last_rebalanced,
            "rebalance_count": len(self.rebalance_history),
        }


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rebalancer = IndexRebalancer()

    # Simulate some risk signals
    fake_signals = [
        {"ticker": "AAPL", "sentiment_score": 0.85},
        {"ticker": "AAPL", "sentiment_score": 0.72},
        {"ticker": "JPM", "sentiment_score": -0.65},
        {"ticker": "JPM", "sentiment_score": -0.80},
        {"ticker": "NVDA", "sentiment_score": 0.90},
        {"ticker": "XOM", "sentiment_score": -0.45},
        {"ticker": "MSFT", "sentiment_score": 0.30},  # Below threshold, won't adjust
    ]

    print("\n" + "=" * 60)
    print("BEFORE REBALANCING")
    print("=" * 60)
    for ticker, weight in rebalancer.get_weights().items():
        print(f"  {ticker}: {weight:.2%}")

    result = rebalancer.rebalance(fake_signals)

    print(f"\n{'=' * 60}")
    print("AFTER REBALANCING")
    print("=" * 60)
    for ticker, weight in result["new_weights"].items():
        print(f"  {ticker}: {weight:.2%}")

    print(f"\nChanges:")
    for ticker, change in result["changes"].items():
        direction = "↑" if change > 0 else "↓"
        print(f"  {ticker}: {direction} {abs(change):.2%}")
