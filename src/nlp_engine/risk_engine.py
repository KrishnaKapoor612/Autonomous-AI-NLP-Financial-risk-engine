"""
Risk Engine — the central orchestrator of the NLP pipeline.

This is the single entry point that ties together:
  1. Sentiment Analysis (FinBERT)
  2. Event Classification (Zero-Shot)
  3. Impact Scoring (Rule-Based)

Input:  raw text + source metadata
Output: a complete, structured RiskSignal
"""

from datetime import datetime, timezone
from typing import Optional

from src.nlp_engine.sentiment import SentimentAnalyzer
from src.nlp_engine.classifier import EventClassifier
from src.nlp_engine.impact_scorer import ImpactScorer


class RiskEngine:
    """
    Central NLP Risk Engine.

    Ingests raw financial text and produces structured risk signals
    with sentiment, event classification, and impact scores.
    """

    def __init__(self):
        """Initialize all NLP sub-components."""
        print("=" * 50)
        print("  Initializing AI/NLP Risk Engine...")
        print("=" * 50)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.event_classifier = EventClassifier()
        self.impact_scorer = ImpactScorer()
        self._signals_history: list[dict] = []
        print("=" * 50)
        print("  Risk Engine ready!")
        print("=" * 50)

    def analyze(
        self,
        text: str,
        source: str = "manual",
        ticker: Optional[str] = None,
        url: Optional[str] = None,
    ) -> dict:
        """
        Analyze a single piece of text and produce a complete risk signal.

        Args:
            text: Raw financial text (headline, article, tweet).
            source: Data source identifier ("newsapi", "gdelt", "manual").
            ticker: Optional stock ticker associated with the text.
            url: Optional source URL.

        Returns:
            A structured risk signal dict.
        """
        # Step 1: Sentiment Analysis
        sentiment_result = self.sentiment_analyzer.analyze(text)

        # Step 2: Event Classification
        classification_result = self.event_classifier.classify(text)

        # Step 3: Impact Scoring
        impact_result = self.impact_scorer.score(
            text=text,
            sentiment_score=sentiment_result["sentiment_score"],
            event_classification=classification_result["event_classification"],
            event_confidence=classification_result["event_confidence"],
        )

        # Assemble the complete risk signal
        signal = {
            "text": text,
            "source": source,
            "ticker": ticker,
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Sentiment
            "sentiment_score": sentiment_result["sentiment_score"],
            "sentiment_label": sentiment_result["sentiment_label"],
            "sentiment_confidence": sentiment_result["confidence"],
            "sentiment_raw_scores": sentiment_result["raw_scores"],
            # Event Classification
            "event_classification": classification_result["event_classification"],
            "event_confidence": classification_result["event_confidence"],
            "event_all_scores": classification_result["all_scores"],
            # Impact
            "impact_score": impact_result["impact_score"],
            "impact_components": impact_result["impact_components"],
        }

        # Store in history
        self._signals_history.append(signal)

        return signal

    def analyze_batch(
        self,
        items: list[dict],
    ) -> list[dict]:
        """
        Analyze a batch of items.

        Args:
            items: List of dicts, each with at least a "text" key.
                   Optional keys: "source", "ticker", "url".

        Returns:
            List of risk signal dicts.
        """
        signals = []
        for item in items:
            signal = self.analyze(
                text=item.get("text", ""),
                source=item.get("source", "manual"),
                ticker=item.get("ticker"),
                url=item.get("url"),
            )
            signals.append(signal)
        return signals

    def get_signals_history(self) -> list[dict]:
        """Return all previously analyzed signals."""
        return self._signals_history

    def get_signals_by_ticker(self, ticker: str) -> list[dict]:
        """Return signals filtered by stock ticker."""
        return [
            s for s in self._signals_history
            if s.get("ticker") and s["ticker"].upper() == ticker.upper()
        ]

    def get_high_impact_signals(self, threshold: int = 7) -> list[dict]:
        """Return signals with impact score >= threshold."""
        return [
            s for s in self._signals_history
            if s["impact_score"] >= threshold
        ]

    def clear_history(self):
        """Clear the signals history."""
        self._signals_history.clear()


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = RiskEngine()

    test_texts = [
        {
            "text": "Apple reports record quarterly revenue of $124 billion, beating analyst expectations",
            "source": "newsapi",
            "ticker": "AAPL",
        },
        {
            "text": "Goldman Sachs warns of potential sovereign debt crisis in Italy as bond yields spike to 7-year highs",
            "source": "gdelt",
        },
        {
            "text": "Federal Reserve signals aggressive rate hike path for 2027, markets tumble",
            "source": "newsapi",
        },
        {
            "text": "Microsoft announces acquisition of AI startup for $2.5 billion",
            "source": "newsapi",
            "ticker": "MSFT",
        },
    ]

    print("\n" + "=" * 70)
    print("FULL RISK ENGINE ANALYSIS")
    print("=" * 70)

    for item in test_texts:
        signal = engine.analyze(**item)
        print(f"\n{'─' * 60}")
        print(f"Text: {signal['text']}")
        print(f"Source: {signal['source']} | Ticker: {signal['ticker']}")
        print(f"Sentiment: {signal['sentiment_label']} ({signal['sentiment_score']})")
        print(f"Event: {signal['event_classification']} (conf: {signal['event_confidence']})")
        print(f"Impact Score: {signal['impact_score']} / 10")

    # Show high impact signals
    high_impact = engine.get_high_impact_signals(threshold=7)
    print(f"\n{'=' * 70}")
    print(f"HIGH IMPACT SIGNALS (>= 7): {len(high_impact)} found")
    for s in high_impact:
        print(f"  [{s['impact_score']}] {s['text'][:80]}...")
