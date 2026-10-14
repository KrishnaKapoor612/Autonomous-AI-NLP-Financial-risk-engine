"""
Impact Scorer — calculates a severity score (1-10) for financial events.

Combines:
  - Sentiment magnitude (how strong is the sentiment?)
  - Event type weight (geopolitical events hit harder than product launches)
  - Urgency keyword detection (words like "crash", "crisis" boost the score)

This is rule-based, not ML — it's a weighted formula that
combines signals from the sentiment analyzer and event classifier.
"""

import re

import sys
sys.path.insert(0, ".")
from config.settings import EVENT_TYPE_WEIGHTS, URGENCY_KEYWORDS


class ImpactScorer:
    """Calculate market impact severity score from NLP signals."""

    def __init__(self):
        self.event_weights = EVENT_TYPE_WEIGHTS
        self.urgency_keywords = URGENCY_KEYWORDS
        # Match complete words and common inflections (e.g., "sanctions",
        # "bankruptcies"). This prevents "war" from matching inside "award".
        inflections = {
            "crash": ("crashes", "crashed", "crashing"),
            "crisis": ("crises",),
            "default": ("defaults", "defaulted", "defaulting"),
            "bankruptcy": ("bankruptcies",),
            "war": ("wars",),
            "sanction": ("sanctions", "sanctioned", "sanctioning"),
            "recession": ("recessions",),
            "collapse": ("collapses", "collapsed", "collapsing"),
            "fraud": ("frauds",),
            "surge": ("surges", "surged", "surging"),
            "plunge": ("plunges", "plunged", "plunging"),
            "halt": ("halts", "halted", "halting"),
            "emergency": ("emergencies",),
            "shutdown": ("shutdowns",),
            "spike": ("spikes", "spiked", "spiking"),
            "record": ("records", "recorded", "recording"),
        }
        self._form_to_keyword = {}
        forms = []
        for keyword in self.urgency_keywords:
            for form in (keyword, *inflections.get(keyword, ())):
                self._form_to_keyword[form] = keyword
                forms.append(re.escape(form))
        pattern = r"(?<!\w)(?:" + "|".join(sorted(forms, key=len, reverse=True)) + r")(?!\w)"
        self._keyword_regex = re.compile(pattern, re.IGNORECASE)

    def score(
        self,
        text: str,
        sentiment_score: float,
        event_classification: str,
        event_confidence: float,
    ) -> dict:
        """
        Calculate the impact score for a piece of financial text.

        Args:
            text: Original text (used for keyword detection).
            sentiment_score: From sentiment analyzer (-1.0 to +1.0).
            event_classification: From event classifier (e.g., "Geopolitical").
            event_confidence: Confidence of the classification (0.0 to 1.0).

        Returns:
            dict with keys:
                - impact_score: int from 1 to 10
                - impact_components: dict showing how the score was calculated
        """
        # ── Component 1: Sentiment magnitude (0.0 to 1.0) ──
        # Stronger sentiment (positive OR negative) = higher potential impact
        sentiment_magnitude = abs(sentiment_score)

        # ── Component 2: Event type weight ──
        event_weight = self.event_weights.get(event_classification, 1.0)

        # ── Component 3: Urgency keyword boost ──
        matched_forms = self._keyword_regex.findall(text) if text else []
        unique_keywords = list(dict.fromkeys(
            self._form_to_keyword[form.lower()] for form in matched_forms
        ))
        # Each unique keyword adds 0.1, capped at 0.5 boost
        keyword_boost = min(len(unique_keywords) * 0.1, 0.5)

        # ── Component 4: Classification confidence factor ──
        # Higher confidence in the event type = more reliable signal
        confidence_factor = 0.5 + (event_confidence * 0.5)  # Range: 0.5 to 1.0

        # ── Combine into raw score ──
        # Base: sentiment magnitude (0-1) × 10 gives us a 0-10 range
        base_score = sentiment_magnitude * 10

        # Apply event weight, keyword boost, and confidence
        raw_score = base_score * event_weight * (1 + keyword_boost) * confidence_factor

        # Normalize to 1-10 range
        # The max possible raw_score is: 10 × 1.6 × 1.5 × 1.0 = 24
        # So we divide by ~2.4 to normalize back to ~10
        normalized = raw_score / 2.4
        impact_score = max(1, min(10, round(normalized)))

        return {
            "impact_score": impact_score,
            "impact_components": {
                "sentiment_magnitude": round(sentiment_magnitude, 4),
                "event_weight": event_weight,
                "keyword_boost": round(keyword_boost, 2),
                "matched_keywords": unique_keywords,
                "confidence_factor": round(confidence_factor, 4),
                "raw_score": round(raw_score, 4),
            },
        }


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = ImpactScorer()

    test_cases = [
        {
            "text": "Goldman Sachs warns of potential sovereign debt crisis in Italy as bond yields spike",
            "sentiment_score": -0.87,
            "event_classification": "Credit Event",
            "event_confidence": 0.82,
        },
        {
            "text": "Apple launches new iPhone with improved camera features",
            "sentiment_score": 0.45,
            "event_classification": "Product Launch",
            "event_confidence": 0.91,
        },
        {
            "text": "Russia threatens nuclear escalation amid collapsing peace talks",
            "sentiment_score": -0.95,
            "event_classification": "Geopolitical",
            "event_confidence": 0.94,
        },
        {
            "text": "The Federal Reserve held interest rates steady",
            "sentiment_score": 0.05,
            "event_classification": "Macroeconomic",
            "event_confidence": 0.78,
        },
    ]

    print("\n" + "=" * 70)
    print("IMPACT SCORING RESULTS")
    print("=" * 70)

    for case in test_cases:
        result = scorer.score(**case)
        print(f"\nText: {case['text']}")
        print(f"  Impact Score: {result['impact_score']} / 10")
        print(f"  Components: {result['impact_components']}")
