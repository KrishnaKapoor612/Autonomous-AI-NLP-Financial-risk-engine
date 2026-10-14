"""
Financial Sentiment Analysis Engine (FinBERT).

Purpose:
  Performs domain-specific sentiment analysis on financial news and earnings text.

Role in Project:
  - Leverages ProsusAI/finbert, a transformer model fine-tuned on financial discourse.
  - Generates discrete sentiment classifications (positive, negative, neutral) with confidence scores.
  - Produces a continuous composite polarity score scaled from -1.0 (extreme bearish) to +1.0 (extreme bullish).
  - Feeds directional sentiment scores into Module A (Tactical Index Rebalancer) and the ImpactScorer.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

import sys
sys.path.insert(0, ".")
from config.settings import SENTIMENT_MODEL


class SentimentAnalyzer:
    """Financial sentiment analysis using FinBERT."""

    def __init__(self):
        """Load FinBERT model and tokenizer."""
        print(f"[SentimentAnalyzer] Loading model: {SENTIMENT_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,  # Return all label scores
        )
        self._label_map = {"positive": 1, "negative": -1, "neutral": 0}
        print("[SentimentAnalyzer] Model loaded successfully.")

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single text string.

        Args:
            text: Financial news headline or article text.

        Returns:
            dict with keys:
                - sentiment_label: "positive" | "negative" | "neutral"
                - sentiment_score: float from -1.0 (very negative) to +1.0 (very positive)
                - confidence: float from 0.0 to 1.0 (model's confidence)
                - raw_scores: dict of all label probabilities
        """
        if not text or not text.strip():
            return {
                "sentiment_label": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "raw_scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            }

        # Truncate to model's max length
        text = text[:512]

        # Run inference
        results = self.pipe(text)

        # Transformers versions differ here: some return a flat list of score
        # dictionaries for a single string; others wrap that list one level deep.
        if isinstance(results, dict):
            scores_list = [results]
        elif results and isinstance(results[0], dict):
            scores_list = results
        elif results and isinstance(results[0], list):
            scores_list = results[0]
        else:
            scores_list = []

        scores_list = [
            item for item in scores_list
            if isinstance(item, dict)
            and isinstance(item.get("label"), str)
            and isinstance(item.get("score"), (int, float))
        ]

        if not scores_list:
            return {
                "sentiment_label": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "raw_scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            }

        # Build raw scores dict
        raw_scores = {}
        for item in scores_list:
            raw_scores[item["label"].lower()] = round(float(item["score"]), 4)

        # Determine the top label
        top = max(scores_list, key=lambda x: x["score"])
        label = top["label"].lower()
        confidence = round(float(top["score"]), 4)

        # Compute a continuous score from -1.0 to +1.0
        # Score = P(positive) - P(negative)
        pos = raw_scores.get("positive", 0.0)
        neg = raw_scores.get("negative", 0.0)
        score = round(pos - neg, 4)

        return {
            "sentiment_label": label,
            "sentiment_score": score,
            "confidence": confidence,
            "raw_scores": raw_scores,
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Analyze sentiment for a batch of texts.

        Args:
            texts: List of text strings.

        Returns:
            List of sentiment result dicts.
        """
        return [self.analyze(text) for text in texts]


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    test_texts = [
        "Apple reports record quarterly revenue of $124 billion",
        "Tesla shares plunge 15% after massive recall announcement",
        "The Federal Reserve held interest rates steady at its latest meeting",
        "Goldman Sachs warns of potential sovereign debt crisis in Italy",
        "Microsoft acquires gaming company in $68 billion deal",
    ]

    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS RESULTS")
    print("=" * 70)

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"  Label: {result['sentiment_label']}")
        print(f"  Score: {result['sentiment_score']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Raw: {result['raw_scores']}")
