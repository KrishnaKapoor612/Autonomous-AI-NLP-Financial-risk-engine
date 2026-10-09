"""
Zero-Shot Event Classification Engine.

Purpose:
  Classifies unstructured financial text into predefined risk categories without domain fine-tuning.

Role in Project:
  - Categorizes ingested news into institutional taxonomies (Geopolitical, Macroeconomic, Credit Event, etc.).
  - Uses the facebook/bart-large-mnli model based on Natural Language Inference (NLI).
  - Supplies confidence scores and top-k category distributions to the ImpactScorer.
  - Determines downstream stress testing activation and shock scenario mappings in Module B.
"""

from transformers import pipeline

import sys
sys.path.insert(0, ".")
from config.settings import CLASSIFIER_MODEL, EVENT_CATEGORIES


class EventClassifier:
    """Zero-shot event classification for financial text."""

    def __init__(self):
        """Load the zero-shot classification pipeline."""
        print(f"[EventClassifier] Loading model: {CLASSIFIER_MODEL}...")
        self.pipe = pipeline(
            "zero-shot-classification",
            model=CLASSIFIER_MODEL,
        )
        self.categories = EVENT_CATEGORIES
        print("[EventClassifier] Model loaded successfully.")

    def classify(self, text: str, top_k: int = 3) -> dict:
        """
        Classify text into event categories.

        Args:
            text: Financial news text to classify.
            top_k: Number of top categories to return.

        Returns:
            dict with keys:
                - event_classification: str — top predicted category
                - event_confidence: float — confidence of top prediction (0.0 to 1.0)
                - all_scores: list of dicts with label + score for top_k categories
        """
        if not text or not text.strip():
            return {
                "event_classification": "Unknown",
                "event_confidence": 0.0,
                "all_scores": [],
            }

        text = text[:512]

        # Run zero-shot classification
        result = self.pipe(
            text,
            candidate_labels=self.categories,
            multi_label=False,
        )

        # result has: {"sequence": ..., "labels": [...], "scores": [...]}
        labels = result["labels"]
        scores = result["scores"]

        # Build the top-k results
        all_scores = []
        for label, score in zip(labels[:top_k], scores[:top_k]):
            all_scores.append({
                "label": label,
                "score": round(score, 4),
            })

        return {
            "event_classification": labels[0],
            "event_confidence": round(scores[0], 4),
            "all_scores": all_scores,
        }

    def classify_batch(self, texts: list[str], top_k: int = 3) -> list[dict]:
        """Classify a batch of texts."""
        return [self.classify(text, top_k) for text in texts]


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    classifier = EventClassifier()

    test_texts = [
        "Russia threatens to cut gas supplies to Europe amid escalating tensions",
        "Federal Reserve raises interest rates by 75 basis points",
        "Lehman Brothers files for bankruptcy in largest US filing ever",
        "Microsoft announces acquisition of Activision Blizzard for $68 billion",
        "Apple launches new iPhone with revolutionary AI chip",
        "EU imposes stricter regulations on Big Tech data practices",
        "Massive earthquake hits Japan's industrial heartland",
    ]

    print("\n" + "=" * 70)
    print("EVENT CLASSIFICATION RESULTS")
    print("=" * 70)

    for text in test_texts:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"  Classification: {result['event_classification']}")
        print(f"  Confidence: {result['event_confidence']}")
        print(f"  Top 3: {result['all_scores']}")
