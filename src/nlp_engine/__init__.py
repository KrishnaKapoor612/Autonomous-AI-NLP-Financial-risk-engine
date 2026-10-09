"""
AI/NLP Risk Engine Core Package.

Purpose:
  Provides the natural language processing, classification, and scoring pipeline.

Role in Project:
  - SentimentAnalyzer: Evaluates financial text using FinBERT (score range -1.0 to +1.0).
  - EventClassifier: Classifies incoming stories into zero-shot risk categories via BART.
  - ImpactScorer: Computes quantitative market impact ratings (1 to 10 scale).
  - RiskEngine: Central pipeline orchestrator synthesizing text into structured Risk Signals.
"""

from src.nlp_engine.sentiment import SentimentAnalyzer
from src.nlp_engine.classifier import EventClassifier
from src.nlp_engine.impact_scorer import ImpactScorer
from src.nlp_engine.risk_engine import RiskEngine

__all__ = [
    "SentimentAnalyzer",
    "EventClassifier",
    "ImpactScorer",
    "RiskEngine",
]