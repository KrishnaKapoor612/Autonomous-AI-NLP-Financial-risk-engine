"""
Central Configuration Module for the AI/NLP Financial Risk Engine.

Purpose:
  Defines all global parameters, model checkpoints, threshold values,
  and portfolio structures used across the ingestion, NLP, and quantitative
  decision modules.

Key Components:
  - API Credentials and environment variable loading (.env)
  - 15-stock S&P benchmark equity basket with baseline weights for Module A
  - 10-holding $40.3M synthetic wholesale banking portfolio and shock scenarios
  - NLP model checkpoints (ProsusAI/finbert, facebook/bart-large-mnli)
  - Event taxonomies, urgency keyword dictionaries, and impact scoring weights
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ───────────────────────────────────────────────────────────────
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# ─── Mock S&P Index (15 stocks from S&P 100) ────────────────────────────────
# ticker → initial equal-ish weight (sums to 1.0)
MOCK_INDEX_TICKERS = {
    "AAPL": 0.10,   # Apple
    "MSFT": 0.10,   # Microsoft
    "GOOGL": 0.08,  # Alphabet
    "AMZN": 0.08,   # Amazon
    "NVDA": 0.08,   # NVIDIA
    "META": 0.06,   # Meta
    "JPM": 0.06,    # JPMorgan Chase
    "JNJ": 0.06,    # Johnson & Johnson
    "V": 0.05,      # Visa
    "PG": 0.05,     # Procter & Gamble
    "XOM": 0.05,    # Exxon Mobil
    "BAC": 0.05,    # Bank of America
    "HD": 0.06,     # Home Depot
    "UNH": 0.06,    # UnitedHealth
    "MA": 0.06,     # Mastercard
}

# ─── Ticker-to-Company Name Mapping ─────────────────────────────────────────
TICKER_TO_NAME = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta",
    "JPM": "JPMorgan Chase", "JNJ": "Johnson & Johnson", "V": "Visa",
    "PG": "Procter & Gamble", "XOM": "Exxon Mobil", "BAC": "Bank of America",
    "HD": "Home Depot", "UNH": "UnitedHealth", "MA": "Mastercard",
}

# Reverse mapping: company name (lowercase) → ticker with common financial media aliases
NAME_TO_TICKER = {name.lower(): ticker for ticker, name in TICKER_TO_NAME.items()}
NAME_TO_TICKER.update({
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "chase": "JPM",
    "bank of america": "BAC",
    "bofa": "BAC",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "j&j": "JNJ",
    "johnson and johnson": "JNJ",
    "unitedhealth": "UNH",
    "united health": "UNH",
    "unitedhealthcare": "UNH",
})

# ─── NLP Engine Settings ────────────────────────────────────────────────────
SENTIMENT_MODEL = "ProsusAI/finbert"
CLASSIFIER_MODEL = "facebook/bart-large-mnli"

EVENT_CATEGORIES = [
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Merger/Acquisition",
    "Product Launch",
    "Regulatory",
    "Earnings Report",
    "Natural Disaster",
    "Legal/Lawsuit",
    "Market Volatility",
]

# Weights for impact score calculation per event type
EVENT_TYPE_WEIGHTS = {
    "Geopolitical": 1.5,
    "Macroeconomic": 1.4,
    "Credit Event": 1.6,
    "Merger/Acquisition": 1.2,
    "Product Launch": 0.8,
    "Regulatory": 1.3,
    "Earnings Report": 1.0,
    "Natural Disaster": 1.5,
    "Legal/Lawsuit": 1.1,
    "Market Volatility": 1.3,
}

# Keywords that boost impact score
URGENCY_KEYWORDS = [
    "crash", "crisis", "default", "bankruptcy", "war", "sanction",
    "recession", "collapse", "fraud", "surge", "plunge", "halt",
    "emergency", "shutdown", "spike", "record", "unprecedented",
]

# ─── Rebalancer Settings (Module A) ─────────────────────────────────────────
REBALANCE_ADJUSTMENT_FACTOR = 0.05   # How aggressively to adjust weights
REBALANCE_ADJUSTMENT_MIN = 0.01
REBALANCE_ADJUSTMENT_MAX = 0.20
REBALANCE_SENTIMENT_THRESHOLD = 0.05  # Keep API, dashboard, and CLI defaults aligned
REBALANCE_SENTIMENT_THRESHOLD_MIN = 0.01
REBALANCE_SENTIMENT_THRESHOLD_MAX = 0.50
REBALANCE_MIN_WEIGHT = 0.02          # 2% minimum weight per stock
REBALANCE_MAX_WEIGHT = 0.20          # 20% maximum weight per stock

# ─── Stress Test Settings (Module B) ────────────────────────────────────────
STRESS_TEST_IMPACT_THRESHOLD = 7  # Minimum impact score to trigger stress test

SHOCK_SCENARIOS = {
    "Geopolitical": {
        "name": "Geopolitical Crisis",
        "equity_shock": -0.15,
        "corp_bond_spread_shock": 0.02,
        "govt_bond_rate_shock": -0.005,
        "loan_default_shock": 0.05,
        "derivative_vol_shock": 0.30,
    },
    "Macroeconomic": {
        "name": "Rate Hike Shock",
        "equity_shock": -0.08,
        "corp_bond_spread_shock": 0.01,
        "govt_bond_rate_shock": 0.02,
        "loan_default_shock": 0.03,
        "derivative_vol_shock": 0.15,
    },
    "Credit Event": {
        "name": "Credit Crisis",
        "equity_shock": -0.20,
        "corp_bond_spread_shock": 0.05,
        "govt_bond_rate_shock": -0.01,
        "loan_default_shock": 0.10,
        "derivative_vol_shock": 0.40,
    },
    "Merger/Acquisition": {
        "name": "M&A Disruption",
        "equity_shock": -0.05,
        "corp_bond_spread_shock": 0.005,
        "govt_bond_rate_shock": 0.0,
        "loan_default_shock": 0.01,
        "derivative_vol_shock": 0.10,
    },
    "Regulatory": {
        "name": "Regulatory Crackdown",
        "equity_shock": -0.10,
        "corp_bond_spread_shock": 0.015,
        "govt_bond_rate_shock": 0.0,
        "loan_default_shock": 0.04,
        "derivative_vol_shock": 0.20,
    },
    "Natural Disaster": {
        "name": "Natural Disaster Impact",
        "equity_shock": -0.12,
        "corp_bond_spread_shock": 0.02,
        "govt_bond_rate_shock": -0.005,
        "loan_default_shock": 0.06,
        "derivative_vol_shock": 0.25,
    },
}

# Default shock for event types not explicitly mapped
DEFAULT_SHOCK = {
    "name": "General Market Stress",
    "equity_shock": -0.05,
    "corp_bond_spread_shock": 0.01,
    "govt_bond_rate_shock": 0.0,
    "loan_default_shock": 0.02,
    "derivative_vol_shock": 0.10,
}

# ─── Synthetic Portfolio for Stress Testing (Module B) ──────────────────────
SYNTHETIC_PORTFOLIO = [
    {
        "id": 1, "name": "US Treasury 10Y Bond",
        "type": "Government Bond", "value": 10_000_000,
        "rate": 0.025, "sector": "Government", "rating": "AAA",
    },
    {
        "id": 2, "name": "TechCorp Senior Bond",
        "type": "Corporate Bond", "value": 5_000_000,
        "rate": 0.045, "sector": "Technology", "rating": "A",
    },
    {
        "id": 3, "name": "EnergyCo Bond",
        "type": "Corporate Bond", "value": 4_000_000,
        "rate": 0.055, "sector": "Energy", "rating": "BBB",
    },
    {
        "id": 4, "name": "AAPL Equity Holding",
        "type": "Equity", "value": 3_000_000,
        "sector": "Technology", "rating": "N/A",
    },
    {
        "id": 5, "name": "JPM Equity Holding",
        "type": "Equity", "value": 2_500_000,
        "sector": "Financials", "rating": "N/A",
    },
    {
        "id": 6, "name": "XOM Equity Holding",
        "type": "Equity", "value": 2_000_000,
        "sector": "Energy", "rating": "N/A",
    },
    {
        "id": 7, "name": "RetailCo Term Loan",
        "type": "Loan", "value": 8_000_000,
        "rate": 0.065, "sector": "Consumer", "rating": "BBB",
    },
    {
        "id": 8, "name": "MfgCorp Revolving Credit",
        "type": "Loan", "value": 5_000_000,
        "rate": 0.055, "sector": "Industrials", "rating": "BB",
    },
    {
        "id": 9, "name": "IR Swap (Pay Fixed)",
        "type": "Interest Rate Swap", "notional": 20_000_000,
        "value": 500_000, "fixed_rate": 0.035,
        "sector": "Derivatives", "rating": "N/A",
    },
    {
        "id": 10, "name": "Credit Default Swap",
        "type": "Credit Default Swap", "notional": 10_000_000,
        "value": 300_000, "sector": "Derivatives", "rating": "N/A",
    },
]
