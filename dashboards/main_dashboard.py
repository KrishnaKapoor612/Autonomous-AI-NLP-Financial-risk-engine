"""
Streamlit Dashboard — Main Entry Point (Multi-Page App).

This is the unified dashboard that combines all three views:
  1. Risk Engine Monitor
  2. Index Rebalancer (Module A)
  3. Stress Tester (Module B)

Run: streamlit run dashboards/main_dashboard.py
"""

import streamlit as st
import requests

st.set_page_config(
    page_title="AI/NLP Financial Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("Risk Engine")
st.sidebar.markdown("---")
st.sidebar.info(
    "**AI/NLP Financial Risk Engine**\n\n"
    "Real-time risk intelligence powered by FinBERT and Zero-Shot NLP.\n\n"
    "Navigate using the pages in the sidebar."
)

st.sidebar.markdown("### API Status")
API_BASE = st.sidebar.text_input("API URL", value="http://localhost:8000", key="api_url")

# Check API status
try:
    r = requests.get(f"{API_BASE}/health", timeout=15)
    if r.status_code == 200:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.error("API Error")
except Exception:
    st.sidebar.error("API Not Running")
    st.sidebar.caption("Start the API first:\n`uvicorn src.api.main:app --reload`")

# ─── Main Page ───────────────────────────────────────────────────────────────
st.title("AI/NLP Financial Risk Engine")
st.markdown("### Real-Time Risk Intelligence Platform")

st.markdown("""
Welcome to the **AI/NLP Financial Risk Engine** — a platform that ingests 
unstructured financial text data and converts it into actionable risk signals.

#### Architecture
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ##### Data Sources
    - **NewsAPI** — Financial news articles
    - **GDELT** — Global event monitoring
    """)

with col2:
    st.markdown("""
    ##### NLP Engine
    - **FinBERT** — Sentiment Analysis
    - **BART-MNLI** — Event Classification
    - **Rule Engine** — Impact Scoring
    """)

with col3:
    st.markdown("""
    ##### Applications
    - **Module A** — Index Rebalancer
    - **Module B** — Stress Tester
    """)

st.markdown("---")
st.markdown("""
#### Quick Start

1. **Start the API**: `uvicorn src.api.main:app --reload --port 8000`
2. **Navigate** to the pages in the sidebar to:
   - **Risk Engine** — Analyze text and view signals
   - **Index Rebalancer** — Rebalance your benchmark portfolio
   - **Stress Test** — Run wholesale stress scenarios
""")

# Quick stats from API
try:
    signals_resp = requests.get(f"{API_BASE}/api/signals?limit=100", timeout=15)
    if signals_resp.status_code == 200:
        data = signals_resp.json()
        total_signals = data.get("total", 0)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals", total_signals)

        if total_signals > 0:
            signals = data.get("signals", [])
            avg_sentiment = sum(s.get("sentiment_score", 0) for s in signals) / len(signals)
            avg_impact = sum(s.get("impact_score", 0) for s in signals) / len(signals)
            high_impact = sum(1 for s in signals if s.get("impact_score", 0) >= 7)

            m2.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
            m3.metric("Avg Impact", f"{avg_impact:.1f}")
            m4.metric("High Impact (>= 7)", high_impact)
except Exception:
    pass
