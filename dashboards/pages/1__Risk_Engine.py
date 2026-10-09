"""
Streamlit Page 1: NLP Risk Engine Monitor.

Purpose:
  Provides the real-time operational dashboard for financial risk monitoring.

Role in Project:
  - Facilitates interactive manual text input for ad-hoc FinBERT risk evaluations.
  - Queries live external news from NewsAPI and macro events from GDELT 2.0.
  - Visualizes real-time sentiment distribution, event classifications, and impact scores.
  - Displays structured Risk Signal history streamed from the FastAPI backend.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Risk Engine", layout="wide")

API_BASE = st.session_state.get("api_url", "http://localhost:8000")

st.title("NLP Risk Engine Monitor")
st.markdown("Analyze financial text and view structured risk signals in real-time.")

st.markdown("### Analyze Text")

col_input, col_options = st.columns([3, 1])
with col_input:
    text_input = st.text_area(
        "Enter financial text to analyze:",
        placeholder="e.g., Apple shares tumble 5% after China bans iPhones in government offices",
        height=100,
    )
with col_options:
    source = st.selectbox("Source", ["manual", "newsapi", "gdelt"])
    ticker = st.text_input("Ticker (optional)", placeholder="AAPL")

if st.button("Analyze", type="primary", use_container_width=True):
    if text_input:
        with st.spinner("Running NLP analysis..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/analyze",
                    json={
                        "text": text_input,
                        "source": source,
                        "ticker": ticker if ticker else None,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    signal = resp.json()

                    # Display results in columns
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(
                            "Sentiment",
                            f"{signal['sentiment_score']:.3f}",
                            delta=signal["sentiment_label"],
                        )
                    with c2:
                        st.metric(
                            "Event Type",
                            signal["event_classification"],
                            delta=f"conf: {signal['event_confidence']:.2f}",
                        )
                    with c3:
                        impact = signal["impact_score"]
                        st.metric(
                            "Impact Score",
                            f"{impact} / 10",
                        )

                    # Show raw JSON
                    with st.expander("Full Signal JSON"):
                        st.json(signal)
                else:
                    st.error(f"API Error: {resp.status_code}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Start it with: `uvicorn src.api.main:app --reload`")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")

st.markdown("### Ingest Data")

tab_news, tab_gdelt, tab_tickers = st.tabs(["NewsAPI", "GDELT", "By Tickers"])

with tab_news:
    col_nq, col_ndays = st.columns([3, 1])
    with col_nq:
        news_query = st.text_input("Search query", value="stock market", key="news_q")
    with col_ndays:
        news_days = st.selectbox(
            "Lookback",
            [7, 1, 3, 14, 30],
            index=0,
            format_func=lambda d: f"{d} days",
            key="news_days",
        )
    news_count = st.slider("Articles (recommend 5-10 for fast analysis)", 1, 20, 5, key="news_count")
    if st.button("Fetch & Analyze News", key="fetch_news"):
        with st.spinner(f"Fetching and analyzing {news_count} articles with FinBERT & BART..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/ingest/news",
                    json={
                        "query": news_query,
                        "page_size": news_count,
                        "days_back": news_days,
                    },
                    timeout=180,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    article_count = data.get("count", 0)
                    if article_count > 0:
                        st.success(f"Analyzed {article_count} articles")
                        st.rerun()
                    else:
                        st.info("No news articles found matching your query.")
                elif resp.status_code == 401:
                    st.warning("**NewsAPI Key Missing:** Please add your free API key in `.env` (`NEWSAPI_KEY=your_key`). Get one for free at [newsapi.org/register](https://newsapi.org/register).")
                elif resp.status_code == 429:
                    st.warning("**NewsAPI Quota Exceeded:** Daily limit of 100 requests reached on the free tier.")
                else:
                    try:
                        error_detail = resp.json().get("detail", resp.text)
                    except Exception:
                        error_detail = resp.text
                    st.error(f"Error: {error_detail}")
            except requests.ConnectionError:
                st.error("API not running")
            except requests.Timeout:
                st.warning(
                    f"**Analysis Timed Out:** Processing {news_count} articles through FinBERT and BART on CPU exceeded the deadline. "
                    "Try selecting 5 to 10 articles for fast sub-minute processing."
                )
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab_gdelt:
    col_q, col_days = st.columns([3, 1])
    with col_q:
        gdelt_query = st.text_input("Keyword", value="financial crisis", key="gdelt_q")
    with col_days:
        gdelt_days = st.selectbox(
            "Lookback",
            [7, 3, 1, 14],
            index=0,
            format_func=lambda d: f"{d} days",
            key="gdelt_days",
        )
    gdelt_count = st.slider("Max articles", 1, 30, 10, key="gdelt_count")
    if st.button("Fetch & Analyze GDELT", key="fetch_gdelt"):
        with st.spinner("Fetching from GDELT..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/ingest/gdelt",
                    json={
                        "query": gdelt_query,
                        "page_size": gdelt_count,
                        "days_back": gdelt_days,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    article_count = data.get("count", 0)
                    if article_count > 0:
                        st.success(f"Analyzed {article_count} articles")
                        st.rerun()
                    else:
                        st.info("No articles found on GDELT matching this keyword and timespan.")
                elif resp.status_code == 429:
                    error_detail = "GDELT is temporarily rate-limiting requests. Please wait a few seconds and try again."
                    try:
                        error_detail = resp.json().get("detail", error_detail)
                    except Exception:
                        pass
                    st.warning(
                        f"{error_detail}\n\n"
                        "**Tips to resolve:**\n"
                        "- GDELT enforces a strict minimum 5-second interval between requests. Wait 30-60 seconds before retrying.\n"
                        "- Try broader macro terms (e.g. 'financial crisis', 'interest rate') or company names (e.g. 'Nvidia').\n"
                        "- Or use the **By Tickers** tab or **NewsAPI** tab to analyze news immediately."
                    )
                else:
                    try:
                        error_detail = resp.json().get("detail", resp.text)
                    except Exception:
                        error_detail = resp.text
                    st.error(f"Error: {error_detail}")
            except requests.ConnectionError:
                st.error("API not running")
            except requests.Timeout:
                st.warning("Request timed out waiting for GDELT response. Please try again.")
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab_tickers:
    if st.button("Fetch News for All Index Stocks", key="fetch_tickers"):
        with st.spinner("Fetching news for all 15 stocks..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/ingest/tickers",
                    json={"page_size": 3},
                    timeout=240,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    article_count = data.get("count", 0)
                    if article_count > 0:
                        st.success(f"Analyzed {article_count} articles for {len(data.get('tickers_searched', []))} tickers")
                        st.rerun()
                    else:
                        st.info("No news articles found for index tickers.")
                elif resp.status_code == 401:
                    st.warning("**NewsAPI Key Missing:** Please add your free API key in `.env` (`NEWSAPI_KEY=your_key`). Get one for free at [newsapi.org/register](https://newsapi.org/register).")
                elif resp.status_code == 429:
                    st.warning("**NewsAPI Quota Exceeded:** Daily limit of 100 requests reached on the free tier.")
                else:
                    try:
                        error_detail = resp.json().get("detail", resp.text)
                    except Exception:
                        error_detail = resp.text
                    st.error(f"Error: {error_detail}")
            except requests.ConnectionError:
                st.error("API not running")
            except requests.Timeout:
                st.warning("Analysis timed out. Analyzing news for all 15 stocks sequentially took longer than expected.")
            except Exception as e:
                st.error(f"Request failed: {e}")

st.markdown("---")

# ─── Signals Feed ────────────────────────────────────────────────────────────
st.markdown("### Signals Feed")

try:
    resp = requests.get(f"{API_BASE}/api/signals?limit=100", timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        signals = data.get("signals", [])

        if signals:
            st.info(f"Showing {len(signals)} of {data['total']} total signals")

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            sentiments = [s["sentiment_score"] for s in signals]
            impacts = [s["impact_score"] for s in signals]
            m1.metric("Total Signals", len(signals))
            m2.metric("Avg Sentiment", f"{sum(sentiments)/len(sentiments):.3f}")
            m3.metric("Avg Impact", f"{sum(impacts)/len(impacts):.1f}")
            m4.metric("High Impact (>= 7)", sum(1 for i in impacts if i >= 7))

            # Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Sentiment distribution
                labels = [s["sentiment_label"] for s in signals]
                label_counts = pd.Series(labels).value_counts()
                fig = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    title="Sentiment Distribution",
                    color=label_counts.index,
                    color_discrete_map={
                        "positive": "#22c55e",
                        "negative": "#ef4444",
                        "neutral": "#eab308",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                # Event classification breakdown
                events = [s["event_classification"] for s in signals]
                event_counts = pd.Series(events).value_counts()
                fig = px.bar(
                    x=event_counts.index,
                    y=event_counts.values,
                    title="Event Classification Breakdown",
                    labels={"x": "Event Type", "y": "Count"},
                    color=event_counts.values,
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Signals table
            def _format_time(ts_str):
                if not ts_str:
                    return "-"
                clean = str(ts_str).replace("T", " ")
                return clean[:16]

            df = pd.DataFrame([
                {
                    "Time": _format_time(s.get("timestamp")),
                    "Text": s["text"][:85] + "..." if len(s["text"]) > 85 else s["text"],
                    "Source": s["source"],
                    "Ticker": s.get("ticker") if s.get("ticker") else "-",
                    "Sentiment": f"{s['sentiment_score']:+.2f}",
                    "Label": s["sentiment_label"],
                    "Event": s["event_classification"],
                    "Impact": s["impact_score"],
                }
                for s in reversed(signals)
            ])
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("No signals yet. Ingest some data above to get started.")
except requests.ConnectionError:
    st.warning("Cannot connect to API. Start it first.")
except requests.Timeout:
    st.info("Signals feed is updating. Please refresh in a moment.")
except Exception as e:
    st.error(f"Error: {e}")
