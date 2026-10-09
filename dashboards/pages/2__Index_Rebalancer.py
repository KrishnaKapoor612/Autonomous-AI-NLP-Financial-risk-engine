"""
Streamlit Page 2: Tactical Index Rebalancer (Module A).

Purpose:
  Provides an interactive interface to monitor and execute dynamic portfolio rebalancing.

Role in Project:
  - Displays current allocations and baseline weights for the 15-stock S&P benchmark basket.
  - Triggers tactical rebalancing runs powered by incoming NLP sentiment scores.
  - Renders comparison charts (before vs after) and historical rebalancing trajectories.
  - Enforces portfolio normalization constraints (weights sum to 1.0) and allocation guardrails.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Index Rebalancer", layout="wide")

API_BASE = st.session_state.get("api_url", "http://localhost:8000")

st.title("Tactical Index Rebalancer")
st.markdown("Dynamically rebalance a benchmark S&P equity basket based on NLP sentiment signals.")

# ─── Portfolio Summary ───────────────────────────────────────────────────────
try:
    resp = requests.get(f"{API_BASE}/api/portfolio/summary", timeout=25)
    if resp.status_code == 200:
        summary = resp.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks in Index", summary["total_stocks"])
        c2.metric("Total Weight", f"{summary['total_weight']:.2%}")
        c3.metric("Rebalance Count", summary["rebalance_count"])
        c4.metric("Last Rebalanced", summary.get("last_rebalanced", "Never")[:19] if summary.get("last_rebalanced") else "Never")
except Exception:
    st.warning("Cannot fetch portfolio summary. Is the API running?")

st.markdown("---")

# ─── Current Weights ─────────────────────────────────────────────────────────
st.markdown("### Current Portfolio Weights")

try:
    resp = requests.get(f"{API_BASE}/api/portfolio/weights", timeout=25)
    if resp.status_code == 200:
        data = resp.json()
        weights = data["weights"]

        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            # Donut chart
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.4,
                textinfo="label+percent",
                marker=dict(
                    colors=px.colors.qualitative.Set3[:len(weights)]
                ),
            )])
            fig.update_layout(
                title="Portfolio Composition",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            df = pd.DataFrame([
                {"Ticker": t, "Weight": f"{w:.2%}", "Weight_raw": w}
                for t, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(
                df[["Ticker", "Weight"]],
                use_container_width=True,
                height=400,
                hide_index=True,
            )

except requests.ConnectionError:
    st.error("API not running")
except Exception as e:
    st.error(f"Error: {e}")

st.markdown("---")

# ─── Rebalance Controls ─────────────────────────────────────────────────────
st.markdown("### Rebalance Portfolio")

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    adj_factor = st.slider(
        "Adjustment Factor (Multiplier)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="Controls how aggressively portfolio weights shift in response to sentiment.",
    )
with col_ctrl2:
    sentiment_thresh = st.slider(
        "Sentiment Threshold (Noise Filter)",
        min_value=0.01,
        max_value=0.50,
        value=0.05,
        step=0.01,
        help="Minimum absolute net sentiment required to trigger a weight shift. Lower values allow smaller net sentiment (e.g., +0.05) to rebalance.",
    )

col_act1, col_act2 = st.columns([2, 1])

with col_act1:
    if st.button("Rebalance Now", type="primary", use_container_width=True):
        with st.spinner("Rebalancing based on sentiment signals..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/portfolio/rebalance",
                    json={
                        "adjustment_factor": adj_factor,
                        "sentiment_threshold": sentiment_thresh,
                    },
                    timeout=25,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("changes"):
                        st.success(f"Rebalanced: {len(result['changes'])} stocks adjusted.")

                        # Show before/after comparison
                        prev = result["previous_weights"]
                        new = result["new_weights"]

                        tickers = list(prev.keys())
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name="Before",
                            x=tickers,
                            y=[prev[t] for t in tickers],
                            marker_color="lightblue",
                        ))
                        fig.add_trace(go.Bar(
                            name="After",
                            x=tickers,
                            y=[new[t] for t in tickers],
                            marker_color="darkblue",
                        ))
                        fig.update_layout(
                            title="Weight Changes: Before vs After",
                            barmode="group",
                            yaxis_title="Weight",
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Changes detail
                        changes = result["changes"]
                        if changes:
                            st.markdown("**Weight Changes:**")
                            for ticker, change in sorted(changes.items(), key=lambda x: x[1]):
                                prefix = "[+]" if change > 0 else "[-]"
                                st.write(f"**{ticker}**: {prefix} {change:+.4f} ({change*100:+.2f}%)")
                    else:
                        st.info(result.get("message", "No changes made."))

                        avg_sents = result.get("avg_sentiments", {})
                        if avg_sents:
                            st.markdown("##### Detected Net Sentiment per Ticker:")
                            for t, s in avg_sents.items():
                                status = "Triggered" if abs(s) >= sentiment_thresh else "Filtered (Below Threshold)"
                                st.markdown(f"- **{t}**: Net Sentiment = `{s:+.4f}` — Status: *{status}* (Required: `±{sentiment_thresh:.2f}`)")
                            st.caption("**Note**: If a stock was filtered out, reduce the **Sentiment Threshold** slider above to trigger rebalancing on lower net scores.")
                else:
                    st.error(f"Error: {resp.text}")
            except requests.ConnectionError:
                st.error("API not running")
            except requests.Timeout:
                st.error("Request timed out.")

with col_act2:
    if st.button("Reset to Default", use_container_width=True):
        try:
            resp = requests.post(f"{API_BASE}/api/portfolio/reset", timeout=25)
            if resp.status_code == 200:
                st.success("Portfolio reset to initial weights.")
                st.rerun()
        except Exception:
            st.error("Failed to reset")

st.markdown("---")

# ─── Rebalance History ───────────────────────────────────────────────────────
st.markdown("### Rebalance History & Allocation Trajectory")

try:
    resp = requests.get(f"{API_BASE}/api/portfolio/history", timeout=25)
    if resp.status_code == 200:
        history = resp.json().get("history", [])
        if history:
            records = []

            # Step 0: Prepend initial baseline weights before first rebalancing round
            first_entry = history[0]
            if "previous_weights" in first_entry and first_entry["previous_weights"]:
                for ticker, weight in first_entry["previous_weights"].items():
                    records.append({
                        "Step": "0 (Baseline)",
                        "Ticker": ticker,
                        "Weight": weight,
                        "Allocation": f"{weight*100:.2f}%",
                    })

            for idx, entry in enumerate(history, 1):
                raw_ts = str(entry.get("timestamp", ""))
                # Clean timestamp formatting: HH:MM:SS
                time_str = raw_ts[11:19] if len(raw_ts) >= 19 else f"T+{idx}"
                step_label = f"Round {idx} ({time_str})"

                for ticker, weight in entry.get("new_weights", {}).items():
                    records.append({
                        "Step": step_label,
                        "Ticker": ticker,
                        "Weight": weight,
                        "Allocation": f"{weight*100:.2f}%",
                    })

            if records:
                df_hist = pd.DataFrame(records)
                fig = px.line(
                    df_hist,
                    x="Step",
                    y="Weight",
                    color="Ticker",
                    markers=True,
                    title="Portfolio Weights Trajectory Across Rebalancing Rounds",
                    height=450,
                    hover_data={"Allocation": True, "Weight": False, "Step": True},
                )
                fig.update_xaxes(
                    type="category",
                    title_text="Rebalancing Sequence",
                )
                fig.update_yaxes(
                    title_text="Portfolio Allocation",
                    tickformat=".1%",
                )
                fig.update_layout(
                    hovermode="x unified",
                    legend_title_text="Ticker",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rebalance history recorded yet. Click 'Run Tactical Rebalancing' above to generate allocation shifts.")
except Exception:
    pass
