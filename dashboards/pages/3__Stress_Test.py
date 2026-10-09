"""
Streamlit Page 3: Strategic Portfolio Stress Tester (Module B).

Purpose:
  Provides an interactive workspace for simulating macro shocks on institutional balance sheets.

Role in Project:
  - Displays the $40.3M multi-asset synthetic wholesale banking book (bonds, loans, equities, derivatives).
  - Simulates severity shocks triggered by high-impact NLP events (Impact Score >= 7).
  - Visualizes pre- and post-stress valuations, dollar drawdowns, and loss breakdowns by asset class.
  - Generates institutional risk summaries to assess capital adequacy under severe market distress.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Stress Test", layout="wide")

API_BASE = st.session_state.get("api_url", "http://localhost:8000")

st.title("Strategic Portfolio Stress Tester")
st.markdown("Simulate the impact of major financial events on a synthetic banking portfolio.")

# ─── Portfolio Overview ──────────────────────────────────────────────────────
st.markdown("### Synthetic Portfolio")

try:
    resp = requests.get(f"{API_BASE}/api/stress-test/portfolio", timeout=25)
    if resp.status_code == 200:
        data = resp.json()
        portfolio = data["portfolio"]
        total_value = data["total_value"]

        st.metric("Total Portfolio Value", f"${total_value:,.0f}")

        # Portfolio table
        df_portfolio = pd.DataFrame([
            {
                "Asset": a["name"],
                "Type": a["type"],
                "Sector": a.get("sector", "-"),
                "Value": f"${a.get('value', 0):,.0f}",
                "Rating": a.get("rating", "-"),
            }
            for a in portfolio
        ])
        st.dataframe(df_portfolio, use_container_width=True, hide_index=True)

        # Portfolio composition by type
        type_values = {}
        for a in portfolio:
            t = a["type"]
            type_values[t] = type_values.get(t, 0) + a.get("value", 0)

        fig = px.pie(
            values=list(type_values.values()),
            names=list(type_values.keys()),
            title="Portfolio Composition by Asset Type",
            hole=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

except requests.ConnectionError:
    st.error("API not running. Start it with: `uvicorn src.api.main:app --reload`")
except Exception as e:
    st.error(f"Error: {e}")

st.markdown("---")

# ─── Stress Test Controls ───────────────────────────────────────────────────
st.markdown("### Run Stress Test")

col_event, col_impact, col_btn = st.columns([2, 1, 1])

with col_event:
    event_type = st.selectbox(
        "Event Type",
        [
            "Geopolitical",
            "Macroeconomic",
            "Credit Event",
            "Merger/Acquisition",
            "Regulatory",
            "Natural Disaster",
        ],
    )

with col_impact:
    impact_score = st.slider("Impact Score", 1, 10, 8)

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_manual = st.button("Run Stress Test", type="primary", use_container_width=True)

# Auto stress test based on signals
col_auto, _ = st.columns([1, 2])
with col_auto:
    run_auto = st.button("Auto Stress Test (from signals)", use_container_width=True)

# ─── Execute Stress Test ─────────────────────────────────────────────────────
result = None

if run_manual:
    with st.spinner(f"Running {event_type} stress test (Impact: {impact_score})..."):
        try:
            resp = requests.post(
                f"{API_BASE}/api/stress-test/run",
                json={"event_type": event_type, "impact_score": impact_score},
                timeout=10,
            )
            if resp.status_code == 200:
                result = resp.json()
        except Exception as e:
            st.error(f"Error: {e}")

if run_auto:
    with st.spinner("Checking for high-impact signals..."):
        try:
            resp = requests.post(f"{API_BASE}/api/stress-test/auto", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "stress_test_result" in data:
                    result = data["stress_test_result"]
                    st.info(
                        f"Triggered by: **{data['triggered_by']['text'][:100]}...** "
                        f"(Impact: {data['triggered_by']['impact_score']})"
                    )
                else:
                    st.info(data.get("message", "No high-impact signals found."))
        except Exception as e:
            st.error(f"Error: {e}")

# ─── Display Results ─────────────────────────────────────────────────────────
if result:
    st.markdown("---")
    st.markdown(f"### Stress Test Results: {result['scenario_name']}")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pre-Stress Value", f"${result['total_pre_stress']:,.0f}")
    m2.metric("Post-Stress Value", f"${result['total_post_stress']:,.0f}")
    m3.metric("Total Loss", f"${result['total_loss']:,.0f}", delta=f"-{result['total_loss_pct']:.1f}%")
    m4.metric("Impact Score", f"{result['impact_score']} / 10")

    # Before/After waterfall chart
    assets = result["asset_results"]

    fig = go.Figure()

    asset_names = [a["name"] for a in assets]
    pre_values = [a["pre_stress_value"] for a in assets]
    post_values = [a["post_stress_value"] for a in assets]

    fig.add_trace(go.Bar(
        name="Pre-Stress",
        x=asset_names,
        y=pre_values,
        marker_color="#3b82f6",
    ))
    fig.add_trace(go.Bar(
        name="Post-Stress",
        x=asset_names,
        y=post_values,
        marker_color="#ef4444",
    ))

    fig.update_layout(
        title="Portfolio Value: Before vs After Stress Test",
        barmode="group",
        yaxis_title="Value ($)",
        height=450,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Loss breakdown chart
    fig_loss = px.bar(
        x=[a["name"] for a in assets],
        y=[a["loss"] for a in assets],
        color=[a["loss_pct"] for a in assets],
        color_continuous_scale="Reds",
        title="Loss by Asset",
        labels={"x": "Asset", "y": "Loss ($)", "color": "Loss %"},
    )
    fig_loss.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_loss, use_container_width=True)

    # Detailed table
    st.markdown("#### Per-Asset Breakdown")
    df_results = pd.DataFrame([
        {
            "Asset": a["name"],
            "Type": a["asset_type"],
            "Pre-Stress": f"${a['pre_stress_value']:,.0f}",
            "Post-Stress": f"${a['post_stress_value']:,.0f}",
            "Loss": f"${a['loss']:,.0f}",
            "Loss %": f"{a['loss_pct']:.1f}%",
        }
        for a in assets
    ])
    st.dataframe(df_results, use_container_width=True, hide_index=True)

st.markdown("---")

# ─── Stress Test History ─────────────────────────────────────────────────────
st.markdown("### Stress Test History")

try:
    resp = requests.get(f"{API_BASE}/api/stress-test/history", timeout=25)
    if resp.status_code == 200:
        history = resp.json().get("history", [])
        if history:
            df_hist = pd.DataFrame([
                {
                    "Time": h["timestamp"][:19],
                    "Scenario": h["scenario_name"],
                    "Event": h["event_type"],
                    "Impact": h["impact_score"],
                    "Pre-Stress": f"${h['total_pre_stress']:,.0f}",
                    "Post-Stress": f"${h['total_post_stress']:,.0f}",
                    "Loss": f"${h['total_loss']:,.0f}",
                    "Loss %": f"{h['total_loss_pct']:.1f}%",
                }
                for h in reversed(history)
            ])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("No stress tests run yet.")
except Exception:
    pass
