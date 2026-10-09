"""
Root entry point for the Autonomous AI/NLP Financial Risk Engine.
S&P Global & CRISIL Campus Hackathon.

Usage:
    python main.py             # Runs end-to-end CLI demo across Ingestion -> NLP -> Module A -> Module B
    python main.py --api       # Launches the FastAPI REST backend server
    python main.py --dashboard # Launches the Streamlit interactive dashboard
    python main.py --all       # Launches both API backend and Dashboard concurrently
    python main.py --test      # Runs test suite with pytest
"""

import sys
import os
import argparse
import subprocess
import json

def run_cli_demo():
    print("=" * 80)
    print("  AUTONOMOUS AI/NLP FINANCIAL RISK ENGINE & DECISION PLATFORM")
    print("  S&P Global & CRISIL Campus Hackathon Prototype")
    print("=" * 80)
    
    # 1. Initialize Engines
    print("\n[1/4] Initializing AI/NLP Risk Engine & Quantitative Modules...")
    from src.nlp_engine.risk_engine import RiskEngine
    from src.module_a.rebalancer import IndexRebalancer
    from src.module_b.stress_tester import StressTester

    engine = RiskEngine()
    rebalancer = IndexRebalancer()
    stress_tester = StressTester()
    print("[OK] FinBERT sentiment model loaded")
    print("[OK] BART-large-MNLI zero-shot classifier loaded")
    print("[OK] Module A Index Rebalancer initialized (15 S&P stocks)")
    print("[OK] Module B Wholesale Banking Stress Tester initialized ($40.3M balance sheet)")

    # 2. Ingest Sample News
    print("\n[2/4] Ingesting Sample Financial Feeds...")
    data_path = os.path.join("data", "raw", "sample_financial_news.json")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
    else:
        articles = [
            {"title": "Federal Reserve signals aggressive 50 bps rate hike amid inflation", "ticker": "BAC"},
            {"title": "Middle East escalating geopolitical conflict disrupts key oil pipeline", "ticker": "XOM"},
            {"title": "Microsoft reports record Azure cloud AI revenue growth surging 32%", "ticker": "MSFT"},
        ]

    print(f"[OK] Ingested {len(articles)} headlines for processing.")

    # 3. Analyze with NLP Engine
    print("\n[3/4] Processing Headlines through AI/NLP Pipeline...")
    normalized_articles = [
        {
            "text": a.get("text") or a.get("title", ""),
            "ticker": a.get("ticker"),
            "source": a.get("source", "NewsAPI"),
        }
        for a in articles
    ]
    signals = engine.analyze_batch(normalized_articles)

    print("\n" + "-" * 80)
    print(f"{'Ticker':<8} | {'Sentiment':<10} | {'Event Category':<18} | {'Impact (1-10)':<14} | Headline")
    print("-" * 80)
    for sig in signals:
        ticker = sig.get("ticker", "N/A")
        sent = f"{sig.get('sentiment_score', 0):+.2f} ({sig.get('sentiment_label', '')})"
        event = sig.get("event_classification") or sig.get("event_type", "General")
        impact = f"{sig.get('impact_score', 0)}/10"
        title = sig.get("text", "")[:40] + "..."
        print(f"{ticker:<8} | {sent:<10} | {event:<18} | {impact:<14} | {title}")
    print("-" * 80)

    # 4. Trigger Downstream Modules
    print("\n[4/4] Executing Downstream Quantitative Modules...")
    
    # Module A: Tactical Rebalancing
    print("\n--- [Module A: S&P Benchmark Rebalancing] ---")
    rebalance_res = rebalancer.rebalance(signals, adjustment_factor=0.05, sentiment_threshold=0.05)
    print(f"Status: {rebalance_res.get('message')}")
    if rebalance_res.get("changes"):
        for t, chg in sorted(rebalance_res["changes"].items(), key=lambda x: x[1]):
            arrow = "[+]" if chg > 0 else "[-]"
            print(f"  {arrow} {t:<6}: {chg:+.4f} (New Weight: {rebalance_res['new_weights'][t]:.2%})")

    # Module B: Wholesale Banking Stress Test
    print("\n--- [Module B: Wholesale Banking Stress Test] ---")
    high_impact_signals = [s for s in signals if s.get("impact_score", 0) >= 7]
    if high_impact_signals:
        top_sig = high_impact_signals[0]
        event_name = top_sig.get("event_classification") or top_sig.get("event_type", "Macroeconomic")
        stress_res = stress_tester.run_stress_test(
            event_type=event_name,
            impact_score=top_sig.get("impact_score", 8)
        )
        print(f"Triggered by High-Impact Event: {top_sig.get('text')}")
        print(f"Pre-Stress Value:  ${stress_res['total_pre_stress']:,.0f}")
        print(f"Post-Stress Value: ${stress_res['total_post_stress']:,.0f}")
        print(f"Total Loss:        -${stress_res['total_loss']:,.0f} ({stress_res['total_loss_pct']:.2f}%)")
    else:
        stress_res = stress_tester.run_stress_test("Macroeconomic", impact_score=8)
        print("Running Baseline Macroeconomic Shock:")
        print(f"Pre-Stress Value:  ${stress_res['total_pre_stress']:,.0f}")
        print(f"Post-Stress Value: ${stress_res['total_post_stress']:,.0f}")
        print(f"Total Loss:        -${stress_res['total_loss']:,.0f} ({stress_res['total_loss_pct']:.2f}%)")

    print("\n" + "=" * 80)
    print("[SUCCESS] Full pipeline execution complete!")
    print("To launch the web dashboards and interactive REST API:")
    print("  1. REST API:    uvicorn src.api.main:app --reload --port 8000")
    print("  2. Dashboard:   streamlit run dashboards/main_dashboard.py")
    print("  3. Or run:      python main.py --all")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI/NLP Financial Risk Engine Entry Point")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI REST backend server")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit multi-page dashboard")
    parser.add_argument("--all", action="store_true", help="Launch both API server and Streamlit dashboard")
    parser.add_argument("--test", action="store_true", help="Run full test suite via pytest")
    args = parser.parse_args()

    if args.test:
        subprocess.run([sys.executable, "-m", "pytest", "tests/"])
    elif args.api:
        subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", "--port", "8000"])
    elif args.dashboard:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboards/main_dashboard.py"])
    elif args.all:
        import threading
        api_thread = threading.Thread(
            target=lambda: subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", "--port", "8000"])
        )
        api_thread.daemon = True
        api_thread.start()
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboards/main_dashboard.py"])
    else:
        run_cli_demo()

if __name__ == "__main__":
    main()
