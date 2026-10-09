"""
Module B: Strategic Portfolio Stress Tester.

Simulates the impact of major events on a synthetic portfolio
of wholesale banking assets (bonds, loans, equities, derivatives).

When the NLP Engine detects a high-impact event (Impact Score >= 7),
this module applies predefined shock scenarios to the portfolio and
calculates the resulting losses.
"""

from datetime import datetime, timezone
from copy import deepcopy

import sys
sys.path.insert(0, ".")
from config.settings import (
    SYNTHETIC_PORTFOLIO,
    SHOCK_SCENARIOS,
    DEFAULT_SHOCK,
    STRESS_TEST_IMPACT_THRESHOLD,
)


class StressTester:
    """
    Simulates stress tests on a synthetic banking portfolio.

    Each stress test applies shock factors (equity drops, spread widening,
    rate changes) to different asset types based on the detected event.
    """

    def __init__(self):
        """Initialize with the synthetic portfolio."""
        self.portfolio = deepcopy(SYNTHETIC_PORTFOLIO)
        self.test_history: list[dict] = []
        total_value = sum(a.get("value", 0) for a in self.portfolio)
        print(f"[StressTester] Portfolio initialized: {len(self.portfolio)} assets, "
              f"total value: ${total_value:,.0f}")

    def get_portfolio(self) -> list[dict]:
        """Return the current portfolio."""
        return deepcopy(self.portfolio)

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(asset.get("value", 0) for asset in self.portfolio)

    def run_stress_test(
        self,
        event_type: str,
        impact_score: int = 8,
    ) -> dict:
        """
        Run a stress test based on an event type.

        Args:
            event_type: The event classification (e.g., "Geopolitical").
            impact_score: Severity 1-10 (scales the shock magnitude).

        Returns:
            dict with pre/post values, losses, and per-asset breakdown.
        """
        # Get the shock scenario for this event type
        scenario = SHOCK_SCENARIOS.get(event_type, DEFAULT_SHOCK)

        # Scale shocks by impact severity (impact_score / 10)
        scale_factor = impact_score / 10.0

        # Apply shocks to each asset
        asset_results = []
        total_pre = 0.0
        total_post = 0.0

        for asset in self.portfolio:
            pre_value = asset.get("value", 0)
            asset_type = asset.get("type", "")
            total_pre += pre_value

            # Calculate post-stress value based on asset type
            post_value = self._apply_shock(asset, scenario, scale_factor)
            loss = pre_value - post_value
            loss_pct = (loss / pre_value * 100) if pre_value > 0 else 0

            total_post += post_value

            asset_results.append({
                "name": asset.get("name", "Unknown"),
                "asset_type": asset_type,
                "sector": asset.get("sector", ""),
                "pre_stress_value": round(pre_value, 2),
                "post_stress_value": round(post_value, 2),
                "loss": round(loss, 2),
                "loss_pct": round(loss_pct, 2),
            })

        total_loss = total_pre - total_post
        total_loss_pct = (total_loss / total_pre * 100) if total_pre > 0 else 0

        timestamp = datetime.now(timezone.utc).isoformat()

        result = {
            "event_type": event_type,
            "scenario_name": scenario["name"],
            "impact_score": impact_score,
            "scale_factor": scale_factor,
            "total_pre_stress": round(total_pre, 2),
            "total_post_stress": round(total_post, 2),
            "total_loss": round(total_loss, 2),
            "total_loss_pct": round(total_loss_pct, 2),
            "asset_results": asset_results,
            "timestamp": timestamp,
        }

        self.test_history.append(result)
        return result

    def _apply_shock(self, asset: dict, scenario: dict, scale: float) -> float:
        """
        Apply a shock to a single asset based on its type.

        Args:
            asset: Portfolio asset dict.
            scenario: Shock scenario dict.
            scale: Scale factor from impact score.

        Returns:
            Post-stress asset value.
        """
        value = asset.get("value", 0)
        asset_type = asset.get("type", "")

        if asset_type == "Equity":
            # Direct equity price shock
            shock = scenario.get("equity_shock", 0) * scale
            return value * (1 + shock)

        elif asset_type == "Corporate Bond":
            # Spread widening reduces bond value
            # Simplified: each 100bps spread increase = ~X% price drop
            # Duration approximation: ~5 years
            spread_shock = scenario.get("corp_bond_spread_shock", 0) * scale
            price_impact = -spread_shock * 5  # duration × spread change
            return value * (1 + price_impact)

        elif asset_type == "Government Bond":
            # Rate change affects bond price (inversely)
            rate_shock = scenario.get("govt_bond_rate_shock", 0) * scale
            # Government bonds have longer duration (~7 years)
            price_impact = -rate_shock * 7
            return value * (1 + price_impact)

        elif asset_type == "Loan":
            # Loans suffer from increased defaults
            default_shock = scenario.get("loan_default_shock", 0) * scale
            return value * (1 - default_shock)

        elif asset_type in ("Interest Rate Swap", "Credit Default Swap"):
            # Derivatives: volatility shock on MTM value
            vol_shock = scenario.get("derivative_vol_shock", 0) * scale
            # Can lose or gain value; for stress test, assume adverse movement
            return value * (1 - vol_shock)

        else:
            # Unknown asset type: apply a moderate general shock
            return value * (1 + scenario.get("equity_shock", 0) * scale * 0.5)

    def should_trigger(self, impact_score: int) -> bool:
        """Check if an impact score should trigger a stress test."""
        return impact_score >= STRESS_TEST_IMPACT_THRESHOLD

    def get_history(self) -> list[dict]:
        """Return stress test history."""
        return self.test_history


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tester = StressTester()

    print(f"\nPortfolio Value: ${tester.get_portfolio_value():,.0f}")
    print(f"Assets: {len(tester.get_portfolio())}")

    # Run a geopolitical stress test
    print("\n" + "=" * 70)
    print("STRESS TEST: Geopolitical Crisis (Impact Score: 9)")
    print("=" * 70)

    result = tester.run_stress_test("Geopolitical", impact_score=9)

    print(f"\nScenario: {result['scenario_name']}")
    print(f"Pre-Stress:  ${result['total_pre_stress']:>14,.2f}")
    print(f"Post-Stress: ${result['total_post_stress']:>14,.2f}")
    print(f"Total Loss:  ${result['total_loss']:>14,.2f} ({result['total_loss_pct']:.1f}%)")

    print(f"\nPer-Asset Breakdown:")
    print(f"{'Asset':<30} {'Type':<20} {'Loss':>12} {'Loss %':>8}")
    print("-" * 72)
    for a in result["asset_results"]:
        print(f"{a['name']:<30} {a['asset_type']:<20} ${a['loss']:>10,.2f} {a['loss_pct']:>7.1f}%")
