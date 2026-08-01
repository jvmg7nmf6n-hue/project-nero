from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.hypothesis_quality_gate import build_hypothesis_quality_gate


class HypothesisQualityGateTests(unittest.TestCase):
    def test_rework_hypothesis_with_failure_fix_is_approved_for_shadow_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            variants = base / "variants.csv"
            recommendations = base / "recommendations.csv"
            verification = base / "verification.csv"
            quarantine = base / "quarantine.csv"
            edge = base / "edge.json"
            pd.DataFrame([
                {
                    "Parent": "OLD_BREAKOUT",
                    "Proposed Variant": "FIX_BREAKOUT_QUALITY_V2",
                    "Mode": "SHADOW_TEST_ONLY",
                    "Hypothesis": "Breakout losses shrink after retest confirmation and volatility filter.",
                    "Proposed Changes": "Add retest confirmation; require trend support; block volatility shock regimes; target at 1.5R.",
                }
            ]).to_csv(variants, index=False)
            pd.DataFrame([
                {
                    "Candidate": "OLD_BREAKOUT",
                    "Family": "Momentum",
                    "Trades": 92,
                    "Expectancy R": -0.66,
                    "Profit Factor": 0.59,
                    "Action": "REWORK",
                }
            ]).to_csv(recommendations, index=False)
            pd.DataFrame([
                {
                    "candidate_id": "OLD_BREAKOUT",
                    "net_pnl": -1549.48,
                    "total_trades": 92,
                    "expectancy_r": -0.66,
                    "profit_factor": 0.59,
                }
            ]).to_csv(verification, index=False)
            pd.DataFrame([
                {"candidate_id": "OLD_BREAKOUT", "net_pnl": -1549.48}
            ]).to_csv(quarantine, index=False)
            edge.write_text('{"rows": []}', encoding="utf-8")

            report, summary = build_hypothesis_quality_gate(
                variants, recommendations, verification, quarantine, edge, base / "out.csv", base / "out.json"
            )

            self.assertEqual(summary.approved_shadow_tests, 1)
            self.assertEqual(report.iloc[0]["decision"], "APPROVE_SHADOW_TEST")
            self.assertTrue(report.iloc[0]["fixes_known_failure"])

    def test_high_overfit_small_sample_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            variants = base / "variants.csv"
            recommendations = base / "recommendations.csv"
            verification = base / "verification.csv"
            quarantine = base / "quarantine.csv"
            edge = base / "edge.json"
            pd.DataFrame([
                {
                    "Parent": "TINY",
                    "Proposed Variant": "TINY_V2",
                    "Mode": "SHADOW_TEST_ONLY",
                    "Hypothesis": "Tune RSI ADX MA ATR volatility target retest filter for a tiny sample.",
                    "Proposed Changes": "RSI, ADX, MA, ATR, volatility, target, retest, reward filter.",
                }
            ]).to_csv(variants, index=False)
            pd.DataFrame([
                {
                    "Candidate": "TINY",
                    "Family": "Mean Reversion",
                    "Trades": 4,
                    "Expectancy R": 1.1,
                    "Profit Factor": 4.0,
                    "Action": "COLLECT_MORE_DATA",
                }
            ]).to_csv(recommendations, index=False)
            pd.DataFrame([
                {"candidate_id": "TINY", "net_pnl": 200.0, "total_trades": 4}
            ]).to_csv(verification, index=False)
            pd.DataFrame().to_csv(quarantine, index=False)
            edge.write_text('{"rows": []}', encoding="utf-8")

            report, summary = build_hypothesis_quality_gate(
                variants, recommendations, verification, quarantine, edge, base / "out.csv", base / "out.json"
            )

            self.assertEqual(summary.rejected, 1)
            self.assertEqual(report.iloc[0]["decision"], "REJECT_WEAK_IDEA")
            self.assertEqual(report.iloc[0]["overfit_risk"], "HIGH")


if __name__ == "__main__":
    unittest.main()
