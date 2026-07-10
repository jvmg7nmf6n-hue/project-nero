from __future__ import annotations

import unittest
from unittest.mock import patch

from nero_app.core.schema import VerdictOutput
from nero_app.core.verdict_modifiers import apply_white_house_modifier
from nero_app.core.white_house_impact import WhiteHouseImpactResult


class VerdictModifierTest(unittest.TestCase):
    def test_white_house_crypto_policy_can_lift_btc_verdict(self) -> None:
        base = VerdictOutput(
            direction="neutral",
            confidence=0.40,
            risk_score=0.60,
            summary="Base verdict.",
            drivers=[],
        )
        impact = WhiteHouseImpactResult(
            query_tags={"crypto_friendly_policy", "strategic_bitcoin_reserve"},
            matched_events=3,
            btc_average_impact=86.0,
            gold_average_impact=15.0,
            btc_direction="bullish/high positive impact",
            gold_direction="low impact",
            confidence=0.82,
            top_events=[],
            notes=[],
        )

        with patch("nero_app.core.verdict_modifiers.score_white_house_impact", return_value=impact):
            modified, returned_impact = apply_white_house_modifier("BTC", "White House strategic bitcoin reserve", base)

        self.assertIs(returned_impact, impact)
        self.assertEqual(modified.direction, "bullish")
        self.assertGreater(modified.confidence, base.confidence)
        self.assertTrue(any("White House impact" in driver for driver in modified.drivers))
        self.assertIn("White House impact memory", modified.summary)

    def test_unsupported_asset_is_not_modified(self) -> None:
        base = VerdictOutput(
            direction="neutral",
            confidence=0.40,
            risk_score=0.60,
            summary="Base verdict.",
            drivers=[],
        )

        modified, impact = apply_white_house_modifier("OIL", "White House sanctions on energy", base)

        self.assertIs(modified, base)
        self.assertIsNone(impact)


if __name__ == "__main__":
    unittest.main()
