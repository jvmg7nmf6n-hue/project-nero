from __future__ import annotations

import unittest

import pandas as pd

from nero_app.core.white_house_impact import (
    classify_white_house_text,
    format_white_house_impact_report,
    score_white_house_impact,
)


class WhiteHouseImpactTest(unittest.TestCase):
    def test_classifies_crypto_policy_text(self) -> None:
        tags = classify_white_house_text("President announces strategic bitcoin reserve and stablecoin framework.")

        self.assertIn("strategic_bitcoin_reserve", tags)
        self.assertIn("crypto_friendly_policy", tags)
        self.assertIn("policy_clarity", tags)

    def test_crypto_friendly_policy_scores_btc_impact(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "date": "2025-03-06",
                    "headline": "Strategic Bitcoin Reserve established",
                    "tags": "crypto_friendly_policy|strategic_bitcoin_reserve|institutional_legitimacy|structural_adoption",
                    "btc_impact_score": 86,
                    "gold_impact_score": 15,
                    "confidence": 0.78,
                },
                {
                    "date": "2025-01-23",
                    "headline": "Digital financial technology order",
                    "tags": "crypto_friendly_policy|policy_clarity|anti_cbdc|regulatory_framework",
                    "btc_impact_score": 78,
                    "gold_impact_score": 12,
                    "confidence": 0.72,
                },
            ]
        )

        result = score_white_house_impact("White House announces strategic bitcoin reserve and digital asset framework", events)

        self.assertGreaterEqual(result.btc_average_impact, 70)
        self.assertLess(result.gold_average_impact, 35)
        self.assertIn("BTC-specific", " ".join(result.notes))
        self.assertIn("NERO White House Market Impact", format_white_house_impact_report(result))

    def test_geopolitical_sanctions_scores_gold_impact(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "date": "2022-02-24",
                    "headline": "Russia sanctions after Ukraine invasion",
                    "tags": "sanctions|war|geopolitical_risk|risk_off|safe_haven|oil_supply_risk",
                    "btc_impact_score": 55,
                    "gold_impact_score": 70,
                    "confidence": 0.65,
                }
            ]
        )

        result = score_white_house_impact("President announces sanctions after geopolitical conflict and war risk", events)

        self.assertGreaterEqual(result.gold_average_impact, 65)
        self.assertIn("safe haven", " ".join(result.notes))
        self.assertIn("sanctions", result.query_tags)


if __name__ == "__main__":
    unittest.main()
