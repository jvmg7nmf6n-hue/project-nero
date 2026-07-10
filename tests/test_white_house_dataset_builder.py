from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from nero_app.core.white_house_dataset_builder import (
    build_impact_summary,
    build_white_house_dataset,
    enrich_events_with_returns,
)


class WhiteHouseDatasetBuilderTest(unittest.TestCase):
    def test_enrich_events_with_forward_returns(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": "E1",
                    "date": "2025-01-01",
                    "event_type": "crypto_policy",
                    "tags": "crypto_friendly_policy|policy_clarity",
                    "confidence": 0.7,
                }
            ]
        )
        btc_prices = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=40, freq="D"),
                "close": [100 + i for i in range(40)],
            }
        )
        gold_prices = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=40, freq="D"),
                "close": [200 + i * 0.5 for i in range(40)],
            }
        )

        enriched = enrich_events_with_returns(events, btc_prices=btc_prices, gold_prices=gold_prices)

        self.assertAlmostEqual(float(enriched.iloc[0]["btc_return_7d"]), 0.07)
        self.assertAlmostEqual(float(enriched.iloc[0]["gold_return_7d"]), 3.5 / 200)
        self.assertGreater(float(enriched.iloc[0]["btc_impact_score"]), 0)

    def test_build_impact_summary_groups_by_tags_and_event_type(self) -> None:
        events = pd.DataFrame(
            [
                {"event_type": "crypto_policy", "tags": "crypto_friendly_policy|policy_clarity", "btc_return_7d": 0.05, "gold_return_7d": 0.01, "confidence": 0.8},
                {"event_type": "crypto_policy", "tags": "crypto_friendly_policy", "btc_return_7d": -0.01, "gold_return_7d": 0.00, "confidence": 0.6},
            ]
        )

        summary = build_impact_summary(events)

        crypto = summary[summary["group"] == "crypto_friendly_policy"].iloc[0]
        event_type = summary[summary["group"] == "event_type:crypto_policy"].iloc[0]
        self.assertEqual(int(crypto["events"]), 2)
        self.assertAlmostEqual(float(crypto["btc_avg_7d"]), 0.02)
        self.assertEqual(int(event_type["events"]), 2)

    def test_build_white_house_dataset_writes_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "events.csv"
            btc_path = root / "btc.csv"
            gold_path = root / "gold.csv"
            output_path = root / "enriched.csv"
            summary_path = root / "summary.csv"

            pd.DataFrame(
                [
                    {
                        "event_id": "E1",
                        "date": "2025-01-01",
                        "event_type": "sanctions_geopolitics",
                        "tags": "sanctions|war|safe_haven",
                        "confidence": 0.65,
                    }
                ]
            ).to_csv(input_path, index=False)
            pd.DataFrame({"date": pd.date_range("2025-01-01", periods=40, freq="D"), "close": [100 + i for i in range(40)]}).to_csv(btc_path, index=False)
            pd.DataFrame({"date": pd.date_range("2025-01-01", periods=40, freq="D"), "close": [200 + i for i in range(40)]}).to_csv(gold_path, index=False)

            result = build_white_house_dataset(
                input_path=input_path,
                output_path=output_path,
                summary_path=summary_path,
                btc_price_path=btc_path,
                gold_price_path=gold_path,
            )

            self.assertEqual(result.events, 1)
            self.assertEqual(result.btc_enriched, 1)
            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
