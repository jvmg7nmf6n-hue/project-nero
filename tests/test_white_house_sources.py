from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from nero_app.core.white_house_sources import fetch_source_snapshot, list_official_sources


class WhiteHouseSourcesTest(unittest.TestCase):
    def test_source_registry_contains_official_sources(self) -> None:
        sources = list_official_sources()
        names = {source.name for source in sources}

        self.assertIn("White House Briefing Room", names)
        self.assertIn("GovInfo Presidential Documents", names)
        self.assertIn("American Presidency Project", names)

    def test_fetch_source_snapshot_counts_relevant_links(self) -> None:
        html = """
        <html><body>
          <a href="/briefing-room/presidential-actions/test">Strategic Bitcoin Reserve announcement</a>
          <a href="/briefing-room/statements/test">A ceremonial event</a>
        </body></html>
        """
        response = Mock()
        response.text = html
        response.raise_for_status.return_value = None

        with patch("nero_app.core.white_house_sources.requests.get", return_value=response):
            frame = fetch_source_snapshot(max_links_per_source=5)

        self.assertEqual(set(frame["status"]), {"ok"})
        self.assertGreaterEqual(int(frame["relevant_links"].sum()), 1)
        self.assertTrue(frame["sample_relevant_text"].astype(str).str.contains("Bitcoin Reserve").any())


if __name__ == "__main__":
    unittest.main()
