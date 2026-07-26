from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nero_app.core.research_integrity import (
    audit_research_dependencies,
    build_indicator_reference_report,
    build_metric_reference_report,
    build_optimizer_guardrail_report,
    write_research_integrity_reports,
)


class ResearchIntegrityTests(unittest.TestCase):
    def test_dependency_audit_includes_required_reference_libraries(self) -> None:
        report = audit_research_dependencies()

        self.assertIn("arch", set(report["library"]))
        self.assertIn("statsmodels", set(report["library"]))
        self.assertIn("optuna", set(report["library"]))
        self.assertIn("RESEARCH_ONLY_WITH_GUARDRAILS", set(report["decision"]))

    def test_indicator_report_is_safe_without_optional_dependencies(self) -> None:
        report = build_indicator_reference_report()

        self.assertGreaterEqual(len(report), 2)
        self.assertIn("status", report.columns)
        self.assertIn("RSI(14)", set(report["check"]))

    def test_metric_report_is_safe_without_empyrical(self) -> None:
        report = build_metric_reference_report()

        self.assertEqual(set(report["metric"]), {"Sharpe", "Sortino", "Max Drawdown"})
        self.assertIn("status", report.columns)

    def test_optimizer_guardrails_block_silent_overfit(self) -> None:
        report = build_optimizer_guardrail_report()

        mandatory = report[report["status"] == "MANDATORY"]
        self.assertGreaterEqual(len(mandatory), 4)
        self.assertIn("Manual versioning", set(report["guardrail"]))

    def test_write_reports_creates_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = write_research_integrity_reports(Path(tmp))

            self.assertIn("summary", written)
            self.assertTrue((Path(tmp) / "research_integrity_dependencies.csv").exists())
            self.assertTrue((Path(tmp) / "research_integrity_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
