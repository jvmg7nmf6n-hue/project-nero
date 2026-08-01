from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.hypothesis_quality_gate import build_hypothesis_quality_gate


def main() -> None:
    report, summary = build_hypothesis_quality_gate()
    print(
        "Hypothesis quality gate complete. "
        f"hypotheses={summary.total_hypotheses} "
        f"approved={summary.approved_shadow_tests} "
        f"repair_first={summary.repair_first} "
        f"collect={summary.collect_evidence} "
        f"rejected={summary.rejected} "
        f"top={summary.top_hypothesis} "
        f"avg_score={summary.average_score:.1f}"
    )
    if report.empty:
        return
    for _, row in report.sort_values("gate_score", ascending=False).head(5).iterrows():
        print(
            f"{row['proposed_variant']}: decision={row['decision']} "
            f"score={row['gate_score']:.0f} parent={row['parent']} "
            f"trades={int(row['parent_trades'])} risk={row['overfit_risk']}"
        )


if __name__ == "__main__":
    main()
