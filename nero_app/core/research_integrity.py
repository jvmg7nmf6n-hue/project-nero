from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd

from nero_app.core.quant_intelligence import max_drawdown, sharpe_ratio, sortino_ratio
from nero_app.core.technical_analysis import _rsi


REPORT_DIR = Path("reports")


@dataclass(frozen=True)
class DependencyAuditRow:
    library: str
    package: str
    role: str
    status: str
    decision: str
    notes: str


def audit_research_dependencies() -> pd.DataFrame:
    rows = [
        _dependency_row(
            "arch",
            "arch",
            "GARCH volatility validation",
            "HARD_DEPENDENCY_ALREADY_IN_REQUIREMENTS",
            "Used by Quant Intel for GARCH(1,1); EWMA fallback remains available if fitting fails.",
        ),
        _dependency_row(
            "statsmodels",
            "statsmodels",
            "Cointegration and Granger validation",
            "HARD_DEPENDENCY_ALREADY_IN_REQUIREMENTS",
            "Used as the reference implementation for Engle-Granger/ADF and Granger tests.",
        ),
        _dependency_row(
            "pandas-ta",
            "pandas_ta",
            "Technical indicator reference checks",
            "OPTIONAL_REFERENCE_ADAPTER",
            "Preferred over TA-Lib for cloud portability; NERO can compare RSI/MACD/Bollinger when installed.",
        ),
        _dependency_row(
            "TA-Lib",
            "talib",
            "Technical indicator reference checks",
            "NOT_HARD_DEPENDENCY",
            "Binary wheels can break Windows/Streamlit deployment; use only if already available locally.",
        ),
        _dependency_row(
            "empyrical",
            "empyrical",
            "Financial metric cross-checks",
            "OPTIONAL_REFERENCE_ADAPTER",
            "Cross-checks Sharpe, Sortino, and max drawdown without replacing NERO's stored calculations.",
        ),
        _dependency_row(
            "optuna",
            "optuna",
            "Disciplined hyperparameter research",
            "RESEARCH_ONLY_WITH_GUARDRAILS",
            "Never auto-deploy optimized parameters; use only with chronological split, OOS, CI, and manual versioning.",
        ),
        _dependency_row(
            "hyperopt",
            "hyperopt",
            "Alternative hyperparameter search",
            "NOT_SELECTED_INITIALLY",
            "Optuna is the cleaner first adapter; adding both increases complexity and overfit risk.",
        ),
    ]
    return pd.DataFrame([asdict(row) for row in rows])


def build_indicator_reference_report() -> pd.DataFrame:
    closes = pd.Series(
        [
            100.0,
            101.5,
            100.8,
            102.1,
            103.0,
            102.6,
            104.2,
            105.1,
            104.7,
            106.0,
            107.4,
            106.9,
            108.2,
            109.0,
            108.4,
            110.3,
            111.1,
            110.6,
            112.0,
            113.2,
            112.7,
            114.0,
            115.1,
            114.5,
            116.4,
            117.0,
            116.2,
            118.5,
            119.3,
            118.8,
        ]
    )
    local_rsi = _rsi(closes, period=14)
    rows: list[dict[str, Any]] = []

    pandas_ta = _optional_import("pandas_ta")
    if pandas_ta is None:
        rows.append(
            {
                "check": "RSI(14)",
                "local_value": round(local_rsi, 6),
                "reference_library": "pandas-ta",
                "reference_value": "",
                "absolute_diff": "",
                "status": "REFERENCE_NOT_INSTALLED",
                "notes": "Install pandas-ta locally to compare indicator formulas against a reference implementation.",
            }
        )
    else:
        ref_value = _latest_numeric(pandas_ta.rsi(closes, length=14))
        diff = abs(local_rsi - ref_value)
        rows.append(
            {
                "check": "RSI(14)",
                "local_value": round(local_rsi, 6),
                "reference_library": "pandas-ta",
                "reference_value": round(ref_value, 6),
                "absolute_diff": round(diff, 8),
                "status": "PASS" if diff <= 1.0 else "REVIEW",
                "notes": "Small differences can occur from smoothing convention; large differences indicate formula drift.",
            }
        )

    talib = _optional_import("talib")
    rows.append(
        {
            "check": "TA-Lib availability",
            "local_value": "",
            "reference_library": "TA-Lib",
            "reference_value": "",
            "absolute_diff": "",
            "status": "AVAILABLE" if talib is not None else "NOT_INSTALLED",
            "notes": "TA-Lib remains optional because it is a deployment-risky binary dependency.",
        }
    )
    return pd.DataFrame(rows)


def build_metric_reference_report() -> pd.DataFrame:
    returns = pd.Series([0.012, -0.008, 0.004, 0.018, -0.011, 0.006, 0.003, -0.004, 0.014, -0.006] * 12)
    equity = (1 + returns).cumprod()
    local = {
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(equity),
    }
    empyrical = _optional_import("empyrical")
    rows: list[dict[str, Any]] = []
    if empyrical is None:
        for metric, value in local.items():
            rows.append(
                {
                    "metric": metric,
                    "local_value": round(float(value), 6),
                    "reference_library": "empyrical",
                    "reference_value": "",
                    "absolute_diff": "",
                    "status": "REFERENCE_NOT_INSTALLED",
                    "notes": "Install empyrical/maintained fork locally to cross-check financial metrics.",
                }
            )
        return pd.DataFrame(rows)

    reference = {
        "Sharpe": float(empyrical.sharpe_ratio(returns, annualization=252)),
        "Sortino": float(empyrical.sortino_ratio(returns, annualization=252)),
        "Max Drawdown": float(empyrical.max_drawdown(equity)),
    }
    for metric, value in local.items():
        ref_value = reference[metric]
        diff = abs(float(value) - ref_value)
        rows.append(
            {
                "metric": metric,
                "local_value": round(float(value), 6),
                "reference_library": "empyrical",
                "reference_value": round(ref_value, 6),
                "absolute_diff": round(diff, 8),
                "status": "PASS" if diff <= 0.05 else "REVIEW",
                "notes": "Reference check against standard finance metrics implementation.",
            }
        )
    return pd.DataFrame(rows)


def build_optimizer_guardrail_report() -> pd.DataFrame:
    optuna = _optional_import("optuna")
    rows = [
        {
            "guardrail": "Optimizer dependency",
            "status": "AVAILABLE" if optuna is not None else "NOT_INSTALLED",
            "required_before_use": "Install optuna only for research runs; never required for live/paper execution.",
        },
        {
            "guardrail": "Chronological split",
            "status": "MANDATORY",
            "required_before_use": "Train/test must be time-ordered; random shuffle is not allowed for trading systems.",
        },
        {
            "guardrail": "Out-of-sample survival",
            "status": "MANDATORY",
            "required_before_use": "Optimized parameters must improve or hold up in the unseen test half.",
        },
        {
            "guardrail": "Random-entry baseline",
            "status": "MANDATORY",
            "required_before_use": "Strategy timing must beat a random entry baseline inside the same regime pool.",
        },
        {
            "guardrail": "Minimum sample",
            "status": "MANDATORY",
            "required_before_use": "No optimized variant can be promoted before at least 30-50 closed trades.",
        },
        {
            "guardrail": "Manual versioning",
            "status": "MANDATORY",
            "required_before_use": "NERO can recommend parameter changes, but cannot silently auto-change live/paper strategy parameters.",
        },
    ]
    return pd.DataFrame(rows)


def write_research_integrity_reports(report_dir: Path = REPORT_DIR) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dependencies": audit_research_dependencies(),
        "indicators": build_indicator_reference_report(),
        "metrics": build_metric_reference_report(),
        "optimizer_guardrails": build_optimizer_guardrail_report(),
    }
    written: dict[str, str] = {}
    for name, frame in outputs.items():
        csv_path = report_dir / f"research_integrity_{name}.csv"
        json_path = report_dir / f"research_integrity_{name}.json"
        frame.to_csv(csv_path, index=False)
        frame.to_json(json_path, orient="records", indent=2)
        written[name] = str(csv_path)

    summary = {
        "dependency_rows": len(outputs["dependencies"]),
        "indicator_checks": len(outputs["indicators"]),
        "metric_checks": len(outputs["metrics"]),
        "optimizer_guardrails": len(outputs["optimizer_guardrails"]),
        "hard_dependencies": ["arch", "statsmodels"],
        "optional_reference_adapters": ["pandas-ta", "TA-Lib", "empyrical", "optuna"],
        "policy": "Third-party libraries verify NERO's calculations; they do not auto-authorize real-money trades.",
    }
    summary_path = report_dir / "research_integrity_summary.json"
    pd.Series(summary, dtype="object").to_json(summary_path, indent=2)
    written["summary"] = str(summary_path)
    return written


def _dependency_row(library: str, package: str, role: str, decision: str, notes: str) -> DependencyAuditRow:
    return DependencyAuditRow(
        library=library,
        package=package,
        role=role,
        status="INSTALLED" if _optional_import(package) is not None else "NOT_INSTALLED",
        decision=decision,
        notes=notes,
    )


def _optional_import(package: str) -> Any | None:
    try:
        return import_module(package)
    except Exception:
        return None


def _latest_numeric(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return 0.0 if clean.empty else float(clean.iloc[-1])
