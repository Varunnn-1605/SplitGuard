"""JSON report writing and a concise terminal summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from splitguard.schemas import AuditResult, Severity


def report_to_json(result: AuditResult, *, include_generated_at: bool = True) -> str:
    """Serialise a result. ``allow_nan=False`` guarantees strict JSON."""
    return json.dumps(
        result.to_dict(include_generated_at=include_generated_at),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def write_json_report(
    result: AuditResult, path: str | Path, *, include_generated_at: bool = True
) -> Path:
    """Write the report to ``path`` (parent directories must exist)."""
    target = Path(path)
    target.write_text(
        report_to_json(result, include_generated_at=include_generated_at) + "\n", encoding="utf-8"
    )
    return target


def load_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_summary(result: AuditResult, *, max_findings: int = 20) -> str:
    """Plain-ASCII terminal summary (safe on any console encoding)."""
    summary = result.summary()
    lines = [f"SplitGuard {result.tool_version} | mode: {result.mode}"]
    for item in result.inputs:
        lines.append(
            f"  input ({item['role']}): {item['name']} - {item['n_rows']} rows x {item['n_columns']} columns"
        )
    for name, shape in result.partitions.items():
        lines.append(f"  partition {name}: {shape['n_rows']} rows")
    lines.append(
        f"Checks: {summary['checks_run']} run, {summary['checks_skipped']} skipped, "
        f"{summary['checks_failed']} failed"
    )
    counts = summary["by_severity"]
    lines.append(
        f"Findings: {summary['total_findings']} total | "
        + " | ".join(f"{s.value} {counts[s.value]}" for s in Severity)
    )
    for finding in result.findings[:max_findings]:
        lines.append(
            f"  [{finding.severity.value.upper()}, {finding.confidence.value} confidence] "
            f"{finding.finding_id}: {_ascii(finding.title)}"
        )
    if len(result.findings) > max_findings:
        lines.append(f"  ... and {len(result.findings) - max_findings} more (see the JSON report)")
    if result.split_recommendation is not None:
        rec = result.split_recommendation
        if rec.status == "recommended":
            lines.append(f"Split recommendation: {rec.strategy}")
        else:
            lines.append("Split recommendation: needs user input")
            lines += [f"  missing: {_ascii(m)}" for m in rec.missing_information]
    for check in result.checks:
        if check.status.value == "failed":
            lines.append(f"  FAILED {check.check_id}: {check.reason}")
    for warning in result.warnings:
        lines.append(f"Warning: {_ascii(warning)}")
    lines.append("Note: no findings is not proof that the data is free of leakage.")
    return "\n".join(lines)


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")
