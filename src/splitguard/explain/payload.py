"""Build the minimal, allowlisted input the model is allowed to see.

The input is derived from the report JSON structure (``AuditResult.to_dict()``)
only. It never touches DataFrames or CSV files, so raw rows cannot reach the
model. Of the report, it keeps:

* per finding: ``finding_id``, ``rule_id``, ``category``, ``severity``,
  ``confidence``, ``title``, ``description``, aggregate ``evidence``,
  ``affected_columns``, ``recommended_action`` and ``limitations``;
* checks that were skipped or failed, with their reasons;
* report warnings and limitations, the summary counts, partition row counts,
  the declared scenario, target and task type, and the split recommendation's
  status, strategy, missing information, warnings and limitations.

It drops ``inputs`` (file names, hashes, column lists), ``configuration`` apart
from the four fields above, ``feature_columns``, ``generated_at``, details of
checks that ran, and split parameters. Every string is treated as untrusted:
control characters are removed and lengths, list sizes and nesting depth are
capped. Evidence keys that could hold row-level values are dropped by name.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from splitguard.explain.errors import ExplanationInputError
from splitguard.schemas import AuditResult

# Rules the engine documents as heuristics (name or association based): their
# findings are review prompts, never direct evidence.
HEURISTIC_RULES = frozenset({"TGT-005", "TGT-006", "TMP-005", "ID-001"})
DIRECT = "direct measurement"
HEURISTIC = "heuristic review prompt"
SEVERITIES = ("critical", "high", "medium", "low", "info")
CONFIDENCES = ("high", "medium", "low")
# Never forwarded even if a future detector adds them to evidence.
EXCLUDED_EVIDENCE_KEYS = frozenset(
    {
        "rows",
        "row",
        "records",
        "raw",
        "sample",
        "samples",
        "examples",
        "example_rows",
        "values",
        "example_values",
        "row_indices",
        "indices",
        "index",
        "data",
        "dataframe",
        "path",
        "file_path",
    }
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁦-⁩]")


@dataclass(frozen=True)
class PayloadLimits:
    max_findings: int = 12
    max_string: int = 300
    max_list: int = 12
    max_keys: int = 30
    max_depth: int = 4
    max_notes: int = 15
    max_chars: int = 14_000


def clean_text(value: Any, limit: int) -> str:
    """Untrusted string -> single-line-safe text of at most ``limit`` characters."""
    text = _CONTROL.sub(" ", str(value))
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) > limit:
        text = text[: max(0, limit - 13)].rstrip() + " [truncated]"
    return text


def _clean(value: Any, limits: PayloadLimits, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return clean_text(value, limits.max_string)
    if depth >= limits.max_depth:
        return "[nested detail omitted]"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[: limits.max_keys]:
            name = clean_text(key, 60)
            if name.lower() in EXCLUDED_EVIDENCE_KEYS:
                continue
            out[name] = _clean(item, limits, depth + 1)
        if len(items) > limits.max_keys:
            out["_omitted_keys"] = len(items) - limits.max_keys
        return out
    if isinstance(value, (list, tuple)):
        cleaned = [_clean(v, limits, depth + 1) for v in value[: limits.max_list]]
        if len(value) > limits.max_list:
            cleaned.append(f"... {len(value) - limits.max_list} more")
        return cleaned
    return clean_text(value, limits.max_string)


def _strings(value: Any, limits: PayloadLimits, cap: int | None = None) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    cap = cap or limits.max_notes
    out = [clean_text(v, limits.max_string) for v in value[:cap] if v is not None]
    if len(value) > cap:
        out.append(f"... {len(value) - cap} more")
    return out


def evidence_kind(rule_id: str, confidence: str) -> str:
    return HEURISTIC if rule_id in HEURISTIC_RULES or confidence == "low" else DIRECT


def report_dict(report: AuditResult | Mapping[str, Any]) -> Mapping[str, Any]:
    """The report JSON structure for an ``AuditResult`` or an already-loaded report."""
    if isinstance(report, AuditResult):
        return report.to_dict(include_generated_at=False)
    if isinstance(report, Mapping):
        return report
    raise ExplanationInputError("Expected a SplitGuard AuditResult or a JSON report object.")


def _finding(raw: Any, limits: PayloadLimits) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ExplanationInputError("Each finding in the report must be an object.")
    fid, rule_id = raw.get("finding_id"), raw.get("rule_id")
    severity, confidence = raw.get("severity"), raw.get("confidence")
    if not isinstance(fid, str) or not fid.strip() or not isinstance(rule_id, str):
        raise ExplanationInputError("A finding in the report has no finding_id or rule_id.")
    if severity not in SEVERITIES or confidence not in CONFIDENCES:
        raise ExplanationInputError("A finding in the report has an invalid severity/confidence.")
    evidence = raw.get("evidence")
    return {
        "finding_id": clean_text(fid, 160),
        "rule_id": clean_text(rule_id, 20),
        "category": clean_text(raw.get("category", ""), 40),
        "severity": severity,
        "confidence": confidence,
        "evidence_kind": evidence_kind(rule_id, confidence),
        "title": clean_text(raw.get("title", ""), limits.max_string),
        "description": clean_text(raw.get("description", ""), limits.max_string * 2),
        "evidence": _clean(evidence if isinstance(evidence, Mapping) else {}, limits),
        "affected_columns": _strings(raw.get("affected_columns", []), limits, limits.max_list),
        "recommended_action": clean_text(raw.get("recommended_action", ""), limits.max_string),
        "limitations": _strings(raw.get("limitations", []), limits, 5),
    }


def _checks_not_run(checks: Any, limits: PayloadLimits) -> list[dict[str, Any]]:
    out = []
    for check in checks if isinstance(checks, list) else []:
        if not isinstance(check, Mapping) or check.get("status") not in {"skipped", "failed"}:
            continue
        out.append(
            {
                "check_id": clean_text(check.get("check_id", ""), 20),
                "name": clean_text(check.get("name", ""), 100),
                "status": check["status"],
                "reason": clean_text(check.get("reason") or "", limits.max_string),
            }
        )
    return out


def _split(raw: Any, limits: PayloadLimits) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    return {
        "status": clean_text(raw.get("status", ""), 40),
        "strategy": None if raw.get("strategy") is None else clean_text(raw["strategy"], 60),
        "missing_information": _strings(raw.get("missing_information", []), limits, 8),
        "warnings": _strings(raw.get("warnings", []), limits, 8),
        "limitations": _strings(raw.get("limitations", []), limits, 8),
    }


def _size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False))


def build_explanation_input(
    report: AuditResult | Mapping[str, Any], limits: PayloadLimits | None = None
) -> dict[str, Any]:
    """Return the allowlisted, size-bounded explanation input for ``report``.

    The report is only read; it is never modified.
    """
    limits = limits or PayloadLimits()
    data = report_dict(report)
    findings_raw = data.get("findings")
    if not isinstance(findings_raw, list):
        raise ExplanationInputError("The report has no 'findings' list; is it a SplitGuard report?")
    findings = [_finding(f, limits) for f in findings_raw]
    ids = [f["finding_id"] for f in findings]
    if len(ids) != len(set(ids)):
        raise ExplanationInputError("The report contains duplicate finding IDs.")
    config = data.get("configuration") if isinstance(data.get("configuration"), Mapping) else {}
    partitions = data.get("partitions") if isinstance(data.get("partitions"), Mapping) else {}
    summary = data.get("summary") if isinstance(data.get("summary"), Mapping) else {}
    payload: dict[str, Any] = {
        "report": {
            "tool_version": clean_text((data.get("tool") or {}).get("version", ""), 20),
            "mode": clean_text(data.get("mode", ""), 30),
            "task_type": None if not data.get("task_type") else clean_text(data["task_type"], 30),
            "scenario": None if not config.get("scenario") else clean_text(config["scenario"], 40),
            "target": None if not config.get("target") else clean_text(config["target"], 120),
            "partition_rows": {
                clean_text(name, 40): (part.get("n_rows") if isinstance(part, Mapping) else None)
                for name, part in list(partitions.items())[:4]
            },
        },
        "summary": {
            k: _clean(summary.get(k), limits)
            for k in (
                "total_findings",
                "by_severity",
                "checks_run",
                "checks_skipped",
                "checks_failed",
            )
        },
        "findings": findings[: limits.max_findings],
        "findings_not_sent": ids[limits.max_findings :],
        "checks_not_run": _checks_not_run(data.get("checks"), limits),
        "warnings": _strings(data.get("warnings", []), limits),
        "split_recommendation": _split(data.get("split_recommendation"), limits),
        "engine_limitations": _strings(data.get("limitations", []), limits, 8),
    }
    _fit(payload, limits)
    return payload


def _fit(payload: dict[str, Any], limits: PayloadLimits) -> None:
    """Shrink the payload in place until it fits ``limits.max_chars``.

    Order: shorten evidence to top-level scalars, then trim notes and checks, then
    move the least severe findings to ``findings_not_sent``.
    """
    if _size(payload) <= limits.max_chars:
        return
    for f in payload["findings"]:
        f["evidence"] = {
            k: v for k, v in f["evidence"].items() if not isinstance(v, (dict, list))
        } | {"_note": "nested evidence omitted to fit the size limit"}
    for key, keep in (("warnings", 5), ("checks_not_run", 10), ("engine_limitations", 3)):
        if _size(payload) <= limits.max_chars:
            return
        items = payload[key]
        if len(items) > keep:
            payload[key] = items[:keep] + [f"... {len(items) - keep} more not sent"]
    while _size(payload) > limits.max_chars and payload["findings"]:
        dropped = payload["findings"].pop()
        payload["findings_not_sent"].insert(0, dropped["finding_id"])
    if _size(payload) > limits.max_chars:
        raise ExplanationInputError("The report is too large to explain within the size limit.")
