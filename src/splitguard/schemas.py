"""Typed, validated result schemas shared by every detector.

All public objects expose ``to_dict()`` returning JSON-serialisable values only.
``to_jsonable`` is the single conversion point for NumPy scalars, pandas
timestamps, missing values and non-finite floats.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import PurePath
from typing import Any

import numpy as np
import pandas as pd

from splitguard._version import REPORT_FORMAT_VERSION, __version__


class Category(StrEnum):
    """What kind of problem a finding describes."""

    DUPLICATE_CONTAMINATION = "duplicate_contamination"
    TARGET_LEAKAGE = "target_leakage"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    IDENTIFIER_RISK = "identifier_risk"
    SCHEMA_ISSUE = "schema_issue"
    SPLIT_RECOMMENDATION = "split_recommendation"


class Severity(StrEnum):
    """Potential impact *if* the suspected issue is real."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        """0 = most severe."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


class Confidence(StrEnum):
    """How strongly the available evidence supports the finding."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CheckStatus(StrEnum):
    RUN = "run"
    SKIPPED = "skipped"
    FAILED = "failed"


def to_jsonable(value: Any) -> Any:
    """Convert ``value`` into plain JSON types.

    * NumPy scalars become Python scalars.
    * NaN, +/-inf, ``None``, ``pd.NA`` and ``NaT`` become ``None``.
    * Timestamps become ISO-8601 strings; timedeltas become pandas strings.
    * Sets are sorted for determinism. Paths are reduced to their file name so
      reports never leak absolute local paths.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts.isoformat()
    if isinstance(value, (pd.Timedelta, _dt.timedelta, np.timedelta64)):
        td = pd.Timedelta(value)
        return None if pd.isna(td) else str(td)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [to_jsonable(v) for v in list(value)]
    if isinstance(value, PurePath):
        return value.name
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    return str(value)


@dataclass(frozen=True)
class Finding:
    """A single suspicious pattern with its supporting evidence.

    ``finding_id`` is ``"<rule_id>:<key>"``; ``key`` is a deterministic,
    rule-specific identifier (usually a column or partition name), so the same
    input and configuration always yield the same IDs.
    """

    rule_id: str
    key: str
    category: Category
    severity: Severity
    confidence: Confidence
    title: str
    description: str
    evidence: dict[str, Any]
    affected_columns: list[str]
    recommended_action: str
    limitations: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", Category(self.category))
        object.__setattr__(self, "severity", Severity(self.severity))
        object.__setattr__(self, "confidence", Confidence(self.confidence))
        for name in ("rule_id", "key", "title", "description", "recommended_action"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Finding.{name} must be a non-empty string")
        if not isinstance(self.evidence, dict):
            raise TypeError("Finding.evidence must be a dict")
        object.__setattr__(self, "affected_columns", [str(c) for c in self.affected_columns])
        if isinstance(self.limitations, str):
            object.__setattr__(self, "limitations", [self.limitations])
        if not self.limitations:
            raise ValueError("Finding.limitations must explain why this is not proof of leakage")

    @property
    def finding_id(self) -> str:
        return f"{self.rule_id}:{self.key}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "rule_id": self.rule_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "title": self.title,
            "description": self.description,
            "evidence": to_jsonable(self.evidence),
            "affected_columns": list(self.affected_columns),
            "recommended_action": self.recommended_action,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class CheckRecord:
    """Whether a check ran, was skipped (and why), or failed."""

    check_id: str
    name: str
    status: CheckStatus
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CheckStatus(self.status))
        if self.status is not CheckStatus.RUN and not self.reason:
            raise ValueError("Skipped or failed checks must state a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "details": to_jsonable(self.details),
        }


@dataclass(frozen=True)
class SplitRecommendation:
    """Advice on how to split an unsplit dataset.

    ``status`` is ``"recommended"`` or ``"needs_user_input"``. A recommendation
    is derived from the declared scenario and metadata; it is never claimed to
    be optimal or leakage-free.
    """

    status: str
    strategy: str | None
    rationale: list[str]
    assumptions: list[str]
    limitations: list[str]
    missing_information: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"recommended", "needs_user_input"}:
            raise ValueError(f"Invalid split recommendation status: {self.status!r}")
        if self.status == "needs_user_input" and not self.missing_information:
            raise ValueError("needs_user_input requires missing_information")
        if self.status == "recommended" and not self.strategy:
            raise ValueError("A recommendation requires a strategy")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "status": self.status,
                "strategy": self.strategy,
                "rationale": self.rationale,
                "assumptions": self.assumptions,
                "limitations": self.limitations,
                "missing_information": self.missing_information,
                "warnings": self.warnings,
                "parameters": self.parameters,
                "evidence": self.evidence,
            }
        )


def _sort_key(finding: Finding) -> tuple[int, str, str]:
    return (finding.severity.rank, finding.rule_id, finding.key)


@dataclass
class AuditResult:
    """Everything an audit produced. Serialise with ``to_dict()``."""

    mode: str
    inputs: list[dict[str, Any]]
    partitions: dict[str, dict[str, int]]
    config: dict[str, Any]
    task_type: str | None
    feature_columns: list[str]
    checks: list[CheckRecord]
    findings: list[Finding]
    warnings: list[str]
    limitations: list[str]
    split_recommendation: SplitRecommendation | None = None
    generated_at: str | None = None
    tool_version: str = __version__

    def __post_init__(self) -> None:
        self.findings = sorted(self.findings, key=_sort_key)
        ids = [f.finding_id for f in self.findings]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate finding ids: {dupes}")

    @property
    def counts_by_severity(self) -> dict[str, int]:
        counts = Counter(f.severity.value for f in self.findings)
        return {s.value: counts.get(s.value, 0) for s in Severity}

    @property
    def counts_by_category(self) -> dict[str, int]:
        counts = Counter(f.category.value for f in self.findings)
        return {c.value: counts.get(c.value, 0) for c in Category}

    def checks_with_status(self, status: CheckStatus) -> list[CheckRecord]:
        return [c for c in self.checks if c.status is status]

    def has_severity_at_least(self, severity: Severity | str) -> bool:
        threshold = Severity(severity).rank
        return any(f.severity.rank <= threshold for f in self.findings)

    def summary(self) -> dict[str, Any]:
        return {
            "total_findings": len(self.findings),
            "by_severity": self.counts_by_severity,
            "by_category": self.counts_by_category,
            "checks_run": len(self.checks_with_status(CheckStatus.RUN)),
            "checks_skipped": len(self.checks_with_status(CheckStatus.SKIPPED)),
            "checks_failed": len(self.checks_with_status(CheckStatus.FAILED)),
        }

    def to_dict(self, include_generated_at: bool = True) -> dict[str, Any]:
        """Return the stable JSON report structure.

        Pass ``include_generated_at=False`` for byte-identical reports across
        runs on the same inputs and configuration.
        """
        report: dict[str, Any] = {
            "tool": {"name": "splitguard", "version": self.tool_version},
            "report_format_version": REPORT_FORMAT_VERSION,
        }
        if include_generated_at:
            report["generated_at"] = self.generated_at
        report.update(
            {
                "mode": self.mode,
                "inputs": to_jsonable(self.inputs),
                "partitions": to_jsonable(self.partitions),
                "configuration": to_jsonable(self.config),
                "task_type": self.task_type,
                "feature_columns": list(self.feature_columns),
                "summary": self.summary(),
                "checks": [c.to_dict() for c in self.checks],
                "findings": [f.to_dict() for f in self.findings],
                "split_recommendation": (
                    self.split_recommendation.to_dict() if self.split_recommendation else None
                ),
                "warnings": list(self.warnings),
                "limitations": list(self.limitations),
            }
        )
        return report
