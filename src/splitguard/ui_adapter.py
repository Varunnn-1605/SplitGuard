"""Thin, Streamlit-free adapter between the UI and the audit engine.

Everything the UI shows comes from the engine's own objects
(:class:`~splitguard.schemas.AuditResult`, its ``to_dict()`` report and
:class:`~splitguard.schemas.SplitRecommendation`). This module only:

* loads uploaded bytes through the engine's validated loader,
* turns form selections into an :class:`~splitguard.config.AuditConfig`,
* calls the public audit / split functions,
* filters and groups results for display, and
* renders a concise Markdown summary from the JSON report structure.

It never re-implements detection logic and imports nothing from Streamlit,
so it can be unit-tested without the UI dependency.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

import pandas as pd

from splitguard.audit import audit_split_column, audit_train_test, inspect_unsplit
from splitguard.config import AuditConfig, ConfigError, Limits
from splitguard.hashing import canonical_series
from splitguard.reporting import report_to_json
from splitguard.schemas import (
    AuditResult,
    Category,
    CheckStatus,
    Confidence,
    Finding,
    Severity,
    SplitRecommendation,
    to_jsonable,
)
from splitguard.splitting import SplitConfirmationError, create_split
from splitguard.validation import InputValidationError, LoadedTable, load_table

MODES: dict[str, str] = {
    "train_test": "Separate train and test files",
    "split_column": "One file with a split column",
    "unsplit": "One file that has not been split yet",
}
MODE_ROLES: dict[str, tuple[str, ...]] = {
    "train_test": ("train", "test"),
    "split_column": ("data",),
    "unsplit": ("data",),
}

SCENARIO_HELP: dict[str, str] = {
    "unknown": "Not declared - findings that depend on the use case get reduced confidence.",
    "independent": "New rows from the same population; rows are independent of each other.",
    "new_entities": "The model must work for customers/patients/devices it has never seen.",
    "forecasting": "The model predicts future periods from past data.",
    "new_entities_forecasting": "Both: unseen entities and future periods.",
}
SCENARIO_LABELS: dict[str, str] = {
    "unknown": "not declared",
    "independent": "independent rows",
    "new_entities": "new entities",
    "forecasting": "forecasting",
    "new_entities_forecasting": "new entities + forecasting",
}

STRATEGY_HELP: dict[str, str] = {
    "random": "Rows are assigned to train or test at random (seeded). Suitable when rows are "
    "independent observations.",
    "stratified_random": "Random, but keeps each class's share the same in train and test. "
    "Suitable for independent classification rows.",
    "group": "All rows of an entity go to the same side, so test entities are unseen in "
    "training. Needed when the model must handle new entities.",
    "time": "Train on earlier rows, test on later rows. Needed when the model forecasts "
    "future periods.",
    "group_and_time": "Test on later periods *and* unseen entities. Must be designed "
    "deliberately; SplitGuard does not create it automatically.",
}

EVIDENCE_LABELS: dict[Confidence, str] = {
    Confidence.HIGH: "Strong evidence",
    Confidence.MEDIUM: "Moderate evidence",
    Confidence.LOW: "Review prompt",
}

NO_PROOF_NOTE = (
    "No findings does not prove the data is free of leakage. Only the listed checks were "
    "run, and each looks for specific, explainable patterns."
)
OVERLAP_NOTE = (
    "Several findings can describe the same rows from different angles (for example "
    "duplicate and identifier findings). They are not independent confirmations."
)
PRIVACY_NOTE = (
    "Files are processed locally by this app and held in its memory while you use it. "
    "Nothing is uploaded to any external service, and no analytics are sent by the "
    "splitguard-ui launcher."
)

# --------------------------------------------------------------------- loading


@dataclass(frozen=True)
class LoadedFile:
    """A validated CSV: the original bytes (passed to the engine) plus a parsed frame
    used only for column choices and previews."""

    name: str
    data: bytes
    table: LoadedTable

    @property
    def frame(self) -> pd.DataFrame:
        return self.table.frame

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.table.frame.columns]


def safe_name(name: str | None, fallback: str) -> str:
    """Browser-supplied names are reduced to a bare file name."""
    base = PurePath(str(name or "")).name.strip()
    return base[:120] or fallback


def load_file(name: str | None, data: bytes, limits: Limits | None = None) -> LoadedFile:
    """Validate uploaded bytes with the engine's loader (size, rows, header, encoding).

    Raises :class:`InputValidationError` with actionable, row-free messages.
    """
    label = safe_name(name, "uploaded.csv")
    table = load_table(bytes(data), name=label, limits=limits or Limits())
    return LoadedFile(label, bytes(data), table)


def safe_preview(frame: pd.DataFrame, n: int = 5, max_chars: int = 60) -> pd.DataFrame:
    """First ``n`` rows as truncated text (a copy; the source frame is untouched)."""
    head = frame.head(n).copy()
    for column in head.columns:
        head[column] = [
            ""
            if pd.isna(v)
            else (s if len(s := str(v)) <= max_chars else s[: max_chars - 3] + "...")
            for v in head[column].to_numpy(dtype=object)
        ]
    return head


def split_labels(frame: pd.DataFrame, column: str, max_labels: int = 20) -> list[str] | None:
    """Distinct labels of a candidate split column, or None if there are too many."""
    labels = canonical_series(frame[column]).dropna().unique()
    if len(labels) > max_labels:
        return None
    return sorted(str(v) for v in labels)


def default_split_label(labels: Sequence[str], wanted: str) -> str | None:
    """Pre-select a label only on an exact (case-insensitive) match; never guess."""
    for label in labels:
        if label.strip().lower() == wanted:
            return label
    return None


# ---------------------------------------------------------------- configuration


@dataclass
class FormState:
    """Everything the user chose in the form. ``None``/empty means 'not set'."""

    mode: str
    target: str | None = None
    task_type: str = "auto"
    scenario: str = "unknown"
    entity_id: str | None = None
    prediction_timestamp: str | None = None
    outcome_timestamp: str | None = None
    feature_availability: dict[str, str] = field(default_factory=dict)
    exclude_columns: list[str] = field(default_factory=list)
    min_lead_time: str | None = None
    expect_target_in_test: bool = True
    test_target_diagnostics: bool = False
    split_column: str | None = None
    train_value: str | None = None
    test_value: str | None = None
    test_size: float = 0.2
    seed: int = 42

    def fingerprint(self) -> str:
        return json.dumps(to_jsonable(self.__dict__), sort_keys=True)


def build_config(state: FormState) -> AuditConfig:
    """Map form state onto the real AuditConfig schema (validated by AuditConfig)."""
    lead = (state.min_lead_time or "").strip() or None
    return AuditConfig(
        target=state.target or None,
        entity_id=state.entity_id or None,
        prediction_timestamp=state.prediction_timestamp or None,
        outcome_timestamp=state.outcome_timestamp or None,
        feature_availability={k: v for k, v in state.feature_availability.items() if k and v},
        exclude_columns=tuple(state.exclude_columns),
        scenario=state.scenario,
        task_type=state.task_type,
        min_lead_time=lead,
        expect_target_in_test=state.expect_target_in_test,
        test_target_diagnostics=state.test_target_diagnostics,
    )


def missing_inputs(state: FormState, files: dict[str, LoadedFile]) -> list[str]:
    """Things that must be provided before an audit can run (no guessing)."""
    missing = [f"the {role} file" for role in MODE_ROLES[state.mode] if role not in files]
    if state.mode == "split_column":
        if not state.split_column:
            missing.append("the split column")
        elif not state.train_value or not state.test_value:
            missing.append("which split labels mean train and test")
        elif state.train_value == state.test_value:
            missing.append("different labels for train and test")
    return missing


# ------------------------------------------------------------------------- running


def run_audit(state: FormState, files: dict[str, LoadedFile]) -> AuditResult:
    """Call the public engine API for the selected mode with the original bytes."""
    config = build_config(state)
    if state.mode == "train_test":
        train, test = files["train"], files["test"]
        return audit_train_test(
            train.data, test.data, config, train_name=train.name, test_name=test.name
        )
    data = files["data"]
    if state.mode == "split_column":
        return audit_split_column(
            data.data,
            state.split_column,
            state.train_value,
            state.test_value,
            config,
            name=data.name,
        )
    if state.mode == "unsplit":
        return inspect_unsplit(
            data.data, config, name=data.name, test_size=state.test_size, seed=state.seed
        )
    raise ConfigError(f"Unknown mode {state.mode!r}")


@dataclass(frozen=True)
class UIError:
    title: str
    messages: list[str]
    hints: list[str] = field(default_factory=list)


def describe_error(exc: BaseException) -> UIError:
    """User-facing description. Unexpected errors expose only their type, because
    exception messages from parsing libraries can contain data values."""
    if isinstance(exc, InputValidationError):
        return UIError(
            "The input could not be used",
            [f"[{i.code}] {i.message}" for i in exc.issues],
            [i.hint for i in exc.issues if i.hint],
        )
    if isinstance(exc, ConfigError):
        return UIError("The settings are not valid", [str(exc)])
    if isinstance(exc, SplitConfirmationError):
        return UIError("The split was not created", [str(exc)])
    if isinstance(exc, NotImplementedError):
        return UIError("Not available", [str(exc)])
    return UIError(
        "Something went wrong inside SplitGuard",
        [f"Internal error ({type(exc).__name__}). No details are shown to avoid exposing data."],
        ["Try the CLI on the same files, or report the error type."],
    )


# ------------------------------------------------------------------------- results


def filter_findings(
    findings: Iterable[Finding],
    severities: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    confidences: Iterable[str] | None = None,
) -> list[Finding]:
    """Keep findings matching every non-None filter (empty filter = show none)."""
    sev = None if severities is None else set(severities)
    cat = None if categories is None else set(categories)
    conf = None if confidences is None else set(confidences)
    return [
        f
        for f in findings
        if (sev is None or f.severity.value in sev)
        and (cat is None or f.category.value in cat)
        and (conf is None or f.confidence.value in conf)
    ]


def evidence_label(finding: Finding) -> str:
    return EVIDENCE_LABELS[finding.confidence]


def category_label(value: str) -> str:
    return Category(value).value.replace("_", " ").capitalize()


_TIE_WORDS = ("exactly equal", "shared boundary")
_COVERAGE_WORDS = (
    "could not be compared",
    "not checked",
    "not assessed",
    "too few non-missing",
    "uninformative",
    "not audited",
)
_METADATA_PREFIXES = ("No ", "Evaluation scenario not declared")
WARNING_GROUP_TITLES: dict[str, str] = {
    "coverage": "Partial coverage - some rows or checks were not assessed",
    "timestamp_ties": "Timestamp ties and boundaries - ambiguous, not flagged",
    "missing_metadata": "Missing metadata - checks that could not run",
    "other": "Other notes",
}


def group_warnings(warnings: Iterable[str]) -> dict[str, list[str]]:
    """Group engine warnings for display; every warning lands in exactly one group."""
    groups: dict[str, list[str]] = {key: [] for key in WARNING_GROUP_TITLES}
    for warning in warnings:
        if any(w in warning for w in _TIE_WORDS):
            key = "timestamp_ties"
        elif warning.startswith(_METADATA_PREFIXES):
            key = "missing_metadata"
        elif any(w in warning for w in _COVERAGE_WORDS):
            key = "coverage"
        else:
            key = "other"
        groups[key].append(warning)
    return groups


def checks_by_status(result: AuditResult) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {s.value: [] for s in CheckStatus}
    for c in result.checks:
        out[c.status.value].append({"check": c.check_id, "name": c.name, "reason": c.reason or ""})
    return out


def skipped_by_reason(result: AuditResult) -> list[tuple[str, str, list[str]]]:
    """Checks that did not run, grouped by (status, reason), in report order.

    Many checks share one reason (for example "no split exists"), so grouping keeps
    the list short and readable on narrow screens.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    for c in result.checks:
        if c.status is not CheckStatus.RUN:
            groups.setdefault((c.status.value, c.reason or ""), []).append(c.check_id)
    return [(status, reason, ids) for (status, reason), ids in groups.items()]


def partial_note_count(result: AuditResult) -> int:
    """Warnings saying rows/checks were only partly assessed or timestamps were tied."""
    groups = group_warnings(result.warnings)
    return len(groups["coverage"]) + len(groups["timestamp_ties"])


def headline(result: AuditResult) -> tuple[str, str]:
    """(level, message) for the status banner. Levels: error, warning, info.

    A result without findings is never shown as "success": no findings is not
    evidence that the data is free of leakage.
    """
    summary = result.summary()
    sev = summary["by_severity"]
    partial = partial_note_count(result)
    partial_text = (
        f" {partial} note(s) describe rows or checks that were only partly assessed or "
        "ambiguous; see 'What was not fully assessed'."
        if partial
        else ""
    )
    if summary["checks_failed"]:
        return "error", (
            f"{summary['checks_failed']} check(s) failed to run, so this audit is incomplete."
        )
    serious = sev["critical"] + sev["high"]
    if serious:
        return "warning", (
            f"{summary['total_findings']} finding(s) to review, including {sev['critical']} "
            f"critical and {sev['high']} high severity. Severity is the impact if the issue is "
            "real; check each finding's confidence." + partial_text
        )
    if summary["total_findings"]:
        return "warning" if partial else "info", (
            f"{summary['total_findings']} lower-severity finding(s) to review." + partial_text
        )
    return "warning" if partial else "info", (
        "No findings from the checks that ran." + partial_text + " " + NO_PROOF_NOTE
    )


def finding_view(finding: Finding) -> dict[str, Any]:
    """The fields of one finding exactly as they appear in the JSON report."""
    return finding.to_dict()


# ------------------------------------------------------------------------- reports


def json_report_bytes(result: AuditResult, include_generated_at: bool = True) -> bytes:
    """The engine's own JSON report, unchanged."""
    return report_to_json(result, include_generated_at=include_generated_at).encode("utf-8")


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


def md_escape(text: Any) -> str:
    """Escape Markdown so column names or labels from the data cannot inject markup."""
    return _MD_SPECIAL.sub(r"\\\1", str(text)).replace("\n", " ")


def markdown_report(report: dict[str, Any]) -> str:
    """Concise human-readable report built only from the JSON report structure."""
    e = md_escape
    summary = report["summary"]
    lines = [
        "# SplitGuard audit report",
        "",
        f"- Tool: splitguard {e(report['tool']['version'])} (report format "
        f"{e(report['report_format_version'])})",
    ]
    if report.get("generated_at"):
        lines.append(f"- Generated: {e(report['generated_at'])}")
    lines += [
        f"- Mode: {e(report['mode'])}; task type: {e(report['task_type'] or 'n/a')}",
        f"- Scenario: {e(report['configuration']['scenario'])}",
        "",
        "## Inputs",
        "",
    ]
    for item in report["inputs"]:
        lines.append(
            f"- {e(item['role'])}: {e(item['name'])} - {item['n_rows']} rows, "
            f"{item['n_columns']} columns, sha256 `{item['sha256'][:16]}...`"
        )
    sev = summary["by_severity"]
    lines += [
        "",
        "## Summary",
        "",
        f"- Findings: {summary['total_findings']} (critical {sev['critical']}, high {sev['high']}, "
        f"medium {sev['medium']}, low {sev['low']}, info {sev['info']})",
        f"- Checks: {summary['checks_run']} run, {summary['checks_skipped']} skipped, "
        f"{summary['checks_failed']} failed",
        "",
        f"> {NO_PROOF_NOTE}",
        f"> {OVERLAP_NOTE}",
        "",
    ]
    if report["warnings"]:
        lines += ["## Warnings and coverage", ""]
        lines += [f"- {e(w)}" for w in report["warnings"]]
        lines.append("")
    skipped = [c for c in report["checks"] if c["status"] != "run"]
    if skipped:
        lines += ["## Checks not run", ""]
        lines += [f"- {c['check_id']} ({c['status']}): {e(c['reason'])}" for c in skipped]
        lines.append("")
    lines += ["## Findings", ""]
    if not report["findings"]:
        lines += ["No findings from the checks that ran.", ""]
    for f in report["findings"]:
        label = EVIDENCE_LABELS[Confidence(f["confidence"])]
        lines += [
            f"### {e(f['finding_id'])}: {e(f['title'])}",
            "",
            f"- Severity: **{f['severity']}** | Confidence: **{f['confidence']}** ({label})",
            f"- Category: {e(f['category'])}",
            f"- Affected columns: {', '.join(e(c) for c in f['affected_columns']) or 'none'}",
            "",
            e(f["description"]),
            "",
            f"**Recommended action:** {e(f['recommended_action'])}",
            "",
            "**Why this is not proof:** " + " ".join(e(x) for x in f["limitations"]),
            "",
            "Evidence:",
            "",
            "```json",
            json.dumps(f["evidence"], indent=1, ensure_ascii=False).replace("```", "'''"),
            "```",
            "",
        ]
    rec = report.get("split_recommendation")
    if rec:
        lines += ["## Split recommendation", "", f"- Status: {e(rec['status'])}"]
        if rec["strategy"]:
            lines.append(f"- Strategy: {e(rec['strategy'])}")
        lines += [f"- Missing: {e(m)}" for m in rec["missing_information"]]
        lines += [f"- {e(r)}" for r in rec["rationale"]]
        lines.append("")
    lines += ["## Limitations", ""] + [f"- {e(x)}" for x in report["limitations"]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- split


@dataclass(frozen=True)
class SplitDownload:
    train_csv: bytes
    test_csv: bytes
    report_json: bytes
    report: dict[str, Any]


def create_split_files(
    data: LoadedFile,
    recommendation: SplitRecommendation,
    *,
    confirmed_strategy: str,
    state: FormState,
) -> SplitDownload:
    """Create a split only through the engine's confirmed ``create_split``."""
    config = build_config(state)
    result = create_split(
        data.frame.copy(deep=True),
        recommendation,
        confirmed_strategy=confirmed_strategy.strip(),
        config=config,
        test_size=state.test_size,
        seed=state.seed,
    )
    payload = {"recommendation": recommendation.to_dict(), "split": result.report}

    def to_csv(df: pd.DataFrame) -> bytes:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    return SplitDownload(
        train_csv=to_csv(result.train),
        test_csv=to_csv(result.test),
        report_json=json.dumps(to_jsonable(payload), indent=2, allow_nan=False).encode("utf-8"),
        report=result.report,
    )


SEVERITIES = [s.value for s in Severity]
CONFIDENCES = [c.value for c in Confidence]
CATEGORIES = [c.value for c in Category]
