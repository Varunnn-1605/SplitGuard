"""Public audit API. The CLI is a thin wrapper around these functions.

Three input modes:

* :func:`audit_train_test` - separate train and test tables (Mode A).
* :func:`audit_split_column` - one table with an explicit split column (Mode B).
* :func:`inspect_unsplit` - one unsplit table (Mode C): within-dataset checks
  plus a split-strategy recommendation. Train/test contamination is *not*
  audited, and the data is never split during an audit.

Each source may be a CSV path, CSV bytes, or a pandas DataFrame.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from splitguard.config import AuditConfig, TaskType
from splitguard.context import RULES, AuditContext, DetectorResult
from splitguard.detectors import duplicates, identifiers, schema, target_leakage, temporal
from splitguard.schemas import (
    AuditResult,
    CheckRecord,
    CheckStatus,
    Confidence,
    Finding,
    Severity,
    SplitRecommendation,
)
from splitguard.splitting.recommendations import DEFAULT_SEED, DEFAULT_TEST_SIZE, recommend_split
from splitguard.validation import (
    TableSource,
    ValidationIssue,
    check_missing_rate,
    feature_columns,
    load_table,
    parse_timestamps,
    raise_if_issues,
    require_columns,
    resolve_task_type,
    split_frame,
    validate_target,
)

# Detector order is fixed so reports are deterministic.
_DETECTORS: list[tuple[str, Callable[[AuditContext], DetectorResult], list[str]]] = [
    ("schema", schema.run, [r for r in RULES if r.startswith("SCH")]),
    ("identifiers", identifiers.run, [r for r in RULES if r.startswith("ID")]),
    ("duplicates", duplicates.run, [r for r in RULES if r.startswith("DUP")]),
    ("target_leakage", target_leakage.run, [r for r in RULES if r.startswith("TGT")]),
    ("temporal", temporal.run, [r for r in RULES if r.startswith("TMP")]),
]

GLOBAL_LIMITATIONS = [
    "No findings does not mean no leakage: only the listed checks were run, and each "
    "detects specific, explainable patterns.",
    "Only exact duplicates are detected; near-duplicates (fuzzy matches, re-scaled or "
    "re-formatted copies) are not.",
    "Findings can describe the same underlying problem from different angles (for "
    "example DUP-002, DUP-004 and ID-003); they are not independent confirmations.",
    "Statistical associations are review signals, not proof of leakage.",
    "Leakage spread across combinations of features is not detected (no model is trained).",
]


def audit_train_test(
    train: TableSource,
    test: TableSource,
    config: AuditConfig | None = None,
    *,
    train_name: str | None = None,
    test_name: str | None = None,
) -> AuditResult:
    """Audit separate train and test tables (Mode A)."""
    config = config or AuditConfig()
    train_t = load_table(
        train, name=train_name or _default_name(train, "train"), limits=config.limits
    )
    test_t = load_table(test, name=test_name or _default_name(test, "test"), limits=config.limits)
    inputs = [{"role": "train", **train_t.metadata}, {"role": "test", **test_t.metadata}]
    return _run(
        "train_test", {"train": train_t.frame, "test": test_t.frame}, inputs, config, warnings=[]
    )


def audit_split_column(
    data: TableSource,
    split_column: str,
    train_value: Any = "train",
    test_value: Any = "test",
    config: AuditConfig | None = None,
    *,
    name: str | None = None,
) -> AuditResult:
    """Audit one table whose ``split_column`` labels each row train or test (Mode B)."""
    config = config or AuditConfig()
    if split_column in {config.target, *config.metadata_columns()}:
        raise_if_issues(
            [ValidationIssue("ROLE_CONFLICT", f"Split column {split_column!r} has another role")]
        )
    table = load_table(data, name=name or _default_name(data, "data"), limits=config.limits)
    label = table.metadata["name"]
    train, test = split_frame(table.frame, split_column, train_value, test_value, label)
    inputs = [
        {
            "role": "combined",
            **table.metadata,
            "split_column": split_column,
            "train_value": str(train_value),
            "test_value": str(test_value),
        }
    ]
    return _run("split_column", {"train": train, "test": test}, inputs, config, warnings=[])


def inspect_unsplit(
    data: TableSource,
    config: AuditConfig | None = None,
    *,
    name: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> AuditResult:
    """Run within-dataset checks and recommend a split strategy (Mode C).

    The dataset is not split. Cross-partition checks are reported as skipped.
    """
    config = config or AuditConfig()
    table = load_table(data, name=name or _default_name(data, "data"), limits=config.limits)
    inputs = [{"role": "unsplit", **table.metadata}]
    warnings = [
        "No train/test split exists: train/test contamination was not audited. "
        "Use `audit` once a split has been created."
    ]
    return _run(
        "unsplit",
        {"unsplit": table.frame},
        inputs,
        config,
        warnings=warnings,
        recommend={"test_size": test_size, "seed": seed},
    )


def _default_name(source: TableSource, fallback: str) -> str:
    if isinstance(source, (pd.DataFrame, bytes, bytearray)):
        return fallback
    return Path(source).name


# --------------------------------------------------------------------- validation


def _validate(
    partitions: dict[str, pd.DataFrame], config: AuditConfig
) -> tuple[dict[str, dict[str, pd.Series]], list[str], TaskType | None]:
    th = config.thresholds
    primary_name = "train" if "test" in partitions else "unsplit"
    primary = partitions[primary_name]
    issues: list[ValidationIssue] = []

    issues += require_columns(
        primary,
        {
            "target": config.target,
            **{f"availability feature {k!r}": k for k in config.feature_availability},
        },
        primary_name,
    )
    for name, df in partitions.items():
        roles = {
            "entity_id": config.entity_id,
            "prediction_timestamp": config.prediction_timestamp,
            "outcome_timestamp": config.outcome_timestamp,
        }
        if name == primary_name:
            roles.update(
                {f"availability for {k!r}": v for k, v in config.feature_availability.items()}
            )
            roles.update({f"excluded column {c!r}": c for c in config.exclude_columns})
        else:
            # Availability columns are only required where their feature exists.
            roles.update(
                {
                    f"availability for {k!r}": v
                    for k, v in config.feature_availability.items()
                    if k in df.columns
                }
            )
        issues += require_columns(df, roles, name)
    raise_if_issues(issues)

    features = feature_columns(primary.columns, config)
    if not features:
        issues.append(
            ValidationIssue(
                "NO_FEATURE_COLUMNS",
                f"{primary_name}: no feature columns remain after excluding the target and metadata",
            )
        )
    if "test" in partitions:
        test_cols = set(partitions["test"].columns)
        if features and not (set(features) & test_cols):
            issues.append(
                ValidationIssue(
                    "INCOMPATIBLE_SCHEMA",
                    "train and test share no feature columns",
                    "Check that both files come from the same pipeline.",
                )
            )

    task_type = None
    if config.target:
        task_type = resolve_task_type(primary[config.target], config.task_type)
        issues += validate_target(primary, config.target, primary_name, task_type, th)
    if config.entity_id:
        for name, df in partitions.items():
            issues += check_missing_rate(
                df[config.entity_id], "entity_id", name, th.max_missing_rate_required
            )

    timestamps: dict[str, dict[str, pd.Series]] = {}
    for name, df in partitions.items():
        timestamps[name] = {}
        for column in config.timestamp_columns():
            if column not in df.columns:
                continue
            parsed, ts_issues = parse_timestamps(
                df[column], f"timestamp ({column})", name, th.max_missing_rate_required
            )
            issues += ts_issues
            timestamps[name][column] = parsed
    raise_if_issues(issues)
    return timestamps, features, task_type


# ----------------------------------------------------------------------------- run


def _run(
    mode: str,
    partitions: dict[str, pd.DataFrame],
    inputs: list[dict[str, Any]],
    config: AuditConfig,
    *,
    warnings: list[str],
    recommend: dict[str, Any] | None = None,
) -> AuditResult:
    timestamps, features, task_type = _validate(partitions, config)
    ctx = AuditContext(
        mode=mode,
        config=config,
        partitions=partitions,
        timestamps=timestamps,
        feature_columns=features,
        task_type=task_type,
    )
    primary = ctx.primary
    ctx.target_usable = bool(
        config.target and ctx.canonical(ctx.primary_name, config.target).nunique(dropna=True) >= 2
    )
    findings: list[Finding] = []
    checks: list[CheckRecord] = []
    detector_warnings: list[str] = []
    for name, detector, rule_ids in _DETECTORS:
        try:
            out = detector(ctx)
        except Exception as exc:  # noqa: BLE001 - isolate detector failures
            # Only the exception type is recorded: messages can echo data values.
            reason = (
                f"Detector {name!r} failed with {type(exc).__name__}; its checks did not complete."
            )
            checks += [CheckRecord(r, RULES[r][0], CheckStatus.FAILED, reason) for r in rule_ids]
            continue
        findings += out.findings
        checks += out.checks
        detector_warnings += out.warnings

    recommendation = None
    if recommend is not None:
        recommendation = recommend_split(primary, config, **recommend)
        findings.append(_recommendation_finding(recommendation))
        checks.append(CheckRecord("SPL-001", RULES["SPL-001"][0], CheckStatus.RUN))

    return AuditResult(
        mode=mode,
        inputs=inputs,
        partitions={
            k: {"n_rows": len(v), "n_columns": len(v.columns)} for k, v in partitions.items()
        },
        config=config.to_dict(),
        task_type=task_type.value if task_type else None,
        feature_columns=features,
        checks=checks,
        findings=findings,
        warnings=warnings + _context_warnings(ctx) + detector_warnings,
        limitations=list(GLOBAL_LIMITATIONS),
        split_recommendation=recommendation,
        generated_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    )


def _context_warnings(ctx: AuditContext) -> list[str]:
    cfg = ctx.config
    warnings = []
    if not cfg.target:
        warnings.append("No target configured: target leakage checks were not run.")
    if not cfg.prediction_timestamp:
        warnings.append(
            "No prediction_timestamp configured: temporal leakage was not assessed. "
            "This is not evidence that the data is temporally safe."
        )
    elif not cfg.feature_availability:
        warnings.append(
            "No feature_availability metadata: whether individual features are known at "
            "prediction time was not assessed."
        )
    if cfg.scenario.value == "unknown":
        warnings.append(
            "Evaluation scenario not declared: entity-overlap and time-overlap findings are "
            "reported at reduced confidence."
        )
    if ctx.has_split and cfg.entity_id is None:
        warnings.append(
            "No entity_id configured: entity overlap across partitions was not checked."
        )
    return warnings


def _recommendation_finding(rec: SplitRecommendation) -> Finding:
    if rec.status == "recommended":
        title = f"Recommended split strategy: {rec.strategy}"
        description = " ".join(rec.rationale)
        action = (
            f"If this matches your deployment scenario, create the split explicitly "
            f"(CLI: splitguard split --confirm-strategy {rec.strategy}) and audit the result."
        )
        confidence = Confidence.MEDIUM
    else:
        title = "Split strategy needs more information"
        description = "Missing: " + " | ".join(rec.missing_information)
        action = "Provide the missing information and re-run inspect."
        confidence = Confidence.LOW
    return Finding(
        rule_id="SPL-001",
        key="dataset",
        category=RULES["SPL-001"][1],
        severity=Severity.INFO,
        confidence=confidence,
        title=title,
        description=description,
        evidence={"status": rec.status, "strategy": rec.strategy, "parameters": rec.parameters},
        affected_columns=[],
        recommended_action=action,
        limitations=list(rec.limitations),
    )
