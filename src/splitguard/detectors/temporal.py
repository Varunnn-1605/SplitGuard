"""Detector C: temporal leakage risks backed by explicit timestamps.

Every rule compares configured timestamp columns; none infers timing from
column names except TMP-005, which is a low-confidence review prompt. When
the metadata a rule needs is missing, the rule is skipped and the report says
temporal safety was not assessed.

All timestamps are parsed to UTC; naive values are treated as UTC.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from splitguard.context import AuditContext, DetectorResult
from splitguard.schemas import Confidence, Severity
from splitguard.validation import name_tokens

_ASSUMPTION = (
    "All timestamps share one clock; naive timestamps were interpreted as UTC. "
    "Rows with a missing timestamp on either side are not compared."
)
BOUNDARY_RULE = (
    "Strict comparison: a value is flagged only if its timestamp is strictly later. Equal "
    "timestamps are counted in n_equal but not flagged, because the data cannot show whether "
    "a value stamped at exactly that instant was usable."
)
_TEMPORAL_WORDS = {
    "future",
    "next",
    "after",
    "outcome",
    "final",
    "resolved",
    "settled",
    "closed",
    "followup",
    "later",
    "result",
    "results",
    "posthoc",
}


def run(ctx: AuditContext) -> DetectorResult:
    result = DetectorResult()
    _availability_vs_prediction(ctx, result)
    _availability_vs_outcome(ctx, result)
    _outcome_vs_prediction(ctx, result)
    _split_overlap(ctx, result)
    _name_clues(ctx, result)
    return result


def _lag_stats(lags: pd.Series) -> dict[str, Any]:
    if lags.empty:
        return {"max_lag": None, "median_lag": None}
    return {"max_lag": lags.max(), "median_lag": lags.median()}


def _compare(
    ctx: AuditContext, left: str, right: str, strict: bool = True
) -> tuple[dict[str, Any], int]:
    """Count rows where ``left > right`` (or ``>=``) per partition."""
    per_partition: dict[str, Any] = {}
    total = 0
    for name, stamps in ctx.timestamps.items():
        if left not in stamps or right not in stamps:
            per_partition[name] = {"skipped": "column not present in this partition"}
            continue
        a, b = stamps[left], stamps[right]
        valid = a.notna() & b.notna()
        later = valid & ((a > b) if strict else (a >= b))
        n_valid, n_later = int(valid.sum()), int(later.sum())
        per_partition[name] = {
            "n_rows": len(a),
            "n_compared": n_valid,
            "n_not_compared_missing": int((~valid).sum()),
            "n_violations": n_later,
            "violation_rate": n_later / n_valid if n_valid else None,
            "n_equal": int((valid & (a == b)).sum()),
            **_lag_stats((a - b)[later]),
        }
        total += n_later
    return per_partition, total


def _coverage_warnings(
    result: DetectorResult, rule_id: str, feature: str, column: str, other: str, per_partition: dict
) -> None:
    """State what a comparison could not establish: missing timestamps and exact ties."""
    parts = [p for p in per_partition.values() if "skipped" not in p]
    n_missing = sum(p["n_not_compared_missing"] for p in parts)
    n_equal = sum(p["n_equal"] for p in parts)
    if n_missing:
        result.warnings.append(
            f"{rule_id}: {n_missing} row(s) for {feature!r} could not be compared because "
            f"{column!r} or {other!r} is missing; their timing was not assessed."
        )
    if n_equal:
        result.warnings.append(
            f"{rule_id}: {n_equal} row(s) have {column!r} exactly equal to {other!r}. They are "
            f"not flagged, but SplitGuard cannot tell whether {feature!r} was usable at that "
            "instant; review these rows."
        )


def _availability_vs_prediction(ctx: AuditContext, result: DetectorResult) -> None:
    cfg = ctx.config
    if not cfg.feature_availability:
        result.skipped(
            "TMP-001",
            "No feature_availability metadata configured; feature timing was not assessed.",
        )
        return
    if not cfg.prediction_timestamp:
        result.skipped(
            "TMP-001", "No prediction_timestamp configured; availability cannot be compared."
        )
        return
    details = {}
    for feature, column in cfg.feature_availability.items():
        per_partition, total = _compare(ctx, column, cfg.prediction_timestamp)
        details[feature] = {
            "availability_column": column,
            "boundary_rule": BOUNDARY_RULE,
            "partitions": per_partition,
        }
        _coverage_warnings(
            result, "TMP-001", feature, column, cfg.prediction_timestamp, per_partition
        )
        if not total:
            continue
        result.add(
            "TMP-001",
            feature,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            title=f"{feature!r} becomes available after prediction time on {total} row(s)",
            description=(
                f"For {total} rows the availability timestamp {column!r} is later than the "
                f"prediction timestamp {cfg.prediction_timestamp!r}. At prediction time the "
                "model could not have known this value."
            ),
            evidence={
                "comparison_rule": f"{column} > {cfg.prediction_timestamp}",
                "assumptions": _ASSUMPTION,
                "partitions": per_partition,
            },
            affected_columns=[feature, column, cfg.prediction_timestamp],
            recommended_action=(
                f"Use only values of {feature!r} recorded at or before prediction time "
                "(an as-of join), or remove the feature."
            ),
            limitations=[
                "The finding is only as reliable as the availability metadata supplied.",
                "If the availability column records something other than when the value was "
                "known (for example a last-modified time), the comparison may be wrong.",
            ],
        )
    result.ran("TMP-001", details)


def _availability_vs_outcome(ctx: AuditContext, result: DetectorResult) -> None:
    cfg = ctx.config
    if not cfg.feature_availability or not cfg.outcome_timestamp:
        result.skipped(
            "TMP-002", "Requires both feature_availability metadata and an outcome_timestamp."
        )
        return
    details = {}
    for feature, column in cfg.feature_availability.items():
        per_partition, total = _compare(ctx, column, cfg.outcome_timestamp)
        details[feature] = {
            "availability_column": column,
            "boundary_rule": BOUNDARY_RULE,
            "partitions": per_partition,
        }
        _coverage_warnings(result, "TMP-002", feature, column, cfg.outcome_timestamp, per_partition)
        if not total:
            continue
        result.add(
            "TMP-002",
            feature,
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            title=f"{feature!r} is recorded after the outcome on {total} row(s)",
            description=(
                f"For {total} rows {column!r} is later than the outcome timestamp "
                f"{cfg.outcome_timestamp!r}: the value was created after the label was known "
                "and may encode it."
            ),
            evidence={
                "comparison_rule": f"{column} > {cfg.outcome_timestamp}",
                "assumptions": _ASSUMPTION,
                "partitions": per_partition,
            },
            affected_columns=[feature, column, cfg.outcome_timestamp],
            recommended_action=f"Remove {feature!r} or rebuild it from pre-outcome data only.",
            limitations=["Relies entirely on the supplied availability and outcome timestamps."],
        )
    result.ran("TMP-002", details)


def _outcome_vs_prediction(ctx: AuditContext, result: DetectorResult) -> None:
    cfg = ctx.config
    pred, outcome = cfg.prediction_timestamp, cfg.outcome_timestamp
    if not pred or not outcome:
        result.skipped("TMP-003", "Requires both prediction_timestamp and outcome_timestamp.")
        return
    lead = cfg.min_lead_time_delta
    details: dict[str, Any] = {"min_lead_time": cfg.min_lead_time, "partitions": {}}
    n_before = n_short = 0
    for name, stamps in ctx.timestamps.items():
        p, o = stamps[pred], stamps[outcome]
        valid = p.notna() & o.notna()
        gap = (o - p)[valid]
        before = int((gap < pd.Timedelta(0)).sum())
        short = int(((gap >= pd.Timedelta(0)) & (gap < lead)).sum()) if lead is not None else 0
        details["partitions"][name] = {
            "n_compared": int(valid.sum()),
            "n_not_compared_missing": int((~valid).sum()),
            "n_outcome_before_prediction": before,
            "n_outcome_equal_to_prediction": int((gap == pd.Timedelta(0)).sum()),
            "n_lead_time_below_minimum": short if lead is not None else None,
            "min_gap": gap.min() if len(gap) else None,
        }
        n_before += before
        n_short += short
    details["boundary_rule"] = (
        "An outcome strictly earlier than the prediction is a violation. An outcome at the same "
        "instant is counted but only flagged when min_lead_time is set."
    )
    result.ran("TMP-003", details)
    parts = details["partitions"].values()
    n_missing = sum(p["n_not_compared_missing"] for p in parts)
    n_equal = sum(p["n_outcome_equal_to_prediction"] for p in parts)
    if n_missing:
        result.warnings.append(
            f"TMP-003: {n_missing} row(s) could not be compared because {pred!r} or {outcome!r} "
            "is missing; their timing was not assessed."
        )
    if n_equal and lead is None:
        result.warnings.append(
            f"TMP-003: {n_equal} row(s) have {outcome!r} exactly equal to {pred!r}. They are not "
            "flagged because no min_lead_time was declared; review whether the outcome can be "
            "known at prediction time."
        )
    common = {
        "assumptions": _ASSUMPTION,
        "partitions": details["partitions"],
    }
    if n_before:
        result.add(
            "TMP-003",
            "outcome_before_prediction",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            title=f"Outcome precedes the prediction time on {n_before} row(s)",
            description=(
                f"On {n_before} rows {outcome!r} is earlier than {pred!r}. The label was "
                "already known at the declared prediction time, so features built up to "
                "that time can contain it."
            ),
            evidence={"comparison_rule": f"{outcome} < {pred}", **common},
            affected_columns=[pred, outcome],
            recommended_action=(
                "Check the prediction-time definition. Exclude these rows or move the "
                "prediction time before the outcome."
            ),
            limitations=["Timestamp errors (time zones, placeholder dates) can cause this."],
        )
    if n_short:
        result.add(
            "TMP-003",
            "insufficient_lead_time",
            severity=Severity.MEDIUM,
            confidence=Confidence.HIGH,
            title=f"Outcome occurs within the minimum lead time on {n_short} row(s)",
            description=(
                f"On {n_short} rows the outcome follows the prediction by less than "
                f"{cfg.min_lead_time}, the declared minimum lead time."
            ),
            evidence={
                "comparison_rule": f"0 <= {outcome} - {pred} < {cfg.min_lead_time}",
                **common,
            },
            affected_columns=[pred, outcome],
            recommended_action="Exclude these rows or confirm the declared lead time.",
            limitations=["Only as reliable as the declared min_lead_time."],
        )


def _split_overlap(ctx: AuditContext, result: DetectorResult) -> None:
    pred = ctx.config.prediction_timestamp
    if not ctx.has_split:
        result.skipped("TMP-004", "No train/test split exists (unsplit mode).")
        return
    if not pred:
        result.skipped("TMP-004", "No prediction_timestamp configured.")
        return
    train_ts = ctx.timestamps["train"][pred].dropna()
    test_ts = ctx.timestamps["test"][pred].dropna()
    if train_ts.empty or test_ts.empty:
        result.skipped("TMP-004", "A partition has no non-missing prediction timestamps.")
        return
    train_max, test_min = train_ts.max(), test_ts.min()
    stats = {
        "train_min": train_ts.min(),
        "train_max": train_max,
        "test_min": test_min,
        "test_max": test_ts.max(),
        "n_train_rows_after_test_start": int((train_ts > test_min).sum()),
        "n_test_rows_before_train_end": int((test_ts < train_max).sum()),
        "scenario": ctx.config.scenario.value,
    }
    stats["n_train_rows_at_boundary"] = (
        int((train_ts == test_min).sum()) if train_max == test_min else 0
    )
    stats["n_test_rows_at_boundary"] = (
        int((test_ts == test_min).sum()) if train_max == test_min else 0
    )
    result.ran("TMP-004", stats)
    scenario = ctx.config.scenario
    time_matters = scenario.requires_temporal_order or scenario.value == "unknown"
    if train_max == test_min and time_matters:
        result.warnings.append(
            f"TMP-004: train and test have a shared boundary timestamp {test_min.isoformat()} "
            f"({stats['n_train_rows_at_boundary']} train rows, {stats['n_test_rows_at_boundary']} "
            "test rows). This is not flagged as overlap, but same-instant rows on both sides of a "
            "forecasting split may share information; review the cutoff."
        )
    if train_max <= test_min:
        return
    if scenario.requires_temporal_order:
        severity, confidence = Severity.HIGH, Confidence.HIGH
        context = f"The declared scenario ({scenario.value}) requires training only on the past."
    elif scenario.value == "unknown":
        severity, confidence = Severity.LOW, Confidence.LOW
        context = "This only matters if the model will forecast future periods."
    else:
        return
    result.add(
        "TMP-004",
        "train_test",
        severity=severity,
        confidence=confidence,
        title="Training data extends past the start of the test period",
        description=(
            f"{stats['n_train_rows_after_test_start']} training rows are later than the "
            f"earliest test row ({test_min.isoformat()}). {context}"
        ),
        evidence={"comparison_rule": f"max(train.{pred}) > min(test.{pred})", **stats},
        affected_columns=[pred],
        recommended_action="Use a chronological split: train on rows before a cutoff, test after.",
        limitations=["Overlapping periods are normal when rows are independent of time."],
    )


def _name_clues(ctx: AuditContext, result: DetectorResult) -> None:
    documented = set(ctx.config.feature_availability)
    hits = []
    for column in ctx.feature_columns:
        if column in documented:
            continue
        words = sorted(_TEMPORAL_WORDS & set(name_tokens(column)))
        if not words:
            continue
        hits.append(column)
        result.add(
            "TMP-005",
            column,
            severity=Severity.LOW,
            confidence=Confidence.LOW,
            title=f"Column name {column!r} suggests post-prediction information",
            description=(
                f"The name contains {words}. This is a prompt to check when the value is "
                "recorded, not evidence of leakage."
            ),
            evidence={"rule": "name token match", "matched_words": words},
            affected_columns=[column],
            recommended_action=(
                f"Confirm when {column!r} is recorded. If it can be later than prediction "
                "time, supply its availability timestamp via feature_availability."
            ),
            limitations=["Temporal availability cannot be inferred from a column name."],
        )
    result.ran(
        "TMP-005", {"n_columns_flagged": len(hits), "n_with_availability_metadata": len(documented)}
    )
