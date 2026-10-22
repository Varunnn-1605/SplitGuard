"""Detector D (part 2): schema issues between and within partitions."""

from __future__ import annotations

import pandas as pd

from splitguard.context import AuditContext, DetectorResult
from splitguard.schemas import Confidence, Severity
from splitguard.validation import type_family

_NO_SPLIT = "No train/test split exists (unsplit mode)."


def run(ctx: AuditContext) -> DetectorResult:
    result = DetectorResult()
    _constant_columns(ctx, result)
    _mostly_numeric_text(ctx, result)
    _target_constant(ctx, result)
    if ctx.has_split:
        _column_sets(ctx, result)
        _type_mismatch(ctx, result)
        _target_in_test(ctx, result)
        _unseen_classes(ctx, result)
    else:
        for rule_id in ("SCH-001", "SCH-002", "SCH-003", "SCH-004", "SCH-005"):
            result.skipped(rule_id, _NO_SPLIT)
    return result


def _constant_columns(ctx: AuditContext, result: DetectorResult) -> None:
    empty, constant = [], []
    for column in ctx.feature_columns:
        n_unique = ctx.canonical(ctx.primary_name, column).nunique(dropna=True)
        if n_unique == 0:
            empty.append(column)
        elif n_unique == 1:
            constant.append(column)
    result.ran(
        "SCH-006", {"partition": ctx.primary_name, "all_missing": empty, "constant": constant}
    )
    if not (empty or constant):
        return
    result.add(
        "SCH-006",
        ctx.primary_name,
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        title=f"{len(constant)} constant and {len(empty)} all-missing feature column(s)",
        description=(
            f"In the {ctx.primary_name} partition these columns carry no variation "
            "(missing values ignored when counting distinct values). They cannot help a model "
            "and may indicate a broken extract."
        ),
        evidence={"partition": ctx.primary_name, "constant": constant, "all_missing": empty},
        affected_columns=constant + empty,
        recommended_action="Confirm these columns are expected to be constant; otherwise fix the extract.",
        limitations=["A constant column is not leakage; it is reported as a data-quality signal."],
    )


def _mostly_numeric_text(ctx: AuditContext, result: DetectorResult) -> None:
    df = ctx.primary
    th = ctx.config.thresholds
    flagged = {}
    for column in ctx.feature_columns:
        series = df[column]
        if type_family(series) != "text":
            continue
        values = series.dropna().astype(str)
        parsed = pd.to_numeric(values, errors="coerce")
        rate = float(parsed.notna().mean())
        if th.mostly_numeric_rate <= rate < 1.0:
            flagged[column] = {
                "numeric_share": rate,
                "n_non_numeric_values": int(parsed.isna().sum()),
            }
    result.ran("SCH-007", {"columns": flagged})
    for column, info in flagged.items():
        result.add(
            "SCH-007",
            column,
            severity=Severity.INFO,
            confidence=Confidence.MEDIUM,
            title=f"{column!r} is mostly numeric but stored as text",
            description=(
                f"{info['numeric_share']:.1%} of values parse as numbers but "
                f"{info['n_non_numeric_values']} do not, so the column was treated as text. "
                "Numeric comparisons (tolerances, correlations) were not applied to it."
            ),
            evidence=info,
            affected_columns=[column],
            recommended_action="Check the non-numeric values (placeholders such as 'N/A'?) and fix them upstream.",
            limitations=["Mixed types are a data-quality issue, not leakage."],
        )


def _target_constant(ctx: AuditContext, result: DetectorResult) -> None:
    target = ctx.config.target
    if target is None:
        result.skipped("SCH-008", "No target column configured.")
        return
    n_unique = int(ctx.canonical(ctx.primary_name, target).nunique(dropna=True))
    result.ran("SCH-008", {"n_distinct_target_values": n_unique})
    if ctx.target_usable:
        return
    result.add(
        "SCH-008",
        target,
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        title=f"Target {target!r} has {n_unique} distinct value(s)",
        description="Target leakage checks were skipped because the target does not vary.",
        evidence={"partition": ctx.primary_name, "n_distinct_values": n_unique},
        affected_columns=[target],
        recommended_action="Check that the correct target column and partition were supplied.",
        limitations=["This is a configuration or data problem, not evidence of leakage."],
    )


def _column_sets(ctx: AuditContext, result: DetectorResult) -> None:
    train, test = ctx.partitions["train"], ctx.partitions["test"]
    target = ctx.config.target
    missing = [c for c in train.columns if c not in test.columns and c != target]
    extra = [c for c in test.columns if c not in train.columns]
    result.ran("SCH-001", {"missing_from_test": missing})
    result.ran("SCH-002", {"extra_in_test": extra})
    if missing:
        result.add(
            "SCH-001",
            "test",
            severity=Severity.MEDIUM,
            confidence=Confidence.HIGH,
            title=f"{len(missing)} training column(s) are missing from test",
            description="These columns exist in train but not in test; they were not compared across partitions.",
            evidence={"missing_from_test": missing},
            affected_columns=missing,
            recommended_action="Align the schemas so train and test are built by the same pipeline.",
            limitations=[
                "Schema differences may be intentional (for example dropped audit columns)."
            ],
        )
    if extra:
        result.add(
            "SCH-002",
            "test",
            severity=Severity.LOW,
            confidence=Confidence.HIGH,
            title=f"{len(extra)} column(s) exist only in test",
            description="These test columns have no training counterpart and were not compared.",
            evidence={"extra_in_test": extra},
            affected_columns=extra,
            recommended_action="Check why test has additional columns; a different pipeline may have produced it.",
            limitations=["Extra columns are a consistency issue, not leakage by themselves."],
        )


def _type_mismatch(ctx: AuditContext, result: DetectorResult) -> None:
    train, test = ctx.partitions["train"], ctx.partitions["test"]
    mismatches = {}
    for column in train.columns:
        if column not in test.columns:
            continue
        a, b = type_family(train[column]), type_family(test[column])
        if a != b:
            mismatches[column] = {"train": a, "test": b}
    result.ran("SCH-003", {"mismatches": mismatches})
    for column, families in mismatches.items():
        result.add(
            "SCH-003",
            column,
            severity=Severity.MEDIUM,
            confidence=Confidence.HIGH,
            title=f"{column!r} is {families['train']} in train but {families['test']} in test",
            description=(
                "The column has a different coarse type in each partition. Exact comparisons "
                "of this column across partitions may miss matches."
            ),
            evidence=families,
            affected_columns=[column],
            recommended_action="Make both partitions use the same encoding for this column.",
            limitations=[
                "Type differences can come from missing values or placeholders, not leakage."
            ],
        )


def _target_in_test(ctx: AuditContext, result: DetectorResult) -> None:
    target = ctx.config.target
    if target is None:
        result.skipped("SCH-004", "No target column configured.")
        return
    test = ctx.partitions["test"]
    present = target in test.columns
    all_missing = present and bool(test[target].isna().all())
    result.ran(
        "SCH-004",
        {
            "expected": ctx.config.expect_target_in_test,
            "present": present,
            "all_missing": all_missing,
        },
    )
    if not ctx.config.expect_target_in_test or (present and not all_missing):
        return
    result.add(
        "SCH-004",
        target,
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        title=f"Target {target!r} is {'empty' if present else 'absent'} in test",
        description="The target was expected in test but is unavailable, so test-side label checks could not run.",
        evidence={"present": present, "all_missing": all_missing},
        affected_columns=[target],
        recommended_action="Supply labelled test data, or set expect_target_in_test to false for unlabelled test sets.",
        limitations=["Unlabelled test sets are legitimate in some workflows."],
    )


def _unseen_classes(ctx: AuditContext, result: DetectorResult) -> None:
    target = ctx.config.target
    if not (target and ctx.is_classification and ctx.target_in("test")):
        result.skipped("SCH-005", "Requires a classification target present in both partitions.")
        return
    train_classes = set(ctx.canonical("train", target).dropna())
    test_labels = ctx.canonical("test", target).dropna()
    unseen = set(test_labels) - train_classes
    stats = {
        "n_train_classes": len(train_classes),
        "n_test_classes": int(test_labels.nunique()),
        "n_unseen_classes": len(unseen),
        "n_test_rows_with_unseen_class": int(test_labels.isin(unseen).sum()),
    }
    result.ran("SCH-005", stats)
    if not unseen:
        return
    result.add(
        "SCH-005",
        target,
        severity=Severity.LOW,
        confidence=Confidence.HIGH,
        title=f"{len(unseen)} test target class(es) never occur in train",
        description="A model cannot predict classes it never saw; test metrics will reflect this.",
        evidence=stats,
        affected_columns=[target],
        recommended_action="Check label encoding consistency, or use a stratified split.",
        limitations=["Rare classes can legitimately be absent from a small training set."],
    )
