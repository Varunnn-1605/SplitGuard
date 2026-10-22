"""Detector D (part 1): identifier risks.

High uniqueness alone is not leakage: unique record IDs are often harmless,
and entity IDs can be legitimate grouping variables. These findings ask the
user to decide how each identifier should be treated. Overlap with duplicate
findings (DUP-*) is noted in the evidence; they describe the same rows and are
not independent proof.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from splitguard.context import AuditContext, DetectorResult
from splitguard.schemas import Confidence, Severity
from splitguard.validation import name_tokens

_ID_WORDS = {"id", "ids", "uuid", "guid", "key", "pk", "idx", "index", "record", "rowid"}
_INDEX_ARTIFACT = re.compile(r"^(unnamed: ?\d+|index|level_0)$", re.IGNORECASE)


def _integer_like(series: pd.Series) -> bool:
    if ptypes.is_bool_dtype(series.dtype):
        return False
    if ptypes.is_integer_dtype(series.dtype):
        return True
    if ptypes.is_float_dtype(series.dtype):
        values = series.dropna().to_numpy(dtype=float)
        return len(values) > 0 and bool(np.all(np.isfinite(values) & (values == np.floor(values))))
    return False


def _is_text(series: pd.Series) -> bool:
    return not (
        ptypes.is_numeric_dtype(series.dtype)
        or ptypes.is_bool_dtype(series.dtype)
        or ptypes.is_datetime64_any_dtype(series.dtype)
    )


def _is_row_index(series: pd.Series, name: str) -> tuple[bool, str]:
    if _INDEX_ARTIFACT.match(name):
        return True, "name matches a pandas/export index artefact"
    if not _integer_like(series) or series.isna().any() or len(series) < 2:
        return False, ""
    diffs = np.diff(series.to_numpy(dtype=float))
    if np.all(diffs == 1):
        return True, "values increase by exactly 1 on every row (file order)"
    return False, ""


def run(ctx: AuditContext) -> DetectorResult:
    result = DetectorResult()
    th = ctx.config.thresholds
    df = ctx.primary
    n = len(df)
    candidates: dict[str, dict] = {}
    row_indices: dict[str, dict] = {}
    if n < th.min_rows_for_identifier:
        reason = f"Fewer than {th.min_rows_for_identifier} rows; uniqueness is not meaningful."
        result.skipped("ID-001", reason)
        result.skipped("ID-002", reason)
    else:
        for column in ctx.feature_columns:
            series = df[column]
            is_index, why = _is_row_index(series, column)
            if is_index:
                row_indices[column] = {"reason": why}
                continue
            if not (_integer_like(series) or _is_text(series)):
                continue
            non_null = series.dropna()
            if non_null.empty:
                continue
            ratio = float(non_null.nunique() / len(non_null))
            if ratio < th.identifier_unique_ratio:
                continue
            words = sorted(_ID_WORDS & set(name_tokens(column)))
            candidates[column] = {"unique_ratio": ratio, "id_name_tokens": words}
        _emit_candidates(ctx, result, candidates)
        _emit_row_indices(ctx, result, row_indices)
    _cross_split(ctx, result, candidates, row_indices)
    return result


def _emit_candidates(ctx: AuditContext, result: DetectorResult, candidates: dict) -> None:
    part = ctx.primary_name
    for column, info in candidates.items():
        named = bool(info["id_name_tokens"])
        result.add(
            "ID-001",
            column,
            severity=Severity.LOW,
            confidence=Confidence.MEDIUM if named else Confidence.LOW,
            title=f"{column!r} looks like an identifier ({info['unique_ratio']:.1%} unique)",
            description=(
                f"{info['unique_ratio']:.1%} of non-missing values of {column!r} in the {part} "
                "partition are distinct"
                + (f", and its name contains {info['id_name_tokens']}" if named else "")
                + ". Models can memorise identifiers instead of learning generalisable patterns."
            ),
            evidence={
                "partition": part,
                "unique_ratio": info["unique_ratio"],
                "threshold": ctx.config.thresholds.identifier_unique_ratio,
                "id_name_tokens": info["id_name_tokens"],
            },
            affected_columns=[column],
            recommended_action=(
                f"Decide what {column!r} is. Exclude record keys from model inputs; if it is an "
                "entity ID, configure it as entity_id so cross-split overlap is checked."
            ),
            limitations=[
                "High uniqueness alone is not leakage; free text and fine-grained measurements "
                "can also be unique.",
            ],
        )
    result.ran("ID-001", {"n_candidates": len(candidates)})


def _emit_row_indices(ctx: AuditContext, result: DetectorResult, row_indices: dict) -> None:
    for column, info in row_indices.items():
        result.add(
            "ID-002",
            column,
            severity=Severity.LOW,
            confidence=Confidence.HIGH,
            title=f"{column!r} appears to be a row index",
            description=(
                f"{column!r} looks like a row number: {info['reason']}. If the file is sorted "
                "(by time, outcome or source), a row index can carry that ordering into a model."
            ),
            evidence={"partition": ctx.primary_name, **info},
            affected_columns=[column],
            recommended_action=f"Exclude {column!r} from model inputs (exclude_columns).",
            limitations=["A row index is usually harmless if it is never used as a feature."],
        )
    result.ran("ID-002", {"n_row_index_columns": len(row_indices)})


def _cross_split(
    ctx: AuditContext, result: DetectorResult, candidates: dict, row_indices: dict
) -> None:
    if not ctx.has_split:
        result.skipped("ID-003", "No train/test split exists (unsplit mode).")
        return
    test = ctx.partitions["test"]
    details = {
        "columns_checked": [],
        "row_index_columns_not_checked": sorted(row_indices),
        "configured_entity_id_checked_by": "DUP-003" if ctx.config.entity_id else None,
    }
    for column in candidates:
        if column not in test.columns:
            continue
        details["columns_checked"].append(column)
        train_vals = set(ctx.canonical("train", column).dropna())
        test_vals = ctx.canonical("test", column).dropna()
        shared = set(test_vals) & train_vals
        if not shared:
            continue
        n_rows = int(test_vals.isin(shared).sum())
        result.add(
            "ID-003",
            column,
            severity=Severity.MEDIUM,
            confidence=Confidence.LOW,
            title=f"{len(shared)} value(s) of identifier-like {column!r} occur in both train and test",
            description=(
                f"{n_rows} test rows carry an {column!r} value also present in train. If "
                f"{column!r} is a record key, the same records may be in both partitions."
            ),
            evidence={
                "n_shared_values": len(shared),
                "n_test_rows_with_shared_value": n_rows,
                "test_row_rate": n_rows / len(test),
                "related_rules": ["DUP-002", "DUP-004"],
                "note": "Shares rows with duplicate findings; not independent evidence.",
            },
            affected_columns=[column],
            recommended_action=(
                f"Check what {column!r} identifies. If it is an entity, configure it as "
                "entity_id and decide whether entities may cross the split."
            ),
            limitations=[
                "Identifier semantics are unknown: values may be reused or restart per file.",
            ],
        )
    result.ran("ID-003", details)
