"""Defensive input loading and validation.

Validation never repairs data: it does not drop rows, coerce labels or rename
columns. Problems are collected as :class:`ValidationIssue` objects and raised
together in one :class:`InputValidationError`. Error messages never contain
row contents.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd
from pandas.api import types as ptypes

from splitguard.config import AuditConfig, Limits, TaskType, Thresholds
from splitguard.hashing import canonical_scalar, canonical_series, frame_digest

TableSource = str | os.PathLike[str] | bytes | bytearray | pd.DataFrame


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


class InputValidationError(ValueError):
    """One or more inputs are unusable. ``issues`` lists every problem found."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("InputValidationError requires at least one issue")
        super().__init__("; ".join(f"[{i.code}] {i.message}" for i in self.issues))

    def to_dict(self) -> dict[str, Any]:
        return {"error": "input_validation_failed", "issues": [i.to_dict() for i in self.issues]}


def raise_if_issues(issues: Sequence[ValidationIssue]) -> None:
    if issues:
        raise InputValidationError(issues)


@dataclass(frozen=True)
class LoadedTable:
    frame: pd.DataFrame
    metadata: dict[str, Any]


def _safe_label(value: Any, max_len: int = 40) -> str:
    text = canonical_scalar(value)
    text = "<missing>" if text is None else text
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


# --------------------------------------------------------------------------- loading


def load_table(source: TableSource, *, name: str | None = None, limits: Limits) -> LoadedTable:
    """Load a CSV path, CSV bytes, or an in-memory DataFrame with validation.

    Byte input lets callers (for example a future web UI) pass uploaded
    content without ever handing SplitGuard a file path.
    """
    if isinstance(source, pd.DataFrame):
        return _from_frame(source, name or "dataframe", limits)
    if isinstance(source, (bytes, bytearray)):
        return _from_bytes(bytes(source), name or "uploaded.csv", limits)
    if isinstance(source, (str, os.PathLike)):
        return _from_path(Path(source), name, limits)
    raise InputValidationError(
        [ValidationIssue("UNSUPPORTED_SOURCE", f"Unsupported input type {type(source).__name__}")]
    )


def _from_path(path: Path, name: str | None, limits: Limits) -> LoadedTable:
    label = name or path.name
    if not path.is_file():
        raise InputValidationError([ValidationIssue("FILE_NOT_FOUND", f"{label}: file not found")])
    size = path.stat().st_size
    _check_size(size, label, limits)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    with path.open("rb") as fh:
        frame = _parse_csv(fh, label, limits)
    return LoadedTable(frame, _metadata(label, "csv_file", frame, size, digest.hexdigest()))


def _from_bytes(data: bytes, label: str, limits: Limits) -> LoadedTable:
    _check_size(len(data), label, limits)
    frame = _parse_csv(io.BytesIO(data), label, limits)
    return LoadedTable(
        frame, _metadata(label, "csv_bytes", frame, len(data), hashlib.sha256(data).hexdigest())
    )


def _from_frame(df: pd.DataFrame, label: str, limits: Limits) -> LoadedTable:
    issues = validate_column_names([c for c in df.columns], label)
    if len(df.columns) > limits.max_columns:
        issues.append(_column_limit_issue(label, len(df.columns), limits))
    if len(df) > limits.max_rows:
        issues.append(_row_limit_issue(label, limits))
    raise_if_issues(issues)
    frame = df.reset_index(drop=True)
    raise_if_issues(validate_nonempty(frame, label))
    return LoadedTable(frame, _metadata(label, "dataframe", frame, None, frame_digest(frame)))


def _metadata(
    label: str, kind: str, frame: pd.DataFrame, size: int | None, sha256: str
) -> dict[str, Any]:
    return {
        "name": label,
        "source_type": kind,
        "size_bytes": size,
        "sha256": sha256,
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "columns": [str(c) for c in frame.columns],
    }


def _check_size(size: int, label: str, limits: Limits) -> None:
    if size == 0:
        raise InputValidationError([ValidationIssue("EMPTY_FILE", f"{label}: file is empty")])
    if size > limits.max_file_bytes:
        raise InputValidationError(
            [
                ValidationIssue(
                    "FILE_TOO_LARGE",
                    f"{label}: {size} bytes exceeds the limit of {limits.max_file_bytes} bytes",
                    "Raise limits.max_file_bytes (CLI: --max-file-mb) if this is intended.",
                )
            ]
        )


def _row_limit_issue(label: str, limits: Limits) -> ValidationIssue:
    return ValidationIssue(
        "ROW_LIMIT_EXCEEDED",
        f"{label}: more than {limits.max_rows} rows",
        "Raise limits.max_rows (CLI: --max-rows) or audit a sample.",
    )


def _column_limit_issue(label: str, n: int, limits: Limits) -> ValidationIssue:
    return ValidationIssue(
        "COLUMN_LIMIT_EXCEEDED", f"{label}: {n} columns exceeds the limit of {limits.max_columns}"
    )


def _parse_csv(fh: BinaryIO, label: str, limits: Limits) -> pd.DataFrame:
    # Read the raw header first: pandas silently renames duplicate columns.
    try:
        text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
        header = next(csv.reader(text))
    except StopIteration:
        raise InputValidationError(
            [ValidationIssue("EMPTY_FILE", f"{label}: no header row found")]
        ) from None
    except (UnicodeDecodeError, csv.Error) as exc:
        raise InputValidationError(
            [
                ValidationIssue(
                    "UNREADABLE_CSV",
                    f"{label}: header could not be read ({type(exc).__name__})",
                    "Files must be UTF-8 encoded CSV.",
                )
            ]
        ) from None
    issues = validate_column_names(header, label)
    if len(header) > limits.max_columns:
        issues.append(_column_limit_issue(label, len(header), limits))
    raise_if_issues(issues)

    text.seek(0)
    try:
        frame = pd.read_csv(
            text, nrows=limits.max_rows + 1, low_memory=False, skip_blank_lines=True
        )
    except pd.errors.EmptyDataError:
        raise InputValidationError([ValidationIssue("EMPTY_FILE", f"{label}: no data")]) from None
    except pd.errors.ParserError as exc:
        # Parser messages report line/field counts, not cell contents.
        detail = str(exc).splitlines()[-1][:200]
        raise InputValidationError(
            [ValidationIssue("MALFORMED_CSV", f"{label}: {detail}")]
        ) from None
    except UnicodeDecodeError:
        raise InputValidationError(
            [ValidationIssue("UNREADABLE_CSV", f"{label}: not valid UTF-8", "Re-save as UTF-8.")]
        ) from None
    finally:
        text.detach()
    if len(frame) > limits.max_rows:
        raise InputValidationError([_row_limit_issue(label, limits)])
    raise_if_issues(validate_nonempty(frame, label))
    return frame


# ----------------------------------------------------------------------- structural


def validate_column_names(columns: Sequence[Any], label: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not columns:
        return [ValidationIssue("NO_COLUMNS", f"{label}: no columns")]
    non_string = [i for i, c in enumerate(columns) if not isinstance(c, str)]
    if non_string:
        issues.append(
            ValidationIssue(
                "NON_STRING_COLUMN_NAME",
                f"{label}: column names must be strings (positions {non_string[:10]})",
            )
        )
    blank = [i for i, c in enumerate(columns) if isinstance(c, str) and not c.strip()]
    if blank:
        issues.append(
            ValidationIssue(
                "MISSING_COLUMN_NAME",
                f"{label}: empty column name at position(s) {blank[:10]} (0-based)",
                "Give every column a header.",
            )
        )
    counts = Counter(c for c in columns if isinstance(c, str))
    dupes = sorted(c for c, n in counts.items() if n > 1)
    if dupes:
        issues.append(
            ValidationIssue(
                "DUPLICATE_COLUMN_NAMES",
                f"{label}: duplicate column names {dupes[:10]}",
                "Column names must be unique.",
            )
        )
    return issues


def validate_nonempty(df: pd.DataFrame, label: str) -> list[ValidationIssue]:
    if len(df.columns) == 0:
        return [ValidationIssue("NO_COLUMNS", f"{label}: no columns")]
    if len(df) == 0:
        return [ValidationIssue("EMPTY_DATASET", f"{label}: header present but no data rows")]
    return []


def require_columns(
    df: pd.DataFrame, required: Mapping[str, str | None], label: str
) -> list[ValidationIssue]:
    """``required`` maps a role description to a column name (None = not requested)."""
    issues = []
    for role, column in required.items():
        if column is not None and column not in df.columns:
            issues.append(
                ValidationIssue(
                    "MISSING_COLUMN", f"{label}: configured {role} column {column!r} not found"
                )
            )
    return issues


def check_missing_rate(
    series: pd.Series, role: str, label: str, max_rate: float
) -> list[ValidationIssue]:
    rate = float(series.isna().mean()) if len(series) else 1.0
    if rate > max_rate:
        return [
            ValidationIssue(
                "EXCESSIVE_MISSINGNESS",
                f"{label}: {role} column {series.name!r} is {rate:.1%} missing "
                f"(limit {max_rate:.0%})",
                "Fix the missing values upstream or raise thresholds.max_missing_rate_required.",
            )
        ]
    return []


# --------------------------------------------------------------------------- splits


def split_frame(
    df: pd.DataFrame, split_column: str, train_value: Any, test_value: Any, label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by an explicit label column. The split column is removed from the
    returned partitions because it differs between them by definition."""
    if split_column not in df.columns:
        raise InputValidationError(
            [ValidationIssue("MISSING_COLUMN", f"{label}: split column {split_column!r} not found")]
        )
    train_key, test_key = canonical_scalar(train_value), canonical_scalar(test_value)
    if train_key is None or test_key is None or train_key == test_key:
        raise InputValidationError(
            [
                ValidationIssue(
                    "INVALID_SPLIT_LABELS",
                    "train and test labels must be distinct, non-empty values",
                )
            ]
        )
    labels = canonical_series(df[split_column])
    issues = []
    n_missing = int(labels.isna().sum())
    if n_missing:
        issues.append(
            ValidationIssue("SPLIT_LABEL_MISSING", f"{label}: {n_missing} rows have no split label")
        )
    unknown = labels[labels.notna() & ~labels.isin([train_key, test_key])]
    if len(unknown):
        examples = sorted({_safe_label(v) for v in unknown.unique()})[:5]
        issues.append(
            ValidationIssue(
                "SPLIT_LABEL_INVALID",
                f"{label}: {len(unknown)} rows have unrecognised split labels (e.g. {examples})",
                f"Expected only {train_key!r} or {test_key!r}. Rows are never dropped automatically.",
            )
        )
    is_train, is_test = labels == train_key, labels == test_key
    for name, mask in (("train", is_train), ("test", is_test)):
        if not mask.any():
            issues.append(
                ValidationIssue("EMPTY_PARTITION", f"{label}: the {name} partition has no rows")
            )
    raise_if_issues(issues)
    train = df.loc[is_train.to_numpy()].drop(columns=[split_column]).reset_index(drop=True)
    test = df.loc[is_test.to_numpy()].drop(columns=[split_column]).reset_index(drop=True)
    return train, test


# ----------------------------------------------------------------------- timestamps


def parse_timestamps(
    series: pd.Series, role: str, label: str, max_missing_rate: float
) -> tuple[pd.Series, list[ValidationIssue]]:
    """Parse a timestamp column strictly to UTC.

    Naive timestamps are interpreted as UTC; formats are inferred per value
    (month-first for ambiguous dates). Any non-missing value that fails to
    parse is an error: values are never silently coerced to missing.
    """
    column = series.name
    if ptypes.is_datetime64_any_dtype(series.dtype):
        parsed = pd.to_datetime(series, utc=True)
    elif ptypes.is_numeric_dtype(series.dtype) or ptypes.is_bool_dtype(series.dtype):
        if series.notna().any():
            return series, [
                ValidationIssue(
                    "TIMESTAMP_NOT_TEXT",
                    f"{label}: {role} column {column!r} is numeric; epoch numbers are ambiguous",
                    "Provide ISO-8601 timestamps such as 2024-01-31T12:00:00Z.",
                )
            ]
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    issues = []
    invalid = int((series.notna() & parsed.isna()).sum())
    if invalid:
        issues.append(
            ValidationIssue(
                "INVALID_TIMESTAMP",
                f"{label}: {role} column {column!r} has {invalid} unparseable value(s)",
                "Fix or blank the invalid values; they are not dropped automatically.",
            )
        )
    issues.extend(check_missing_rate(parsed.rename(column), role, label, max_missing_rate))
    return parsed.rename(column), issues


# --------------------------------------------------------------------------- target


def is_numeric_feature(series: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(series.dtype) and not ptypes.is_bool_dtype(series.dtype)


def resolve_task_type(target: pd.Series, requested: TaskType) -> TaskType:
    if requested is not TaskType.AUTO:
        return requested
    if is_numeric_feature(target) and target.nunique(dropna=True) > 2:
        return TaskType.REGRESSION
    return TaskType.CLASSIFICATION


def validate_target(
    df: pd.DataFrame, target: str, label: str, task_type: TaskType, thresholds: Thresholds
) -> list[ValidationIssue]:
    series = df[target]
    issues = check_missing_rate(series, "target", label, thresholds.max_missing_rate_required)
    if task_type is TaskType.REGRESSION and not is_numeric_feature(series):
        issues.append(
            ValidationIssue(
                "TARGET_TYPE",
                f"{label}: regression requires a numeric target but {target!r} is "
                f"{type_family(series)}",
                "Use --task-type classification or fix the target values.",
            )
        )
    return issues


# --------------------------------------------------------------------------- types


_INFERRED_FAMILY = {
    "string": "text",
    "bytes": "text",
    "integer": "numeric",
    "floating": "numeric",
    "mixed-integer-float": "numeric",
    "decimal": "numeric",
    "complex": "numeric",
    "boolean": "boolean",
    "datetime": "datetime",
    "datetime64": "datetime",
    "date": "datetime",
    "empty": "empty",
}


def type_family(series: pd.Series) -> str:
    """Coarse type: numeric, boolean, datetime, text, mixed or empty."""
    if not series.notna().any():
        return "empty"
    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype):
        return "boolean"
    if ptypes.is_numeric_dtype(dtype):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "datetime"
    return _INFERRED_FAMILY.get(ptypes.infer_dtype(series, skipna=True), "mixed")


def feature_columns(columns: Iterable[str], config: AuditConfig) -> list[str]:
    """Columns treated as model features: everything except the target and
    configured metadata, in input order."""
    excluded = config.metadata_columns()
    return [c for c in columns if c != config.target and c not in excluded]


_TOKEN_SPLIT = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def name_tokens(name: str) -> list[str]:
    """``"customerID_2"`` -> ``["customer", "id", "2"]``."""
    spaced = _CAMEL.sub(" ", name)
    return [t.lower() for t in _TOKEN_SPLIT.split(spaced) if t]
