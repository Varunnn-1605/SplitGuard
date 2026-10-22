"""Deterministic value canonicalisation and row hashing.

Comparison semantics (used by every exact-match check):

* Columns are aligned by **name** and sorted, so column order never matters.
* Numbers are compared by value: ``1``, ``1.0`` and an object-column ``1``
  all canonicalise to ``"1"``. Integral floats up to 2**53 are written as
  integers; other floats use NumPy's shortest round-trip representation.
  ``-0.0`` equals ``0.0``.
* Text is compared verbatim: case and surrounding whitespace are significant.
  The text ``"1"`` therefore equals the number ``1``, but ``"1.0"`` (text)
  does not, because text is never re-parsed.
* Booleans canonicalise to ``"True"``/``"False"``; datetimes to ISO-8601.
* Missing values (``NaN``, ``None``, ``NA``, ``NaT``) are equal to each other.
* Rows are compared through pandas' fixed-key 64-bit hash of the canonical
  strings. The hash is deterministic across runs; the chance of a false match
  is about n**2 / 2**65 (negligible below ~10**7 rows).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import numbers
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

NA_TOKEN = "\x00<NA>"
_MAX_EXACT_FLOAT_INT = 2.0**53


def _canonical_float(x: float) -> str | None:
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x.is_integer() and abs(x) < _MAX_EXACT_FLOAT_INT:
        return str(int(x))
    return str(np.float64(x))


def canonical_scalar(value: Any) -> str | None:
    """Canonical string for one value, or ``None`` if it is missing."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return _canonical_float(float(value))
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def canonical_series(series: pd.Series) -> pd.Series:
    """Canonical strings for a column (object dtype, ``None`` for missing)."""
    n = len(series)
    out = np.full(n, None, dtype=object)
    dtype = series.dtype
    present = series.notna().to_numpy()
    if ptypes.is_bool_dtype(dtype):
        values = series.to_numpy(dtype=object)
        out[present] = np.where(values[present].astype(bool), "True", "False")
    elif ptypes.is_integer_dtype(dtype):
        out[present] = series[present].astype(str).to_numpy(dtype=object)
    elif ptypes.is_float_dtype(dtype):
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        finite = np.isfinite(values)
        integral = finite & (np.floor(values) == values) & (np.abs(values) < _MAX_EXACT_FLOAT_INT)
        out[integral] = values[integral].astype(np.int64).astype(str)
        other = finite & ~integral
        out[other] = values[other].astype(str)
        out[np.isposinf(values)] = "inf"
        out[np.isneginf(values)] = "-inf"
    elif ptypes.is_datetime64_any_dtype(dtype):
        out[present] = [ts.isoformat() for ts in series[present]]
    else:
        values = series.to_numpy(dtype=object)
        kind = ptypes.infer_dtype(values, skipna=True)
        if kind == "string":  # fast path: text is compared verbatim
            out[present] = values[present]
        elif kind != "empty":
            out[:] = [canonical_scalar(v) for v in values]
    return pd.Series(out, index=series.index, dtype=object)


def canonical_frame(
    df: pd.DataFrame,
    columns: Sequence[str],
    canonical: Callable[[str], pd.Series] | None = None,
) -> pd.DataFrame:
    """Canonical strings for ``columns`` (sorted by name), missing -> NA_TOKEN.

    ``canonical`` lets callers supply cached per-column results."""
    get = canonical or (lambda column: canonical_series(df[column]))
    data = {}
    for column in sorted(columns):
        values = get(column).to_numpy(dtype=object, copy=True)
        values[pd.isna(values)] = NA_TOKEN
        data[column] = values
    return pd.DataFrame(data, index=df.index)


def row_hashes(
    df: pd.DataFrame,
    columns: Sequence[str],
    canonical: Callable[[str], pd.Series] | None = None,
) -> pd.Series:
    """Deterministic uint64 hash per row over ``columns`` (order-independent)."""
    if not columns:
        raise ValueError("row_hashes requires at least one column")
    return pd.util.hash_pandas_object(canonical_frame(df, columns, canonical), index=False)


def frame_digest(df: pd.DataFrame) -> str:
    """SHA-256 over column names (in order) and pandas' fixed-key row hashes: a
    content fingerprint for in-memory DataFrames, stable across runs."""
    digest = hashlib.sha256()
    digest.update("".join(map(str, df.columns)).encode("utf-8"))
    if len(df.columns) and len(df):
        digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()
