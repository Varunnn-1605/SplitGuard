"""Hand-built fixtures. Nothing here imports SplitGuard detection code, so the
fixtures cannot inherit bugs from the implementation under test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def classification_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Binary target driven weakly by x1 (no leakage)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    color = rng.choice(["red", "green", "blue"], size=n)
    y = (x1 + rng.normal(scale=2.0, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "color": color, "y": y})


def regression_frame(n: int = 200, seed: int = 0, noise: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 2 * x + 1 + rng.normal(scale=noise, size=n)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def iso(ts: pd.Series | pd.DatetimeIndex) -> list[str]:
    return [t.strftime("%Y-%m-%dT%H:%M:%S") for t in ts]


def by_rule(result, rule_id: str) -> list:
    return [f for f in result.findings if f.rule_id == rule_id]


def finding(result, finding_id: str):
    matches = [f for f in result.findings if f.finding_id == finding_id]
    assert matches, f"{finding_id} not found; got {[f.finding_id for f in result.findings]}"
    return matches[0]


def check(result, check_id: str):
    return next(c for c in result.checks if c.check_id == check_id)


def write_csv(directory: Path, name: str, df: pd.DataFrame) -> Path:
    path = directory / name
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def clean_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    return classification_frame(200, seed=1), classification_frame(60, seed=2)
