"""Small, seeded sample datasets for trying the UI without your own data.

Samples are produced as CSV bytes, so they go through exactly the same
loading and validation path as uploaded files. They are synthetic; any
"leakage" in them was planted on purpose and is described in ``description``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Sample:
    key: str
    title: str
    mode: str  # train_test | split_column | unsplit
    description: str
    files: dict[str, bytes]  # role -> CSV bytes (train/test, or data)
    names: dict[str, str]  # role -> file name shown in the UI and report
    suggested: dict[str, Any] = field(default_factory=dict)  # suggested settings


def _customers(n: int, seed: int) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    tenure = g.integers(1, 72, n)
    spend = g.gamma(2.0, 40.0, n).round(2)
    plan = g.choice(["basic", "plus", "pro"], n)
    logit = -0.03 * tenure + 0.01 * spend + (plan == "basic") * 0.5
    churn = (g.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {"tenure_months": tenure, "monthly_spend": spend, "plan": plan, "churn": churn}
    )


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _train_test_sample() -> Sample:
    train, test = _customers(600, 1), _customers(150, 2)
    for df in (train, test):
        df["churn_score_final"] = df["churn"].astype(float)  # planted target copy
    test = pd.concat([test, train.iloc[:12]], ignore_index=True)  # planted cross-split copies
    return Sample(
        key="train_test",
        title="Customer churn: separate train and test files",
        mode="train_test",
        description=(
            "Two files. Planted problems: 'churn_score_final' is a copy of the target, and "
            "12 training rows were also copied into the test file."
        ),
        files={"train": _csv(train), "test": _csv(test)},
        names={"train": "sample_train.csv", "test": "sample_test.csv"},
        suggested={"target": "churn"},
    )


def _split_column_sample() -> Sample:
    g = np.random.default_rng(3)
    df = _customers(700, 4)
    df.insert(0, "customer_id", [f"C{i:04d}" for i in g.integers(0, 260, len(df))])
    df["split"] = np.where(g.random(len(df)) < 0.2, "test", "train")
    return Sample(
        key="split_column",
        title="Customer churn: one file with a split column",
        mode="split_column",
        description=(
            "One file with a 'split' column. Customers have several rows and were split "
            "randomly, so the same customer appears in both partitions. Whether that matters "
            "depends on the scenario you declare."
        ),
        files={"data": _csv(df)},
        names={"data": "sample_with_split.csv"},
        suggested={"target": "churn", "split_column": "split", "entity_id": "customer_id"},
    )


def _temporal_sample() -> Sample:
    g = np.random.default_rng(5)

    def events(n: int, start: str) -> pd.DataFrame:
        df = _customers(n, int(g.integers(0, 10_000)))
        pred = pd.Timestamp(start) + pd.to_timedelta(g.integers(0, 120, n), unit="D")
        outcome = pred + pd.Timedelta(days=30)
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        df["prediction_time"] = pred.strftime(fmt)
        df["outcome_time"] = outcome.strftime(fmt)
        df["spend_recorded_at"] = (pred - pd.Timedelta(days=1)).strftime(fmt)
        # refund_amount is entered during follow-up, after the outcome.
        df["refund_amount"] = (df["churn"] * g.uniform(10, 60, n)).round(2)
        df["refund_recorded_at"] = (outcome + pd.Timedelta(days=10)).strftime(fmt)
        return df

    return Sample(
        key="temporal",
        title="Churn with timestamps: train before test",
        mode="train_test",
        description=(
            "Train covers Jan-Apr, test covers Jul-Oct. 'refund_amount' is recorded 10 days "
            "after the outcome; 'monthly_spend' is recorded the day before prediction."
        ),
        files={"train": _csv(events(500, "2024-01-01")), "test": _csv(events(150, "2024-07-01"))},
        names={"train": "sample_events_train.csv", "test": "sample_events_test.csv"},
        suggested={
            "target": "churn",
            "prediction_timestamp": "prediction_time",
            "outcome_timestamp": "outcome_time",
            "feature_availability": {
                "monthly_spend": "spend_recorded_at",
                "refund_amount": "refund_recorded_at",
            },
            "scenario": "forecasting",
        },
    )


def _unsplit_sample() -> Sample:
    g = np.random.default_rng(6)
    df = _customers(500, 7)
    df.insert(0, "patient_id", [f"P{i:03d}" for i in g.integers(0, 120, len(df))])
    df["visit_date"] = (
        pd.Timestamp("2023-01-01") + pd.to_timedelta(g.integers(0, 365, len(df)), unit="D")
    ).strftime("%Y-%m-%d")
    return Sample(
        key="unsplit",
        title="Clinic visits: one unsplit file",
        mode="unsplit",
        description=(
            "No split yet. Patients have repeated visits with dates. SplitGuard will ask how "
            "the model will be used before recommending a split."
        ),
        files={"data": _csv(df)},
        names={"data": "sample_unsplit.csv"},
        suggested={"target": "churn", "entity_id": "patient_id"},
    )


def samples() -> dict[str, Sample]:
    """All samples, keyed by ``Sample.key`` (deterministic)."""
    items = [_train_test_sample(), _split_column_sample(), _temporal_sample(), _unsplit_sample()]
    return {s.key: s for s in items}
