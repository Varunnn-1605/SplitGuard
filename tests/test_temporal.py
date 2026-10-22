from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import by_rule, check, finding, iso
from splitguard import AuditConfig, Confidence, Severity, audit_train_test, inspect_unsplit

DAY = pd.Timedelta(days=1)


def events(n: int, start: str, seed: int) -> pd.DataFrame:
    """Prediction at a daily cadence, outcome 30 days later, feature snapshotted 1h before."""
    rng = np.random.default_rng(seed)
    pred = pd.Series(pd.date_range(start, periods=n, freq="D"))
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "balance": rng.normal(size=n),
            "y": rng.integers(0, 2, n),
            "pred_at": iso(pred),
            "outcome_at": iso(pred + 30 * DAY),
            "balance_at": iso(pred - pd.Timedelta(hours=1)),
        }
    )


def config(**kwargs) -> AuditConfig:
    base = dict(
        target="y",
        prediction_timestamp="pred_at",
        outcome_timestamp="outcome_at",
        feature_availability={"balance": "balance_at"},
    )
    base.update(kwargs)
    return AuditConfig(**base)


def shift(df: pd.DataFrame, column: str, rows: list[int], base: str, delta: pd.Timedelta) -> None:
    df.loc[rows, column] = iso(pd.to_datetime(df.loc[rows, base]) + delta)


def test_clean_temporal_dataset_has_no_temporal_findings():
    train, test = events(40, "2024-01-01", 1), events(20, "2024-03-01", 2)
    result = audit_train_test(train, test, config(scenario="forecasting"))
    assert [f for f in result.findings if f.rule_id.startswith("TMP")] == []
    for rule_id in ("TMP-001", "TMP-002", "TMP-003", "TMP-004"):
        assert check(result, rule_id).status.value == "run"


def test_feature_available_after_prediction_and_after_outcome():
    train, test = events(40, "2024-01-01", 3), events(20, "2024-03-01", 4)
    shift(train, "balance_at", [0, 1, 2], "pred_at", 2 * DAY)  # after prediction
    shift(train, "balance_at", [10, 11], "outcome_at", DAY)  # after the outcome too
    result = audit_train_test(train, test, config())
    late = finding(result, "TMP-001:balance")
    assert (late.severity, late.confidence) == (Severity.HIGH, Confidence.HIGH)
    assert late.evidence["partitions"]["train"]["n_violations"] == 5
    assert late.evidence["partitions"]["test"]["n_violations"] == 0
    assert late.evidence["comparison_rule"] == "balance_at > pred_at"
    assert late.evidence["partitions"]["train"]["max_lag"] == pd.Timedelta(days=31)
    post = finding(result, "TMP-002:balance")
    assert post.severity is Severity.CRITICAL
    assert post.evidence["partitions"]["train"]["n_violations"] == 2


def test_outcome_before_prediction_and_lead_time():
    train, test = events(40, "2024-01-01", 5), events(20, "2024-03-01", 6)
    shift(train, "outcome_at", [0, 1, 2], "pred_at", -DAY)
    shift(train, "outcome_at", [5, 6, 7, 8], "pred_at", 5 * DAY)
    result = audit_train_test(train, test, config(min_lead_time="10D"))
    before = finding(result, "TMP-003:outcome_before_prediction")
    assert before.evidence["partitions"]["train"]["n_outcome_before_prediction"] == 3
    short = finding(result, "TMP-003:insufficient_lead_time")
    assert short.evidence["partitions"]["train"]["n_lead_time_below_minimum"] == 4
    assert short.severity is Severity.MEDIUM


def test_train_test_time_overlap_depends_on_scenario():
    train, test = events(40, "2024-01-01", 7), events(20, "2024-02-01", 8)
    forecast = audit_train_test(train, test, config(scenario="forecasting"))
    f = finding(forecast, "TMP-004:train_test")
    assert (f.severity, f.confidence) == (Severity.HIGH, Confidence.HIGH)
    assert f.evidence["n_train_rows_after_test_start"] == 8
    unknown = audit_train_test(train, test, config())
    assert finding(unknown, "TMP-004:train_test").confidence is Confidence.LOW
    independent = audit_train_test(train, test, config(scenario="independent"))
    assert by_rule(independent, "TMP-004") == []


def test_missing_metadata_is_reported_as_not_assessed():
    train, test = events(40, "2024-01-01", 9), events(20, "2024-03-01", 10)
    result = audit_train_test(train, test, AuditConfig(target="y"))
    for rule_id in ("TMP-001", "TMP-002", "TMP-003", "TMP-004"):
        assert check(result, rule_id).status.value == "skipped"
    warning = next(w for w in result.warnings if "temporal" in w)
    assert "not assessed" in warning and "not evidence" in warning


def test_name_clues_are_low_confidence_prompts_only():
    train, test = events(40, "2024-01-01", 11), events(20, "2024-03-01", 12)
    for df in (train, test):
        df["future_spend"] = np.arange(len(df), dtype=float)[::-1] * 0.37
    result = audit_train_test(train, test, AuditConfig(target="y"))
    f = finding(result, "TMP-005:future_spend")
    assert (f.severity, f.confidence) == (Severity.LOW, Confidence.LOW)
    assert by_rule(result, "TMP-001") == []


def test_name_clue_suppressed_when_availability_is_documented():
    train, test = events(40, "2024-01-01", 13), events(20, "2024-03-01", 14)
    for df in (train, test):
        df.rename(columns={"balance": "final_balance"}, inplace=True)
    cfg = config(feature_availability={"final_balance": "balance_at"})
    result = audit_train_test(train, test, cfg)
    assert by_rule(result, "TMP-005") == []
    assert by_rule(result, "TMP-001") == []


def test_timezones_are_normalised():
    train, test = events(40, "2024-01-01", 15), events(20, "2024-03-01", 16)
    train.loc[0, "pred_at"] = "2024-01-01T00:00:00"  # naive -> UTC
    train.loc[0, "balance_at"] = "2024-01-01T01:30:00+02:00"  # 23:30 UTC the day before
    result = audit_train_test(train, test, config())
    assert by_rule(result, "TMP-001") == []


def test_unsplit_temporal_checks_run_on_whole_dataset():
    data = events(40, "2024-01-01", 17)
    shift(data, "balance_at", [3], "pred_at", DAY)
    result = inspect_unsplit(data, config())
    f = finding(result, "TMP-001:balance")
    assert f.evidence["partitions"]["unsplit"]["n_violations"] == 1
    assert check(result, "TMP-004").status.value == "skipped"
