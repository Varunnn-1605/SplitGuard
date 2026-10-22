"""Regression tests for defects found in Phase 1.5 adversarial validation.

D1  TGT-002 had no chance correction for regression targets.
D2  TGT-001 fired when the target was constant among the non-missing pairs.
D3  Partially assessed checks (missing timestamps, missing entity IDs,
    too few pairs) were not stated in the report.
D4  Equal-timestamp boundary cases were not stated as ambiguous.
D5  DUP-002 had no small-row-space guard (DUP-004 did).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import by_rule, check, classification_frame, finding, iso
from splitguard import AuditConfig, Confidence, Severity, audit_train_test


def test_d1_zero_inflated_regression_is_not_a_near_copy():
    g = np.random.default_rng(1)
    y = np.zeros(1000)
    y[g.choice(1000, 5, replace=False)] = g.uniform(100, 900, 5)
    feat = np.zeros(1000)
    feat[g.choice(1000, 5, replace=False)] = g.integers(1, 4, 5)
    train = pd.DataFrame({"claims": feat, "age": g.normal(40, 10, 1000), "amount": y})
    test = pd.DataFrame({"claims": [0.0] * 40, "age": g.normal(40, 10, 40), "amount": [1.0] * 40})
    result = audit_train_test(train, test, AuditConfig(target="amount", task_type="regression"))
    assert by_rule(result, "TGT-002") == []


def test_d1_genuine_regression_near_copy_still_detected():
    g = np.random.default_rng(2)
    y = g.normal(size=500)
    copy = y.copy()
    copy[:5] = g.normal(size=5)  # 99% identical
    train = pd.DataFrame({"x": g.normal(size=500), "copy": copy, "y": y})
    f = finding(audit_train_test(train, train.iloc[:50], AuditConfig(target="y")), "TGT-002:copy")
    assert f.evidence["cohens_kappa"] > 0.98
    assert f.evidence["chance_agreement"] < 0.01


def test_d2_constant_target_among_pairs_is_not_a_copy():
    g = np.random.default_rng(0)
    y = (g.random(2000) < 0.99).astype(int)
    flag = np.full(2000, np.nan)
    idx = np.flatnonzero(y == 1)[:40]
    flag[idx] = 1.0
    train = pd.DataFrame({"x": g.normal(size=2000), "flag": flag, "y": y})
    result = audit_train_test(train, train.iloc[:30], AuditConfig(target="y"))
    assert [
        f for f in result.findings if "flag" in f.affected_columns and f.rule_id.startswith("TGT")
    ] == []
    assert check(result, "TGT-001").details["features_with_constant_target_in_pairs"] == ["flag"]
    assert any("single value wherever the feature is present" in w for w in result.warnings)


def test_d3_missing_availability_timestamps_are_stated():
    pred = pd.Series(pd.date_range("2024-01-01", periods=80, freq="D"))
    df = pd.DataFrame(
        {
            "bal": np.arange(80.0),
            "y": [0, 1] * 40,
            "t": iso(pred),
            "bal_at": [None] * 20 + iso(pred[20:] - pd.Timedelta(hours=1)),
        }
    )
    cfg = AuditConfig(target="y", prediction_timestamp="t", feature_availability={"bal": "bal_at"})
    result = audit_train_test(df, df.iloc[40:].reset_index(drop=True), cfg)
    assert by_rule(result, "TMP-001") == []
    assert any(
        w.startswith("TMP-001: 20 row(s) for 'bal' could not be compared") for w in result.warnings
    )


def test_d3_missing_entity_ids_are_stated():
    train = classification_frame(100, 3).assign(c=[f"c{i}" for i in range(70)] + [None] * 30)
    test = classification_frame(40, 4).assign(c=[f"d{i}" for i in range(30)] + [None] * 10)
    result = audit_train_test(
        train, test, AuditConfig(target="y", entity_id="c", scenario="new_entities")
    )
    assert any("30 train and 10 test rows have a missing entity ID" in w for w in result.warnings)


def test_d3_features_with_too_few_pairs_are_stated():
    train = classification_frame(200, 5)
    train["sparse"] = np.nan
    train.loc[:9, "sparse"] = train.loc[:9, "y"]
    result = audit_train_test(train, train.iloc[:30], AuditConfig(target="y"))
    assert any("too few non-missing pairs" in w and "sparse" in w for w in result.warnings)


def test_d4_equal_timestamps_are_counted_and_stated():
    df = pd.DataFrame(
        {
            "bal": np.arange(40.0),
            "y": [0, 1] * 20,
            "t": ["2024-01-01T00:00:00"] * 40,
            "o": ["2024-02-01T00:00:00"] * 40,
            "bal_at": ["2024-01-01T00:00:00"] * 40,
        }
    )
    cfg = AuditConfig(
        target="y",
        prediction_timestamp="t",
        outcome_timestamp="o",
        feature_availability={"bal": "bal_at"},
    )
    result = audit_train_test(df, df, cfg)
    assert by_rule(result, "TMP-001") == []
    details = check(result, "TMP-001").details["bal"]
    assert details["partitions"]["train"]["n_equal"] == 40
    assert "Strict comparison" in details["boundary_rule"]
    assert any(
        "TMP-001: 80 row(s) have 'bal_at' exactly equal to 't'" in w for w in result.warnings
    )


def test_d4_shared_forecasting_boundary_is_stated():
    def frame(dates):
        return pd.DataFrame(
            {"x": np.arange(len(dates), dtype=float), "y": [0, 1] * (len(dates) // 2), "t": dates}
        )

    train = frame([f"2024-01-{d:02d}" for d in range(1, 21)])
    test = frame(["2024-01-20"] + [f"2024-01-{d:02d}" for d in range(21, 30)])
    cfg = AuditConfig(target="y", prediction_timestamp="t", scenario="forecasting")
    result = audit_train_test(train, test, cfg)
    assert by_rule(result, "TMP-004") == []
    assert check(result, "TMP-004").details["n_train_rows_at_boundary"] == 1
    assert any("shared boundary timestamp" in w for w in result.warnings)


def test_d5_small_row_space_cross_duplicates_are_low_confidence():
    g = np.random.default_rng(3)

    def survey(n):
        return pd.DataFrame(
            {"q1": g.integers(1, 4, n), "q2": g.choice(["a", "b"], n), "y": g.integers(0, 2, n)}
        )

    result = audit_train_test(survey(500), survey(100), AuditConfig(target="y"))
    f = finding(result, "DUP-002:train_test")
    assert (f.severity, f.confidence) == (Severity.LOW, Confidence.LOW)
    assert f.evidence["train_row_uniqueness"] < 0.5


def test_d5_high_uniqueness_cross_duplicates_stay_critical():
    train = classification_frame(200, 6)
    test = pd.concat([classification_frame(40, 7), train.iloc[:10]], ignore_index=True)
    f = finding(audit_train_test(train, test, AuditConfig(target="y")), "DUP-002:train_test")
    assert (f.severity, f.confidence) == (Severity.CRITICAL, Confidence.HIGH)
