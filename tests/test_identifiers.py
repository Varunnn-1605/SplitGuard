from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import by_rule, check, classification_frame, finding
from splitguard import AuditConfig, Confidence, Severity, audit_train_test


def with_ids(df: pd.DataFrame, start: int) -> pd.DataFrame:
    return df.assign(transaction_id=[f"T{start + i:06d}" for i in range(len(df))])


def test_unique_legitimate_identifier_is_low_severity():
    train = with_ids(classification_frame(200, seed=1), 0)
    test = with_ids(classification_frame(50, seed=2), 10_000)
    result = audit_train_test(train, test, AuditConfig(target="y"))
    f = finding(result, "ID-001:transaction_id")
    assert (f.severity, f.confidence) == (Severity.LOW, Confidence.MEDIUM)
    assert f.evidence["unique_ratio"] == 1.0
    assert f.evidence["id_name_tokens"] == ["id"]
    assert by_rule(result, "ID-003") == []  # disjoint values
    assert check(result, "ID-003").details["columns_checked"] == ["transaction_id"]


def test_shared_identifier_values_reference_duplicate_rules():
    train = with_ids(classification_frame(200, seed=3), 0)
    test = with_ids(classification_frame(50, seed=4), 190)  # T000190..T000199 shared
    result = audit_train_test(train, test, AuditConfig(target="y"))
    f = finding(result, "ID-003:transaction_id")
    assert f.evidence["n_shared_values"] == 10
    assert f.evidence["n_test_rows_with_shared_value"] == 10
    assert f.evidence["related_rules"] == ["DUP-002", "DUP-004"]
    assert f.confidence is Confidence.LOW


def test_row_index_detection():
    train = classification_frame(100, seed=5)
    train.insert(0, "Unnamed: 0", range(100))
    train["seq"] = np.arange(500, 600)
    result = audit_train_test(train, train.iloc[:30], AuditConfig(target="y"))
    assert finding(result, "ID-002:Unnamed: 0").confidence is Confidence.HIGH
    assert "increase by exactly 1" in finding(result, "ID-002:seq").evidence["reason"]
    assert by_rule(result, "ID-001") == []
    assert sorted(check(result, "ID-003").details["row_index_columns_not_checked"]) == [
        "Unnamed: 0",
        "seq",
    ]


def test_unique_measurements_and_configured_entities_are_not_identifiers():
    train = classification_frame(100, seed=6)
    train["cust"] = [f"c{i}" for i in range(100)]
    result = audit_train_test(train, train.iloc[:30], AuditConfig(target="y", entity_id="cust"))
    assert by_rule(result, "ID-001") == []  # x1/x2 are unique floats; cust is metadata


def test_text_identifier_without_id_name_is_low_confidence():
    train = classification_frame(100, seed=7).assign(ref=[f"r{i}" for i in range(100)])
    result = audit_train_test(train, train.iloc[:10].assign(ref="new"), AuditConfig(target="y"))
    assert finding(result, "ID-001:ref").confidence is Confidence.LOW


def test_identifier_checks_skip_tiny_datasets():
    train = classification_frame(10, seed=8)
    result = audit_train_test(train, train, AuditConfig(target="y"))
    assert check(result, "ID-001").status.value == "skipped"


# ------------------------------------------------------------------- schema


def test_schema_mismatch_findings():
    train = classification_frame(100, seed=9).assign(income=1.5)
    test = classification_frame(30, seed=10).drop(columns=["x2"]).assign(promo="A", income="n/a")
    result = audit_train_test(train, test, AuditConfig(target="y"))
    assert finding(result, "SCH-001:test").evidence["missing_from_test"] == ["x2"]
    assert finding(result, "SCH-002:test").evidence["extra_in_test"] == ["promo"]
    assert finding(result, "SCH-003:income").evidence == {"train": "numeric", "test": "text"}
    assert check(result, "DUP-002").details["columns_not_compared"] == ["promo", "x2"]


def test_target_absent_from_test():
    train, test = (
        classification_frame(100, seed=11),
        classification_frame(30, seed=12).drop(columns=["y"]),
    )
    result = audit_train_test(train, test, AuditConfig(target="y"))
    assert finding(result, "SCH-004:y").evidence == {"present": False, "all_missing": False}
    quiet = audit_train_test(train, test, AuditConfig(target="y", expect_target_in_test=False))
    assert by_rule(quiet, "SCH-004") == []


def test_unseen_test_classes():
    train = classification_frame(100, seed=13)
    test = classification_frame(30, seed=14)
    test.loc[:3, "y"] = 7
    f = finding(audit_train_test(train, test, AuditConfig(target="y")), "SCH-005:y")
    assert f.evidence["n_unseen_classes"] == 1
    assert f.evidence["n_test_rows_with_unseen_class"] == 4


def test_mostly_numeric_text_column(tmp_path):
    train = classification_frame(100, seed=15)
    train["amount"] = [str(i * 1.5) for i in range(97)] + ["N/A", "unknown", "-"]
    path = tmp_path / "train.csv"
    train.to_csv(path, index=False)
    result = audit_train_test(path, train.iloc[:20], AuditConfig(target="y"))
    f = finding(result, "SCH-007:amount")
    # "N/A" is one of pandas' default missing-value markers, so only 2 values are text.
    assert f.evidence["n_non_numeric_values"] == 2


def test_constant_and_empty_columns():
    train = classification_frame(100, seed=16).assign(k="same", empty=np.nan)
    f = finding(audit_train_test(train, train.iloc[:20], AuditConfig(target="y")), "SCH-006:train")
    assert f.evidence["constant"] == ["k"]
    assert f.evidence["all_missing"] == ["empty"]
    assert f.severity is Severity.INFO
