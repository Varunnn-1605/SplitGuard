from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import by_rule, check, classification_frame, finding
from splitguard import AuditConfig, Confidence, Severity, audit_train_test, inspect_unsplit


def test_within_partition_duplicates_exact_counts():
    base = classification_frame(40, seed=3)
    train = pd.concat([base, base.iloc[[0, 0, 1]]], ignore_index=True)
    result = audit_train_test(train, classification_frame(30, seed=4), AuditConfig(target="y"))
    f = finding(result, "DUP-001:train")
    assert f.evidence["n_rows"] == 43
    assert f.evidence["n_duplicate_rows_beyond_first"] == 3
    assert f.evidence["n_duplicate_groups"] == 2
    assert f.evidence["n_rows_in_duplicate_groups"] == 5
    assert f.evidence["largest_group_size"] == 3
    assert f.severity is Severity.MEDIUM  # 3/43 >= 5%
    assert f.confidence is Confidence.HIGH
    assert by_rule(result, "DUP-002") == []  # within-partition only
    assert result.partitions["train"]["n_rows"] == 43  # nothing dropped


def test_cross_split_exact_rows():
    train = classification_frame(100, seed=5)
    test = pd.concat(
        [classification_frame(16, seed=6), train.iloc[[10, 11, 12, 12]]], ignore_index=True
    )
    result = audit_train_test(train, test, AuditConfig(target="y"))
    f = finding(result, "DUP-002:train_test")
    assert f.evidence["n_test_rows_also_in_train"] == 4
    assert f.evidence["test_rate"] == 0.2
    assert f.evidence["n_distinct_shared_rows"] == 3
    assert f.evidence["n_train_rows_also_in_test"] == 3
    assert f.severity is Severity.CRITICAL
    assert finding(result, "DUP-001:test").evidence["n_duplicate_rows_beyond_first"] == 1
    # Feature-row matches are all full-row matches, so DUP-004 adds no separate finding.
    assert by_rule(result, "DUP-004") == []
    assert check(result, "DUP-004").details["n_feature_only_matches"] == 0


def test_column_order_does_not_matter():
    train = classification_frame(100, seed=5)
    test = pd.concat([classification_frame(16, seed=6), train.iloc[[10, 11]]], ignore_index=True)
    shuffled = test[list(reversed(test.columns))]
    a = audit_train_test(train, test, AuditConfig(target="y"))
    b = audit_train_test(train, shuffled, AuditConfig(target="y"))
    assert finding(a, "DUP-002:train_test").evidence["n_test_rows_also_in_train"] == 2
    assert finding(a, "DUP-002:train_test").evidence == finding(b, "DUP-002:train_test").evidence


def test_int_and_float_compare_by_value_text_is_case_sensitive():
    train = pd.DataFrame({"a": np.arange(30), "b": [f"s{i}" for i in range(30)], "y": [0, 1] * 15})
    test = pd.DataFrame({"a": [2.0, np.nan, 2.0], "b": ["s2", "s5", "S2"], "y": [0, 1, 0]})
    result = audit_train_test(train, test, AuditConfig(target="y"))
    assert finding(result, "DUP-002:train_test").evidence["n_test_rows_also_in_train"] == 1


def test_missing_values_compare_equal():
    train = pd.DataFrame({"a": [1.0, np.nan] * 20, "b": list("pq") * 20, "y": [0, 1] * 20})
    train["c"] = np.arange(40.5, 80.5)
    test = pd.DataFrame({"a": [np.nan], "b": ["q"], "y": [1], "c": [41.5]})
    result = audit_train_test(train, test, AuditConfig(target="y"))
    assert finding(result, "DUP-002:train_test").evidence["n_test_rows_also_in_train"] == 1


def _entity_frames():
    train = classification_frame(60, seed=7)
    train["cust"] = [f"c{i % 20}" for i in range(60)]
    test = classification_frame(30, seed=8)
    test["cust"] = [f"c{15 + i % 10}" for i in range(30)]
    return train, test


def test_entity_overlap_counts_and_scenario_dependent_severity():
    train, test = _entity_frames()
    expectations = {
        "new_entities": (Severity.HIGH, Confidence.HIGH),
        "unknown": (Severity.MEDIUM, Confidence.LOW),
        "independent": (Severity.INFO, Confidence.HIGH),
    }
    for scenario, (severity, confidence) in expectations.items():
        result = audit_train_test(
            train, test, AuditConfig(target="y", entity_id="cust", scenario=scenario)
        )
        f = finding(result, "DUP-003:cust")
        assert (f.severity, f.confidence) == (severity, confidence), scenario
        assert f.evidence["n_shared_entities"] == 5
        assert f.evidence["n_test_entities"] == 10
        assert f.evidence["shared_rate_of_test_entities"] == 0.5
        assert f.evidence["n_test_rows_with_shared_entity"] == 15
        assert "c15" not in str(f.to_dict())  # IDs themselves are not reported


def test_disjoint_entities_produce_no_finding():
    train, test = _entity_frames()
    test["cust"] = [f"d{i}" for i in range(30)]
    result = audit_train_test(
        train, test, AuditConfig(target="y", entity_id="cust", scenario="new_entities")
    )
    assert by_rule(result, "DUP-003") == []
    assert check(result, "DUP-003").details["n_shared_entities"] == 0


def test_entity_check_skipped_without_entity_id(clean_pair):
    result = audit_train_test(*clean_pair, AuditConfig(target="y"))
    assert check(result, "DUP-003").status.value == "skipped"


def test_identical_features_with_different_target():
    train = classification_frame(100, seed=9)
    copies = train.iloc[[1, 2, 3]].copy()
    copies["y"] = 1 - copies["y"]
    test = pd.concat([classification_frame(20, seed=10), copies], ignore_index=True)
    result = audit_train_test(train, test, AuditConfig(target="y"))
    assert by_rule(result, "DUP-002") == []
    f = finding(result, "DUP-004:train_test")
    assert f.evidence["n_test_rows_matching_train_features"] == 3
    assert f.evidence["n_feature_only_matches"] == 3
    assert f.evidence["n_matches_with_only_different_target_in_train"] == 3
    assert (f.severity, f.confidence) == (Severity.HIGH, Confidence.MEDIUM)
    assert "y" not in f.affected_columns


def test_small_feature_space_is_low_confidence():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"f": [0, 1] * 50, "g": list("abcd") * 25, "y": rng.integers(0, 2, 100)})
    test = pd.DataFrame({"f": [0, 1] * 10, "g": list("abcd") * 5, "y": [2] * 20})
    result = audit_train_test(train, test, AuditConfig(target="y"))
    f = finding(result, "DUP-004:train_test")
    assert (f.severity, f.confidence) == (Severity.LOW, Confidence.LOW)
    assert f.evidence["train_feature_row_uniqueness"] == 0.04


def test_clean_data_has_no_duplicate_findings(clean_pair):
    result = audit_train_test(*clean_pair, AuditConfig(target="y"))
    assert [f for f in result.findings if f.rule_id.startswith("DUP")] == []


def test_unsplit_mode_skips_cross_partition_checks(clean_pair):
    result = inspect_unsplit(clean_pair[0], AuditConfig(target="y"))
    for rule_id in ("DUP-002", "DUP-003", "DUP-004"):
        c = check(result, rule_id)
        assert c.status.value == "skipped"
        assert "not audited" in c.reason
    assert check(result, "DUP-001").status.value == "run"
