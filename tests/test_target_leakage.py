from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import by_rule, check, classification_frame, finding, regression_frame
from splitguard import AuditConfig, Confidence, Severity, audit_train_test

STRONG = ("TGT-001", "TGT-002", "TGT-003", "TGT-004")


def strong_ids(result, column: str) -> list[str]:
    return [
        f.rule_id for f in result.findings if f.rule_id in STRONG and column in f.affected_columns
    ]


def run(train, test=None, **cfg):
    test = test if test is not None else train.iloc[:40]
    return audit_train_test(train, test, AuditConfig(target="y", **cfg))


def test_exact_numeric_copy():
    train = classification_frame(200, seed=11)
    train["leak"] = train["y"].astype(float)
    result = run(train, classification_frame(50, seed=12).assign(leak=0.0))
    f = finding(result, "TGT-001:leak")
    assert (f.severity, f.confidence) == (Severity.CRITICAL, Confidence.HIGH)
    assert f.evidence["comparison"] == "exact"
    assert f.evidence["n_pairs"] == 200
    assert f.evidence["exact_match_rate"] == 1.0
    assert f.evidence["partition"] == "train"
    assert all(x.key != "leak" for x in by_rule(result, "TGT-005"))  # reported once, as a copy


def test_copy_after_text_normalisation():
    train = classification_frame(200, seed=13)
    train["y"] = np.where(train["y"] == 1, "yes", "no")
    train["leak"] = np.where(train["y"] == "yes", "Yes ", "NO")
    f = finding(run(train), "TGT-001:leak")
    assert f.evidence["exact_match_rate"] == 0.0
    assert f.evidence["comparison"].startswith("normalised")


def test_regression_copy_within_tolerance():
    train = regression_frame(200, seed=14)
    train["leak"] = train["y"] * (1 + 1e-9)
    f = finding(run(train), "TGT-001:leak")
    assert f.evidence["comparison"] == "numeric, within tolerance"
    assert f.evidence["numeric_close_rate"] == 1.0


def test_copy_with_missing_values_uses_pairs():
    train = classification_frame(200, seed=15)
    train["leak"] = train["y"].astype(float)
    train.loc[:19, "leak"] = np.nan
    f = finding(run(train), "TGT-001:leak")
    assert f.evidence["n_pairs"] == 180


def test_near_copy_with_kappa():
    rng = np.random.default_rng(16)
    y = np.array([0, 1] * 100)
    leak = y.copy()
    leak[:2] = 1 - leak[:2]
    train = pd.DataFrame({"x1": rng.normal(size=200), "leak": leak, "y": y})
    f = finding(run(train), "TGT-002:leak")
    assert (f.severity, f.confidence) == (Severity.HIGH, Confidence.MEDIUM)
    assert f.evidence["match_rate"] == 0.99
    assert f.evidence["cohens_kappa"] == pytest.approx(0.98)


def test_chance_agreement_on_imbalanced_target_is_not_a_near_copy():
    rng = np.random.default_rng(17)
    y = np.zeros(200, dtype=int)
    y[[0, 1]] = 1
    leak = np.zeros(200, dtype=int)
    leak[5] = 1  # 98.5% raw agreement, but worse than chance
    train = pd.DataFrame({"x1": rng.normal(size=200), "leak": leak, "y": y})
    result = run(train)
    assert strong_ids(result, "leak") == []
    assert all(f.key != "leak" for f in by_rule(result, "TGT-005"))


def test_one_to_one_relabelling():
    train = classification_frame(200, seed=18)
    train["labelled"] = np.where(train["y"] == 1, "pos", "neg")
    train["inverted"] = 1 - train["y"]
    result = run(train)
    for column in ("labelled", "inverted"):
        f = finding(result, f"TGT-003:{column}")
        assert f.evidence["relationship"] == "one-to-one relabelling"
        assert f.severity is Severity.CRITICAL


def test_affine_transform_of_regression_target():
    train = regression_frame(200, seed=19)
    train["cents"] = train["y"] * 100 + 5
    f = finding(run(train), "TGT-003:cents")
    assert f.evidence["relationship"].startswith("exact affine")
    assert f.evidence["slope"] == pytest.approx(0.01)
    assert f.evidence["intercept"] == pytest.approx(-0.05)


def test_monotonic_transform_is_high_not_critical():
    train = regression_frame(200, seed=20)
    train["expy"] = np.exp(train["y"] / 10)
    f = finding(run(train), "TGT-003:expy")
    assert f.evidence["relationship"].startswith("strictly monotonic")
    assert (f.severity, f.confidence) == (Severity.HIGH, Confidence.MEDIUM)


def test_missingness_determines_binary_target():
    train = classification_frame(200, seed=21)
    train["followup"] = np.random.default_rng(0).normal(size=200)
    train.loc[train["y"] == 1, "followup"] = np.nan
    f = finding(run(train), "TGT-004:followup")
    assert f.evidence["n_missing"] == int((train["y"] == 1).sum())
    assert f.evidence["n_rows"] == 200


def test_strong_legitimate_regression_feature_is_only_a_review_signal():
    train = regression_frame(300, seed=22, noise=0.3)
    result = run(train)
    assert strong_ids(result, "x") == []
    f = finding(result, "TGT-005:x")
    assert (f.severity, f.confidence) == (Severity.MEDIUM, Confidence.LOW)
    assert "not proof" in f.title
    assert 0.9 <= f.evidence["value"] < 1.0
    assert f.evidence["statistic"] == "max(|pearson|, |spearman|)"
    assert any("Correlation is not leakage" in text for text in f.limitations)


def test_strong_legitimate_classification_feature_uses_auc():
    rng = np.random.default_rng(23)
    x = rng.normal(size=300)
    train = pd.DataFrame(
        {
            "x": x,
            "z": rng.normal(size=300),
            "y": (x + rng.normal(scale=0.1, size=300) > 0).astype(int),
        }
    )
    result = run(train)
    assert strong_ids(result, "x") == []
    f = finding(result, "TGT-005:x")
    assert f.evidence["statistic"].startswith("single-feature ROC AUC")
    assert f.evidence["value"] >= 0.95
    assert by_rule(result, "TGT-005") == [f]  # z is noise


def test_categorical_association_uses_cramers_v():
    rng = np.random.default_rng(24)
    city = rng.choice(list("ABCD"), size=400)
    y = np.isin(city, ["A", "B"]).astype(int)
    flip = rng.random(400) < 0.02
    y[flip] = 1 - y[flip]
    train = pd.DataFrame({"city": city, "x": rng.normal(size=400), "y": y})
    f = finding(run(train), "TGT-005:city")
    assert f.evidence["statistic"] == "bias-corrected Cramer's V"
    assert f.evidence["value"] >= 0.9


def test_weak_features_produce_no_target_findings(clean_pair):
    result = audit_train_test(*clean_pair, AuditConfig(target="y"))
    assert [f for f in result.findings if f.rule_id.startswith("TGT")] == []
    details = check(result, "TGT-005").details
    assert details["n_features_evaluated"] == 3


def test_high_cardinality_text_is_skipped_for_association():
    train = classification_frame(100, seed=25)
    train["note"] = [f"free text {i}" for i in range(100)]
    details = check(run(train), "TGT-005").details
    assert "note" in details["skipped"]


def test_too_few_rows_skips_target_checks():
    train = classification_frame(20, seed=26)
    train["leak"] = train["y"]
    result = run(train, test=train)
    assert strong_ids(result, "leak") == []
    assert "leak" in check(result, "TGT-001").details["features_with_too_few_pairs"]


def test_constant_feature_is_not_association(clean_pair):
    train, test = clean_pair
    train = train.assign(const=5)
    result = audit_train_test(train, test.assign(const=5), AuditConfig(target="y"))
    assert all(f.key != "const" for f in result.findings if f.rule_id.startswith("TGT"))
    assert finding(result, "SCH-006:train").evidence["constant"] == ["const"]


def test_target_like_names_are_low_confidence():
    train = classification_frame(200, seed=27).rename(columns={"y": "churn"})
    rng = np.random.default_rng(0)
    train["churn_flag_old"] = rng.normal(size=200)
    train["label_guess"] = rng.normal(size=200)
    result = audit_train_test(train, train.iloc[:50], AuditConfig(target="churn"))
    for column in ("churn_flag_old", "label_guess"):
        f = finding(result, f"TGT-006:{column}")
        assert (f.severity, f.confidence) == (Severity.LOW, Confidence.LOW)


def test_no_target_skips_target_checks(clean_pair):
    result = audit_train_test(*clean_pair)
    for rule_id in ("TGT-001", "TGT-005", "TGT-006"):
        assert check(result, rule_id).status.value == "skipped"
    assert any("target leakage checks were not run" in w for w in result.warnings)


def test_constant_target_is_reported_and_skips_scans(clean_pair):
    train, test = clean_pair
    result = audit_train_test(train.assign(y=0), test, AuditConfig(target="y"))
    assert finding(result, "SCH-008:y").evidence["n_distinct_values"] == 1
    assert check(result, "TGT-001").status.value == "skipped"


def test_test_partition_diagnostics_are_opt_in_and_labelled(clean_pair):
    train, test = clean_pair
    test = test.assign(leak=test["y"])
    train = train.assign(leak=np.random.default_rng(0).integers(0, 2, len(train)))
    plain = audit_train_test(train, test, AuditConfig(target="y"))
    assert by_rule(plain, "TGT-001") == []
    diag = audit_train_test(train, test, AuditConfig(target="y", test_target_diagnostics=True))
    f = finding(diag, "TGT-001:test:leak")
    assert f.evidence["diagnostic_only"] is True
    assert f.evidence["partition"] == "test"
    assert f.title.startswith("[test diagnostic]")


def test_mixed_type_object_column_does_not_crash():
    train = classification_frame(60, seed=28)
    train["mixed"] = pd.Series([1, "a", 2.5, None] * 15, dtype=object)
    result = run(train)
    assert result.summary()["checks_failed"] == 0
