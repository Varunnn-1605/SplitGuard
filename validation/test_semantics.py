"""Rule-semantics audit: properties that must hold for every finding produced by
every adversarial case, plus safety properties (limits, immutability, errors)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import build_cases, clf  # noqa: E402
from splitguard import (  # noqa: E402
    AuditConfig,
    InputValidationError,
    Limits,
    audit_split_column,
    audit_train_test,
    cli,  # noqa: E402
    inspect_unsplit,
)
from splitguard.reporting import report_to_json  # noqa: E402

HEURISTIC_RULES = {"TGT-005", "TGT-006", "TMP-005", "ID-001"}
SAFETY_CLAIMS = (
    "is safe",
    "leakage-free",
    "leak-free",
    "no leakage",
    "confirmed leakage",
    "proves",
)


@pytest.fixture(scope="module")
def results():
    out = []
    for case in build_cases():
        outcome = case.run()
        if outcome.result is not None:
            out.append((case, outcome.result))
    assert len(out) >= 35
    return out


def all_findings(results):
    return [(case, result, f) for case, result in results for f in result.findings]


def test_correlation_is_never_proof(results):
    for case, _, f in all_findings(results):
        if f.rule_id == "TGT-005":
            assert (f.severity.value, f.confidence.value) == ("medium", "low"), case.case_id
            assert "not proof" in f.title
            assert any("not leakage" in lim for lim in f.limitations)


def test_heuristics_never_have_high_confidence(results):
    for case, _, f in all_findings(results):
        if f.rule_id in HEURISTIC_RULES:
            assert f.confidence.value != "high", (case.case_id, f.finding_id)


def test_unique_ids_alone_are_never_severe(results):
    for case, _, f in all_findings(results):
        if f.rule_id in {"ID-001", "ID-002"}:
            assert f.severity.value == "low", (case.case_id, f.finding_id)
        if f.rule_id == "ID-003":
            assert f.confidence.value == "low"
            assert f.evidence["related_rules"], "overlap with duplicate rules must be declared"


def test_entity_overlap_follows_declared_scenario(results):
    expected = {
        "new_entities": ("high", "high"),
        "new_entities_forecasting": ("high", "high"),
        "unknown": ("medium", "low"),
        "independent": ("info", "high"),
        "forecasting": ("info", "high"),
    }
    seen = set()
    for case, result, f in all_findings(results):
        if f.rule_id == "DUP-003":
            scenario = result.config["scenario"]
            assert (f.severity.value, f.confidence.value) == expected[scenario], case.case_id
            seen.add(scenario)
    assert {"new_entities", "independent", "forecasting"} <= seen


def test_temporal_safety_never_asserted_without_metadata(results):
    for case, result in results:
        cfg = result.config
        checks = {c.check_id: c for c in result.checks}
        if not cfg["prediction_timestamp"]:
            for rule in ("TMP-001", "TMP-003", "TMP-004"):
                assert checks[rule].status.value == "skipped", case.case_id
            assert any("not assessed" in w and "not evidence" in w for w in result.warnings)
        if cfg["prediction_timestamp"] and not cfg["feature_availability"]:
            assert checks["TMP-001"].status.value == "skipped"


def test_skips_and_failures_always_explain_themselves(results):
    for case, result in results:
        for c in result.checks:
            if c.status.value != "run":
                assert c.reason and len(c.reason) > 10, (case.case_id, c.check_id)


def test_target_and_metadata_are_never_features(results):
    for case, result in results:
        cfg = result.config
        metadata = {
            cfg["target"],
            cfg["entity_id"],
            cfg["prediction_timestamp"],
            cfg["outcome_timestamp"],
        }
        metadata |= set(cfg["feature_availability"].values()) | set(cfg["exclude_columns"])
        metadata.discard(None)
        assert not metadata & set(result.feature_columns), case.case_id
        for f in result.findings:
            if f.rule_id in {
                "TGT-001",
                "TGT-002",
                "TGT-003",
                "TGT-004",
                "TGT-005",
                "ID-001",
                "ID-002",
            }:
                assert cfg["target"] not in f.affected_columns, (case.case_id, f.finding_id)


def test_findings_carry_explanatory_evidence(results):
    for case, _, f in all_findings(results):
        assert isinstance(f.evidence, dict) and f.evidence, (case.case_id, f.finding_id)
        assert f.limitations and f.recommended_action and f.description
        text = json.dumps(f.to_dict()["evidence"], allow_nan=False)
        if f.rule_id.startswith("TMP-00") and f.rule_id != "TMP-005":
            assert "comparison_rule" in f.evidence
        if f.rule_id in {"DUP-002", "DUP-004"}:
            assert "comparison" in f.evidence
        if f.rule_id == "TGT-005":
            assert {"statistic", "value", "threshold", "n_pairs"} <= set(f.evidence)
        assert "NaN" not in text


def test_severity_and_confidence_vary_independently(results):
    pairs = {(f.severity.value, f.confidence.value) for _, _, f in all_findings(results)}
    assert any(s in {"critical", "high", "medium"} and c == "low" for s, c in pairs)
    assert any(s in {"low", "info"} and c == "high" for s, c in pairs)


def test_no_finding_or_warning_claims_safety_or_proof(results):
    for case, result in results:
        texts = [w for w in result.warnings] + [
            f.title + " " + f.description for f in result.findings
        ]
        for text in texts:
            lowered = text.lower()
            assert not any(claim in lowered for claim in SAFETY_CLAIMS), (case.case_id, text)


def test_overlap_is_declared_not_counted_as_independent(results):
    for _case, result in results:
        assert any("not independent confirmations" in lim for lim in result.limitations)
        ids = {f.rule_id for f in result.findings}
        if {"DUP-002", "DUP-004"} <= ids:
            dup4 = next(f for f in result.findings if f.rule_id == "DUP-004")
            assert "n_also_exact_full_row_matches" in dup4.evidence


def test_reports_are_strict_json_and_deterministic(results):
    for case, result in results[:10]:
        a = report_to_json(result, include_generated_at=False)
        json.loads(a)
        b = report_to_json(case.run().result, include_generated_at=False)
        assert a == b, case.case_id


# --------------------------------------------------------------- safety properties


def test_limits_hold_for_files_and_frames(tmp_path):
    df = clf(500, 1)
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    for source in (path, df, path.read_bytes()):
        with pytest.raises(InputValidationError) as exc:
            audit_train_test(source, df.iloc[:10], AuditConfig(limits=Limits(max_rows=100)))
        assert {i.code for i in exc.value.issues} == {"ROW_LIMIT_EXCEEDED"}
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(path, df, AuditConfig(limits=Limits(max_file_bytes=1024)))
    assert {i.code for i in exc.value.issues} == {"FILE_TOO_LARGE"}


def test_inputs_are_never_modified_in_any_mode():
    df = clf(200, 2)
    df["when"] = pd.date_range("2024-01-01", periods=200, freq="D").strftime("%Y-%m-%d")
    df["split"] = np.where(np.arange(200) < 160, "train", "test")
    snapshot = df.copy(deep=True)
    cfg = AuditConfig(target="y", prediction_timestamp="when")
    audit_train_test(df.drop(columns="split"), df.drop(columns="split"), cfg)
    audit_split_column(df, "split", config=cfg)
    inspect_unsplit(df.drop(columns="split"), cfg)
    pd.testing.assert_frame_equal(df, snapshot)


def test_errors_never_echo_row_contents(tmp_path, capsys):
    df = clf(100, 3)
    df["secret"] = [f"SSN-{i:09d}" for i in range(100)]
    df["split"] = ["train"] * 70 + ["holdout"] * 10 + ["test"] * 20
    df.loc[0, "when"] = "not-a-date SSN-999999999"
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    code = cli.main(
        ["audit", "--data", str(path), "--split-column", "split", "--timestamp", "when"]
    )
    err = capsys.readouterr().err
    assert code == cli.EXIT_INVALID
    assert "SSN-" not in err


def test_partial_assessment_is_always_stated(results):
    for case, result in results:
        tmp1 = next(c for c in result.checks if c.check_id == "TMP-001")
        if tmp1.status.value != "run":
            continue
        for feature, info in tmp1.details.items():
            missing = sum(p.get("n_not_compared_missing", 0) for p in info["partitions"].values())
            equal = sum(p.get("n_equal", 0) for p in info["partitions"].values())
            if missing:
                assert any(
                    "could not be compared" in w and feature in w for w in result.warnings
                ), case.case_id
            if equal:
                assert any("exactly equal" in w for w in result.warnings), case.case_id
            assert "boundary_rule" in info
