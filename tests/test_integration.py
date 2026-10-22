"""End-to-end tests: CLI, input-mode equivalence, examples and failure isolation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import check, classification_frame, write_csv
from splitguard import AuditConfig, audit_split_column, audit_train_test, cli, inspect_unsplit
from splitguard import audit as audit_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def leaky_files(tmp_path):
    train = classification_frame(200, seed=1)
    train["leak"] = train["y"]
    test = classification_frame(50, seed=2)
    test["leak"] = test["y"]
    return write_csv(tmp_path, "train.csv", train), write_csv(tmp_path, "test.csv", test)


def test_cli_audit_writes_report(leaky_files, tmp_path, capsys):
    train, test = leaky_files
    out = tmp_path / "report.json"
    code = cli.main(
        ["audit", "--train", str(train), "--test", str(test), "--target", "y", "-o", str(out)]
    )
    assert code == cli.EXIT_OK
    stdout = capsys.readouterr().out
    assert "critical 1" in stdout and "JSON report written to" in stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["by_severity"]["critical"] == 1


def test_cli_fail_on(leaky_files):
    train, test = leaky_files
    args = ["audit", "--train", str(train), "--test", str(test), "--target", "y"]
    assert cli.main([*args, "--fail-on", "critical"]) == cli.EXIT_FINDINGS
    clean_args = [
        "audit",
        "--train",
        str(train),
        "--test",
        str(test),
        "--target",
        "y",
        "--exclude",
        "leak",
    ]
    assert cli.main([*clean_args, "--fail-on", "high"]) == cli.EXIT_OK


def test_cli_invalid_inputs(tmp_path, leaky_files, capsys):
    train, test = leaky_files
    assert (
        cli.main(["audit", "--train", str(train), "--test", str(test), "--target", "nope"])
        == cli.EXIT_INVALID
    )
    assert "MISSING_COLUMN" in capsys.readouterr().err
    assert cli.main(["audit", "--train", str(train)]) == cli.EXIT_INVALID
    assert (
        cli.main(["audit", "--train", str(train), "--test", str(test), "--data", str(train)])
        == cli.EXIT_INVALID
    )
    bad_config = tmp_path / "c.json"
    bad_config.write_text('{"tagret": "y"}', encoding="utf-8")
    assert (
        cli.main(["audit", "--train", str(train), "--test", str(test), "--config", str(bad_config)])
        == cli.EXIT_INVALID
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["audit", "--scenario", "bogus"])
    assert exc.value.code == 2


def test_cli_config_file_and_flags(tmp_path, leaky_files):
    train, test = leaky_files
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"target": "y", "thresholds": {"strong_auc": 0.6}}), encoding="utf-8"
    )
    out = tmp_path / "r.json"
    cli.main(
        [
            "audit",
            "--train",
            str(train),
            "--test",
            str(test),
            "--config",
            str(config),
            "-o",
            str(out),
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["configuration"]["target"] == "y"
    assert report["configuration"]["thresholds"]["strong_auc"] == 0.6


def test_cli_split_column_and_inspect_and_split(tmp_path, capsys):
    df = classification_frame(150, seed=3)
    df["split"] = ["train"] * 120 + ["test"] * 30
    data = write_csv(tmp_path, "data.csv", df)
    assert (
        cli.main(["audit", "--data", str(data), "--split-column", "split", "--target", "y"])
        == cli.EXIT_OK
    )
    unsplit = write_csv(tmp_path, "unsplit.csv", df.drop(columns=["split"]))
    assert cli.main(["inspect", "--data", str(unsplit), "--target", "y"]) == cli.EXIT_OK
    assert "needs user input" in capsys.readouterr().out
    out_dir = tmp_path / "out"
    base = [
        "split",
        "--data",
        str(unsplit),
        "--target",
        "y",
        "--scenario",
        "independent",
        "--out-dir",
        str(out_dir),
    ]
    assert (
        cli.main([*base, "--confirm-strategy", "random"]) == cli.EXIT_INVALID
    )  # wrong confirmation
    assert not (out_dir / "train.csv").exists()
    assert cli.main([*base, "--confirm-strategy", "stratified_random"]) == cli.EXIT_OK
    assert len(pd.read_csv(out_dir / "train.csv")) + len(pd.read_csv(out_dir / "test.csv")) == 150
    assert (
        cli.main([*base, "--confirm-strategy", "stratified_random"]) == cli.EXIT_INVALID
    )  # no overwrite
    no_scenario = [
        "split",
        "--data",
        str(unsplit),
        "--target",
        "y",
        "--out-dir",
        str(out_dir),
        "--confirm-strategy",
        "x",
    ]
    assert cli.main(no_scenario) == cli.EXIT_INVALID


def test_split_column_mode_matches_separate_files():
    train, test = classification_frame(120, seed=4), classification_frame(30, seed=5)
    test = pd.concat([test, train.iloc[:3]], ignore_index=True)
    combined = pd.concat(
        [train.assign(split="train"), test.assign(split="test")], ignore_index=True
    )
    a = audit_train_test(train, test, AuditConfig(target="y"))
    b = audit_split_column(combined, "split", config=AuditConfig(target="y"))
    assert [f.to_dict() for f in a.findings] == [f.to_dict() for f in b.findings]
    assert b.mode == "split_column"


def test_detector_failure_is_isolated(monkeypatch, clean_pair):
    def boom(ctx):
        raise RuntimeError("secret value 12345")

    detectors = [
        (n, boom if n == "temporal" else fn, rules) for n, fn, rules in audit_module._DETECTORS
    ]
    monkeypatch.setattr(audit_module, "_DETECTORS", detectors)
    result = audit_train_test(*clean_pair, AuditConfig(target="y"))
    failed = check(result, "TMP-001")
    assert failed.status.value == "failed"
    assert "RuntimeError" in failed.reason and "12345" not in failed.reason
    assert check(result, "DUP-001").status.value == "run"


def test_cli_exit_code_for_failed_checks(monkeypatch, leaky_files):
    def boom(ctx):
        raise RuntimeError("x")

    detectors = [
        (n, boom if n == "schema" else fn, rules) for n, fn, rules in audit_module._DETECTORS
    ]
    monkeypatch.setattr(audit_module, "_DETECTORS", detectors)
    train, test = leaky_files
    assert cli.main(["audit", "--train", str(train), "--test", str(test)]) == cli.EXIT_FAILED


def test_module_entry_point():
    out = subprocess.run(
        [sys.executable, "-m", "splitguard", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "splitguard 0.2.0"


def test_synthetic_examples_end_to_end(tmp_path):
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        import generate_synthetic_data as gen
    finally:
        sys.path.pop(0)
    gen.generate(tmp_path, seed=7)
    d = tmp_path

    def ids(name, **cfg):
        cfg.setdefault("target", "churn")
        result = audit_train_test(d / name / "train.csv", d / name / "test.csv", AuditConfig(**cfg))
        return {f.finding_id for f in result.findings}, result

    assert ids("01_clean")[0] == set()
    copy_ids, _ = ids("02_target_copy")
    assert "TGT-001:churn_score_final" in copy_ids
    legit_ids, legit = ids("03_strong_legit", target="price")
    assert legit_ids == {"TGT-005:sqft"}
    assert next(iter(legit.findings)).confidence.value == "low"
    assert "DUP-002:train_test" in ids("04_cross_duplicates")[0]
    assert (
        "DUP-003:customer_id"
        in ids("05_repeated_entities", entity_id="customer_id", scenario="new_entities")[0]
    )
    temporal = dict(
        prediction_timestamp="prediction_time",
        outcome_timestamp="outcome_time",
        scenario="forecasting",
    )
    post_ids, _ = ids(
        "06_post_outcome_feature",
        feature_availability={
            "refund_amount": "refund_amount_available_at",
            "tenure_months": "tenure_available_at",
        },
        **temporal,
    )
    assert {"TMP-001:refund_amount", "TMP-002:refund_amount"} <= post_ids
    assert not any("tenure" in i for i in post_ids)
    clean_ids, _ = ids(
        "07_temporal_clean",
        feature_availability={"tenure_months": "tenure_available_at"},
        **temporal,
    )
    assert clean_ids == set()
    assert ids("08_high_cardinality_id")[0] == {"ID-001:transaction_id"}
    assert {"SCH-001:test", "SCH-002:test", "SCH-003:x2"} <= ids("09_schema_mismatch")[0]
    unsplit = inspect_unsplit(
        d / "10_unsplit" / "data.csv", AuditConfig(target="churn", entity_id="patient_id")
    )
    assert unsplit.split_recommendation.status == "needs_user_input"
