"""Tests for the Streamlit-free UI adapter, sample data and launcher.

These run without Streamlit installed.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import classification_frame
from splitguard import (
    AuditConfig,
    ConfigError,
    InputValidationError,
    Limits,
    audit_split_column,
    audit_train_test,
    inspect_unsplit,
    recommend_split,
)
from splitguard import ui_adapter as ua
from splitguard.reporting import report_to_json
from splitguard.splitting import SplitConfirmationError
from splitguard.ui_launcher import main as launcher_main
from splitguard.ui_launcher import streamlit_args
from splitguard.ui_samples import samples


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def loaded(df: pd.DataFrame, name: str = "f.csv") -> ua.LoadedFile:
    return ua.load_file(name, csv_bytes(df))


# ------------------------------------------------------------------- loading


def test_load_file_validates_and_keeps_bytes():
    df = classification_frame(50, 1)
    f = ua.load_file("C:\\Users\\someone\\secret dir\\train.csv", csv_bytes(df))
    assert f.name == "train.csv"  # browser paths reduced to a bare name
    assert f.columns == ["x1", "x2", "color", "y"]
    assert f.data == csv_bytes(df)
    assert f.table.metadata["source_type"] == "csv_bytes"


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"", "EMPTY_FILE"),
        (b"a,b\n", "EMPTY_DATASET"),
        (b"a,b\n1,2\n3,4,5\n", "MALFORMED_CSV"),
        (b"a,a\n1,2\n", "DUPLICATE_COLUMN_NAMES"),
        ("x\ncaf\xe9\n".encode("latin-1"), "UNREADABLE_CSV"),
    ],
)
def test_load_file_rejects_bad_input(data, code):
    with pytest.raises(InputValidationError) as exc:
        ua.load_file("bad.csv", data)
    assert {i.code for i in exc.value.issues} == {code}


def test_load_file_enforces_engine_limits():
    data = csv_bytes(classification_frame(200, 2))
    with pytest.raises(InputValidationError) as exc:
        ua.load_file("t.csv", data, Limits(max_rows=100))
    assert {i.code for i in exc.value.issues} == {"ROW_LIMIT_EXCEEDED"}
    with pytest.raises(InputValidationError) as exc:
        ua.load_file("t.csv", data, Limits(max_file_bytes=100))
    assert {i.code for i in exc.value.issues} == {"FILE_TOO_LARGE"}


def test_safe_preview_truncates_and_does_not_mutate():
    df = pd.DataFrame({"t": ["x" * 200, None], "n": [1.5, 2.0]})
    before = df.copy(deep=True)
    preview = ua.safe_preview(df, n=1, max_chars=10)
    assert preview["t"].tolist() == ["xxxxxxx..."]
    assert len(preview) == 1
    pd.testing.assert_frame_equal(df, before)


def test_split_labels_and_no_guessing():
    df = pd.DataFrame({"s": ["Train", "TEST", "Train"], "many": range(3)})
    assert ua.split_labels(df, "s") == ["TEST", "Train"]
    assert ua.split_labels(pd.DataFrame({"m": range(50)}), "m") is None
    assert ua.default_split_label(["TEST", "Train"], "train") == "Train"
    assert ua.default_split_label(["holdout", "fit"], "test") is None


# ------------------------------------------------------------- configuration


def test_build_config_maps_to_real_schema():
    state = ua.FormState(
        mode="train_test",
        target="y",
        task_type="classification",
        scenario="new_entities",
        entity_id="cust",
        prediction_timestamp="t",
        outcome_timestamp="o",
        feature_availability={"bal": "bal_at", "ignored": ""},
        exclude_columns=["row"],
        min_lead_time=" 7D ",
        test_target_diagnostics=True,
    )
    cfg = ua.build_config(state)
    assert cfg == AuditConfig(
        target="y",
        task_type="classification",
        scenario="new_entities",
        entity_id="cust",
        prediction_timestamp="t",
        outcome_timestamp="o",
        feature_availability={"bal": "bal_at"},
        exclude_columns=("row",),
        min_lead_time="7D",
        test_target_diagnostics=True,
    )


def test_optional_settings_are_optional():
    cfg = ua.build_config(ua.FormState(mode="unsplit", target="", min_lead_time="  "))
    assert cfg == AuditConfig()


def test_invalid_settings_raise_config_error():
    with pytest.raises(ConfigError):
        ua.build_config(ua.FormState(mode="train_test", target="y", entity_id="y"))
    with pytest.raises(ConfigError):
        ua.build_config(ua.FormState(mode="train_test", min_lead_time="soon"))


def test_missing_inputs_per_mode():
    f = loaded(classification_frame(40, 3))
    assert ua.missing_inputs(ua.FormState(mode="train_test"), {"train": f}) == ["the test file"]
    state = ua.FormState(mode="split_column")
    assert ua.missing_inputs(state, {"data": f}) == ["the split column"]
    state.split_column, state.train_value, state.test_value = "s", "a", "a"
    assert ua.missing_inputs(state, {"data": f}) == ["different labels for train and test"]
    assert ua.missing_inputs(ua.FormState(mode="unsplit"), {"data": f}) == []


# ------------------------------------------------------------------ running


def _ids(result):
    return [x.to_dict() for x in result.findings]


def test_run_audit_matches_engine_for_every_mode():
    s = samples()
    tt = s["train_test"]
    files = {r: ua.load_file(tt.names[r], tt.files[r]) for r in ("train", "test")}
    state = ua.FormState(mode="train_test", target="churn")
    via_ui = ua.run_audit(state, files)
    direct = audit_train_test(
        tt.files["train"],
        tt.files["test"],
        AuditConfig(target="churn"),
        train_name="sample_train.csv",
        test_name="sample_test.csv",
    )
    assert _ids(via_ui) == _ids(direct)
    assert via_ui.to_dict(include_generated_at=False) == direct.to_dict(include_generated_at=False)

    sc = s["split_column"]
    data = {"data": ua.load_file(sc.names["data"], sc.files["data"])}
    state = ua.FormState(
        mode="split_column",
        target="churn",
        split_column="split",
        train_value="train",
        test_value="test",
        entity_id="customer_id",
    )
    direct = audit_split_column(
        sc.files["data"],
        "split",
        "train",
        "test",
        AuditConfig(target="churn", entity_id="customer_id"),
        name="sample_with_split.csv",
    )
    assert _ids(ua.run_audit(state, data)) == _ids(direct)

    un = s["unsplit"]
    data = {"data": ua.load_file(un.names["data"], un.files["data"])}
    state = ua.FormState(
        mode="unsplit", target="churn", scenario="new_entities", entity_id="patient_id"
    )
    result = ua.run_audit(state, data)
    direct = inspect_unsplit(
        un.files["data"],
        AuditConfig(target="churn", scenario="new_entities", entity_id="patient_id"),
    )
    assert _ids(result) == _ids(direct)
    assert result.split_recommendation.strategy == "group"


def test_samples_are_deterministic_and_show_planted_issues():
    a, b = samples(), samples()
    assert {k: v.files for k, v in a.items()} == {k: v.files for k, v in b.items()}
    ids = lambda r: {f.finding_id for f in r.findings}  # noqa: E731
    tt = a["train_test"]
    files = {r: ua.load_file(tt.names[r], tt.files[r]) for r in ("train", "test")}
    assert {"TGT-001:churn_score_final", "DUP-002:train_test"} <= ids(
        ua.run_audit(ua.FormState(mode="train_test", target="churn"), files)
    )
    tm = a["temporal"]
    files = {r: ua.load_file(tm.names[r], tm.files[r]) for r in ("train", "test")}
    sug = tm.suggested
    state = ua.FormState(
        mode="train_test",
        target="churn",
        scenario=sug["scenario"],
        prediction_timestamp=sug["prediction_timestamp"],
        outcome_timestamp=sug["outcome_timestamp"],
        feature_availability=dict(sug["feature_availability"]),
    )
    temporal_ids = ids(ua.run_audit(state, files))
    assert {"TMP-001:refund_amount", "TMP-002:refund_amount"} <= temporal_ids
    assert not any("monthly_spend" in i for i in temporal_ids if i.startswith("TMP"))
    un = a["unsplit"]
    rec = ua.run_audit(
        ua.FormState(mode="unsplit", target="churn"),
        {"data": ua.load_file(un.names["data"], un.files["data"])},
    ).split_recommendation
    assert rec.status == "needs_user_input"


def test_describe_error_hides_unexpected_messages():
    err = ua.describe_error(RuntimeError("row value SSN-123"))
    assert "SSN" not in " ".join(err.messages + err.hints)
    assert "RuntimeError" in err.messages[0]
    with pytest.raises(InputValidationError) as exc:
        ua.load_file("x.csv", b"")
    assert ua.describe_error(exc.value).messages == ["[EMPTY_FILE] x.csv: file is empty"]
    assert ua.describe_error(ConfigError("bad")).title == "The settings are not valid"


def test_engine_errors_surface_through_run_audit():
    train = loaded(classification_frame(50, 4), "train.csv")
    test = loaded(pd.DataFrame({"other": range(10), "y": [0, 1] * 5}), "test.csv")
    with pytest.raises(InputValidationError) as exc:
        ua.run_audit(ua.FormState(mode="train_test", target="y"), {"train": train, "test": test})
    assert "INCOMPATIBLE_SCHEMA" in ua.describe_error(exc.value).messages[0]


# ------------------------------------------------------------------ results


@pytest.fixture(scope="module")
def sample_result():
    tt = samples()["train_test"]
    files = {r: ua.load_file(tt.names[r], tt.files[r]) for r in ("train", "test")}
    return ua.run_audit(ua.FormState(mode="train_test", target="churn"), files)


def test_filter_findings(sample_result):
    all_f = sample_result.findings
    assert ua.filter_findings(all_f) == all_f
    crit = ua.filter_findings(all_f, severities=["critical"])
    assert crit and all(f.severity.value == "critical" for f in crit)
    assert ua.filter_findings(all_f, severities=[]) == []
    low = ua.filter_findings(all_f, confidences=["low"])
    assert all(ua.evidence_label(f) == "Review prompt" for f in low)
    cat = ua.filter_findings(all_f, categories=["target_leakage"], confidences=["high"])
    assert [f.finding_id for f in cat] == ["TGT-001:churn_score_final"]


def test_group_warnings_puts_each_warning_in_one_group():
    warnings = [
        "No prediction_timestamp configured: temporal leakage was not assessed. This is not evidence that the data is temporally safe.",
        "TMP-001: 3 row(s) have 'a_at' exactly equal to 't'. They are not flagged...",
        "TMP-004: train and test have a shared boundary timestamp 2024-01-01",
        "TMP-001: 2 row(s) for 'a' could not be compared because 'a_at' or 't' is missing; their timing was not assessed.",
        "DUP-003: 3 train and 1 test rows have a missing entity ID in 'c'; they were not checked for entity overlap.",
        "TGT-001..005: 1 feature(s) in the train partition have too few non-missing pairs",
        "Evaluation scenario not declared: entity-overlap and time-overlap findings are reported at reduced confidence.",
        "Something unexpected",
    ]
    groups = ua.group_warnings(warnings)
    assert sum(len(v) for v in groups.values()) == len(warnings)
    assert len(groups["timestamp_ties"]) == 2
    assert len(groups["coverage"]) == 3
    assert len(groups["missing_metadata"]) == 2
    assert groups["other"] == ["Something unexpected"]


def test_real_engine_warnings_are_grouped(sample_result):
    groups = ua.group_warnings(sample_result.warnings)
    assert groups["missing_metadata"]
    assert groups["other"] == []


def test_headline_never_claims_safety():
    clean = audit_train_test(
        classification_frame(200, 5), classification_frame(50, 6), AuditConfig(target="y")
    )
    level, text = ua.headline(clean)
    # Never a green "success": no findings is not evidence of no leakage (browser QA fix).
    assert level == "info" and "does not prove" in text


def test_headline_flags_partial_coverage_when_there_are_no_findings():
    def frame(start: str, seed: int, n_missing: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        pred = pd.Series(pd.date_range(start, periods=60, freq="D"))
        avail = (pred - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M").astype(object)
        avail[:n_missing] = None
        return pd.DataFrame(
            {
                "bal": rng.normal(size=60),
                "y": [0, 1] * 30,
                "t": pred.dt.strftime("%Y-%m-%d"),
                "bal_at": avail,
            }
        )

    cfg = AuditConfig(
        target="y",
        prediction_timestamp="t",
        feature_availability={"bal": "bal_at"},
        scenario="forecasting",
    )
    result = audit_train_test(frame("2024-01-01", 1, 10), frame("2024-06-01", 2, 0), cfg)
    assert result.findings == []
    level, text = ua.headline(result)
    assert level == "warning"
    assert ua.partial_note_count(result) >= 1
    assert "partly assessed" in text and "does not prove" in text


def test_skipped_checks_are_grouped_by_reason():
    data = samples()["unsplit"]
    result = ua.run_audit(
        ua.FormState(mode="unsplit", target="churn"),
        {"data": ua.load_file(data.names["data"], data.files["data"])},
    )
    groups = ua.skipped_by_reason(result)
    flat = [i for _, _, ids in groups for i in ids]
    assert len(flat) == len(set(flat)) == result.summary()["checks_skipped"]
    reasons = [reason for _, reason, _ in groups]
    assert len(reasons) == len(set(reasons)) < len(flat)
    assert all(status == "skipped" and reason for status, reason, _ in groups)


def test_checks_by_status_lists_skipped_reasons(sample_result):
    status = ua.checks_by_status(sample_result)
    assert status["skipped"] and all(c["reason"] for c in status["skipped"])
    assert {c["check"] for c in status["skipped"]} >= {"TMP-001", "DUP-003"}


def test_json_download_is_the_engine_report(sample_result):
    assert ua.json_report_bytes(sample_result, False) == report_to_json(
        sample_result, include_generated_at=False
    ).encode("utf-8")
    report = json.loads(ua.json_report_bytes(sample_result))
    assert list(report)[:3] == ["tool", "report_format_version", "generated_at"]
    assert report["tool"]["version"] == "0.2.0"


def test_markdown_report_is_complete_and_escaped(tmp_path):
    train = classification_frame(200, 7)
    train["[click](http://evil.example)"] = train["y"]
    path = tmp_path / "train.csv"
    train.to_csv(path, index=False)
    result = audit_train_test(path, train.iloc[:40], AuditConfig(target="y"))
    md = ua.markdown_report(result.to_dict())
    assert "](http" not in md  # injected link is escaped
    assert "### TGT\\-001:" in md  # ids are escaped like all other text
    assert ua.NO_PROOF_NOTE in md and ua.OVERLAP_NOTE in md
    assert "## Checks not run" in md and "Why this is not proof" in md
    assert str(tmp_path) not in md


def test_md_escape():
    assert ua.md_escape("a_b*[x](y)") == "a\\_b\\*\\[x\\]\\(y\\)"


# -------------------------------------------------------------------- split


def test_split_creation_requires_confirmation():
    un = samples()["unsplit"]
    f = ua.load_file(un.names["data"], un.files["data"])
    state = ua.FormState(
        mode="unsplit", target="churn", scenario="new_entities", entity_id="patient_id"
    )
    rec = recommend_split(f.frame, ua.build_config(state))
    with pytest.raises(SplitConfirmationError):
        ua.create_split_files(f, rec, confirmed_strategy="random", state=state)
    before = f.frame.copy(deep=True)
    out = ua.create_split_files(f, rec, confirmed_strategy=" group ", state=state)
    pd.testing.assert_frame_equal(f.frame, before)
    train, test = pd.read_csv(io.BytesIO(out.train_csv)), pd.read_csv(io.BytesIO(out.test_csv))
    assert len(train) + len(test) == len(f.frame)
    assert set(train["patient_id"]).isdisjoint(test["patient_id"])
    assert json.loads(out.report_json)["split"]["group_overlap"]["n_shared_groups"] == 0


def test_group_and_time_split_is_not_created():
    df = classification_frame(100, 8)
    df["g"] = [f"g{i % 10}" for i in range(100)]
    df["t"] = pd.date_range("2024-01-01", periods=100, freq="D").strftime("%Y-%m-%d")
    f = loaded(df)
    state = ua.FormState(
        mode="unsplit",
        target="y",
        scenario="new_entities_forecasting",
        entity_id="g",
        prediction_timestamp="t",
    )
    rec = recommend_split(f.frame, ua.build_config(state))
    with pytest.raises(NotImplementedError):
        ua.create_split_files(f, rec, confirmed_strategy="group_and_time", state=state)


# ----------------------------------------------------------------- launcher


def test_launcher_args_are_private_by_default():
    args = streamlit_args("localhost", 8501, headless=True)
    assert args[:2] == ["run", args[1]] and args[1].endswith("ui_app.py")
    pairs = dict(zip(args[2::2], args[3::2], strict=True))
    assert pairs["--browser.gatherUsageStats"] == "false"
    assert pairs["--server.address"] == "localhost"
    assert pairs["--server.maxUploadSize"] == "200"
    assert pairs["--client.toolbarMode"] == "viewer"  # no "Deploy" button
    assert pairs["--theme.primaryColor"].startswith("#")


def test_base_package_works_without_streamlit():
    code = (
        "import sys; sys.modules['streamlit'] = None\n"
        "import splitguard, splitguard.cli, splitguard.ui_adapter, splitguard.ui_samples\n"
        "from splitguard.ui_launcher import main\n"
        "assert main(['--headless']) == 2\n"
        "assert splitguard.cli.build_parser().prog == 'splitguard'\n"
        "print('BASE_OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "pip install" in out.stderr
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "BASE_OK"


def test_launcher_reports_missing_streamlit(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "streamlit", None)
    monkeypatch.setitem(sys.modules, "streamlit.web", None)
    assert launcher_main(["--headless"]) == 2
    assert 'pip install "splitguard[ui]"' in capsys.readouterr().err
