"""Streamlit AppTest tests for the UI. Skipped when Streamlit is not installed.

Widgets are located by their stable keys and assertions use user-visible text,
not layout positions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import classification_frame

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import splitguard  # noqa: E402

APP = Path(splitguard.__file__).with_name("ui_app.py")
SAMPLE = "Try a sample dataset"
UPLOAD = "Upload my own CSV files"


def start() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def text(at: AppTest) -> str:
    """All rendered text, with Markdown escapes removed."""
    parts = []
    for kind in (
        "markdown",
        "caption",
        "info",
        "warning",
        "success",
        "error",
        "title",
        "header",
        "subheader",
    ):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts).replace("\\", "").replace("**", "")


def widget(at: AppTest, kind: str, name: str):
    """Configuration widgets are keyed 'cfg:<scope>:<name>'."""
    matches = [w for w in getattr(at, kind) if w.key and w.key.endswith(f":{name}")]
    assert matches, f"no {kind} ending with :{name}"
    return matches[0]


def run_audit(at: AppTest) -> AppTest:
    at.button(key="run_audit").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def use_sample(at: AppTest, key: str) -> AppTest:
    at.radio(key="source").set_value(SAMPLE).run()
    at.selectbox(key="sample_key").set_value(key).run()
    assert not at.exception
    return at


def csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ startup


def test_startup_shows_landing_and_privacy():
    at = start()
    body = text(at)
    assert "SplitGuard" in body
    assert "Try a sample dataset" in body  # landing points to the sample workflow
    assert "Nothing is uploaded to any external service" in body
    assert not at.error
    assert len(at.button) == 0  # no audit button before data exists


# ----------------------------------------------------------------- samples


def test_sample_train_test_workflow():
    at = use_sample(start(), "train_test")
    assert widget(at, "selectbox", "target").value == "churn"  # suggested, user-changeable
    assert widget(at, "selectbox", "scenario").value == "unknown"  # never guessed
    assert "result" not in at.session_state  # nothing runs before the click
    run_audit(at)
    body = text(at)
    assert "TGT-001:churn_score_final" in body
    assert "DUP-002:train_test" in body
    assert "not independent confirmations" in body
    labels = [b.label for b in at.download_button]
    assert "Full JSON report" in labels and "Readable summary (Markdown)" in labels
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Critical / high"].startswith("2")


def test_skipped_checks_and_warnings_are_prominent():
    at = run_audit(use_sample(start(), "train_test"))
    body = text(at)
    assert "Missing metadata - checks that could not run" in body
    assert "temporal leakage was not assessed" in body
    assert "Checks that did not run" in body
    assert any(e.label.startswith("What was not fully assessed") for e in at.expander)


def test_temporal_sample_shows_post_outcome_feature():
    at = run_audit(use_sample(start(), "temporal"))
    body = text(at)
    assert "TMP-002:refund_amount" in body
    assert "TMP-001:refund_amount" in body
    assert widget(at, "selectbox", "scenario").value == "forecasting"


def test_filters_change_visible_findings():
    at = run_audit(use_sample(start(), "train_test"))
    total = len(at.session_state["result"].findings)
    at.multiselect(key="filter_severity").set_value(["critical"]).run()
    assert f"of {total} finding(s)" in text(at)
    shown = [
        m.value for m in at.markdown if str(m.value).startswith("**") and "churn" in str(m.value)
    ]
    assert shown
    at.multiselect(key="filter_confidence").set_value([]).run()
    assert "No findings match these filters." in text(at)


def test_changing_settings_marks_results_stale():
    at = run_audit(use_sample(start(), "train_test"))
    widget(at, "selectbox", "scenario").set_value("independent").run()
    assert "changed since this audit ran" in text(at)


# ----------------------------------------------------------------- uploads


def test_upload_train_test_and_clean_result_states_no_proof():
    at = start()
    at.radio(key="mode").set_value("train_test").run()
    at.file_uploader(key="upload_train_test_train").upload(
        "my_train.csv", csv(classification_frame(200, 1)), "text/csv"
    )
    at.file_uploader(key="upload_train_test_test").upload(
        "my_test.csv", csv(classification_frame(60, 2)), "text/csv"
    )
    at.run()
    body = text(at)
    assert "my_train.csv" in body and "200 rows x 4 columns" in body
    widget(at, "selectbox", "target").set_value("y").run()
    run_audit(at)
    body = text(at)
    assert "does not prove the data is free of leakage" in body
    assert at.session_state["result"].mode == "train_test"


def test_malformed_upload_shows_actionable_error_without_rows():
    at = start()
    at.radio(key="mode").set_value("unsplit").run()
    bad = b"name,ssn\nalice,SSN-111\nbob,SSN-222,extra\n"
    at.file_uploader(key="upload_unsplit_data").upload("bad.csv", bad, "text/csv").run()
    errors = " ".join(str(e.value) for e in at.error).replace("\\", "")
    assert "MALFORMED_CSV" in errors
    assert "SSN-" not in errors
    assert not at.exception


def test_engine_error_is_shown_safely():
    at = start()
    at.radio(key="mode").set_value("train_test").run()
    at.file_uploader(key="upload_train_test_train").upload(
        "a.csv", csv(classification_frame(80, 3)), "text/csv"
    )
    other = pd.DataFrame({"other": range(10), "y": [0, 1] * 5})
    at.file_uploader(key="upload_train_test_test").upload("b.csv", csv(other), "text/csv")
    at.run()
    widget(at, "selectbox", "target").set_value("y").run()
    run_audit(at)
    errors = " ".join(str(e.value) for e in at.error).replace("\\", "")
    assert "INCOMPATIBLE_SCHEMA" in errors
    assert "result" not in at.session_state


def test_split_column_upload_requires_labels():
    df = classification_frame(150, 4)
    df["fold"] = ["fit"] * 120 + ["holdout"] * 30
    at = start()
    at.radio(key="mode").set_value("split_column").run()
    at.file_uploader(key="upload_split_column_data").upload("d.csv", csv(df), "text/csv").run()
    widget(at, "selectbox", "split_column").set_value("fold").run()
    # Labels are not 'train'/'test', so nothing is pre-selected and the run is blocked.
    assert widget(at, "selectbox", "train_value").value == "(not set)"
    assert at.button(key="run_audit").disabled
    assert "which split labels mean train and test" in text(at)
    widget(at, "selectbox", "train_value").set_value("fit").run()
    widget(at, "selectbox", "test_value").set_value("holdout").run()
    widget(at, "selectbox", "target").set_value("y").run()
    assert "fold" not in widget(at, "selectbox", "target").options
    run_audit(at)
    result = at.session_state["result"]
    assert result.mode == "split_column"
    assert result.partitions["test"]["n_rows"] == 30


# ------------------------------------------------------------------- split


def test_unsplit_needs_input_then_recommends_and_confirms_split():
    at = run_audit(use_sample(start(), "unsplit"))
    body = text(at)
    assert "needs more information" in body
    assert "prediction_scenario" in body
    assert "split_download" not in at.session_state  # never split automatically
    widget(at, "selectbox", "scenario").set_value("new_entities").run()
    run_audit(at)
    assert "Recommended strategy: group" in text(at)
    create = at.button(key="create_split")
    assert create.disabled  # requires typed confirmation and acknowledgement
    at.text_input(key="confirm_strategy").set_value("random").run()
    at.checkbox(key="confirm_ack").check().run()
    assert at.button(key="create_split").disabled  # wrong strategy typed
    at.text_input(key="confirm_strategy").set_value("group").run()
    at.button(key="create_split").click().run()
    assert not at.exception
    download = at.session_state["split_download"]
    assert download.report["group_overlap"]["n_shared_groups"] == 0
    assert {"train.csv", "test.csv", "split_report.json"} <= {b.label for b in at.download_button}


def test_strategy_explanations_available():
    at = run_audit(use_sample(start(), "unsplit"))
    assert any(e.label == "How the split strategies differ" for e in at.expander)
