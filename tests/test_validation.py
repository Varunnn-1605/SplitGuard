from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import classification_frame, write_csv
from splitguard import (
    AuditConfig,
    ConfigError,
    InputValidationError,
    Limits,
    audit_split_column,
    audit_train_test,
    inspect_unsplit,
)


def codes(exc_info) -> set[str]:
    return {issue.code for issue in exc_info.value.issues}


def test_empty_file(tmp_path, clean_pair):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(empty, clean_pair[1])
    assert codes(exc) == {"EMPTY_FILE"}


def test_header_only_file(tmp_path, clean_pair):
    path = tmp_path / "header.csv"
    path.write_text("x1,x2,y\n", encoding="utf-8")
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(path, clean_pair[1])
    assert codes(exc) == {"EMPTY_DATASET"}


def test_missing_file(tmp_path, clean_pair):
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(tmp_path / "nope.csv", clean_pair[1])
    assert codes(exc) == {"FILE_NOT_FOUND"}


def test_duplicate_and_blank_column_names(tmp_path, clean_pair):
    path = tmp_path / "bad.csv"
    path.write_text("a,a,,b\n1,2,3,4\n", encoding="utf-8")
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(path, clean_pair[1])
    assert codes(exc) == {"DUPLICATE_COLUMN_NAMES", "MISSING_COLUMN_NAME"}


def test_malformed_csv(tmp_path, clean_pair):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(path, clean_pair[1])
    assert codes(exc) == {"MALFORMED_CSV"}


def test_non_utf8_csv(tmp_path, clean_pair):
    path = tmp_path / "latin.csv"
    path.write_bytes("name,y\ncaf\xe9,1\n".encode("latin-1"))
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(path, clean_pair[1])
    assert codes(exc) == {"UNREADABLE_CSV"}


def test_file_size_limit_checked_before_parsing(tmp_path, clean_pair):
    path = write_csv(tmp_path, "train.csv", clean_pair[0])
    config = AuditConfig(limits=Limits(max_file_bytes=100))
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(path, clean_pair[1], config)
    assert codes(exc) == {"FILE_TOO_LARGE"}


def test_row_limit(tmp_path, clean_pair):
    path = write_csv(tmp_path, "train.csv", clean_pair[0])
    config = AuditConfig(limits=Limits(max_rows=50))
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(path, clean_pair[1], config)
    assert codes(exc) == {"ROW_LIMIT_EXCEEDED"}
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(clean_pair[0], clean_pair[1], config)
    assert codes(exc) == {"ROW_LIMIT_EXCEEDED"}


def test_column_limit(clean_pair):
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(*clean_pair, AuditConfig(limits=Limits(max_columns=2)))
    assert codes(exc) == {"COLUMN_LIMIT_EXCEEDED"}


def test_bytes_input_and_bom(clean_pair):
    data = ("﻿" + clean_pair[0].to_csv(index=False)).encode("utf-8")
    result = audit_train_test(data, clean_pair[1], AuditConfig(target="y"))
    assert result.inputs[0]["columns"] == ["x1", "x2", "color", "y"]
    assert result.inputs[0]["source_type"] == "csv_bytes"


def test_missing_configured_columns(clean_pair):
    train, test = clean_pair
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(train, test, AuditConfig(target="label", entity_id="cust"))
    assert codes(exc) == {"MISSING_COLUMN"}
    messages = " ".join(i.message for i in exc.value.issues)
    assert "'label'" in messages and "'cust'" in messages


def test_entity_missing_from_test_only(clean_pair):
    train, test = clean_pair
    train = train.assign(cust=[f"c{i}" for i in range(len(train))])
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(train, test, AuditConfig(target="y", entity_id="cust"))
    assert [i.message for i in exc.value.issues] == [
        "test: configured entity_id column 'cust' not found"
    ]


def test_excluded_column_must_exist(clean_pair):
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(*clean_pair, AuditConfig(exclude_columns=("typo",)))
    assert codes(exc) == {"MISSING_COLUMN"}


def test_no_feature_columns():
    df = pd.DataFrame({"id": range(40), "y": [0, 1] * 20})
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(df, df, AuditConfig(target="y", entity_id="id"))
    assert codes(exc) == {"NO_FEATURE_COLUMNS"}


def test_incompatible_schemas(clean_pair):
    train = clean_pair[0]
    test = pd.DataFrame({"other": range(10), "y": [0, 1] * 5})
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(train, test, AuditConfig(target="y"))
    assert codes(exc) == {"INCOMPATIBLE_SCHEMA"}


def test_excessive_target_missingness(clean_pair):
    train = clean_pair[0].astype({"y": float})
    train.loc[: int(len(train) * 0.6), "y"] = np.nan
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(train, clean_pair[1], AuditConfig(target="y"))
    assert codes(exc) == {"EXCESSIVE_MISSINGNESS"}


def test_regression_requires_numeric_target(clean_pair):
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(*clean_pair, AuditConfig(target="color", task_type="regression"))
    assert codes(exc) == {"TARGET_TYPE"}


def test_invalid_timestamps_are_counted_not_dropped(clean_pair):
    train, test = clean_pair
    stamps = pd.date_range("2024-01-01", periods=len(train), freq="h").strftime("%Y-%m-%d %H:%M")
    train = train.assign(t=list(stamps))
    train.loc[[3, 7], "t"] = "not a date"
    test = test.assign(t="2024-06-01")
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(train, test, AuditConfig(prediction_timestamp="t"))
    (issue,) = exc.value.issues
    assert issue.code == "INVALID_TIMESTAMP"
    assert "2 unparseable" in issue.message


def test_numeric_timestamps_rejected(clean_pair):
    train, test = clean_pair
    train, test = train.assign(t=range(len(train))), test.assign(t=range(len(test)))
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(train, test, AuditConfig(prediction_timestamp="t"))
    assert codes(exc) == {"TIMESTAMP_NOT_TEXT"}


def test_excessive_timestamp_missingness(clean_pair):
    train, test = clean_pair
    train = train.assign(t=[None] * 150 + ["2024-01-01"] * 50)
    test = test.assign(t="2024-02-01")
    with pytest.raises(InputValidationError) as exc:
        audit_train_test(train, test, AuditConfig(prediction_timestamp="t"))
    assert codes(exc) == {"EXCESSIVE_MISSINGNESS"}


# ----------------------------------------------------------------- split column


def _split_frame() -> pd.DataFrame:
    df = classification_frame(100, seed=5)
    df["split"] = ["train"] * 80 + ["test"] * 20
    df["secret"] = [f"SECRET{i}" for i in range(100)]
    return df


def test_invalid_split_labels_reported_without_row_contents():
    df = _split_frame()
    df.loc[[1, 2], "split"] = "validation"
    with pytest.raises(InputValidationError) as exc:
        audit_split_column(df, "split")
    assert codes(exc) == {"SPLIT_LABEL_INVALID"}
    assert "2 rows" in str(exc.value)
    assert "SECRET" not in str(exc.value)


def test_missing_split_labels():
    df = _split_frame()
    df.loc[5, "split"] = None
    with pytest.raises(InputValidationError) as exc:
        audit_split_column(df, "split")
    assert codes(exc) == {"SPLIT_LABEL_MISSING"}


def test_empty_test_partition():
    df = _split_frame()
    df["split"] = "train"
    with pytest.raises(InputValidationError) as exc:
        audit_split_column(df, "split")
    assert codes(exc) == {"EMPTY_PARTITION"}


def test_numeric_split_labels():
    df = _split_frame()
    df["split"] = [0] * 80 + [1] * 20
    result = audit_split_column(df, "split", train_value="0", test_value="1")
    assert result.partitions == {
        "train": {"n_rows": 80, "n_columns": 5},
        "test": {"n_rows": 20, "n_columns": 5},
    }


def test_identical_split_labels_rejected():
    with pytest.raises(InputValidationError) as exc:
        audit_split_column(_split_frame(), "split", "train", "train")
    assert codes(exc) == {"INVALID_SPLIT_LABELS"}


def test_split_column_cannot_have_another_role():
    with pytest.raises(InputValidationError):
        audit_split_column(_split_frame(), "split", config=AuditConfig(target="split"))


# ----------------------------------------------------------------------- config


def test_input_frames_are_not_modified(clean_pair):
    train, test = clean_pair
    before_train, before_test = train.copy(deep=True), test.copy(deep=True)
    audit_train_test(train, test, AuditConfig(target="y"))
    inspect_unsplit(train, AuditConfig(target="y"))
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(test, before_test)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario": "sometimes"},
        {"task_type": "ranking"},
        {"target": "y", "entity_id": "y"},
        {"target": "y", "exclude_columns": ("y",)},
        {"min_lead_time": "soon"},
        {"min_lead_time": "-1D"},
        {"exclude_columns": "abc"},
        {"thresholds": {"strong_correlation": 1.5}},
        {"thresholds": {"strong_auc": 0.3}},
        {"thresholds": {"min_rows_for_target_checks": 0}},
        {"thresholds": {"strong_eta": "high"}},
        {"limits": {"max_rows": 0}},
        {"feature_availability": {"x1": ""}},
        {"target": "y", "feature_availability": {"y": "t"}},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ConfigError):
        AuditConfig(**kwargs)


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ConfigError, match="tagret"):
        AuditConfig.from_dict({"tagret": "y"})
    with pytest.raises(ConfigError, match="strong_corr"):
        AuditConfig.from_dict({"thresholds": {"strong_corr": 0.5}})


def test_config_round_trip():
    config = AuditConfig(
        target="y",
        entity_id="c",
        exclude_columns=("b", "a"),
        feature_availability={"f": "f_at"},
        scenario="forecasting",
        prediction_timestamp="t",
        thresholds={"strong_correlation": 0.8},
    )
    again = AuditConfig.from_dict(config.to_dict())
    assert again == config
    assert again.exclude_columns == ("a", "b")
    assert again.thresholds.strong_correlation == 0.8
