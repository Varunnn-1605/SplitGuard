from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest

from conftest import classification_frame, write_csv
from splitguard import AuditConfig, Finding, audit_train_test
from splitguard.reporting import load_json_report, render_summary, report_to_json, write_json_report
from splitguard.schemas import CheckRecord, to_jsonable


def leaky_pair():
    train = classification_frame(200, seed=1)
    train["leak"] = train["y"]
    train["note"] = [f"ZZSECRET{i}" for i in range(200)]
    test = pd.concat(
        [classification_frame(40, seed=2).assign(leak=0, note="x"), train.iloc[:5]],
        ignore_index=True,
    )
    return train, test


def test_report_is_strict_json_with_expected_sections():
    result = audit_train_test(*leaky_pair(), AuditConfig(target="y"))
    report = json.loads(report_to_json(result))
    assert list(report) == [
        "tool",
        "report_format_version",
        "generated_at",
        "mode",
        "inputs",
        "partitions",
        "configuration",
        "task_type",
        "feature_columns",
        "summary",
        "checks",
        "findings",
        "split_recommendation",
        "warnings",
        "limitations",
    ]
    assert report["tool"] == {"name": "splitguard", "version": "0.2.0"}
    assert report["summary"]["total_findings"] == len(report["findings"])
    assert set(report["summary"]["by_severity"]) == {"critical", "high", "medium", "low", "info"}
    for f in report["findings"]:
        assert set(f) >= {
            "finding_id",
            "category",
            "severity",
            "confidence",
            "title",
            "description",
            "evidence",
            "affected_columns",
            "recommended_action",
            "limitations",
        }
    severities = [f["severity"] for f in report["findings"]]
    order = ["critical", "high", "medium", "low", "info"]
    assert severities == sorted(severities, key=order.index)


def test_reports_are_deterministic(tmp_path):
    train, test = leaky_pair()
    paths = write_csv(tmp_path, "train.csv", train), write_csv(tmp_path, "test.csv", test)
    a = report_to_json(
        audit_train_test(*paths, AuditConfig(target="y")), include_generated_at=False
    )
    b = report_to_json(
        audit_train_test(*paths, AuditConfig(target="y")), include_generated_at=False
    )
    assert a == b
    assert "generated_at" not in json.loads(a)


def test_report_has_no_paths_or_row_contents(tmp_path):
    train, test = leaky_pair()
    paths = write_csv(tmp_path, "train.csv", train), write_csv(tmp_path, "test.csv", test)
    text = report_to_json(audit_train_test(*paths, AuditConfig(target="y")))
    assert str(tmp_path) not in text
    assert str(tmp_path).replace("\\", "\\\\") not in text
    assert "ZZSECRET" not in text
    report = json.loads(text)
    assert report["inputs"][0]["name"] == "train.csv"
    assert len(report["inputs"][0]["sha256"]) == 64


def test_to_jsonable_handles_awkward_values():
    value = {
        "np_int": np.int64(3),
        "np_float": np.float32(1.5),
        "nan": float("nan"),
        "inf": np.inf,
        "na": pd.NA,
        "nat": pd.NaT,
        "ts": pd.Timestamp("2024-01-02", tz="UTC"),
        "date": dt.date(2024, 1, 3),
        "td": pd.Timedelta(days=2),
        "bool": np.bool_(True),
        "set": {"b", "a"},
        "arr": np.array([1, 2]),
        5: "int key",
    }
    out = to_jsonable(value)
    json.dumps(out, allow_nan=False)
    assert out["np_int"] == 3 and out["np_float"] == 1.5
    assert out["nan"] is None and out["inf"] is None and out["na"] is None and out["nat"] is None
    assert out["ts"] == "2024-01-02T00:00:00+00:00"
    assert out["date"] == "2024-01-03T00:00:00"
    assert out["td"] == "2 days 00:00:00"
    assert out["bool"] is True and out["set"] == ["a", "b"] and out["arr"] == [1, 2]
    assert out["5"] == "int key"


def test_finding_schema_is_validated():
    kwargs = dict(
        rule_id="X-1",
        key="k",
        category="target_leakage",
        severity="high",
        confidence="low",
        title="t",
        description="d",
        evidence={},
        affected_columns=[],
        recommended_action="a",
        limitations=["l"],
    )
    assert Finding(**kwargs).finding_id == "X-1:k"
    with pytest.raises(ValueError):
        Finding(**{**kwargs, "severity": "severe"})
    with pytest.raises(ValueError):
        Finding(**{**kwargs, "limitations": []})
    with pytest.raises(ValueError):
        Finding(**{**kwargs, "title": " "})
    with pytest.raises(ValueError):
        CheckRecord("X-1", "name", "skipped")  # skipped checks need a reason


def test_non_finite_evidence_serialises_as_null():
    train = classification_frame(100, seed=3)
    train["inf_col"] = np.where(np.arange(100) % 2, np.inf, 1.0)
    text = report_to_json(audit_train_test(train, train.iloc[:20], AuditConfig(target="y")))
    assert "NaN" not in text and "Infinity" not in text


def test_write_and_load_report(tmp_path):
    result = audit_train_test(*leaky_pair(), AuditConfig(target="y"))
    path = write_json_report(result, tmp_path / "report.json")
    loaded = load_json_report(path)
    assert loaded["summary"] == result.summary()


def test_summary_is_ascii_and_lists_findings():
    result = audit_train_test(*leaky_pair(), AuditConfig(target="y"))
    text = render_summary(result)
    text.encode("ascii")
    assert "TGT-001:leak" in text
    assert "not proof" in text
    assert not math.isnan(result.summary()["total_findings"])
