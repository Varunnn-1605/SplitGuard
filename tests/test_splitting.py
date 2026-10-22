from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import check, iso
from splitguard import (
    AuditConfig,
    InputValidationError,
    create_split,
    inspect_unsplit,
    recommend_split,
)
from splitguard.splitting import SplitConfirmationError


def dataset(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "group": [f"g{i // 5}" for i in range(n)],
            "when": iso(pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))),
            "y": [1] * 30 + [0] * (n - 30),
            "amount": rng.normal(size=n) * 100,
        }
    )


def rec(df=None, **cfg):
    return recommend_split(df if df is not None else dataset(), AuditConfig(**cfg))


@pytest.mark.parametrize(
    ("cfg", "missing_prefix"),
    [
        ({"target": "y"}, "prediction_scenario"),
        ({"scenario": "independent"}, "target"),
        ({"target": "y", "scenario": "new_entities"}, "entity_id"),
        ({"target": "y", "scenario": "forecasting"}, "prediction_timestamp"),
    ],
)
def test_needs_user_input(cfg, missing_prefix):
    r = rec(**cfg)
    assert r.status == "needs_user_input"
    assert r.strategy is None
    assert any(m.startswith(missing_prefix) for m in r.missing_information)


def test_hints_when_scenario_unknown():
    r = rec(target="y", entity_id="group", prediction_timestamp="when")
    assert r.status == "needs_user_input"
    text = " ".join(r.rationale)
    assert "new_entities" in text and "forecasting" in text


@pytest.mark.parametrize(
    ("cfg", "strategy"),
    [
        ({"target": "y", "scenario": "independent"}, "stratified_random"),
        ({"target": "amount", "scenario": "independent"}, "random"),
        ({"target": "y", "scenario": "new_entities", "entity_id": "group"}, "group"),
        ({"target": "y", "scenario": "forecasting", "prediction_timestamp": "when"}, "time"),
        (
            {
                "target": "y",
                "scenario": "new_entities_forecasting",
                "entity_id": "group",
                "prediction_timestamp": "when",
            },
            "group_and_time",
        ),
    ],
)
def test_recommended_strategies(cfg, strategy):
    r = rec(**cfg)
    assert r.status == "recommended"
    assert r.strategy == strategy
    assert r.rationale and r.assumptions
    assert any("not discovered from the CSV" in lim or "optimal" in lim for lim in r.limitations)


def test_independent_with_repeated_entities_warns():
    r = rec(target="y", scenario="independent", entity_id="group")
    assert any("same entity in both" in w for w in r.warnings)


def test_creation_requires_matching_confirmation():
    df = dataset()
    config = AuditConfig(target="y", scenario="independent")
    r = recommend_split(df, config)
    with pytest.raises(SplitConfirmationError):
        create_split(df, r, confirmed_strategy="random", config=config)
    pending = recommend_split(df, AuditConfig(target="y"))
    with pytest.raises(SplitConfirmationError):
        create_split(df, pending, confirmed_strategy="random", config=config)


def test_stratified_split_is_seeded_and_balanced():
    df = dataset()
    config = AuditConfig(target="y", scenario="independent")
    r = recommend_split(df, config)
    a = create_split(df, r, confirmed_strategy="stratified_random", config=config)
    b = create_split(df, r, confirmed_strategy="stratified_random", config=config)
    pd.testing.assert_frame_equal(a.test, b.test)
    assert len(a.train) + len(a.test) == 100
    assert a.report["class_balance"]["test"] == {"0": 14, "1": 6}
    c = create_split(df, r, confirmed_strategy="stratified_random", config=config, seed=99)
    assert not a.test.equals(c.test)


def test_group_split_has_no_group_overlap():
    df = dataset()
    config = AuditConfig(target="y", scenario="new_entities", entity_id="group")
    r = recommend_split(df, config)
    s = create_split(df, r, confirmed_strategy="group", config=config)
    assert set(s.train["group"]).isdisjoint(set(s.test["group"]))
    assert s.report["group_overlap"]["n_shared_groups"] == 0
    assert len(s.test) == 20


def test_time_split_respects_chronology():
    df = dataset().sample(frac=1, random_state=0)  # shuffled input
    config = AuditConfig(target="y", scenario="forecasting", prediction_timestamp="when")
    r = recommend_split(df, config)
    s = create_split(df, r, confirmed_strategy="time", config=config)
    assert pd.to_datetime(s.train["when"]).max() < pd.to_datetime(s.test["when"]).min()
    assert len(s.test) == 20
    assert s.report["time_boundaries"]["test_min"] == "2024-03-21T00:00:00+00:00"


def test_group_and_time_is_review_only():
    df = dataset()
    config = AuditConfig(
        target="y",
        scenario="new_entities_forecasting",
        entity_id="group",
        prediction_timestamp="when",
    )
    r = recommend_split(df, config)
    with pytest.raises(NotImplementedError):
        create_split(df, r, confirmed_strategy="group_and_time", config=config)


def test_rows_with_missing_group_are_not_dropped():
    df = dataset()
    df.loc[3, "group"] = None
    config = AuditConfig(target="y", scenario="new_entities", entity_id="group")
    r = recommend_split(df, config)
    with pytest.raises(InputValidationError):
        create_split(df, r, confirmed_strategy="group", config=config)


def test_inspect_never_splits():
    df = dataset()
    result = inspect_unsplit(df, AuditConfig(target="y", scenario="independent"))
    assert list(result.partitions) == ["unsplit"]
    assert result.partitions["unsplit"]["n_rows"] == 100
    assert result.split_recommendation.strategy == "stratified_random"
    assert check(result, "DUP-002").status.value == "skipped"
    assert any("contamination was not audited" in w for w in result.warnings)
    spl = next(f for f in result.findings if f.rule_id == "SPL-001")
    assert spl.category.value == "split_recommendation"
