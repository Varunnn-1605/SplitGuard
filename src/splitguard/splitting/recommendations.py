"""Split-strategy recommendation and explicit, confirmed split creation.

The right split depends on how the model will be deployed, which a CSV alone
cannot reveal. ``recommend_split`` therefore maps the *declared* scenario and
metadata to a strategy, and returns ``needs_user_input`` when critical context
is missing. It never claims to find an optimal or leakage-free split.

``create_split`` is a separate operation that only runs when the caller
confirms the recommended strategy by name. It uses no model, no test
performance and no feature selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from splitguard.config import AuditConfig, ConfigError, Scenario, TaskType
from splitguard.hashing import canonical_series
from splitguard.schemas import SplitRecommendation, to_jsonable
from splitguard.validation import (
    InputValidationError,
    ValidationIssue,
    parse_timestamps,
    raise_if_issues,
    require_columns,
    resolve_task_type,
)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42


class SplitStrategy(StrEnum):
    STRATIFIED_RANDOM = "stratified_random"
    RANDOM = "random"
    GROUP = "group"
    TIME = "time"
    GROUP_AND_TIME = "group_and_time"


class SplitConfirmationError(ValueError):
    """Raised when split creation is attempted without matching confirmation."""


_COMMON_LIMITATIONS = [
    "Derived from the declared scenario and metadata, not discovered from the CSV; "
    "SplitGuard cannot know the real deployment setting.",
    "No split is guaranteed to be optimal or free of leakage. Duplicates, target copies "
    "and post-outcome features survive any split; audit the result.",
]

_SCENARIO_HELP = (
    "prediction_scenario: how will the model be used? One of independent (new rows from "
    "the same population), new_entities (unseen customers/patients/devices), forecasting "
    "(future periods), new_entities_forecasting (both)."
)


def _check_test_size(test_size: float) -> None:
    if not 0.0 < float(test_size) < 1.0:
        raise ConfigError("test_size must be strictly between 0 and 1")


def _describe(df: pd.DataFrame, config: AuditConfig) -> tuple[dict[str, Any], pd.Series | None]:
    evidence: dict[str, Any] = {"n_rows": len(df)}
    timestamps = None
    if config.target and config.target in df.columns:
        counts = canonical_series(df[config.target]).value_counts(dropna=True)
        evidence["target"] = {
            "n_distinct": len(counts),
            "n_missing": int(df[config.target].isna().sum()),
            "min_class_count": int(counts.min()) if len(counts) else 0,
        }
    if config.entity_id and config.entity_id in df.columns:
        sizes = canonical_series(df[config.entity_id]).value_counts(dropna=True)
        evidence["entity"] = {
            "column": config.entity_id,
            "n_entities": len(sizes),
            "n_entities_with_multiple_rows": int((sizes > 1).sum()),
            "max_rows_per_entity": int(sizes.max()) if len(sizes) else 0,
            "n_missing": int(df[config.entity_id].isna().sum()),
        }
    if config.prediction_timestamp and config.prediction_timestamp in df.columns:
        timestamps, issues = parse_timestamps(
            df[config.prediction_timestamp],
            "prediction_timestamp",
            "dataset",
            config.thresholds.max_missing_rate_required,
        )
        raise_if_issues(issues)
        valid = timestamps.dropna()
        evidence["time"] = {
            "column": config.prediction_timestamp,
            "min": valid.min() if len(valid) else None,
            "max": valid.max() if len(valid) else None,
            "n_distinct": int(valid.nunique()),
            "n_missing": int(timestamps.isna().sum()),
        }
    return to_jsonable(evidence), timestamps


def recommend_split(
    df: pd.DataFrame,
    config: AuditConfig,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> SplitRecommendation:
    """Recommend a split strategy for an unsplit dataset (no data is modified)."""
    _check_test_size(test_size)
    raise_if_issues(
        require_columns(
            df,
            {
                "target": config.target,
                "entity_id": config.entity_id,
                "prediction_timestamp": config.prediction_timestamp,
            },
            "dataset",
        )
    )
    evidence, _ = _describe(df, config)
    scenario = config.scenario
    missing: list[str] = []
    if config.target is None:
        missing.append(
            "target: the column the model predicts (needed for task type and stratification)."
        )
    if scenario is Scenario.UNKNOWN:
        missing.append(_SCENARIO_HELP)
    if scenario.requires_entity_independence and not config.entity_id:
        missing.append("entity_id: the column identifying entities that must not cross the split.")
    if scenario.requires_temporal_order and not config.prediction_timestamp:
        missing.append(
            "prediction_timestamp: the time each prediction is made, for a chronological split."
        )

    hints = []
    entity = evidence.get("entity")
    if entity and entity["n_entities_with_multiple_rows"]:
        hints.append(
            f"{entity['n_entities_with_multiple_rows']} entities in {entity['column']!r} have "
            "several rows; if the model must handle unseen entities, choose new_entities."
        )
    if "time" in evidence:
        hints.append(
            "Rows carry timestamps; if the model predicts future periods, choose forecasting."
        )

    if missing:
        return SplitRecommendation(
            status="needs_user_input",
            strategy=None,
            rationale=["Critical context is missing, so no strategy is recommended yet."] + hints,
            assumptions=[],
            limitations=list(_COMMON_LIMITATIONS),
            missing_information=missing,
            evidence=evidence,
        )

    task = resolve_task_type(df[config.target], config.task_type)
    params: dict[str, Any] = {"test_size": test_size, "target": config.target}
    warnings: list[str] = []
    assumptions: list[str] = []
    if scenario is Scenario.INDEPENDENT:
        min_class = evidence["target"]["min_class_count"]
        if task is TaskType.CLASSIFICATION and min_class >= 2:
            strategy = SplitStrategy.STRATIFIED_RANDOM
            rationale = [
                "Independent classification rows: a seeded stratified random split keeps "
                "class proportions similar in train and test."
            ]
        else:
            strategy = SplitStrategy.RANDOM
            rationale = ["Independent observations: a seeded random split is appropriate."]
            if task is TaskType.CLASSIFICATION:
                warnings.append(
                    "Some class has fewer than 2 rows, so stratification is not possible."
                )
        params["seed"] = seed
        assumptions.append("Rows are independent and identically distributed.")
        if entity and entity["n_entities_with_multiple_rows"]:
            warnings.append(
                "Entities repeat across rows; a random split will put the same entity in both "
                "partitions. Confirm that rows really are independent."
            )
        if "time" in evidence:
            warnings.append(
                "Rows have timestamps; a random split mixes periods. Confirm this is intended."
            )
    elif scenario is Scenario.NEW_ENTITIES:
        strategy = SplitStrategy.GROUP
        params.update(group_column=config.entity_id, seed=seed)
        rationale = [
            f"The model must generalise to unseen entities, so all rows of an entity in "
            f"{config.entity_id!r} go to the same partition."
        ]
        assumptions.append(
            f"{config.entity_id!r} identifies the unit that must be unseen at test time."
        )
        if entity and entity["n_entities"] < 10:
            warnings.append("Fewer than 10 entities: test metrics will be very noisy.")
    elif scenario is Scenario.FORECASTING:
        strategy = SplitStrategy.TIME
        params["time_column"] = config.prediction_timestamp
        rationale = [
            "The model predicts future periods, so train on earlier rows and test on later rows."
        ]
        assumptions.append("The most recent period is representative of deployment.")
    else:
        strategy = SplitStrategy.GROUP_AND_TIME
        params.update(group_column=config.entity_id, time_column=config.prediction_timestamp)
        rationale = [
            "Both unseen entities and future periods matter; evaluation must respect both."
        ]
        warnings.append(
            "Group-and-time evaluation is a review recommendation: design it explicitly (for "
            "example hold out later periods and unseen entities). Automatic creation is not "
            "implemented in this version."
        )
        assumptions.append("Entity and time constraints are both required at deployment.")
    return SplitRecommendation(
        status="recommended",
        strategy=strategy.value,
        rationale=rationale + hints,
        assumptions=assumptions,
        limitations=list(_COMMON_LIMITATIONS),
        warnings=warnings,
        parameters=params,
        evidence={**evidence, "task_type": task.value},
    )


# ----------------------------------------------------------------------- creation


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    report: dict[str, Any]


def create_split(
    df: pd.DataFrame,
    recommendation: SplitRecommendation,
    *,
    confirmed_strategy: str,
    config: AuditConfig,
    test_size: float | None = None,
    seed: int | None = None,
) -> SplitResult:
    """Create a split, but only for a confirmed recommended strategy.

    Rows are never dropped: an error is raised if any row cannot be placed
    (for example a missing timestamp or group value). Original row order is
    kept within each partition.
    """
    if recommendation.status != "recommended":
        raise SplitConfirmationError(
            "No strategy has been recommended; supply the missing information first."
        )
    if confirmed_strategy != recommendation.strategy:
        raise SplitConfirmationError(
            f"Confirmed strategy {confirmed_strategy!r} does not match the recommended "
            f"strategy {recommendation.strategy!r}."
        )
    strategy = SplitStrategy(recommendation.strategy)
    if strategy is SplitStrategy.GROUP_AND_TIME:
        raise NotImplementedError(
            "group_and_time is a review recommendation only; design this evaluation explicitly."
        )
    params = recommendation.parameters
    test_size = float(
        params.get("test_size", DEFAULT_TEST_SIZE) if test_size is None else test_size
    )
    seed = int(params.get("seed", DEFAULT_SEED) if seed is None else seed)
    _check_test_size(test_size)
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    warnings: list[str] = []
    n = len(df)

    if strategy is SplitStrategy.RANDOM:
        order = rng.permutation(n)
        test_idx = order[: max(1, round(n * test_size))]
    elif strategy is SplitStrategy.STRATIFIED_RANDOM:
        labels = canonical_series(df[config.target])
        if labels.isna().any():
            raise _row_error("target", int(labels.isna().sum()))
        picked = []
        for label in sorted(labels.unique()):
            members = np.flatnonzero((labels == label).to_numpy())
            if len(members) < 2:
                warnings.append("A class with a single row was kept in train.")
                continue
            k = min(len(members) - 1, max(1, round(len(members) * test_size)))
            picked.append(rng.permutation(members)[:k])
        test_idx = np.concatenate(picked) if picked else np.array([], dtype=int)
    elif strategy is SplitStrategy.GROUP:
        groups = canonical_series(df[params["group_column"]])
        if groups.isna().any():
            raise _row_error("group", int(groups.isna().sum()))
        sizes = groups.value_counts().sort_index()
        order = rng.permutation(len(sizes))
        chosen, count = [], 0
        for i in order:
            if count >= n * test_size:
                break
            chosen.append(sizes.index[i])
            count += int(sizes.iloc[i])
        if len(chosen) == len(sizes):
            chosen.pop()
        test_idx = np.flatnonzero(groups.isin(chosen).to_numpy())
    else:  # TIME
        stamps, issues = parse_timestamps(df[params["time_column"]], "time_column", "dataset", 1.0)
        raise_if_issues(issues)
        if stamps.isna().any():
            raise _row_error("timestamp", int(stamps.isna().sum()))
        ordered = stamps.sort_values(kind="mergesort")
        cutoff_pos = min(n - 1, math.floor(n * (1 - test_size)))
        boundary = ordered.iloc[cutoff_pos]
        test_idx = np.flatnonzero((stamps >= boundary).to_numpy())
        if len(test_idx) == n:
            raise InputValidationError(
                [
                    ValidationIssue(
                        "SPLIT_IMPOSSIBLE", "All rows share the earliest cutoff timestamp."
                    )
                ]
            )
        if int((stamps == boundary).sum()) > 1:
            warnings.append(
                "Several rows share the cutoff timestamp; all of them went to test so no "
                "timestamp is split across partitions."
            )

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    if test_mask.all() or not test_mask.any():
        raise InputValidationError(
            [ValidationIssue("SPLIT_IMPOSSIBLE", "The requested split leaves a partition empty.")]
        )
    train_df, test_df = (
        df.loc[~test_mask].reset_index(drop=True),
        df.loc[test_mask].reset_index(drop=True),
    )
    report = _split_report(train_df, test_df, strategy, params, config, test_size, seed, warnings)
    return SplitResult(train_df, test_df, report)


def _row_error(what: str, count: int) -> InputValidationError:
    return InputValidationError(
        [
            ValidationIssue(
                "SPLIT_MISSING_VALUES",
                f"{count} rows have a missing {what} and cannot be placed; rows are never dropped automatically.",
            )
        ]
    )


def _split_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    strategy: SplitStrategy,
    params: dict[str, Any],
    config: AuditConfig,
    test_size: float,
    seed: int,
    warnings: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "strategy": strategy.value,
        "requested_test_size": test_size,
        "seed": seed
        if strategy in (SplitStrategy.RANDOM, SplitStrategy.STRATIFIED_RANDOM, SplitStrategy.GROUP)
        else None,
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "actual_test_fraction": len(test) / (len(train) + len(test)),
        "warnings": warnings,
        "limitations": [
            "The split separates rows only by the chosen rule. It does not remove duplicates, "
            "target copies or post-outcome features. Run `splitguard audit` on the result.",
        ],
    }
    target = config.target
    if target and target in train.columns:
        task = resolve_task_type(pd.concat([train[target], test[target]]), config.task_type)
        if task is TaskType.CLASSIFICATION:
            report["class_balance"] = {
                name: canonical_series(part[target]).value_counts().sort_index().to_dict()
                for name, part in (("train", train), ("test", test))
            }
    if "group_column" in params:
        col = params["group_column"]
        shared = set(canonical_series(train[col])) & set(canonical_series(test[col]))
        report["group_overlap"] = {"column": col, "n_shared_groups": len(shared)}
    if "time_column" in params:
        col = params["time_column"]
        tr, _ = parse_timestamps(train[col], "time", "train", 1.0)
        te, _ = parse_timestamps(test[col], "time", "test", 1.0)
        report["time_boundaries"] = {
            "column": col,
            "train_min": tr.min(),
            "train_max": tr.max(),
            "test_min": te.min(),
            "test_max": te.max(),
        }
    return to_jsonable(report)
