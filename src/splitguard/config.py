"""Audit configuration: column roles, evaluation scenario, thresholds and limits.

Every statistical threshold lives in :class:`Thresholds` so it is visible in
the report and can be changed without touching detector code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any

import pandas as pd


class ConfigError(ValueError):
    """Raised when the configuration itself is invalid."""


class Scenario(StrEnum):
    """How the model will be used, which decides what counts as contamination.

    * ``independent``: rows are independent observations; new rows come from
      the same distribution.
    * ``new_entities``: the model must generalise to entities (customers,
      patients, devices...) never seen in training.
    * ``forecasting``: the model predicts future periods from past data.
    * ``new_entities_forecasting``: both constraints apply.
    * ``unknown``: not declared; findings that depend on the scenario use
      reduced confidence.
    """

    UNKNOWN = "unknown"
    INDEPENDENT = "independent"
    NEW_ENTITIES = "new_entities"
    FORECASTING = "forecasting"
    NEW_ENTITIES_FORECASTING = "new_entities_forecasting"

    @property
    def requires_entity_independence(self) -> bool:
        return self in (Scenario.NEW_ENTITIES, Scenario.NEW_ENTITIES_FORECASTING)

    @property
    def requires_temporal_order(self) -> bool:
        return self in (Scenario.FORECASTING, Scenario.NEW_ENTITIES_FORECASTING)


class TaskType(StrEnum):
    """``auto`` infers: non-numeric or two-valued numeric targets are
    classification; other numeric targets are regression. Override for
    numeric multiclass labels."""

    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


def _check_rate(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ConfigError(f"thresholds.{name} must be between 0 and 1, got {value!r}")


@dataclass(frozen=True)
class Thresholds:
    """Tunable decision thresholds. See README "Detectors" for their rationale."""

    # Equality comparisons for numeric values (numpy.isclose semantics).
    numeric_rtol: float = 1e-6
    numeric_atol: float = 1e-9
    # A feature matching the target on at least this share of rows is a near copy...
    near_copy_match_rate: float = 0.98
    # ...but for categorical targets only if agreement beats chance (Cohen's kappa).
    near_copy_min_kappa: float = 0.9
    # Univariate association review thresholds (never treated as proof).
    strong_correlation: float = 0.9
    strong_auc: float = 0.95
    strong_cramers_v: float = 0.9
    strong_eta: float = 0.9
    # Minimum non-missing (feature, target) pairs before target checks run.
    min_rows_for_target_checks: int = 30
    # Categorical association is only computed for low-cardinality columns.
    max_categories_for_association: int = 50
    max_unique_ratio_for_categorical: float = 0.5
    # Identifier detection.
    identifier_unique_ratio: float = 0.95
    min_rows_for_identifier: int = 20
    # Required fields (target, entity ID, timestamps) may not exceed this missing rate.
    max_missing_rate_required: float = 0.5
    # Cross-split exact-duplicate rate at which severity becomes critical.
    cross_duplicate_critical_rate: float = 0.05
    # If fewer than this share of training feature rows are distinct, identical
    # feature rows across partitions are expected and reported at low confidence.
    feature_row_low_uniqueness: float = 0.5
    # Text columns whose values are at least this share numeric are flagged as mixed.
    mostly_numeric_rate: float = 0.9

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"thresholds.{f.name} must be a number, got {value!r}")
        for name in (
            "near_copy_match_rate",
            "strong_correlation",
            "strong_cramers_v",
            "strong_eta",
            "max_unique_ratio_for_categorical",
            "identifier_unique_ratio",
            "max_missing_rate_required",
            "cross_duplicate_critical_rate",
            "feature_row_low_uniqueness",
            "mostly_numeric_rate",
        ):
            _check_rate(name, getattr(self, name))
        if not 0.5 <= self.strong_auc <= 1.0:
            raise ConfigError("thresholds.strong_auc must be between 0.5 and 1")
        if not -1.0 <= self.near_copy_min_kappa <= 1.0:
            raise ConfigError("thresholds.near_copy_min_kappa must be between -1 and 1")
        if self.numeric_rtol < 0 or self.numeric_atol < 0:
            raise ConfigError("numeric tolerances must be non-negative")
        for name in (
            "min_rows_for_target_checks",
            "max_categories_for_association",
            "min_rows_for_identifier",
        ):
            if int(getattr(self, name)) < 1:
                raise ConfigError(f"thresholds.{name} must be a positive integer")


@dataclass(frozen=True)
class Limits:
    """Resource limits checked *before* expensive work."""

    max_file_bytes: int = 200 * 1024 * 1024
    max_rows: int = 1_000_000
    max_columns: int = 2_000

    def __post_init__(self) -> None:
        for name in ("max_file_bytes", "max_rows", "max_columns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"limits.{name} must be a positive integer")


@dataclass(frozen=True)
class AuditConfig:
    """Column roles and options for an audit.

    Columns given a role here (entity ID, timestamps, availability columns,
    ``exclude_columns``) are metadata: they are not treated as model features.
    The target is also excluded from features.
    """

    target: str | None = None
    entity_id: str | None = None
    prediction_timestamp: str | None = None
    outcome_timestamp: str | None = None
    # feature column -> column holding the per-row time that feature became available.
    feature_availability: Mapping[str, str] = field(default_factory=dict)
    exclude_columns: tuple[str, ...] = ()
    scenario: Scenario = Scenario.UNKNOWN
    task_type: TaskType = TaskType.AUTO
    # Minimum required gap between prediction time and outcome time (pandas
    # Timedelta string such as "30D"). None = only check the outcome is not earlier.
    min_lead_time: str | None = None
    expect_target_in_test: bool = True
    # Also run target-copy checks on the test partition, reported as diagnostics only.
    test_target_diagnostics: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "scenario", Scenario(self.scenario))
        except ValueError as exc:
            valid = ", ".join(s.value for s in Scenario)
            raise ConfigError(f"Unknown scenario {self.scenario!r}; use one of: {valid}") from exc
        try:
            object.__setattr__(self, "task_type", TaskType(self.task_type))
        except ValueError as exc:
            valid = ", ".join(t.value for t in TaskType)
            raise ConfigError(f"Unknown task_type {self.task_type!r}; use one of: {valid}") from exc
        if isinstance(self.thresholds, Mapping):
            object.__setattr__(
                self, "thresholds", _build(Thresholds, self.thresholds, "thresholds")
            )
        if isinstance(self.limits, Mapping):
            object.__setattr__(self, "limits", _build(Limits, self.limits, "limits"))
        if isinstance(self.exclude_columns, str):
            raise ConfigError("exclude_columns must be a list of column names, not a string")
        object.__setattr__(self, "exclude_columns", tuple(sorted(set(self.exclude_columns))))
        availability = dict(self.feature_availability or {})
        for key, value in availability.items():
            if not isinstance(key, str) or not isinstance(value, str) or not key or not value:
                raise ConfigError("feature_availability must map feature names to column names")
        object.__setattr__(self, "feature_availability", dict(sorted(availability.items())))
        for name in ("target", "entity_id", "prediction_timestamp", "outcome_timestamp"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{name} must be a non-empty column name")
        self._check_roles()
        if self.min_lead_time is not None:
            delta = self.min_lead_time_delta
            if delta is None or delta < pd.Timedelta(0):
                raise ConfigError("min_lead_time must be a non-negative duration such as '30D'")

    def _check_roles(self) -> None:
        roles: dict[str, str] = {}
        for role in ("target", "entity_id", "prediction_timestamp", "outcome_timestamp"):
            column = getattr(self, role)
            if column is None:
                continue
            if column in roles:
                raise ConfigError(
                    f"Column {column!r} is assigned to both {roles[column]} and {role}"
                )
            roles[column] = role
        if self.target is not None:
            if self.target in self.exclude_columns:
                raise ConfigError("The target column cannot also be in exclude_columns")
            if self.target in self.feature_availability.values():
                raise ConfigError("The target column cannot be a feature-availability column")
            if self.target in self.feature_availability:
                raise ConfigError("The target column cannot be given feature-availability metadata")
        for feature in self.feature_availability:
            if feature in roles or feature in self.exclude_columns:
                raise ConfigError(
                    f"feature_availability key {feature!r} is a metadata column, not a feature"
                )

    @property
    def min_lead_time_delta(self) -> pd.Timedelta | None:
        if self.min_lead_time is None:
            return None
        try:
            delta = pd.Timedelta(self.min_lead_time)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"Invalid min_lead_time {self.min_lead_time!r}") from exc
        return None if pd.isna(delta) else delta

    def metadata_columns(self) -> set[str]:
        """Configured non-feature columns (the target is handled separately)."""
        cols = {
            c
            for c in (self.entity_id, self.prediction_timestamp, self.outcome_timestamp)
            if c is not None
        }
        cols.update(self.feature_availability.values())
        cols.update(self.exclude_columns)
        return cols

    def timestamp_columns(self) -> list[str]:
        """All configured timestamp columns, deduplicated, in a stable order."""
        cols = [c for c in (self.prediction_timestamp, self.outcome_timestamp) if c]
        for column in self.feature_availability.values():
            if column not in cols:
                cols.append(column)
        return cols

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scenario"] = self.scenario.value
        data["task_type"] = self.task_type.value
        data["exclude_columns"] = list(self.exclude_columns)
        data["feature_availability"] = dict(self.feature_availability)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditConfig:
        """Build a config from a plain mapping (for example a JSON config file).

        Unknown keys are rejected so that typos do not silently disable checks.
        """
        if not isinstance(data, Mapping):
            raise ConfigError("Configuration must be a JSON object")
        return _build(cls, data, "config")


def _build(cls: type, data: Mapping[str, Any], label: str) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown {label} option(s): {', '.join(unknown)}")
    kwargs = dict(data)
    if "exclude_columns" in kwargs and isinstance(kwargs["exclude_columns"], list):
        kwargs["exclude_columns"] = tuple(kwargs["exclude_columns"])
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Invalid {label}: {exc}") from exc
