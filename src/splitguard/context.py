"""Shared state passed to detectors, plus the rule catalogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from splitguard.config import AuditConfig, TaskType
from splitguard.hashing import canonical_series, row_hashes
from splitguard.schemas import Category, CheckRecord, CheckStatus, Finding

# rule_id -> (check name, category). Rule IDs are stable across versions.
RULES: dict[str, tuple[str, Category]] = {
    "DUP-001": ("Exact duplicate rows within a partition", Category.DUPLICATE_CONTAMINATION),
    "DUP-002": ("Exact rows shared by train and test", Category.DUPLICATE_CONTAMINATION),
    "DUP-003": ("Entity IDs shared by train and test", Category.DUPLICATE_CONTAMINATION),
    "DUP-004": (
        "Identical feature rows shared by train and test",
        Category.DUPLICATE_CONTAMINATION,
    ),
    "TGT-001": ("Feature identical to the target", Category.TARGET_LEAKAGE),
    "TGT-002": ("Feature nearly identical to the target", Category.TARGET_LEAKAGE),
    "TGT-003": ("Feature is a deterministic re-encoding of the target", Category.TARGET_LEAKAGE),
    "TGT-004": ("Feature missingness determines the target", Category.TARGET_LEAKAGE),
    "TGT-005": ("Unusually strong univariate association with the target", Category.TARGET_LEAKAGE),
    "TGT-006": ("Target-like column name", Category.TARGET_LEAKAGE),
    "TMP-001": ("Feature available after prediction time", Category.TEMPORAL_LEAKAGE),
    "TMP-002": ("Feature available after the outcome", Category.TEMPORAL_LEAKAGE),
    "TMP-003": ("Outcome timing contradicts the prediction boundary", Category.TEMPORAL_LEAKAGE),
    "TMP-004": ("Train and test time ranges overlap", Category.TEMPORAL_LEAKAGE),
    "TMP-005": ("Temporal wording in a column name", Category.TEMPORAL_LEAKAGE),
    "ID-001": ("High-uniqueness identifier-like column", Category.IDENTIFIER_RISK),
    "ID-002": ("Likely row index column", Category.IDENTIFIER_RISK),
    "ID-003": ("Identifier values shared by train and test", Category.IDENTIFIER_RISK),
    "SCH-001": ("Columns missing from test", Category.SCHEMA_ISSUE),
    "SCH-002": ("Extra columns in test", Category.SCHEMA_ISSUE),
    "SCH-003": ("Column type differs between train and test", Category.SCHEMA_ISSUE),
    "SCH-004": ("Target absent from test", Category.SCHEMA_ISSUE),
    "SCH-005": ("Test target classes unseen in train", Category.SCHEMA_ISSUE),
    "SCH-006": ("Constant or empty columns", Category.SCHEMA_ISSUE),
    "SCH-007": ("Mostly numeric text column (mixed types)", Category.SCHEMA_ISSUE),
    "SCH-008": ("Target has fewer than two distinct values", Category.SCHEMA_ISSUE),
    "SPL-001": ("Split-strategy recommendation", Category.SPLIT_RECOMMENDATION),
}

ROW_COMPARISON_SEMANTICS = (
    "Columns aligned by name (order-independent); numbers compared by value (1 == 1.0); "
    "text compared verbatim (case and whitespace significant); missing values equal each "
    "other; rows matched with a deterministic 64-bit hash."
)


@dataclass
class AuditContext:
    """Validated inputs for one audit. Partitions are ``train``/``test`` or ``unsplit``."""

    mode: str
    config: AuditConfig
    partitions: dict[str, pd.DataFrame]
    timestamps: dict[str, dict[str, pd.Series]]
    feature_columns: list[str]
    task_type: TaskType | None
    target_usable: bool = False
    _canonical_cache: dict[tuple[str, str], pd.Series] = field(default_factory=dict, repr=False)

    def canonical(self, partition: str, column: str) -> pd.Series:
        """Canonical strings for one column (cached; see :mod:`splitguard.hashing`)."""
        key = (partition, column)
        if key not in self._canonical_cache:
            self._canonical_cache[key] = canonical_series(self.partitions[partition][column])
        return self._canonical_cache[key]

    def row_hashes(self, partition: str, columns: list[str]) -> pd.Series:
        return row_hashes(
            self.partitions[partition], columns, lambda c: self.canonical(partition, c)
        )

    @property
    def has_split(self) -> bool:
        return "test" in self.partitions

    @property
    def primary_name(self) -> str:
        return "train" if self.has_split else "unsplit"

    @property
    def primary(self) -> pd.DataFrame:
        return self.partitions[self.primary_name]

    @property
    def test(self) -> pd.DataFrame | None:
        return self.partitions.get("test")

    @property
    def is_classification(self) -> bool:
        return self.task_type is TaskType.CLASSIFICATION

    def common_feature_columns(self) -> list[str]:
        test = self.test
        if test is None:
            return list(self.feature_columns)
        return [c for c in self.feature_columns if c in test.columns]

    def target_in(self, partition: str) -> bool:
        target = self.config.target
        return target is not None and target in self.partitions[partition].columns


@dataclass
class DetectorResult:
    """Findings and check records produced by one detector."""

    findings: list[Finding] = field(default_factory=list)
    checks: list[CheckRecord] = field(default_factory=list)
    # Statements about what could not be (fully) assessed; surfaced in the report.
    warnings: list[str] = field(default_factory=list)

    def ran(self, rule_id: str, details: dict[str, Any] | None = None) -> None:
        self.checks.append(
            CheckRecord(rule_id, RULES[rule_id][0], CheckStatus.RUN, None, details or {})
        )

    def skipped(self, rule_id: str, reason: str, details: dict[str, Any] | None = None) -> None:
        self.checks.append(
            CheckRecord(rule_id, RULES[rule_id][0], CheckStatus.SKIPPED, reason, details or {})
        )

    def add(self, rule_id: str, key: str, **kwargs: Any) -> None:
        self.findings.append(
            Finding(rule_id=rule_id, key=key, category=RULES[rule_id][1], **kwargs)
        )
