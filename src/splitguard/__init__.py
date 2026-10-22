"""SplitGuard: evidence-driven audits for ML data leakage and train/test contamination."""

from splitguard._version import __version__
from splitguard.audit import audit_split_column, audit_train_test, inspect_unsplit
from splitguard.config import AuditConfig, ConfigError, Limits, Scenario, TaskType, Thresholds
from splitguard.schemas import (
    AuditResult,
    Category,
    CheckRecord,
    CheckStatus,
    Confidence,
    Finding,
    Severity,
    SplitRecommendation,
)
from splitguard.splitting import SplitStrategy, create_split, recommend_split
from splitguard.validation import InputValidationError, ValidationIssue

__all__ = [
    "AuditConfig",
    "AuditResult",
    "Category",
    "CheckRecord",
    "CheckStatus",
    "ConfigError",
    "Confidence",
    "Finding",
    "InputValidationError",
    "Limits",
    "Scenario",
    "Severity",
    "SplitRecommendation",
    "SplitStrategy",
    "TaskType",
    "Thresholds",
    "ValidationIssue",
    "__version__",
    "audit_split_column",
    "audit_train_test",
    "create_split",
    "inspect_unsplit",
    "recommend_split",
]
