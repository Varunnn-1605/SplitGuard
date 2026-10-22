"""Independent detectors. Each exposes ``run(ctx) -> DetectorResult``."""

from splitguard.detectors import duplicates, identifiers, schema, target_leakage, temporal

__all__ = ["duplicates", "identifiers", "schema", "target_leakage", "temporal"]
