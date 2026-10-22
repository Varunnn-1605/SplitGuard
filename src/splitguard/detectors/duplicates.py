"""Detector A: exact duplicates and cross-split contamination.

Only *exact* matches are detected (see :mod:`splitguard.hashing` for the
comparison semantics). Near-duplicate / fuzzy matching is future work.
Nothing is ever dropped: the detector only counts.
"""

from __future__ import annotations

import numpy as np

from splitguard.context import ROW_COMPARISON_SEMANTICS, AuditContext, DetectorResult
from splitguard.schemas import Confidence, Severity

_NO_SPLIT = (
    "No train/test split exists (unsplit mode); cross-partition contamination was not audited."
)


def run(ctx: AuditContext) -> DetectorResult:
    result = DetectorResult()
    _within_partitions(ctx, result)
    if not ctx.has_split:
        for rule_id in ("DUP-002", "DUP-003", "DUP-004"):
            result.skipped(rule_id, _NO_SPLIT)
        return result
    exact_mask = _cross_exact(ctx, result)
    _entity_overlap(ctx, result)
    _cross_features(ctx, result, exact_mask)
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _within_partitions(ctx: AuditContext, result: DetectorResult) -> None:
    details = {}
    for name, df in ctx.partitions.items():
        columns = list(df.columns)
        hashes = ctx.row_hashes(name, columns)
        extra = int(hashes.duplicated(keep="first").sum())
        in_groups = hashes.duplicated(keep=False)
        stats = {
            "partition": name,
            "n_rows": len(df),
            "n_columns_compared": len(columns),
            "n_duplicate_rows_beyond_first": extra,
            "duplicate_rate": _rate(extra, len(df)),
            "n_rows_in_duplicate_groups": int(in_groups.sum()),
            "n_duplicate_groups": int(hashes[in_groups].nunique()),
            "largest_group_size": int(hashes[in_groups].value_counts().max()) if extra else 0,
        }
        details[name] = stats
        if not extra:
            continue
        rate = stats["duplicate_rate"]
        result.add(
            "DUP-001",
            name,
            severity=Severity.MEDIUM if rate >= 0.05 else Severity.LOW,
            confidence=Confidence.HIGH,
            title=f"{extra} exact duplicate row(s) within the {name} partition ({rate:.1%})",
            description=(
                f"{stats['n_rows_in_duplicate_groups']} rows of the {name} partition fall into "
                f"{stats['n_duplicate_groups']} groups of identical rows across all "
                f"{len(columns)} columns. This is a within-partition count, not contamination "
                "between train and test."
            ),
            evidence={**stats, "comparison": ROW_COMPARISON_SEMANTICS},
            affected_columns=[],
            recommended_action=(
                "Check whether repeated rows are genuine repeated observations or a collection "
                "or join artefact. Decide deliberately whether to keep, weight or deduplicate them."
            ),
            limitations=[
                "Repeated rows can be legitimate (for example identical transactions).",
                "Rows that differ only in a unique ID column are not counted as duplicates.",
            ],
        )
    result.ran("DUP-001", details)


def _cross_exact(ctx: AuditContext, result: DetectorResult) -> np.ndarray | None:
    train, test = ctx.partitions["train"], ctx.partitions["test"]
    common = sorted(set(train.columns) & set(test.columns))
    not_compared = sorted(set(train.columns) ^ set(test.columns))
    train_h, test_h = ctx.row_hashes("train", common), ctx.row_hashes("test", common)
    in_train = test_h.isin(set(train_h)).to_numpy()
    n_test = int(in_train.sum())
    shared = set(test_h[in_train])
    stats = {
        "n_columns_compared": len(common),
        "columns_not_compared": not_compared,
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_test_rows_also_in_train": n_test,
        "test_rate": _rate(n_test, len(test)),
        "n_train_rows_also_in_test": int(train_h.isin(shared).sum()),
        "n_distinct_shared_rows": len(shared),
        "train_row_uniqueness": train_h.nunique() / len(train),
    }
    result.ran("DUP-002", stats)
    if not n_test:
        return in_train
    rate = stats["test_rate"]
    th = ctx.config.thresholds
    if stats["train_row_uniqueness"] < th.feature_row_low_uniqueness:
        # Same guard as DUP-004: with few possible rows, repeats are expected by chance.
        severity, confidence = Severity.LOW, Confidence.LOW
        note = (
            f" Only {stats['train_row_uniqueness']:.1%} of training rows are distinct, so the "
            "row space is small and identical rows across partitions are expected by chance."
        )
    else:
        critical = rate >= th.cross_duplicate_critical_rate
        severity, confidence = (Severity.CRITICAL if critical else Severity.HIGH), Confidence.HIGH
        note = " A model can score well on these rows by memorisation, inflating test metrics."
    result.add(
        "DUP-002",
        "train_test",
        severity=severity,
        confidence=confidence,
        title=f"{n_test} test row(s) ({rate:.1%}) also appear exactly in train",
        description=(
            f"{n_test} of {len(test)} test rows are identical to at least one training row "
            f"across all {len(common)} shared columns ({len(shared)} distinct rows)." + note
        ),
        evidence={**stats, "comparison": ROW_COMPARISON_SEMANTICS},
        affected_columns=[],
        recommended_action=(
            "Find out why the same records are in both partitions (duplicated source data, "
            "overlapping extracts, oversampling before splitting). Re-split so identical "
            "records stay in one partition, or report metrics with and without them."
        ),
        limitations=[
            "Identical rows can be genuinely distinct observations when the feature space "
            "is small or values are coarse.",
            "Columns present in only one partition are not compared.",
        ],
    )
    return in_train


def _entity_overlap(ctx: AuditContext, result: DetectorResult) -> None:
    entity = ctx.config.entity_id
    if entity is None:
        result.skipped("DUP-003", "No entity_id configured.")
        return
    test = ctx.partitions["test"]
    train_ids, test_ids = ctx.canonical("train", entity), ctx.canonical("test", entity)
    train_set, test_set = set(train_ids.dropna()), set(test_ids.dropna())
    shared = train_set & test_set
    test_rows = int(test_ids.isin(shared).sum())
    scenario = ctx.config.scenario
    stats = {
        "entity_column": entity,
        "scenario": scenario.value,
        "n_train_entities": len(train_set),
        "n_test_entities": len(test_set),
        "n_shared_entities": len(shared),
        "shared_rate_of_test_entities": _rate(len(shared), len(test_set)),
        "n_test_rows_with_shared_entity": test_rows,
        "test_row_rate": _rate(test_rows, len(test)),
        "n_missing_entity_train": int(train_ids.isna().sum()),
        "n_missing_entity_test": int(test_ids.isna().sum()),
    }
    result.ran("DUP-003", stats)
    n_missing = stats["n_missing_entity_train"] + stats["n_missing_entity_test"]
    if n_missing:
        result.warnings.append(
            f"DUP-003: {stats['n_missing_entity_train']} train and {stats['n_missing_entity_test']} "
            f"test rows have a missing entity ID in {entity!r}; they were not checked for entity "
            "overlap."
        )
    if not shared:
        return
    if scenario.requires_entity_independence:
        severity, confidence = Severity.HIGH, Confidence.HIGH
        context = (
            f"The declared scenario ({scenario.value}) requires generalising to unseen "
            "entities, so test metrics on these rows overstate real performance."
        )
    elif scenario.value == "unknown":
        severity, confidence = Severity.MEDIUM, Confidence.LOW
        context = (
            "No evaluation scenario was declared. This only matters if the model must "
            "work for entities it has never seen."
        )
    else:
        severity, confidence = Severity.INFO, Confidence.HIGH
        context = (
            f"For the declared scenario ({scenario.value}) repeated entities across "
            "partitions are expected and not by themselves leakage."
        )
    result.add(
        "DUP-003",
        entity,
        severity=severity,
        confidence=confidence,
        title=f"{len(shared)} entity ID(s) in {entity!r} appear in both train and test",
        description=(
            f"{len(shared)} of {len(test_set)} test entities ({stats['shared_rate_of_test_entities']:.1%}) "
            f"also occur in train, covering {test_rows} test rows. {context}"
        ),
        evidence=stats,
        affected_columns=[entity],
        recommended_action=(
            "If the model must handle new entities, use a group-based split on "
            f"{entity!r} so each entity appears in only one partition."
        ),
        limitations=[
            "A repeated entity ID is not proof of leakage; it depends on the deployment scenario.",
            "Entity IDs are compared exactly; different spellings of the same entity are not matched.",
        ],
    )


def _cross_features(
    ctx: AuditContext, result: DetectorResult, exact_mask: np.ndarray | None
) -> None:
    features = ctx.common_feature_columns()
    if not features:
        result.skipped("DUP-004", "No feature columns are shared by train and test.")
        return
    train, test = ctx.partitions["train"], ctx.partitions["test"]
    train_h, test_h = ctx.row_hashes("train", features), ctx.row_hashes("test", features)
    matched = test_h.isin(set(train_h)).to_numpy()
    n_matched = int(matched.sum())
    also_exact = int((matched & exact_mask).sum()) if exact_mask is not None else 0
    feature_only = n_matched - also_exact
    train_uniqueness = train_h.nunique() / len(train)
    stats = {
        "n_feature_columns_compared": len(features),
        "excluded_from_comparison": sorted(set(test.columns) - set(features)),
        "n_test_rows_matching_train_features": n_matched,
        "test_rate": _rate(n_matched, len(test)),
        "n_also_exact_full_row_matches": also_exact,
        "n_feature_only_matches": feature_only,
        "train_feature_row_uniqueness": train_uniqueness,
    }
    target = ctx.config.target
    if n_matched and target and ctx.target_in("train") and ctx.target_in("test"):
        train_pairs = set(zip(train_h, ctx.canonical("train", target), strict=True))
        test_t = ctx.canonical("test", target)
        same = sum(
            (h, t) in train_pairs for h, t in zip(test_h[matched], test_t[matched], strict=True)
        )
        stats["n_matches_with_same_target_in_train"] = int(same)
        stats["n_matches_with_only_different_target_in_train"] = int(n_matched - same)
    result.ran("DUP-004", stats)
    if not feature_only:
        return
    natural = train_uniqueness < ctx.config.thresholds.feature_row_low_uniqueness
    if natural:
        severity, confidence = Severity.LOW, Confidence.LOW
        note = (
            f"Only {train_uniqueness:.1%} of training feature rows are distinct, so the feature "
            "space is small and identical feature rows are expected by chance."
        )
    else:
        severity, confidence = Severity.HIGH, Confidence.MEDIUM
        note = (
            f"{train_uniqueness:.1%} of training feature rows are distinct, so repeated "
            "feature rows across partitions are unlikely to be coincidental."
        )
    result.add(
        "DUP-004",
        "train_test",
        severity=severity,
        confidence=confidence,
        title=f"{n_matched} test row(s) share an identical feature vector with train",
        description=(
            f"{n_matched} test rows ({stats['test_rate']:.1%}) have exactly the same values "
            f"as some training row on all {len(features)} feature columns (target and "
            f"metadata excluded); {feature_only} of them are not full-row duplicates (DUP-002). "
            + note
        ),
        evidence={**stats, "comparison": ROW_COMPARISON_SEMANTICS},
        affected_columns=features,
        recommended_action=(
            "Check whether these are the same underlying records (for example re-labelled "
            "or re-timestamped copies). If so, keep each record in one partition."
        ),
        limitations=[
            "Distinct real-world observations can have identical features.",
            "Overlaps with DUP-002 describe the same rows and are not independent evidence.",
        ],
    )


__all__ = ["run"]
