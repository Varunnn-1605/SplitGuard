"""Detector B: target leakage candidates.

Rules, from strongest to weakest evidence:

* TGT-001  feature equals the target on every row (exactly, after safe
  normalisation, or within numeric tolerance).
* TGT-002  feature equals the target on nearly every row and agreement is
  far above chance (Cohen's kappa), for classification and regression alike.

Equality rules are not applied when the target has a single value among the
rows where the feature is present: agreement with one value is uninformative.
* TGT-003  feature is a deterministic re-encoding: a one-to-one relabelling
  (classification), an exact affine transform, or a strictly monotonic
  transform (regression).
* TGT-004  for binary targets, whether the feature is missing determines the class.
* TGT-005  unusually strong univariate association. **Review signal only**:
  strong legitimate predictors look exactly like this.
* TGT-006  target-like column names. Low-confidence prompt only.

All scans run on the training partition (or the whole dataset in unsplit
mode). Nothing is fitted and no thresholds are tuned on the data. With
``test_target_diagnostics`` enabled, TGT-001..004 are also run on the test
partition and reported as diagnostics only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from splitguard.config import Thresholds
from splitguard.context import AuditContext, DetectorResult
from splitguard.schemas import Confidence, Severity
from splitguard.validation import is_numeric_feature, name_tokens

_STRONG_RULES = ("TGT-001", "TGT-002", "TGT-003", "TGT-004")
_TARGET_WORDS = {"target", "label", "labels", "outcome", "groundtruth", "truth"}
_MONOTONIC_TOL = 1e-9
_AFFINE_TOL = 1e-6


def run(ctx: AuditContext) -> DetectorResult:
    result = DetectorResult()
    cfg = ctx.config
    if cfg.target is None:
        for rule_id in (*_STRONG_RULES, "TGT-005", "TGT-006"):
            result.skipped(rule_id, "No target column configured.")
        return result
    if not ctx.target_usable:
        reason = "Target has fewer than two distinct values (see SCH-008)."
        for rule_id in (*_STRONG_RULES, "TGT-005"):
            result.skipped(rule_id, reason)
        _name_check(ctx, result, flagged=set())
        return result

    scan = _scan(ctx, ctx.primary_name, result, diagnostic=False)
    diagnostics: dict[str, Any] = {}
    if cfg.test_target_diagnostics:
        if ctx.has_split and ctx.target_in("test"):
            diagnostics = _scan(ctx, "test", result, diagnostic=True)
        else:
            diagnostics = {"skipped": "No test partition with a target column."}

    for rule_id in _STRONG_RULES:
        details = {"partition": ctx.primary_name, **scan["common"]}
        if diagnostics:
            details["test_diagnostic"] = diagnostics.get("common", diagnostics)
        if rule_id == "TGT-004" and not scan["binary"]:
            result.skipped(rule_id, "Missingness rule applies to binary targets only.", details)
        else:
            result.ran(rule_id, details)
    result.ran("TGT-005", {"partition": ctx.primary_name, **scan["association"]})
    _name_check(ctx, result, flagged=scan["flagged"])
    return result


# ------------------------------------------------------------------------- scanning


def _scan(ctx: AuditContext, partition: str, result: DetectorResult, diagnostic: bool) -> dict:
    cfg, th = ctx.config, ctx.config.thresholds
    df = ctx.partitions[partition]
    target = df[cfg.target]
    is_cls = ctx.is_classification
    target_canon = ctx.canonical(partition, cfg.target)
    target_norm = _normalise(target_canon)
    n_classes = int(target_canon.nunique(dropna=True))
    binary = is_cls and n_classes == 2
    features = [c for c in ctx.feature_columns if c in df.columns]

    flagged: set[str] = set()
    too_few: list[str] = []
    constant_target: list[str] = []
    associations: list[dict[str, Any]] = []
    assoc_skipped: dict[str, str] = {}

    for column in features:
        feature = df[column]
        if binary:
            missing = _missingness_rule(feature, target_canon, th)
            if missing:
                _emit(result, "TGT-004", column, partition, diagnostic, missing)
                flagged.add(column)
                continue
        mask = (feature.notna() & target.notna()).to_numpy()
        n_pairs = int(mask.sum())
        if n_pairs < th.min_rows_for_target_checks:
            too_few.append(column)
            continue
        if target_canon[mask].nunique() < 2:
            # Equality with a single class value carries no information about a copy.
            constant_target.append(column)
            continue
        f, t = feature[mask], target[mask]
        f_canon = ctx.canonical(partition, column)[mask]
        hit = _copy_or_encoding(f, t, f_canon, target_canon[mask], target_norm[mask], is_cls, th)
        if hit:
            _emit(result, hit.pop("rule_id"), column, partition, diagnostic, hit)
            flagged.add(column)
            continue
        if diagnostic:
            continue
        stat = _association(f, t, f_canon, target_canon[mask], is_cls, binary, n_classes, th)
        if "skipped" in stat:
            assoc_skipped[column] = stat["skipped"]
            continue
        associations.append({"column": column, **stat})
        if stat["flagged"]:
            _emit_association(result, column, partition, stat)

    associations.sort(key=lambda a: (-abs(a["value"]), a["column"]))
    label = f"{partition} partition" + (" (test diagnostic)" if diagnostic else "")
    if too_few:
        result.warnings.append(
            f"TGT-001..005: {len(too_few)} feature(s) in the {label} have too few non-missing "
            f"pairs with the target (< {th.min_rows_for_target_checks}) and were not assessed: "
            f"{too_few[:10]}"
        )
    if constant_target:
        result.warnings.append(
            f"TGT-001..005: for {len(constant_target)} feature(s) in the {label} the target "
            "has a single value wherever the feature is present, so equality and association "
            f"checks are uninformative and were not applied: {constant_target[:10]}"
        )
    return {
        "binary": binary,
        "flagged": flagged,
        "common": {
            "n_features_scanned": len(features),
            "features_with_too_few_pairs": too_few,
            "features_with_constant_target_in_pairs": constant_target,
            "min_pairs": th.min_rows_for_target_checks,
            "n_flagged": len(flagged),
        },
        "association": {
            "n_features_evaluated": len(associations),
            "strongest": associations[:10],
            "skipped": assoc_skipped,
            "note": "Associations are review signals, never proof of leakage.",
        },
    }


def _copy_or_encoding(
    f: pd.Series,
    t: pd.Series,
    f_canon: pd.Series,
    t_canon: pd.Series,
    t_norm: pd.Series,
    is_cls: bool,
    th: Thresholds,
) -> dict[str, Any] | None:
    """``f``/``t`` are aligned non-missing pairs; ``*_canon`` their canonical strings."""
    n = len(f)
    exact = float((f_canon.to_numpy() == t_canon.to_numpy()).mean())
    # Canonical numbers and booleans are already normalised; only text needs trimming.
    f_norm = (
        f_canon if is_numeric_feature(f) or ptypes.is_bool_dtype(f.dtype) else _normalise(f_canon)
    )
    normalised = float((f_norm.to_numpy() == t_norm.to_numpy()).mean())
    close = None
    if is_numeric_feature(f) and is_numeric_feature(t):
        fx, tx = f.to_numpy(dtype=float), t.to_numpy(dtype=float)
        close = float(np.isclose(fx, tx, rtol=th.numeric_rtol, atol=th.numeric_atol).mean())
    base = {
        "n_pairs": n,
        "exact_match_rate": exact,
        "normalised_match_rate": normalised,
        "numeric_close_rate": close,
        "tolerance": {"rtol": th.numeric_rtol, "atol": th.numeric_atol}
        if close is not None
        else None,
    }
    if exact == 1.0:
        return {"rule_id": "TGT-001", "comparison": "exact", **base}
    if normalised == 1.0:
        return {
            "rule_id": "TGT-001",
            "comparison": "normalised (trimmed, case-folded text)",
            **base,
        }
    if close == 1.0:
        return {"rule_id": "TGT-001", "comparison": "numeric, within tolerance", **base}

    best = max(normalised, close or 0.0)
    if best >= th.near_copy_match_rate:
        # Chance-correct the agreement for every task: a zero-inflated regression target
        # agrees with any mostly-zero feature on most rows without being copied.
        chance = _chance_agreement(f_norm, t_norm)
        kappa = None if chance >= 1.0 else (best - chance) / (1.0 - chance)
        if kappa is not None and kappa >= th.near_copy_min_kappa:
            return {
                "rule_id": "TGT-002",
                "match_rate": best,
                "match_rate_threshold": th.near_copy_match_rate,
                "chance_agreement": chance,
                "cohens_kappa": kappa,
                "min_kappa": th.near_copy_min_kappa,
                **base,
            }

    if is_cls:
        if _is_bijection(f_norm, t_norm):
            return {
                "rule_id": "TGT-003",
                "relationship": "one-to-one relabelling",
                "n_distinct_values": int(f_norm.nunique()),
                **base,
            }
        return None
    if not is_numeric_feature(f):
        return None
    fx, tx = f.to_numpy(dtype=float), t.to_numpy(dtype=float)
    if not (np.isfinite(fx).all() and np.isfinite(tx).all()) or fx.std() == 0 or tx.std() == 0:
        return None
    design = np.column_stack([fx, np.ones_like(fx)])
    (slope, intercept), *_ = np.linalg.lstsq(design, tx, rcond=None)
    max_rel_residual = float(np.max(np.abs(tx - (slope * fx + intercept))) / tx.std())
    if max_rel_residual <= _AFFINE_TOL:
        return {
            "rule_id": "TGT-003",
            "relationship": "exact affine transform (target = slope * feature + intercept)",
            "slope": float(f"{slope:.10g}"),
            "intercept": float(f"{intercept:.10g}"),
            "max_relative_residual": max_rel_residual,
            **base,
        }
    spearman = _pearson(_ranks(fx), _ranks(tx))
    if (
        spearman is not None
        and abs(spearman) >= 1 - _MONOTONIC_TOL
        and len(np.unique(tx)) >= 10
        and len(np.unique(fx)) >= 10
    ):
        return {
            "rule_id": "TGT-003",
            "relationship": "strictly monotonic transform (Spearman rho = +/-1)",
            "spearman": spearman,
            **base,
        }
    return None


def _missingness_rule(feature: pd.Series, t_canon: pd.Series, th: Thresholds) -> dict | None:
    has_target = t_canon.notna().to_numpy()
    if has_target.sum() < th.min_rows_for_target_checks:
        return None
    missing = feature.isna().to_numpy()[has_target]
    if missing.all() or not missing.any():
        return None
    pairs = pd.DataFrame({"m": missing, "t": t_canon.to_numpy()[has_target]}).drop_duplicates()
    if len(pairs) != 2 or pairs["t"].nunique() != 2:
        return None
    return {
        "n_rows": int(has_target.sum()),
        "n_missing": int(missing.sum()),
        "missing_rate": float(missing.mean()),
        "relationship": (
            "every row where the feature is missing has one target class and every row "
            "where it is present has the other"
        ),
    }


# ------------------------------------------------------------------------- statistics


def _normalise(canon: pd.Series) -> pd.Series:
    """Trim and case-fold canonical strings (missing values stay missing)."""
    present = canon.notna()
    if not present.any():
        return canon
    out = canon.copy()
    out[present] = canon[present].str.strip().str.casefold()
    return out


def _is_bijection(a: pd.Series, b: pd.Series) -> bool:
    pairs = pd.DataFrame({"a": a.to_numpy(), "b": b.to_numpy()}).drop_duplicates()
    n_a, n_b = pairs["a"].nunique(), pairs["b"].nunique()
    return n_a >= 2 and len(pairs) == n_a == n_b


def _chance_agreement(a: pd.Series, b: pd.Series) -> float:
    """Probability that independent draws from the two value distributions are equal."""
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return float(sum(pa.get(k, 0.0) * pb.get(k, 0.0) for k in pa.index))


def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom == 0:
        return None
    return max(-1.0, min(1.0, float((x * y).sum()) / denom))


def _auc(x: np.ndarray, positive: np.ndarray) -> float | None:
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = _ranks(x)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _eta(groups: np.ndarray, values: np.ndarray) -> float | None:
    """Correlation ratio: share of variance in ``values`` explained by ``groups``."""
    overall = values.mean()
    total = float(((values - overall) ** 2).sum())
    if total == 0:
        return None
    stats = pd.DataFrame({"g": groups, "v": values}).groupby("g")["v"].agg(["count", "mean"])
    between = float((stats["count"] * (stats["mean"] - overall) ** 2).sum())
    return math.sqrt(min(1.0, between / total))


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float | None:
    """Bias-corrected Cramer's V (Bergsma 2013)."""
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    n = table.sum()
    r, k = table.shape
    if n < 2 or r < 2 or k < 2:
        return None
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return None
    return math.sqrt(min(1.0, phi2 / denom))


def _association(
    f: pd.Series,
    t: pd.Series,
    f_canon: pd.Series,
    t_canon: pd.Series,
    is_cls: bool,
    binary: bool,
    n_classes: int,
    th: Thresholds,
) -> dict[str, Any]:
    n = len(f)
    n_unique = int(f_canon.nunique())
    if n_unique <= 1:
        return {"skipped": "constant within non-missing pairs"}
    numeric = is_numeric_feature(f)
    categorical_ok = (
        n_unique <= th.max_categories_for_association
        and n_unique / n <= th.max_unique_ratio_for_categorical
    )
    fx = f.to_numpy(dtype=float) if numeric else None

    if not is_cls:
        tx = t.to_numpy(dtype=float)
        if numeric:
            pearson = _pearson(fx, tx)
            spearman = _pearson(_ranks(fx), _ranks(tx))
            if pearson is None or spearman is None:
                return {"skipped": "zero variance"}
            value = max(abs(pearson), abs(spearman))
            return _stat(
                "max(|pearson|, |spearman|)",
                value,
                th.strong_correlation,
                n,
                pearson=pearson,
                spearman=spearman,
            )
        if not categorical_ok:
            return {"skipped": "high-cardinality text; association not computed"}
        eta = _eta(f_canon.to_numpy(), tx)
        if eta is None:
            return {"skipped": "zero variance"}
        return _stat("correlation ratio (eta)", eta, th.strong_eta, n)

    if n_classes > th.max_categories_for_association:
        return {"skipped": "target has too many classes for association measures"}
    if numeric:
        if binary:
            labels = sorted(t_canon.unique())
            if len(labels) < 2:
                return {"skipped": "single class within pairs"}
            auc = _auc(fx, (t_canon == labels[1]).to_numpy())
            if auc is None:
                return {"skipped": "single class within pairs"}
            return _stat(
                "single-feature ROC AUC, max(auc, 1 - auc)",
                max(auc, 1 - auc),
                th.strong_auc,
                n,
                auc=auc,
            )
        eta = _eta(t_canon.to_numpy(), fx)
        if eta is None:
            return {"skipped": "zero variance"}
        return _stat("correlation ratio (eta)", eta, th.strong_eta, n)
    if not categorical_ok:
        return {"skipped": "high-cardinality text; association not computed"}
    v = _cramers_v(f_canon.to_numpy(), t_canon.to_numpy())
    if v is None:
        return {"skipped": "contingency table too small"}
    return _stat("bias-corrected Cramer's V", v, th.strong_cramers_v, n)


def _stat(name: str, value: float, threshold: float, n: int, **extra: float) -> dict[str, Any]:
    return {
        "statistic": name,
        "value": float(value),
        "threshold": threshold,
        "n_pairs": n,
        "flagged": bool(value >= threshold),
        **{k: float(v) for k, v in extra.items()},
    }


# --------------------------------------------------------------------------- emitting

_COPY_TEXT = {
    "TGT-001": dict(
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        title="Feature {col!r} is identical to the target",
        action="Remove {col!r} from model inputs unless you can show it is known at prediction "
        "time by an independent process.",
        limits=[
            "Equality shows the column duplicates the label, not how it got there; it may be "
            "a bookkeeping copy that was never meant to be a feature.",
        ],
    ),
    "TGT-002": dict(
        severity=Severity.HIGH,
        confidence=Confidence.MEDIUM,
        title="Feature {col!r} nearly equals the target",
        action="Find out how {col!r} is produced. If it is derived from the label or recorded "
        "after the outcome, remove it.",
        limits=[
            "A very accurate legitimate measurement (for example a preliminary reading of the "
            "same quantity) can agree with the target almost always.",
        ],
    ),
    "TGT-003": dict(
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        title="Feature {col!r} is a deterministic re-encoding of the target",
        action="Treat {col!r} as a copy of the label unless its derivation is known to be "
        "independent of the outcome.",
        limits=[
            "A deterministic relationship can be legitimate when the target is itself defined "
            "by a business rule over this feature; then the task may be trivial rather than leaked.",
        ],
    ),
    "TGT-004": dict(
        severity=Severity.HIGH,
        confidence=Confidence.MEDIUM,
        title="Whether {col!r} is missing determines the target class",
        action="Check whether {col!r} is only filled in after the outcome is known (for example "
        "a field populated during follow-up). If so, remove it and its missingness indicator.",
        limits=[
            "Missingness can legitimately depend on the class (for example a test only ordered "
            "for one group) if it is decided before prediction time.",
        ],
    ),
}


def _emit(
    result: DetectorResult,
    rule_id: str,
    column: str,
    partition: str,
    diagnostic: bool,
    evidence: dict,
) -> None:
    text = _COPY_TEXT[rule_id]
    if rule_id == "TGT-003" and evidence.get("relationship", "").startswith("strictly monotonic"):
        severity, confidence = Severity.HIGH, Confidence.MEDIUM
    else:
        severity, confidence = text["severity"], text["confidence"]
    title = text["title"].format(col=column)
    relationship = evidence.get("comparison") or evidence.get("relationship") or "near equality"
    description = (
        f"In the {partition} partition, {column!r} relates to the target as: {relationship}. "
    )
    if diagnostic:
        title = "[test diagnostic] " + title
        description += (
            "This scan used test-partition labels for diagnosis only; do not use it to select "
            "features or tune models."
        )
    else:
        description += "No model was trained; this is a direct comparison of values."
    result.add(
        rule_id,
        f"test:{column}" if diagnostic else column,
        severity=severity,
        confidence=confidence,
        title=title,
        description=description,
        evidence={"partition": partition, "diagnostic_only": diagnostic, **evidence},
        affected_columns=[column],
        recommended_action=text["action"].format(col=column),
        limitations=list(text["limits"]),
    )


def _emit_association(result: DetectorResult, column: str, partition: str, stat: dict) -> None:
    result.add(
        "TGT-005",
        column,
        severity=Severity.MEDIUM,
        confidence=Confidence.LOW,
        title=f"Unusually strong association between {column!r} and the target (review; not proof)",
        description=(
            f"{stat['statistic']} = {stat['value']:.3f} over {stat['n_pairs']} rows of the "
            f"{partition} partition (review threshold {stat['threshold']}). Strong association "
            "alone does not establish leakage: good legitimate predictors look the same."
        ),
        evidence={"partition": partition, **stat},
        affected_columns=[column],
        recommended_action=(
            f"Confirm that {column!r} is recorded before prediction time and is not derived "
            "from the outcome. Keep it if its provenance is sound."
        ),
        limitations=[
            "Correlation is not leakage. Many strong features are legitimate.",
            "Univariate statistics miss leakage that only appears in combinations of features.",
        ],
    )


def _name_check(ctx: AuditContext, result: DetectorResult, flagged: set[str]) -> None:
    target = ctx.config.target
    target_norm = "".join(name_tokens(target))
    hits = []
    for column in ctx.feature_columns:
        if column in flagged:
            continue
        tokens = name_tokens(column)
        reasons = []
        if len(target_norm) >= 3 and target_norm in "".join(tokens):
            reasons.append(f"contains the target name {target!r}")
        words = sorted(_TARGET_WORDS & set(tokens))
        if words:
            reasons.append(f"contains target-like word(s) {words}")
        if not reasons:
            continue
        hits.append(column)
        result.add(
            "TGT-006",
            column,
            severity=Severity.LOW,
            confidence=Confidence.LOW,
            title=f"Column name {column!r} resembles the target",
            description=f"The name {column!r} " + " and ".join(reasons) + ". Names are only hints.",
            evidence={"rule": "name token match", "reasons": reasons, "target": target},
            affected_columns=[column],
            recommended_action=f"Check what {column!r} records and when it is populated.",
            limitations=["A column name says nothing reliable about its content or timing."],
        )
    result.ran(
        "TGT-006", {"n_columns_flagged": len(hits), "skipped_already_flagged": sorted(flagged)}
    )
