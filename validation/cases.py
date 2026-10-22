"""Adversarial validation cases for SplitGuard.

Each case builds a deterministic synthetic dataset (numpy/pandas only, fixed
seeds), runs one audit, and states the expected outcome with a reason.

Unit of evaluation
------------------
One case = one injected (or deliberately absent) condition, scored once
against its *designated rule*, regardless of how many other findings the same
rows trigger. Overlapping findings (for example DUP-002 and DUP-004 on the same
rows) are never counted as extra true positives.

* ``positive``  - a leakage condition was injected. TP if the designated rule
  fires on the designated key with the expected severity/confidence, else FN.
* ``negative``  - no leakage was injected. TN if no *forbidden* finding
  appears, else FP. Low-confidence review prompts may be explicitly allowed.
* ``boundary``  - ambiguous or degenerate input. Scored pass/fail against the
  documented expected behaviour; not part of the TP/FP counts.
* ``diagnostic`` - opt-in diagnostics (test-label scans); scored pass/fail.

These are synthetic controls. They check that rules behave as specified; they
are not estimates of real-world detection accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from splitguard import (
    AuditConfig,
    AuditResult,
    InputValidationError,
    ValidationIssue,
    audit_split_column,
    audit_train_test,
    inspect_unsplit,
)

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
DAY = pd.Timedelta(days=1)


# ---------------------------------------------------------------- expectations


@dataclass
class Outcome:
    result: AuditResult | None
    error: InputValidationError | None


@dataclass(frozen=True)
class Fires:
    """Designated rule fires on ``key`` with the given ratings."""

    rule: str
    key: str
    severity: str | None = None
    confidence: str | None = None

    def evaluate(self, o: Outcome) -> tuple[bool, str]:
        if o.result is None:
            return False, f"audit raised {o.error}"
        hits = [f for f in o.result.findings if f.finding_id == f"{self.rule}:{self.key}"]
        if not hits:
            got = sorted(f.finding_id for f in o.result.findings)
            return False, f"{self.rule}:{self.key} did not fire (got {got})"
        f = hits[0]
        if self.severity and f.severity.value != self.severity:
            return False, f"severity {f.severity.value} != {self.severity}"
        if self.confidence and f.confidence.value != self.confidence:
            return False, f"confidence {f.confidence.value} != {self.confidence}"
        return True, f"{f.finding_id} {f.severity.value}/{f.confidence.value}"


@dataclass(frozen=True)
class Absent:
    """None of ``rules`` fires (optionally only for findings touching ``column``)."""

    rules: tuple[str, ...]
    column: str | None = None

    def evaluate(self, o: Outcome) -> tuple[bool, str]:
        if o.result is None:
            return False, f"audit raised {o.error}"
        bad = [
            f.finding_id
            for f in o.result.findings
            if f.rule_id in self.rules
            and (self.column is None or self.column in f.affected_columns)
        ]
        return (not bad), (f"unexpected {bad}" if bad else f"none of {list(self.rules)}")


@dataclass(frozen=True)
class AtMost:
    """Any finding of ``rule`` (on ``column``) is no worse than the given ratings."""

    rule: str
    severity: str
    confidence: str
    column: str | None = None

    def evaluate(self, o: Outcome) -> tuple[bool, str]:
        if o.result is None:
            return False, f"audit raised {o.error}"
        conf_order = ["high", "medium", "low"]
        for f in o.result.findings:
            if f.rule_id != self.rule or (self.column and self.column not in f.affected_columns):
                continue
            if SEVERITY_ORDER.index(f.severity.value) < SEVERITY_ORDER.index(self.severity):
                return False, f"{f.finding_id} severity {f.severity.value} > {self.severity}"
            if conf_order.index(f.confidence.value) < conf_order.index(self.confidence):
                return False, f"{f.finding_id} confidence {f.confidence.value} > {self.confidence}"
        return True, f"{self.rule} at most {self.severity}/{self.confidence}"


@dataclass(frozen=True)
class Skipped:
    rule: str
    reason_contains: str = ""

    def evaluate(self, o: Outcome) -> tuple[bool, str]:
        if o.result is None:
            return False, f"audit raised {o.error}"
        c = next(c for c in o.result.checks if c.check_id == self.rule)
        ok = c.status.value == "skipped" and self.reason_contains in (c.reason or "")
        return ok, f"{self.rule} {c.status.value}: {c.reason}"


@dataclass(frozen=True)
class Warns:
    contains: str

    def evaluate(self, o: Outcome) -> tuple[bool, str]:
        if o.result is None:
            return False, f"audit raised {o.error}"
        ok = any(self.contains in w for w in o.result.warnings)
        return ok, f"warning containing {self.contains!r}" + ("" if ok else " missing")


@dataclass(frozen=True)
class Detail:
    """``check(details_of_rule) -> bool`` for numeric evidence in check details."""

    rule: str
    description: str
    check: Callable[[dict], bool]

    def evaluate(self, o: Outcome) -> tuple[bool, str]:
        if o.result is None:
            return False, f"audit raised {o.error}"
        c = next(c for c in o.result.checks if c.check_id == self.rule)
        try:
            ok = bool(self.check(c.details))
        except (KeyError, TypeError, IndexError) as exc:
            return False, f"{self.description}: {type(exc).__name__} {exc}"
        return ok, self.description


@dataclass(frozen=True)
class FindingEvidence:
    rule: str
    key: str
    description: str
    check: Callable[[dict], bool]

    def evaluate(self, o: Outcome) -> tuple[bool, str]:
        if o.result is None:
            return False, f"audit raised {o.error}"
        hits = [f for f in o.result.findings if f.finding_id == f"{self.rule}:{self.key}"]
        if not hits:
            return False, f"{self.rule}:{self.key} absent"
        return bool(self.check(hits[0].evidence)), self.description


@dataclass(frozen=True)
class RaisesCode:
    code: str

    def evaluate(self, o: Outcome) -> tuple[bool, str]:
        if o.error is None:
            return False, "no validation error raised"
        codes = {i.code for i in o.error.issues}
        return self.code in codes, f"validation codes {sorted(codes)}"


@dataclass
class Case:
    case_id: str
    detector: str  # duplicates | target | temporal | identifiers | schema | validation
    kind: str  # positive | negative | boundary | diagnostic
    description: str
    reason: str
    run: Callable[[], Outcome]
    expect: list[Any]
    designated: str = ""  # rule:key (positive) or guarded rules (negative)
    tags: list[str] = field(default_factory=list)

    def evaluate(self) -> tuple[bool, list[tuple[bool, str]], Outcome]:
        outcome = self.run()
        results = [e.evaluate(outcome) for e in self.expect]
        return all(ok for ok, _ in results), results, outcome


# --------------------------------------------------------------------- helpers


def _audit(train, test, **cfg) -> Callable[[], Outcome]:
    def run() -> Outcome:
        try:
            return Outcome(audit_train_test(train, test, AuditConfig(**cfg)), None)
        except InputValidationError as exc:
            return Outcome(None, exc)

    return run


def _audit_split(data, split_column, **cfg) -> Callable[[], Outcome]:
    def run() -> Outcome:
        try:
            return Outcome(audit_split_column(data, split_column, config=AuditConfig(**cfg)), None)
        except InputValidationError as exc:
            return Outcome(None, exc)

    return run


def _inspect(data, **cfg) -> Callable[[], Outcome]:
    def run() -> Outcome:
        try:
            return Outcome(inspect_unsplit(data, AuditConfig(**cfg)), None)
        except InputValidationError as exc:
            return Outcome(None, exc)

    return run


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def clf(n: int, seed: int, signal: float = 0.5) -> pd.DataFrame:
    """Binary classification with weak signal in x1; no leakage."""
    g = rng(seed)
    x1, x2 = g.normal(size=n), g.normal(size=n)
    y = (signal * x1 + g.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "plan": g.choice(["basic", "plus", "pro"], n), "y": y})


def reg(n: int, seed: int, noise: float = 3.0) -> pd.DataFrame:
    g = rng(seed)
    x = g.normal(size=n)
    return pd.DataFrame({"x": x, "z": g.normal(size=n), "y": 2 * x + g.normal(scale=noise, size=n)})


def iso(values: pd.Series) -> list[str | None]:
    return [None if pd.isna(v) else v.strftime("%Y-%m-%dT%H:%M:%S") for v in values]


def events(n: int, start: str, seed: int) -> pd.DataFrame:
    """Prediction daily from ``start``; outcome +30 days; balance snapshotted 1h earlier."""
    g = rng(seed)
    pred = pd.Series(pd.date_range(start, periods=n, freq="D"))
    return pd.DataFrame(
        {
            "balance": g.normal(size=n),
            "x": g.normal(size=n),
            "y": g.integers(0, 2, n),
            "pred_at": iso(pred),
            "outcome_at": iso(pred + 30 * DAY),
            "balance_at": iso(pred - pd.Timedelta(hours=1)),
        }
    )


TEMPORAL = dict(
    target="y",
    prediction_timestamp="pred_at",
    outcome_timestamp="outcome_at",
    feature_availability={"balance": "balance_at"},
)
STRONG = ("TGT-001", "TGT-002", "TGT-003", "TGT-004")


def _with(df: pd.DataFrame, **cols: Any) -> pd.DataFrame:
    out = df.copy()
    for name, values in cols.items():
        out[name] = values
    return out


# ------------------------------------------------------------------- builders


def build_cases() -> list[Case]:
    cases: list[Case] = []
    add = cases.append

    # ======================= A. positive controls ===========================
    tr, te = clf(400, 1), clf(100, 2)
    add(
        Case(
            "A1",
            "target",
            "positive",
            "Numeric feature is an exact copy of a binary target",
            "Equality on every row is direct evidence of a label copy.",
            _audit(
                _with(tr, leak=tr.y.astype(float)), _with(te, leak=te.y.astype(float)), target="y"
            ),
            [Fires("TGT-001", "leak", "critical", "high")],
            "TGT-001:leak",
        )
    )
    tr, te = reg(400, 3), reg(100, 4)
    add(
        Case(
            "A2",
            "target",
            "positive",
            "Regression target copied into a feature",
            "Equality on every row.",
            _audit(_with(tr, y_copy=tr.y), _with(te, y_copy=te.y), target="y"),
            [Fires("TGT-001", "y_copy", "critical", "high")],
            "TGT-001:y_copy",
        )
    )
    tr = clf(400, 5)
    tr["y"] = np.where(tr.y == 1, "churned", "stayed")
    tr["status"] = np.where(tr.y == "churned", " CHURNED", "Stayed ")
    add(
        Case(
            "A3",
            "target",
            "positive",
            "Text target copied with different case/whitespace",
            "Equality after documented safe normalisation (trim + case-fold).",
            _audit(tr, tr.iloc[:80], target="y"),
            [Fires("TGT-001", "status", "critical", "high")],
            "TGT-001:status",
        )
    )
    tr, te = clf(400, 6), clf(90, 7)
    te = pd.concat([te, tr.iloc[:10]], ignore_index=True)
    add(
        Case(
            "A4",
            "duplicates",
            "positive",
            "10 of 100 test rows are exact copies of training rows",
            "Identical records in both partitions allow memorisation.",
            _audit(tr, te, target="y"),
            [
                Fires("DUP-002", "train_test", "critical", "high"),
                FindingEvidence(
                    "DUP-002",
                    "train_test",
                    "exactly 10 test rows",
                    lambda e: e["n_test_rows_also_in_train"] == 10,
                ),
            ],
            "DUP-002:train_test",
        )
    )
    te2 = te.copy()
    te2["x1"] = te2["x1"].astype(object)  # same values, object dtype
    te2 = te2[["y", "plan", "x2", "x1"]]  # reordered
    add(
        Case(
            "A5",
            "duplicates",
            "positive",
            "Cross-split copies with reordered columns and object dtype",
            "Comparison is by column name and canonical value, not position or dtype.",
            _audit(tr, te2, target="y"),
            [
                FindingEvidence(
                    "DUP-002",
                    "train_test",
                    "exactly 10 test rows despite reordering",
                    lambda e: e["n_test_rows_also_in_train"] == 10,
                )
            ],
            "DUP-002:train_test",
        )
    )
    tr, te = clf(300, 8), clf(100, 9)
    tr["cust"] = [f"c{i % 100}" for i in range(300)]
    te["cust"] = [f"c{80 + i % 40}" for i in range(100)]  # c80..c99 shared (20)
    add(
        Case(
            "A6",
            "duplicates",
            "positive",
            "Entities cross the split; scenario requires unseen entities",
            "Declared scenario new_entities requires entity independence.",
            _audit(tr, te, target="y", entity_id="cust", scenario="new_entities"),
            [
                Fires("DUP-003", "cust", "high", "high"),
                FindingEvidence(
                    "DUP-003", "cust", "20 shared entities", lambda e: e["n_shared_entities"] == 20
                ),
            ],
            "DUP-003:cust",
        )
    )
    add(
        Case(
            "A7",
            "duplicates",
            "positive",
            "Entities cross the split under new_entities_forecasting",
            "Both constraints declared; entity independence still required.",
            _audit(tr, te, target="y", entity_id="cust", scenario="new_entities_forecasting"),
            [Fires("DUP-003", "cust", "high", "high")],
            "DUP-003:cust",
        )
    )
    tr, te = events(60, "2024-01-01", 10), events(30, "2024-04-01", 11)
    late = tr.copy()
    late.loc[:5, "balance_at"] = iso(pd.to_datetime(late.loc[:5, "pred_at"]) + 2 * DAY)
    add(
        Case(
            "A8",
            "temporal",
            "positive",
            "6 rows: feature recorded 2 days after prediction time",
            "Explicit availability metadata later than prediction time.",
            _audit(late, te, **TEMPORAL),
            [
                Fires("TMP-001", "balance", "high", "high"),
                FindingEvidence(
                    "TMP-001",
                    "balance",
                    "6 train violations",
                    lambda e: e["partitions"]["train"]["n_violations"] == 6,
                ),
            ],
            "TMP-001:balance",
        )
    )
    post = tr.copy()
    post.loc[:3, "balance_at"] = iso(pd.to_datetime(post.loc[:3, "outcome_at"]) + DAY)
    add(
        Case(
            "A9",
            "temporal",
            "positive",
            "4 rows: feature recorded 1 day after the outcome",
            "Explicit availability metadata later than the outcome timestamp.",
            _audit(post, te, **TEMPORAL),
            [
                Fires("TMP-002", "balance", "critical", "high"),
                FindingEvidence(
                    "TMP-002",
                    "balance",
                    "4 train violations",
                    lambda e: e["partitions"]["train"]["n_violations"] == 4,
                ),
            ],
            "TMP-002:balance",
        )
    )
    before = tr.copy()
    before.loc[:2, "outcome_at"] = iso(pd.to_datetime(before.loc[:2, "pred_at"]) - DAY)
    add(
        Case(
            "A10",
            "temporal",
            "positive",
            "3 rows: outcome recorded before the prediction time",
            "Contradicts the declared prediction boundary.",
            _audit(before, te, **TEMPORAL),
            [Fires("TMP-003", "outcome_before_prediction", "high", "high")],
            "TMP-003:outcome_before_prediction",
        )
    )
    add(
        Case(
            "A11",
            "temporal",
            "positive",
            "Forecasting: training period overlaps the test period",
            "Scenario forecasting requires train strictly before test.",
            _audit(
                events(60, "2024-01-01", 12),
                events(30, "2024-02-15", 13),
                **TEMPORAL,
                scenario="forecasting",
            ),
            [
                Fires("TMP-004", "train_test", "high", "high"),
                FindingEvidence(
                    "TMP-004",
                    "train_test",
                    "14 train rows after test start (Feb 16-29)",
                    lambda e: e["n_train_rows_after_test_start"] == 14,
                ),
            ],
            "TMP-004:train_test",
        )
    )
    tr, te = reg(400, 14), reg(100, 15)
    add(
        Case(
            "A12",
            "target",
            "positive",
            "Feature is the target in cents (affine re-encoding)",
            "Exact affine transform is a deterministic re-encoding.",
            _audit(_with(tr, cents=tr.y * 100), _with(te, cents=te.y * 100), target="y"),
            [Fires("TGT-003", "cents", "critical", "high")],
            "TGT-003:cents",
        )
    )
    tr = clf(400, 16)
    g = rng(16)
    follow = g.normal(size=400)
    follow[tr.y.to_numpy() == 1] = np.nan
    add(
        Case(
            "A13",
            "target",
            "positive",
            "Follow-up field is only filled in for one class",
            "Missingness exactly separates the classes.",
            _audit(_with(tr, followup=follow), clf(80, 17).assign(followup=0.0), target="y"),
            [Fires("TGT-004", "followup", "high", "medium")],
            "TGT-004:followup",
        )
    )

    # ======================= B. negative controls ===========================
    tr, te = reg(500, 20, noise=0.6), reg(100, 21, noise=0.6)
    add(
        Case(
            "B1",
            "target",
            "negative",
            "Strong legitimate regression predictor (|r| ~ 0.95)",
            "Signal plus independent noise: association is expected and is not leakage. "
            "Only a low-confidence TGT-005 review prompt is allowed.",
            _audit(tr, te, target="y"),
            [Absent(STRONG, "x"), AtMost("TGT-005", "medium", "low", "x")],
            "TGT-001..004 on x",
        )
    )
    g = rng(22)
    x = g.normal(size=500)
    add(
        Case(
            "B2",
            "target",
            "negative",
            "Strong legitimate classification predictor (AUC ~ 0.97)",
            "Noisy threshold of x; not a copy. Only a review prompt is allowed.",
            _audit(
                pd.DataFrame(
                    {
                        "x": x,
                        "z": g.normal(size=500),
                        "y": (x + g.normal(scale=0.15, size=500) > 0).astype(int),
                    }
                ),
                clf(100, 23).rename(columns={"x1": "x", "x2": "z"}).drop(columns=["plan"]),
                target="y",
            ),
            [Absent(STRONG, "x"), AtMost("TGT-005", "medium", "low", "x")],
            "TGT-001..004 on x",
        )
    )
    tr = clf(500, 24)
    tr["noise"] = rng(25).normal(size=500)
    add(
        Case(
            "B3",
            "target",
            "negative",
            "Random feature with no target association",
            "Independent noise must produce no target finding at all.",
            _audit(tr, clf(100, 26).assign(noise=0.0), target="y"),
            [Absent((*STRONG, "TGT-005", "TGT-006"), "noise")],
            "all TGT on noise",
        )
    )
    g = rng(1)
    y = np.zeros(1000)
    y[g.choice(1000, 5, replace=False)] = g.uniform(100, 900, 5)
    claims = np.zeros(1000)
    claims[g.choice(1000, 5, replace=False)] = g.integers(1, 4, 5)
    zi = pd.DataFrame({"n_prior_claims": claims, "age": g.normal(40, 10, 1000), "claim_amount": y})
    add(
        Case(
            "B4",
            "target",
            "negative",
            "Zero-inflated regression target and an unrelated zero-inflated feature",
            "99% raw agreement comes from both columns being mostly zero; chance agreement is ~98%, "
            "so this is not evidence of a copy.",
            _audit(
                zi,
                zi.iloc[:0].reindex(range(50)).fillna(0.0),
                target="claim_amount",
                task_type="regression",
            ),
            [Absent(STRONG, "n_prior_claims")],
            "TGT-001..004 on n_prior_claims",
            tags=["defect-D1"],
        )
    )
    g = rng(0)
    yy = (g.random(2000) < 0.99).astype(int)
    flag = np.full(2000, np.nan)
    flag[g.choice(2000, 40, replace=False)] = 1.0
    add(
        Case(
            "B5",
            "target",
            "negative",
            "Sparse constant flag in a 99%-positive target",
            "The 40 non-missing values all equal the majority class by chance; equality on a "
            "single class value carries no information.",
            _audit(
                pd.DataFrame({"x": g.normal(size=2000), "promo_flag": flag, "y": yy}),
                pd.DataFrame({"x": [0.0] * 5, "promo_flag": [1.0] * 5, "y": [1, 0, 1, 0, 1]}),
                target="y",
            ),
            [Absent(STRONG, "promo_flag")],
            "TGT-001..004 on promo_flag",
            tags=["defect-D2"],
        )
    )
    tr, te = clf(400, 27), clf(100, 28)
    tr["order_id"] = [f"ORD-{i:06d}" for i in rng(29).permutation(10**6)[:400]]
    te["order_id"] = [f"ORD-{i:06d}" for i in range(900_000, 900_100)]
    add(
        Case(
            "B6",
            "identifiers",
            "negative",
            "Unique, disjoint order IDs (legitimate identifier)",
            "High uniqueness alone is not leakage: only a low-severity ID-001 prompt is allowed.",
            _audit(tr, te, target="y"),
            [
                AtMost("ID-001", "low", "medium", "order_id"),
                Absent(("ID-003", *STRONG, "TGT-005"), "order_id"),
            ],
            "ID-003/TGT on order_id",
        )
    )
    tr, te = clf(300, 30), clf(100, 31)
    tr["cust"] = [f"c{i % 100}" for i in range(300)]
    te["cust"] = [f"c{i % 100}" for i in range(100)]
    for scenario in ("independent", "forecasting"):
        add(
            Case(
                f"B7-{scenario}",
                "duplicates",
                "negative",
                f"Entities cross the split; scenario {scenario} permits repeated entities",
                "Overlap is expected under the declared design; at most an info-level note.",
                _audit(tr, te, target="y", entity_id="cust", scenario=scenario),
                [AtMost("DUP-003", "info", "high")],
                "DUP-003 above info",
            )
        )
    g = rng(32)

    def survey(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "q1": g.integers(1, 4, n),
                "q2": g.integers(1, 4, n),
                "q3": g.choice(["a", "b"], n),
                "y": g.integers(0, 2, n),
            }
        )

    add(
        Case(
            "B8",
            "duplicates",
            "negative",
            "Survey with 3 discrete questions (36 possible rows): repeated rows are inevitable",
            "With so few possible rows, identical rows across partitions are expected and "
            "should be at most a low-confidence prompt.",
            _audit(survey(800), survey(200), target="y", scenario="independent"),
            [AtMost("DUP-002", "low", "low"), AtMost("DUP-004", "low", "low")],
            "DUP-002/DUP-004 above low/low",
            tags=["defect-D5"],
        )
    )
    add(
        Case(
            "B9",
            "temporal",
            "negative",
            "Chronological split, features snapshotted before prediction",
            "All availability times precede prediction; outcome follows by 30 days.",
            _audit(
                events(60, "2024-01-01", 33),
                events(30, "2024-04-01", 34),
                **TEMPORAL,
                scenario="forecasting",
            ),
            [Absent(("TMP-001", "TMP-002", "TMP-003", "TMP-004", "TMP-005"))],
            "any TMP",
        )
    )
    add(
        Case(
            "B10",
            "temporal",
            "negative",
            "Overlapping periods under an independent scenario",
            "Time overlap is irrelevant when rows are independent of time.",
            _audit(
                events(60, "2024-01-01", 35),
                events(30, "2024-02-01", 36),
                **TEMPORAL,
                scenario="independent",
            ),
            [Absent(("TMP-004",))],
            "TMP-004",
        )
    )
    tr, te = clf(300, 37), clf(100, 38)
    te.loc[3, "x1"] = np.nan  # float stays float; int column stays compatible
    te = te.astype({"y": "float64"})[["plan", "y", "x2", "x1"]]
    add(
        Case(
            "B11",
            "schema",
            "negative",
            "Compatible dtypes (int vs float target) and reordered columns",
            "Same coarse types and same column set: no schema finding.",
            _audit(tr, te, target="y"),
            [Absent(("SCH-001", "SCH-002", "SCH-003"))],
            "SCH-001..003",
        )
    )

    # ===================== C. boundary / ambiguity cases ====================
    tr, te = events(80, "2024-01-01", 40), events(20, "2024-04-01", 41)
    tr.loc[:29, "balance_at"] = None
    add(
        Case(
            "C1",
            "temporal",
            "boundary",
            "30 of 80 availability timestamps are missing",
            "Rows without timestamps cannot be assessed; the report must say so rather than "
            "implying they are safe.",
            _audit(tr, te, **TEMPORAL),
            [
                Absent(("TMP-001",)),
                Detail(
                    "TMP-001",
                    "30 rows reported as not compared",
                    lambda d: d["balance"]["partitions"]["train"]["n_not_compared_missing"] == 30,
                ),
                Warns("could not be compared"),
            ],
            tags=["defect-D3"],
        )
    )
    tr = events(60, "2024-01-01", 42)
    tr.loc[:9, "balance_at"] = tr.loc[:9, "pred_at"]
    add(
        Case(
            "C2",
            "temporal",
            "boundary",
            "10 rows: availability exactly equals prediction time",
            "SplitGuard cannot know whether a value stamped at the prediction instant was usable. "
            "Documented rule: equal is not a violation, but equal rows are counted and a warning "
            "states the ambiguity.",
            _audit(tr, events(20, "2024-04-01", 43), **TEMPORAL),
            [
                Absent(("TMP-001",)),
                Detail(
                    "TMP-001",
                    "n_equal == 10",
                    lambda d: d["balance"]["partitions"]["train"]["n_equal"] == 10,
                ),
                Warns("exactly equal"),
            ],
            tags=["ambiguity-equal-timestamps"],
        )
    )
    tr = events(60, "2024-01-01", 44)
    tr.loc[:4, "balance_at"] = tr.loc[:4, "outcome_at"]
    add(
        Case(
            "C3",
            "temporal",
            "boundary",
            "5 rows: availability exactly equals the outcome time",
            "Such values are after prediction (TMP-001 fires). Whether a value stamped at the "
            "outcome instant encodes the outcome is undeterminable: TMP-002 does not fire, the "
            "equal count is reported and a warning states the ambiguity.",
            _audit(tr, events(20, "2024-04-01", 45), **TEMPORAL),
            [
                Fires("TMP-001", "balance", "high", "high"),
                Absent(("TMP-002",)),
                Detail(
                    "TMP-002",
                    "n_equal == 5",
                    lambda d: d["balance"]["partitions"]["train"]["n_equal"] == 5,
                ),
                Warns("exactly equal"),
            ],
            tags=["ambiguity-equal-timestamps"],
        )
    )
    tr = events(40, "2024-01-01", 46)
    # Two rows are genuinely late once offsets are applied; one only *looks* late.
    tr.loc[0, ["pred_at", "balance_at"]] = [
        "2024-01-01 10:00:00",
        "2024-01-01T09:30:00-02:00",
    ]  # 11:30Z: late
    tr.loc[1, ["pred_at", "balance_at"]] = [
        "2024-01-02T10:00:00Z",
        "01/02/2024 10:30",
    ]  # late (naive=UTC)
    tr.loc[2, ["pred_at", "balance_at"]] = [
        "2024-01-03T10:00:00+00:00",
        "2024-01-03T11:00:00+02:00",
    ]  # 09:00Z: ok
    add(
        Case(
            "C4",
            "temporal",
            "boundary",
            "Mixed time zones and formats (ISO, offsets, US-style dates)",
            "All values are normalised to UTC (naive = UTC, month-first); exactly 2 rows are late.",
            _audit(tr, events(20, "2024-04-01", 47), **TEMPORAL),
            [
                FindingEvidence(
                    "TMP-001",
                    "balance",
                    "exactly 2 late rows after UTC normalisation",
                    lambda e: e["partitions"]["train"]["n_violations"] == 2,
                )
            ],
            tags=["ambiguity-date-format"],
        )
    )
    tr = events(40, "2024-03-10", 48)
    tr.loc[0, ["pred_at", "balance_at"]] = ["2024-03-10T00:00:00", "03/04/2024"]
    add(
        Case(
            "C5",
            "temporal",
            "boundary",
            "Ambiguous date '03/04/2024' (Mar 4 or Apr 3)",
            "Documented rule: ambiguous dates are read month-first (Mar 4, before prediction). "
            "A day-first source would be misread; this is a documented limitation.",
            _audit(tr, events(20, "2024-06-01", 49), **TEMPORAL),
            [Absent(("TMP-001",))],
            tags=["ambiguity-date-format"],
        )
    )
    tr, te = clf(100, 50), clf(40, 51)
    tr["cust"] = [f"c{i}" for i in range(60)] + [None] * 40
    te["cust"] = [f"d{i}" for i in range(25)] + [None] * 15
    add(
        Case(
            "C6",
            "duplicates",
            "boundary",
            "40% of entity IDs missing (new_entities)",
            "Rows with no entity ID cannot be checked for overlap; the report must say so.",
            _audit(tr, te, target="y", entity_id="cust", scenario="new_entities"),
            [
                Absent(("DUP-003",)),
                Detail(
                    "DUP-003",
                    "40 + 15 missing IDs counted",
                    lambda d: (
                        d["n_missing_entity_train"] == 40 and d["n_missing_entity_test"] == 15
                    ),
                ),
                Warns("missing entity ID"),
            ],
            tags=["defect-D3"],
        )
    )
    tr2 = tr.copy()
    tr2.loc[:69, "cust"] = None
    add(
        Case(
            "C7",
            "validation",
            "boundary",
            "70% of entity IDs missing",
            "Exceeds the 50% limit for required fields: fail clearly rather than audit a fraction.",
            _audit(tr2, te, target="y", entity_id="cust"),
            [RaisesCode("EXCESSIVE_MISSINGNESS")],
        )
    )
    sp = clf(100, 52).assign(split="train")
    add(
        Case(
            "C8",
            "validation",
            "boundary",
            "Split column has no test rows",
            "An empty partition cannot be audited.",
            _audit_split(sp, "split", target="y"),
            [RaisesCode("EMPTY_PARTITION")],
        )
    )
    add(
        Case(
            "C9",
            "validation",
            "boundary",
            "Train partition has a header but no rows",
            "Empty dataset must be rejected, not reported clean.",
            _audit(clf(10, 53).iloc[:0], clf(40, 54), target="y"),
            [RaisesCode("EMPTY_DATASET")],
        )
    )
    tr, te = clf(300, 55), clf(90, 56)
    te = pd.concat([te, tr.iloc[:10]], ignore_index=True)
    te_str = te.astype({"y": str})
    add(
        Case(
            "C10",
            "schema",
            "boundary",
            "Target stored as numbers in train and as strings in test",
            "Canonical comparison treats 1 and '1' as equal, so the 10 copied rows are still found; "
            "the type difference is reported as SCH-003.",
            _audit(tr, te_str, target="y"),
            [
                FindingEvidence(
                    "DUP-002",
                    "train_test",
                    "10 copied rows found across dtypes",
                    lambda e: e["n_test_rows_also_in_train"] == 10,
                ),
                Fires("SCH-003", "y", "medium", "high"),
                Absent(("SCH-005",)),
            ],
        )
    )
    tr = clf(300, 57).assign(region="EU", blank=np.nan)
    add(
        Case(
            "C11",
            "schema",
            "boundary",
            "Constant and all-missing feature columns",
            "Reported once as an info-level data-quality note; never as target leakage.",
            _audit(tr, clf(60, 58).assign(region="EU", blank=np.nan), target="y"),
            [
                Fires("SCH-006", "train", "info", "high"),
                Absent((*STRONG, "TGT-005"), "region"),
                Absent((*STRONG, "TGT-005"), "blank"),
            ],
        )
    )
    tr = clf(300, 59)
    sparse = np.full(300, np.nan)
    sparse[:20] = tr.y[:20]
    add(
        Case(
            "C12",
            "target",
            "boundary",
            "Target copy present on only 20 rows (93% missing)",
            "Below the 30-pair minimum the rule cannot be assessed; the report must list the "
            "feature as not assessed rather than clean.",
            _audit(
                _with(tr, sparse_copy=sparse), clf(60, 60).assign(sparse_copy=np.nan), target="y"
            ),
            [
                Absent(STRONG, "sparse_copy"),
                Detail(
                    "TGT-001",
                    "listed as too few pairs",
                    lambda d: "sparse_copy" in d["features_with_too_few_pairs"],
                ),
                Warns("too few non-missing"),
            ],
            tags=["defect-D3"],
        )
    )
    base_tr, base_te = clf(300, 61), clf(80, 62)
    base_te = pd.concat([base_te, base_tr.iloc[:5]], ignore_index=True)

    def reorder_equivalent() -> Outcome:
        a = audit_train_test(base_tr, base_te, AuditConfig(target="y"))
        b = audit_train_test(
            base_tr[base_tr.columns[::-1]],
            base_te[["y", "x2", "plan", "x1"]],
            AuditConfig(target="y"),
        )
        same = [f.to_dict() for f in a.findings] == [f.to_dict() for f in b.findings]
        if same:
            return Outcome(b, None)
        return Outcome(
            None, InputValidationError([ValidationIssue("MISMATCH", "reordered findings differ")])
        )

    add(
        Case(
            "C13",
            "duplicates",
            "boundary",
            "Column order differs between and within runs",
            "Findings must be identical when only column order changes.",
            reorder_equivalent,
            [Fires("DUP-002", "train_test")],
        )
    )
    add(
        Case(
            "C14",
            "duplicates",
            "boundary",
            "Unsplit dataset",
            "No split exists: cross-partition checks are skipped with a reason, never reported clean.",
            _inspect(clf(200, 63), target="y"),
            [
                Skipped("DUP-002", "not audited"),
                Skipped("ID-003"),
                Warns("contamination was not audited"),
            ],
        )
    )
    add(
        Case(
            "C15",
            "temporal",
            "boundary",
            "No timing metadata at all",
            "Temporal safety must be reported as not assessed.",
            _audit(clf(200, 64), clf(50, 65), target="y"),
            [Skipped("TMP-001"), Skipped("TMP-003"), Skipped("TMP-004"), Warns("not evidence")],
        )
    )
    g = rng(66)
    price = pd.DataFrame(
        {
            "sqft": g.uniform(500, 3000, 300).round(),
            "price": g.permutation(np.arange(100_000, 100_300)),
        }
    )
    add(
        Case(
            "C16",
            "identifiers",
            "boundary",
            "Unique integer regression target",
            "The target must never be treated as a feature or an identifier.",
            _audit(price, price.iloc[:50], target="price"),
            [
                Absent(("ID-001", "ID-002", "SCH-006"), "price"),
                Detail(
                    "TGT-001",
                    "target not scanned as a feature",
                    lambda d: d["n_features_scanned"] == 1,
                ),
            ],
        )
    )
    tr, te = events(60, "2024-01-01", 67), events(30, "2024-03-01", 68)
    te.loc[0, "pred_at"] = tr["pred_at"].iloc[-1]  # test starts exactly at train end
    add(
        Case(
            "C17",
            "temporal",
            "boundary",
            "Forecasting: last train timestamp equals first test timestamp",
            "Train does not extend *past* test start, so TMP-004 does not fire; the shared "
            "boundary instant is ambiguous and must be stated.",
            _audit(tr, te, **TEMPORAL, scenario="forecasting"),
            [Absent(("TMP-004",)), Warns("shared boundary")],
            tags=["ambiguity-equal-timestamps"],
        )
    )
    tr = events(60, "2024-01-01", 69)
    tr.loc[:4, "outcome_at"] = tr.loc[:4, "pred_at"]
    add(
        Case(
            "C18",
            "temporal",
            "boundary",
            "5 rows: outcome timestamp equals prediction timestamp",
            "Without a declared minimum lead time, equal is not 'before'; counted and warned.",
            _audit(tr, events(20, "2024-04-01", 70), **TEMPORAL),
            [
                Absent(("TMP-003",)),
                Detail(
                    "TMP-003",
                    "5 equal",
                    lambda d: d["partitions"]["train"]["n_outcome_equal_to_prediction"] == 5,
                ),
                Warns("exactly equal"),
            ],
            tags=["ambiguity-equal-timestamps"],
        )
    )
    add(
        Case(
            "C19",
            "temporal",
            "boundary",
            "Same as C18 with min_lead_time=1D",
            "With a declared lead time, a zero gap is a violation.",
            _audit(tr, events(20, "2024-04-01", 71), **TEMPORAL, min_lead_time="1D"),
            [Fires("TMP-003", "insufficient_lead_time", "medium", "high")],
        )
    )

    # ============================ D. diagnostics ============================
    tr, te = clf(300, 72), clf(100, 73)
    tr["leak"] = rng(74).integers(0, 2, 300)
    te["leak"] = te["y"]
    add(
        Case(
            "D1",
            "target",
            "diagnostic",
            "Copy exists only in the test partition (opt-in scan)",
            "Test labels are scanned only when requested, and findings are marked diagnostic_only.",
            _audit(tr, te, target="y", test_target_diagnostics=True),
            [
                Fires("TGT-001", "test:leak", "critical", "high"),
                FindingEvidence(
                    "TGT-001",
                    "test:leak",
                    "diagnostic_only flag",
                    lambda e: e["diagnostic_only"] is True,
                ),
                Absent(("TGT-005",), "leak"),
            ],
        )
    )
    add(
        Case(
            "D2",
            "target",
            "diagnostic",
            "Same data without the opt-in flag",
            "Test labels are not scanned by default.",
            _audit(tr, te, target="y"),
            [Absent(STRONG, "leak")],
        )
    )
    return cases
