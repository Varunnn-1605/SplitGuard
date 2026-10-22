# SplitGuard

SplitGuard audits machine-learning datasets for **data leakage** and **train/test
contamination**. It compares values, counts and timestamps, and returns structured
findings with the evidence behind each one.

The engine runs locally with `pandas` and `numpy` only. It needs no LLM, network access,
paid API or trained model, and it never uploads data or sends telemetry.

> **Status (0.2.0):** core audit engine (Python API + CLI + JSON report) and an optional
> local Streamlit UI. There is no LLM-written report, no learned leakage classifier and no
> model evaluation yet. See [Future work](#known-limitations-and-future-work).

## What it does and does not do

| SplitGuard does | SplitGuard does not |
| --- | --- |
| Find exact duplicate rows within and across partitions | Detect near-duplicates or fuzzy matches |
| Find features that equal, nearly equal or re-encode the target | Train models or select features |
| Flag unusually strong univariate associations **for review** | Treat correlation as proof of leakage |
| Compare explicit availability / prediction / outcome timestamps | Infer timing from column names (names are low-confidence hints only) |
| Flag identifiers, schema mismatches and entity overlap | Prove a dataset is leakage-free |
| Recommend a split strategy from your declared scenario | Find a "universally optimal" split from the CSV |
| Create a split, if you confirm the strategy explicitly | Split data silently during an audit |

**No findings does not mean no leakage.** It means none of the listed checks fired.
Every report lists which checks ran, which were skipped and why.

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Dependencies are constrained in `pyproject.toml` (`pandas>=2.1,<3`, `numpy>=1.24,<3`).

## Quick start

```bash
python examples/generate_synthetic_data.py          # writes examples/data/
splitguard audit \
  --train examples/data/02_target_copy/train.csv \
  --test  examples/data/02_target_copy/test.csv \
  --target churn --output report.json
```

```text
SplitGuard 0.2.0 | mode: train_test
  input (train): train.csv - 800 rows x 6 columns
  input (test): test.csv - 200 rows x 6 columns
Checks: 21 run, 5 skipped, 0 failed
Findings: 2 total | critical 1 | high 0 | medium 0 | low 1 | info 0
  [CRITICAL, high confidence] TGT-001:churn_score_final: Feature 'churn_score_final' is identical to the target
  [LOW, low confidence] TMP-005:churn_score_final: Column name 'churn_score_final' suggests post-prediction information
...
JSON report written to: report.json
```

## Input modes

| Mode | CLI | Python |
| --- | --- | --- |
| A. Separate train/test CSVs | `splitguard audit --train a.csv --test b.csv` | `audit_train_test(train, test, config)` |
| B. One CSV with a split column | `splitguard audit --data d.csv --split-column split [--train-value train --test-value test]` | `audit_split_column(data, "split", "train", "test", config)` |
| C. One unsplit CSV | `splitguard inspect --data d.csv` | `inspect_unsplit(data, config)` |

* Sources can be a CSV path, CSV **bytes** (no file path needed, e.g. from an upload),
  or a `pandas.DataFrame`.
* **Mode B** rejects missing or unrecognised split labels and empty partitions instead
  of dropping rows. The split column is excluded from comparisons.
* **Mode C** runs the within-dataset checks and returns a split recommendation. The
  report states that train/test contamination was **not** audited. Splitting is a
  separate, explicit command (`splitguard split`).

### Python API

```python
from splitguard import AuditConfig, audit_train_test
from splitguard.reporting import write_json_report

config = AuditConfig(
    target="churn",
    entity_id="customer_id",                # repeated entities
    prediction_timestamp="prediction_time",
    outcome_timestamp="outcome_time",
    feature_availability={"refund_amount": "refund_amount_available_at"},
    exclude_columns=("row_number",),        # non-feature metadata
    scenario="forecasting",                 # independent | new_entities | forecasting | new_entities_forecasting
    min_lead_time="7D",
)
result = audit_train_test("train.csv", "test.csv", config)
for f in result.findings:
    print(f.finding_id, f.severity.value, f.confidence.value, f.title)
write_json_report(result, "report.json")
```

### CLI reference

```text
splitguard audit   (--train T --test S | --data D --split-column C) [options]
splitguard inspect --data D [options] [--test-size 0.2 --seed 42]
splitguard split   --data D --confirm-strategy NAME --out-dir DIR [options] [--overwrite]

options: --target --entity-id --timestamp (prediction time) --outcome-timestamp
         --availability FEATURE=COLUMN (repeatable) --exclude COLUMN (repeatable)
         --scenario --task-type --min-lead-time --config config.json
         --max-file-mb --max-rows
audit/inspect: --output report.json --fail-on {critical,high,medium,low,info}
               --no-timestamp --test-diagnostics (audit only)
```

`--config` takes a JSON object with `AuditConfig` fields (including `thresholds` and
`limits`). Command-line flags override it. Unknown keys are rejected.

**Exit codes:** `0` audit completed (findings or not), `1` a finding met `--fail-on`,
`2` invalid input/configuration/usage, `3` a check failed internally. A valid dataset
with findings exits `0` unless you set `--fail-on`.

## Web UI (optional)

A local Streamlit interface wraps the same engine. It calls the public API
(`audit_train_test`, `audit_split_column`, `inspect_unsplit`, `create_split`) through a
small adapter (`splitguard/ui_adapter.py`); it does not reimplement any detector.

```bash
pip install -e ".[ui]"        # or, from a wheel: pip install "splitguard-0.2.0-py3-none-any.whl[ui]"
splitguard-ui                 # opens http://localhost:8501
splitguard-ui --port 8600 --headless
```

The base package and CLI do not need Streamlit. Without it, `splitguard-ui` prints an
install hint and exits with code 2.

**Workflow**

1. **Data.** Upload CSVs for one of the three input modes, or pick a built-in sample
   dataset with planted problems. Each file shows its name, row/column counts and an
   optional truncated preview of the first 5 rows. Files are validated by the engine
   (size, row and column limits, header, encoding), and errors show the engine's
   messages, never row contents.
2. **Configure.** Target, task type, use-case scenario (never guessed; defaults to "not
   declared"), entity ID, prediction and outcome time, per-feature "recorded at"
   columns, excluded columns, minimum lead time, and test-label diagnostics. For a split
   column you pick which labels mean train and test; labels are only pre-selected on an
   exact `train`/`test` match. For unsplit data you set the test share and seed used for
   split advice.
3. **Run.** Nothing runs until you click **Run audit**. If data or settings change
   afterwards, the results are marked as out of date.
4. **Results.**
   - A status banner.
   - Severity and confidence badges side by side. Low-confidence findings are labelled
     *review prompts*.
   - Filters by severity, category and confidence.
   - Each finding's description, affected columns, evidence, recommended action and
     limitations.
   - A prominent panel for skipped checks, partial coverage, missing metadata and tied
     timestamps.
   - For unsplit data: the engine's recommendation, or what information it still needs,
     plus an optional split creation that requires typing the strategy name and
     acknowledging that a split does not remove leakage.
   - Downloads: the engine's full JSON report (unchanged schema) and a Markdown summary
     generated from it.

**Privacy and deployment limitations**

* Processing is local to the running app. Uploaded files are held in the app's memory
  for the browser session; nothing is written to disk by SplitGuard or sent to any
  external service.
* `splitguard-ui` binds to `localhost`, disables Streamlit's anonymous usage statistics
  (on by default in Streamlit), caps uploads at the engine's 200 MB limit, disables
  the file watcher, hides Streamlit's developer toolbar (including its "Deploy" button)
  and uses a neutral accent colour. If you run `streamlit run src/splitguard/ui_app.py`
  yourself, pass `--browser.gatherUsageStats false --server.address localhost
  --client.toolbarMode viewer` to get the same privacy behaviour.
* A run with no findings is shown as a neutral or amber notice, never as a green
  "success": absence of findings is not evidence that the data is free of leakage.
* There is no authentication and no isolation between users. It is meant for a single
  user on their own machine; it has not been hardened or tested for shared or public
  deployment.

## Severity versus confidence

Each finding has two separate ratings:

* **Severity**: the potential impact *if the issue is real* (`critical` > `high` >
  `medium` > `low` > `info`). A feature that copies the target is critical because it
  would make every test metric meaningless.
* **Confidence**: how strongly the evidence supports the finding (`high`, `medium`,
  `low`). Exact equality across 800 rows is high confidence; a column name containing
  "label" is low.

A finding can be high-severity and low-confidence. For example, shared entity IDs when
the scenario is undeclared would matter a lot if entities must be unseen, but SplitGuard
can't know that they must be.

## Evidence versus heuristics

| Evidence (direct measurements) | Heuristics (review prompts) |
| --- | --- |
| Exact row/feature/ID matches with counts and rates | Target-like column names (TGT-006) |
| Feature equals / re-encodes the target on every row | Temporal words in column names (TMP-005) |
| Timestamp comparisons on supplied metadata | Identifier-like uniqueness (ID-001) |
| Missingness that exactly separates the classes | Strong univariate associations (TGT-005) |

Every finding includes `evidence` (counts, rates, statistics, comparison rules),
`affected_columns`, a concrete `recommended_action` and `limitations` explaining why it
isn't proof. By default, findings never contain raw row values.

## Detectors

Rule IDs are stable. A finding ID is `RULE:key`, for example `TGT-001:leak`.

### A. Duplicates and cross-split contamination (`detectors/duplicates.py`)

| Rule | Check |
| --- | --- |
| DUP-001 | Exact duplicate rows **within** each partition (count beyond first, groups, largest group) |
| DUP-002 | Exact rows present in **both** train and test (test rows affected, rate, distinct rows). Like DUP-004, reported at low/low when fewer than 50% of training rows are distinct, because repeats are then expected by chance |
| DUP-003 | Configured `entity_id` values in both partitions. Severity depends on the scenario: high for `new_entities`, info for `independent`/`forecasting`, medium/low-confidence when unknown |
| DUP-004 | Identical **feature** vectors across partitions (target and metadata excluded). Reports how many are already full-row matches, and how many carry a different label. Reported at low confidence when the feature space is small (fewer than 50% distinct training feature rows) |

**Row comparison semantics:** columns are aligned by name, so order doesn't matter.
Numbers compare by value (`1 == 1.0`). Text compares verbatim, so case and whitespace
count. Missing values equal each other. Rows are matched with a deterministic 64-bit
hash (false-match probability ≈ n²/2⁶⁵). Duplicates are counted, never dropped. A
unique record-ID column makes otherwise identical rows look distinct, so configure it
via `exclude_columns` or `entity_id`.

### B. Target leakage candidates (`detectors/target_leakage.py`)

Requires `target`. Scans the **training** partition (or the whole dataset in Mode C),
using only non-missing (feature, target) pairs, and only when there are at least 30 of
them.

| Rule | Check | Severity / confidence |
| --- | --- | --- |
| TGT-001 | Feature equals target on every row: exactly, after safe normalisation (canonical numbers; trimmed, case-folded text), or within numeric tolerance (`rtol=1e-6`) | critical / high |
| TGT-002 | Equal on ≥98% of rows **and** chance-corrected agreement (Cohen's κ) ≥ 0.9, for classification and regression. Agreement between two mostly-zero or mostly-majority-class columns therefore doesn't count | high / medium |
| TGT-003 | Deterministic re-encoding: one-to-one relabelling (classification); exact affine transform; strictly monotonic transform (regression) | critical / high (monotonic: high / medium) |
| TGT-004 | Binary target: whether the feature is missing determines the class | high / medium |
| TGT-005 | Unusually strong univariate association: max(\|Pearson\|, \|Spearman\|) ≥ 0.9, single-feature AUC ≥ 0.95, correlation ratio η ≥ 0.9, or bias-corrected Cramér's V ≥ 0.9 | **medium / low, never confirmed** |
| TGT-006 | Column name contains the target name or words such as `label`, `target`, `outcome` | low / low |

Features already flagged by TGT-001..004 are not re-reported by TGT-005 or TGT-006.
TGT-001..005 are not applied to a feature when the target has a single value on every
row where that feature is present; the report lists such features and states that
they were not assessed. The same statement is made for features with too few pairs.
Categorical association is only computed for columns with ≤50 categories and ≤50%
unique values. Above that, every column would look "perfectly associated" by chance.
Constant and all-missing features are reported by SCH-006.

**The test target is never used for tuning.** `--test-diagnostics` (or
`test_target_diagnostics=True`) also runs TGT-001..004 on the test partition. Those
findings are keyed `test:<column>` and marked `diagnostic_only`. Association scans are
never run on test labels. No model is fitted and no threshold is chosen from data.

#### Why correlation does not establish leakage

The most useful legitimate features are, by definition, strongly associated with the
target. A house's floor area predicts its price. That is signal, not leakage. Leakage
is about **availability**: whether the value could have been known at prediction time
without knowing the outcome. A statistic can't answer that, so TGT-005 is always a
low-confidence review prompt. Only deterministic relationships (TGT-001/003) are
treated as strong evidence, and even those come with legitimate explanations in their
limitations (bookkeeping copies, targets defined by business rules).

### C. Temporal leakage (`detectors/temporal.py`)

Uses only timestamps you supply. Values are parsed strictly: an unparseable non-missing
value is a validation error, never silently turned into missing. Naive timestamps are
treated as UTC, and ambiguous dates are read month-first.

| Rule | Needs | Check |
| --- | --- | --- |
| TMP-001 | `prediction_timestamp`, `feature_availability` | availability time > prediction time (high / high) |
| TMP-002 | `outcome_timestamp`, `feature_availability` | availability time > outcome time (critical / high) |
| TMP-003 | `prediction_timestamp`, `outcome_timestamp` [, `min_lead_time`] | outcome before prediction (high / high); outcome sooner than the minimum lead time (medium / high) |
| TMP-004 | split + `prediction_timestamp` | training rows later than the first test row: high for forecasting scenarios, low when the scenario is unknown, not reported for `independent`/`new_entities` |
| TMP-005 | — | temporal words (`future`, `final`, `outcome`, `after`...) in names of features without availability metadata (low / low) |

Each finding records the comparison rule, per-partition counts, maximum/median lag and
its assumptions.

**Boundary semantics (documented ambiguity).** Comparisons are strict: a value is flagged
only if its timestamp is *strictly later* than the prediction (TMP-001) or outcome
(TMP-002). Equal timestamps are counted (`n_equal`) but not flagged, because the data
cannot show whether a value stamped at exactly the prediction instant was usable. The
report always adds a warning when equal timestamps occur. The same applies to an outcome
at the same instant as the prediction (TMP-003, flagged only when `min_lead_time` is set),
and to a forecasting split where the last training timestamp equals the first test
timestamp (TMP-004 warns about the shared boundary but does not flag it as overlap).
Rows whose timestamps are missing are not compared; the report states how many.
Ambiguous dates such as `03/04/2024` are read month-first.

#### Why temporal leakage can't be inferred without prediction-time information

Whether a value is leakage depends on **when it was recorded relative to the
prediction**, and a CSV doesn't say. `account_balance` is fine if it was snapshotted
before the prediction and leaked if it was refreshed after the outcome, and both
versions look identical in the file. Without a prediction timestamp and availability
metadata, SplitGuard marks the temporal checks as skipped and says that temporal
safety **was not assessed**. It never reports the data as safe.

### D. Identifiers and schema (`detectors/identifiers.py`, `detectors/schema.py`)

| Rule | Check |
| --- | --- |
| ID-001 | Integer/text feature with ≥95% unique values (confidence rises if the name contains `id`, `key`, `uuid`...) |
| ID-002 | Likely row index: `Unnamed: 0`/`index`, or integers increasing by exactly 1 in file order |
| ID-003 | Values of an ID-001 column present in both partitions (references DUP-002/DUP-004 because they describe the same rows) |
| SCH-001/002 | Columns missing from test / only in test |
| SCH-003 | Different coarse type (numeric, text, boolean, datetime, mixed, empty) between partitions |
| SCH-004 | Target absent or empty in test (when `expect_target_in_test`) |
| SCH-005 | Test classes never seen in train |
| SCH-006 | Constant or all-missing feature columns |
| SCH-007 | Text column whose values are ≥90% numeric (mixed types) |
| SCH-008 | Target with fewer than two distinct values (target scans skipped) |

High uniqueness is not leakage. Record IDs are usually harmless and entity IDs are
legitimate grouping keys. These findings ask you to decide how each identifier should
be treated.

## Split-strategy recommendations

`inspect` (Mode C) and `recommend_split()` map your **declared** scenario to a strategy:

| Scenario | Strategy |
| --- | --- |
| `independent`, classification | `stratified_random` (seeded) |
| `independent`, regression | `random` (seeded) |
| `new_entities` | `group` on `entity_id` |
| `forecasting` | `time` on `prediction_timestamp` |
| `new_entities_forecasting` | `group_and_time` (review recommendation only) |

If the target or scenario is missing, or the scenario needs an entity or timestamp
column that wasn't given, the result is `needs_user_input` with a list of what is
missing. It also gives hints, such as entities repeating or timestamps being present.

#### Why the split depends on the deployment scenario

The same CSV needs different splits for different uses. Customer-level rows split
randomly are fine for scoring new transactions from known customers. The same split
overstates performance for a model meant for brand-new customers, which needs a group
split. A model used to forecast next quarter needs a time split. The data alone can't
tell these apart, so SplitGuard asks.

**Creating a split** is separate and needs explicit confirmation:

```bash
splitguard split --data d.csv --target churn --scenario independent \
  --confirm-strategy stratified_random --out-dir split/
```

It writes `train.csv`, `test.csv` and `split_report.json`, which covers class balance,
group overlap, time boundaries, row counts and warnings. The command refuses to
overwrite files without `--overwrite`, and it never drops rows: rows with a missing
group or timestamp raise an error. `group_and_time` is not created automatically.
A split doesn't remove duplicates or leaked features, so audit the result.

## JSON report

Stable top-level structure (`report_format_version: "1.0"`):

```text
tool, report_format_version, generated_at, mode, inputs, partitions, configuration,
task_type, feature_columns, summary, checks, findings, split_recommendation,
warnings, limitations
```

* `inputs` records the file name only (no directory), size, SHA-256, row and column
  counts, and column names.
* `checks` lists every rule as `run`, `skipped` (with a reason) or `failed`. A failing
  detector is isolated, and only its exception *type* is recorded.
* Reports are deterministic for the same inputs and configuration. `generated_at` is
  the only variable field; omit it with `--no-timestamp`.
* Values are strict JSON: NaN/inf become `null`, and NumPy scalars and timestamps are
  converted.

A full sample is in [`examples/sample_report.json`](examples/sample_report.json).
One finding from it:

```json
{
  "finding_id": "TMP-002:refund_amount",
  "rule_id": "TMP-002",
  "category": "temporal_leakage",
  "severity": "critical",
  "confidence": "high",
  "title": "'refund_amount' is recorded after the outcome on 1000 row(s)",
  "description": "For 1000 rows 'refund_amount_available_at' is later than the outcome timestamp 'outcome_time': the value was created after the label was known and may encode it.",
  "evidence": {
    "comparison_rule": "refund_amount_available_at > outcome_time",
    "assumptions": "All timestamps share one clock; naive timestamps were interpreted as UTC. Rows with a missing timestamp on either side are not compared.",
    "partitions": {
      "train": {"n_rows": 800, "n_compared": 800, "n_not_compared_missing": 0,
                "n_violations": 800, "violation_rate": 1.0, "n_equal": 0,
                "max_lag": "15 days 00:00:00", "median_lag": "15 days 00:00:00"},
      "test": {"...": "..."}
    }
  },
  "affected_columns": ["refund_amount", "refund_amount_available_at", "outcome_time"],
  "recommended_action": "Remove 'refund_amount' or rebuild it from pre-outcome data only.",
  "limitations": ["Relies entirely on the supplied availability and outcome timestamps."]
}
```

## Input validation and safety

* Errors name the problem and the column, never row contents. Examples: empty file or
  dataset; missing or duplicate column names; configured columns not found;
  unrecognised split labels; empty partitions; unparseable or numeric timestamps;
  required fields that are more than 50% missing; non-numeric regression targets; no
  usable feature columns; train and test sharing no features.
* Limits are checked before expensive work: file size (default 200 MB, checked before
  parsing), column count (checked from the header), and rows (default 1,000,000; the
  parser stops reading at the limit).
* Data is never repaired, relabelled or dropped. Input DataFrames are not modified.
* CSVs must be UTF-8 (a BOM is fine). pandas' default missing markers (`""`, `NA`,
  `N/A`, `null`, `NaN`...) are read as missing. No `eval` and no execution of input
  content.

## Running the tests

```bash
pip install -e ".[dev]"
pytest              # 249 tests (tests/ + validation/; UI tests skip without Streamlit)
python validation/run_validation.py --markdown validation/RESULTS.md
ruff check src tests examples
```

Test fixtures are built by hand with numpy/pandas and don't use SplitGuard code. They
check both detection and **non-detection**: a strong legitimate association must come
out as TGT-005 medium/low and never as a copy. Edge cases covered include missing
values, constant columns, empty partitions, shuffled column order, int/float
equivalence, mixed types, time zones and imbalanced labels. A passing suite shows the
rules behave as specified. It doesn't show that all real-world leakage is detected.

### Adversarial validation suite (`validation/`)

An independent suite of 46 deterministic synthetic cases challenges the rules. It has
its own data builders and does not reuse the unit-test fixtures:

* **positive controls**: injected leakage (target copies, cross-split duplicates,
  entity overlap under `new_entities`, post-prediction/post-outcome features, ...);
* **negative controls**: strong legitimate predictors, random features, legitimate IDs,
  permitted entity overlap, expected duplicate rows, zero-inflated targets, ...;
* **boundary cases**: missing or equal timestamps, time zones and mixed formats,
  missing entity IDs, empty partitions, string/number targets, constant features, ...;
* **diagnostics**: opt-in test-label scans.

Each case states its expected outcome and the reason, and is scored once against its
designated rule (overlapping findings are not counted as extra detections). Current
synthetic results and the per-case table are in
[`validation/RESULTS.md`](validation/RESULTS.md). The results before the Phase 1.5 fixes
are in `validation/RESULTS_before_fixes.md`. These are synthetic controls, not estimates
of real-world accuracy.

## Known limitations and future work

**Limitations**

* Only exact duplicates; no near-duplicate or fuzzy matching.
* Univariate only: leakage spread across combinations of features isn't detected.
* Temporal checks are only as good as the supplied metadata; feature availability
  can't be inferred from data.
* Findings can overlap (for example DUP-002, DUP-004, ID-003 on the same rows). They
  are not independent confirmations.
* Equal timestamps are not flagged (see boundary semantics); day-first date sources
  are misread unless converted to ISO-8601.
* Expected duplicates can only be recognised through the small-row-space guard. There
  is no option to declare that repeated rows are intended, so with highly unique
  features and deliberately repeated records DUP-002 still reports high severity.
* Entity overlap (DUP-003) is judged per declared scenario, but DUP-002 and DUP-004 are
  not: under `independent`, mostly-unique entities with static features that appear in
  both partitions still produce DUP-002 critical/high and DUP-004 high/medium, while
  DUP-003 is info. SplitGuard cannot tell a duplicated record from a repeat observation
  of the same entity, so this is left for the user to judge.
* A feature present only on rows of one class (without perfectly separating the classes)
  is not flagged by TGT-004; the report lists it as not assessed.
* Timestamps are assumed to share one clock. Naive values are treated as UTC.
* Single-machine, in-memory processing (14-40 s for 250k rows × 19 columns on a
  laptop, depending on machine load).

**Future work**

* Near-duplicate detection (normalised text, rounding-tolerant numeric matching, MinHash).
* Optional evaluation module comparing cross-validation and holdout scores. This needs
  a model-evaluation workflow, so it is deliberately not in the data-only audit. Phase 1
  never invents or generates scores.
* `group_and_time` split creation; stratified group splits.
* An optional LLM-written narrative report on top of the JSON.
* Chunked processing for files beyond memory.

## Project layout

```text
src/splitguard/
  audit.py            public API (3 input modes) and orchestration
  config.py           AuditConfig, Scenario, TaskType, Thresholds, Limits
  schemas.py          Finding, CheckRecord, SplitRecommendation, AuditResult, to_jsonable
  validation.py       loading, limits, strict validation
  hashing.py          canonical values and deterministic row hashing
  context.py          shared audit context and rule catalogue
  detectors/          duplicates, target_leakage, temporal, identifiers, schema
  splitting/          recommend_split, create_split
  reporting/          JSON report and terminal summary
  cli.py              thin argparse CLI
  ui.py, ui_app.py    optional Streamlit UI (page and `streamlit run` entry)
  ui_adapter.py       Streamlit-free glue between the UI and the public API
  ui_samples.py       seeded sample datasets for the UI
  ui_launcher.py      `splitguard-ui` launcher (localhost, usage stats off)
examples/             seeded synthetic data generator, sample report
tests/                pytest suite
```
