# SplitGuard

**A deterministic auditor for machine-learning dataset leakage and train/test contamination.**

SplitGuard checks datasets for duplicate records, target leakage candidates, temporal risks, identifier issues, and train/test schema mismatches. It produces evidence-backed findings, confidence and severity ratings, skipped-check explanations, and split-strategy recommendations.

**Python API · CLI · JSON reports · Optional local Streamlit UI**

> **Important:** No findings does not mean a dataset is leakage-free. SplitGuard reports what its configured checks detect and clearly identifies checks that were skipped or could not be assessed.

## Features

- **Duplicate detection:** Find exact duplicates within and across train/test partitions, including configured entity overlap.
- **Target leakage checks:** Identify target copies, deterministic re-encodings, and unusually strong associations that warrant review.
- **Temporal checks:** Compare feature-availability, prediction, and outcome timestamps when you supply the relevant metadata.
- **Schema and identifier checks:** Flag column mismatches, suspicious identifiers, constant features, and other structural issues.
- **Split recommendations:** Recommend a strategy based on your declared prediction scenario. Creating a split requires explicit confirmation.
- **Evidence-backed reports:** Inspect findings, severity, confidence, evidence, recommended actions, limitations, and skipped checks.
- **Local UI:** Run audits through an optional Streamlit interface without sending uploaded datasets to an external service.

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Varunnn-1605/SplitGuard.git
cd SplitGuard

python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# If script execution is blocked, allow it for this session only:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Generate a sample dataset containing a planted target-leakage issue and run an audit:

```bash
python examples/generate_synthetic_data.py

splitguard audit \
  --train examples/data/02_target_copy/train.csv \
  --test examples/data/02_target_copy/test.csv \
  --target churn \
  --output report.json
```

On Windows PowerShell, put the `splitguard audit` command on one line (PowerShell does not use `\` for line continuation).

The audit prints a terminal summary and writes a structured JSON report. It should report `TGT-001` (critical) for `churn_score_final`, a copy of the target. Exit codes: `0` audit completed, `1` a finding met `--fail-on`, `2` invalid input or usage, `3` a check failed internally.

## Input modes

| Mode | Command | Purpose |
|---|---|---|
| Train + test files | `splitguard audit --train train.csv --test test.csv` | Compare existing partitions |
| One file with a split column | `splitguard audit --data data.csv --split-column split` | Audit declared train/test rows |
| Unsplit dataset | `splitguard inspect --data data.csv` | Run available checks and recommend a split strategy |

To create a split after reviewing a recommendation, run `splitguard split --data data.csv --confirm-strategy <strategy> --out-dir <dir>`; it will not split without the explicit strategy name.

Run `splitguard --help` or `splitguard audit --help` for all options (target, entity ID, timestamps, feature availability, scenario, `--fail-on`, `--config`). [`examples/README.md`](examples/README.md) lists ready-to-run commands for every sample dataset.

### Python API

```python
from splitguard import AuditConfig, audit_train_test
from splitguard.reporting import write_json_report

result = audit_train_test("train.csv", "test.csv", AuditConfig(target="churn"))
for f in result.findings:
    print(f.finding_id, f.severity.value, f.confidence.value, f.title)
write_json_report(result, "report.json")
```

`audit_split_column`, `inspect_unsplit`, `recommend_split` and `create_split` are also exported from `splitguard`.

### Optional Streamlit UI

```bash
python -m pip install -e ".[ui]"
splitguard-ui
```

Open the local address printed by the launcher, normally `http://localhost:8501`. The UI wraps the same audit engine; it does not implement a separate set of detection rules.

## How findings work

Every finding separates two concepts:

- **Severity:** The potential impact if the issue is real.
- **Confidence:** How strongly the available evidence supports the finding.

Exact target copies can provide strong evidence. A highly correlated feature or a suspicious column name is only a review prompt; correlation alone does not establish leakage.

Temporal checks depend on metadata you supply. SplitGuard does not infer when a feature became available merely from its name.

## Validation

The project includes unit tests and an independent, deterministic synthetic validation suite covering injected leakage, clean controls, and boundary cases.

Latest recorded checkpoint:

- **251 tests passed** with the `dev` and `ui` extras installed (without Streamlit, the UI test module is skipped)
- **13/13** injected leakage scenarios detected
- **0/12** false alarms across clean controls
- **21/21** boundary cases passed

These are results on designed synthetic cases, **not estimates of real-world detection accuracy**. See [`validation/RESULTS.md`](validation/RESULTS.md) for details.

Run the tests and validation suite:

```bash
python -m pip install -e ".[dev,ui]"
pytest
python validation/run_validation.py --markdown validation/RESULTS.md
ruff check .
ruff format --check .
```

## Privacy and limitations

- Audits run locally. The core engine does not require an LLM, network access, or a paid API.
- Findings report counts, rates and statistics; they do not include raw row values by default.
- The optional UI keeps uploaded files in application memory and does not send them to an external service. The `splitguard-ui` launcher binds to `localhost` and turns off Streamlit's usage statistics.
- The UI is intended for a single user on their own machine. It has no authentication and is not hardened for public or shared deployment.
- Detection focuses on exact duplicates and individual-feature checks; near-duplicate matching and interactions among multiple features are not covered.
- Temporal findings are only as reliable as the supplied timestamps and availability metadata.
- Findings can overlap and should be interpreted in context.

## Project status

**Version 0.2.0.** The deterministic audit engine, CLI, Python API, JSON reporting, optional local UI, and synthetic validation suite are implemented.

An optional LLM-generated explanation layer is planned. It is not part of this release.

## License

No license has been selected yet. Until a license is added, do not assume others have permission to reuse, modify, or redistribute this code.