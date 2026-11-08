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

### Optional plain-language explanations (local LLM)

SplitGuard can ask a local language model, served by [Ollama](https://ollama.com), to explain a finished report in plain language: why each finding matters, what its evidence shows, what to inspect next, and what was not assessed. The deterministic findings stay authoritative. The model cannot add, remove, re-rate or change findings, and its text is never written into the audit report.

**What the model sees.** Only an allowlisted summary of the report:
- per finding: ID, rule, category, severity, confidence, title, description, aggregate evidence, affected column names, recommended action and limitations;
- skipped or failed checks with their reasons;
- coverage warnings.

No CSV rows, DataFrames, file names, hashes or paths are sent. Run `splitguard explain report.json --show-input` to see exactly what would be sent. Column names come from your data and could contain misleading text; the explainer passes them as escaped data and tells the model not to follow them, but no prompt can rule this out completely.

**Setup (all manual; SplitGuard never downloads models or starts Ollama):**

1. Install Ollama from https://ollama.com/download and start it.
2. Download the default model yourself:
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```
3. Check that SplitGuard can reach it:
   ```bash
   splitguard explain --check
   ```

No extra Python packages are needed; the explainer uses the standard library to call Ollama's local API.

**Use it:**

```bash
splitguard audit --train train.csv --test test.csv --target churn --output report.json
splitguard explain report.json
```

In the UI, run an audit, then click **Explain these results** under the findings.

**Configure:**
- Model: set `--model` or `SPLITGUARD_LLM_MODEL`, for example `qwen2.5:7b-instruct` (pull it first).
- Timeout: set `--timeout` or `SPLITGUARD_LLM_TIMEOUT` (seconds; default 300).
- Endpoint: set `--host` or `SPLITGUARD_OLLAMA_HOST`. Only `localhost`, `127.0.0.1` or `::1` are accepted. Proxies, redirects and Ollama's own `OLLAMA_HOST` and `OLLAMA_API_KEY` variables are ignored.

**Resources (approximate, from Ollama's model listings; not measured here):**
- `qwen2.5:3b-instruct` is about a 2 GB download and needs roughly 3 GB of free RAM.
- `qwen2.5:7b-instruct` is about a 4.7 GB download and needs roughly 6 GB, which is tight on an 8 GB machine.
- Memory use and speed depend on context length (the explainer uses 6,144 tokens), quantization and available CPU/GPU. On a CPU, one explanation can take minutes.

The 3B model is the default because it fits modest hardware. It has not been benchmarked for explanation quality, and it is not claimed to be as accurate as 7B.

**Limits:**
- Explanations can be wrong or incomplete. Always check them against the findings and evidence they sit next to.
- Answers that are malformed, oversized or refer to unknown findings are rejected or trimmed.
- Simple checks flag text that overstates certainty (for example "the data is safe") or names a severity that differs from the report. These checks reduce the risk but do not remove it.
- Large reports are capped: the most severe 12 findings are sent, and the rest are listed as not explained.

## How findings work

Every finding separates two concepts:

- **Severity:** The potential impact if the issue is real.
- **Confidence:** How strongly the available evidence supports the finding.

Exact target copies can provide strong evidence. A highly correlated feature or a suspicious column name is only a review prompt; correlation alone does not establish leakage.

Temporal checks depend on metadata you supply. SplitGuard does not infer when a feature became available merely from its name.

## Validation

The project includes unit tests and an independent, deterministic synthetic validation suite covering injected leakage, clean controls, and boundary cases.

Latest recorded checkpoint:

- **320 tests passed** with the `dev` and `ui` extras installed (without Streamlit, the UI test modules are skipped). The explainer tests use a mock provider and a fake local server; they need no Ollama or model.
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

- Audits run locally. The core engine does not require an LLM, network access, or a paid API. The optional explainer only talks to a local Ollama endpoint, and only when you ask for an explanation.
- Findings report counts, rates and statistics; they do not include raw row values by default.
- The optional UI keeps uploaded files in application memory and does not send them to an external service. The `splitguard-ui` launcher binds to `localhost` and turns off Streamlit's usage statistics.
- The UI is intended for a single user on their own machine. It has no authentication and is not hardened for public or shared deployment.
- Detection focuses on exact duplicates and individual-feature checks; near-duplicate matching and interactions among multiple features are not covered.
- Temporal findings are only as reliable as the supplied timestamps and availability metadata.
- Findings can overlap and should be interpreted in context.

## Project status

**Version 0.2.0.** The deterministic audit engine, CLI, Python API, JSON reporting, optional local UI, and synthetic validation suite are implemented.

An optional local LLM explanation layer (Ollama, default `qwen2.5:3b-instruct`) is available but experimental. Its explanation quality has not yet been evaluated.

## License

No license has been selected yet. Until a license is added, do not assume others have permission to reuse, modify, or redistribute this code.