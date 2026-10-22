"""Command-line interface: a thin layer over :mod:`splitguard.audit`.

Exit codes:
    0  audit completed (findings may exist)
    1  a finding met the --fail-on severity
    2  invalid input, configuration or usage
    3  a check failed internally, or an unexpected error occurred
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from splitguard._version import __version__
from splitguard.audit import audit_split_column, audit_train_test, inspect_unsplit
from splitguard.config import AuditConfig, ConfigError, Scenario, TaskType
from splitguard.reporting import render_summary, write_json_report
from splitguard.schemas import AuditResult, CheckStatus, Severity, to_jsonable
from splitguard.splitting import SplitConfirmationError, create_split, recommend_split
from splitguard.validation import InputValidationError, load_table

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_INVALID = 2
EXIT_FAILED = 3


def _availability(value: str) -> tuple[str, str]:
    feature, sep, column = value.partition("=")
    if not sep or not feature or not column:
        raise argparse.ArgumentTypeError("expected FEATURE=AVAILABILITY_COLUMN")
    return feature, column


def _common_options() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("column roles and options")
    g.add_argument(
        "--config", type=Path, help="JSON file with AuditConfig fields; flags override it"
    )
    g.add_argument("--target", help="target column")
    g.add_argument("--entity-id", help="entity/group ID column (e.g. customer_id)")
    g.add_argument("--timestamp", dest="prediction_timestamp", help="prediction timestamp column")
    g.add_argument("--outcome-timestamp", help="outcome timestamp column")
    g.add_argument(
        "--availability",
        action="append",
        type=_availability,
        metavar="FEATURE=COLUMN",
        help="column holding when FEATURE became available (repeatable)",
    )
    g.add_argument(
        "--exclude", action="append", metavar="COLUMN", help="non-feature column (repeatable)"
    )
    g.add_argument("--scenario", choices=[s.value for s in Scenario], help="evaluation scenario")
    g.add_argument(
        "--task-type", choices=[t.value for t in TaskType], help="override task type inference"
    )
    g.add_argument("--min-lead-time", help="minimum prediction-to-outcome gap, e.g. 30D")
    g.add_argument("--max-file-mb", type=float, help="maximum input file size in MB (default 200)")
    g.add_argument("--max-rows", type=int, help="maximum rows per input (default 1,000,000)")
    return p


def _report_options() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("output")
    g.add_argument("--output", "-o", type=Path, help="write the JSON report to this file")
    g.add_argument(
        "--fail-on",
        choices=[s.value for s in Severity],
        help="exit with code 1 if any finding has this severity or worse",
    )
    g.add_argument(
        "--no-timestamp", action="store_true", help="omit generated_at (byte-stable reports)"
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splitguard",
        description="Audit ML datasets for data leakage and train/test contamination.",
    )
    parser.add_argument("--version", action="version", version=f"splitguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    common, report = _common_options(), _report_options()

    audit = sub.add_parser(
        "audit",
        parents=[common, report],
        help="audit train/test files or one file with a split column",
    )
    audit.add_argument("--train", type=Path, help="training CSV (Mode A)")
    audit.add_argument("--test", type=Path, help="testing CSV (Mode A)")
    audit.add_argument("--data", type=Path, help="single CSV with a split column (Mode B)")
    audit.add_argument("--split-column", help="column labelling rows train/test (Mode B)")
    audit.add_argument("--train-value", default="train", help="split label for training rows")
    audit.add_argument("--test-value", default="test", help="split label for testing rows")
    audit.add_argument(
        "--test-diagnostics",
        action="store_true",
        help="also run target-copy checks on test labels (diagnostic only)",
    )

    inspect = sub.add_parser(
        "inspect",
        parents=[common, report],
        help="check an unsplit CSV and recommend a split strategy",
    )
    inspect.add_argument("--data", type=Path, required=True, help="unsplit CSV (Mode C)")
    inspect.add_argument("--test-size", type=float, default=0.2)
    inspect.add_argument("--seed", type=int, default=42)

    split = sub.add_parser(
        "split", parents=[common], help="create a split after confirming the strategy"
    )
    split.add_argument("--data", type=Path, required=True, help="unsplit CSV")
    split.add_argument(
        "--confirm-strategy",
        required=True,
        help="the recommended strategy name, typed to confirm it",
    )
    split.add_argument(
        "--out-dir", type=Path, required=True, help="directory for train.csv, test.csv"
    )
    split.add_argument("--test-size", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--overwrite", action="store_true", help="replace existing output files")
    return parser


def _config_from_args(args: argparse.Namespace) -> AuditConfig:
    data: dict[str, Any] = {}
    if args.config:
        try:
            data = json.loads(args.config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(
                f"Cannot read config file {args.config.name}: {type(exc).__name__}"
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError("Config file must contain a JSON object")
    for key in (
        "target",
        "entity_id",
        "prediction_timestamp",
        "outcome_timestamp",
        "scenario",
        "task_type",
        "min_lead_time",
    ):
        value = getattr(args, key, None)
        if value is not None:
            data[key] = value
    if args.availability:
        data["feature_availability"] = {
            **data.get("feature_availability", {}),
            **dict(args.availability),
        }
    if args.exclude:
        data["exclude_columns"] = sorted(set(data.get("exclude_columns", [])) | set(args.exclude))
    if getattr(args, "test_diagnostics", False):
        data["test_target_diagnostics"] = True
    limits = dict(data.get("limits", {}))
    if args.max_file_mb is not None:
        limits["max_file_bytes"] = int(args.max_file_mb * 1024 * 1024)
    if args.max_rows is not None:
        limits["max_rows"] = args.max_rows
    if limits:
        data["limits"] = limits
    return AuditConfig.from_dict(data)


def _run_audit(args: argparse.Namespace, config: AuditConfig) -> AuditResult:
    mode_a = args.train is not None or args.test is not None
    mode_b = args.data is not None or args.split_column is not None
    if mode_a == mode_b:
        raise ConfigError("Use either --train and --test, or --data with --split-column.")
    if mode_a:
        if args.train is None or args.test is None:
            raise ConfigError("--train and --test must be given together.")
        return audit_train_test(args.train, args.test, config)
    if args.data is None or args.split_column is None:
        raise ConfigError("--data and --split-column must be given together.")
    return audit_split_column(
        args.data, args.split_column, args.train_value, args.test_value, config
    )


def _finish(result: AuditResult, args: argparse.Namespace) -> int:
    print(render_summary(result))
    if args.output:
        write_json_report(result, args.output, include_generated_at=not args.no_timestamp)
        print(f"JSON report written to: {args.output}")
    else:
        print("Tip: add --output report.json to save the full report with evidence.")
    if result.checks_with_status(CheckStatus.FAILED):
        return EXIT_FAILED
    if args.fail_on and result.has_severity_at_least(args.fail_on):
        print(f"Exiting with code {EXIT_FINDINGS}: findings at or above '{args.fail_on}'.")
        return EXIT_FINDINGS
    return EXIT_OK


def _run_split(args: argparse.Namespace, config: AuditConfig) -> int:
    table = load_table(args.data, limits=config.limits)
    recommendation = recommend_split(table.frame, config, test_size=args.test_size, seed=args.seed)
    if recommendation.status != "recommended":
        print("Cannot split yet; missing information:", file=sys.stderr)
        for item in recommendation.missing_information:
            print(f"  - {item}", file=sys.stderr)
        return EXIT_INVALID
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: args.out_dir / f"{name}.csv" for name in ("train", "test")}
    paths["report"] = args.out_dir / "split_report.json"
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        print(f"Refusing to overwrite {existing}; pass --overwrite.", file=sys.stderr)
        return EXIT_INVALID
    result = create_split(
        table.frame,
        recommendation,
        confirmed_strategy=args.confirm_strategy,
        config=config,
        test_size=args.test_size,
        seed=args.seed,
    )
    result.train.to_csv(paths["train"], index=False)
    result.test.to_csv(paths["test"], index=False)
    payload = {"recommendation": recommendation.to_dict(), "split": result.report}
    paths["report"].write_text(
        json.dumps(to_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"Split created with strategy {recommendation.strategy}: "
        f"{result.report['n_train_rows']} train rows, {result.report['n_test_rows']} test rows."
    )
    for warning in result.report["warnings"]:
        print(f"Warning: {warning}")
    print(f"Wrote {paths['train']}, {paths['test']} and {paths['report']}.")
    print("This split does not remove leakage; run `splitguard audit` on the new files.")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.command == "audit":
            return _finish(_run_audit(args, config), args)
        if args.command == "inspect":
            result = inspect_unsplit(args.data, config, test_size=args.test_size, seed=args.seed)
            return _finish(result, args)
        return _run_split(args, config)
    except InputValidationError as exc:
        print("Input validation failed:", file=sys.stderr)
        for issue in exc.issues:
            print(f"  [{issue.code}] {issue.message}", file=sys.stderr)
            if issue.hint:
                print(f"      hint: {issue.hint}", file=sys.stderr)
        return EXIT_INVALID
    except (ConfigError, SplitConfirmationError, NotImplementedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except OSError as exc:
        print(
            f"Error: cannot read or write a file ({type(exc).__name__}: {exc.strerror})",
            file=sys.stderr,
        )
        return EXIT_INVALID
    except Exception as exc:  # noqa: BLE001 - never echo internal messages (may hold data)
        print(f"Unexpected internal error: {type(exc).__name__}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
