"""Command-line interface for local ModelBench runs."""
import argparse
import json
import sys
from pathlib import Path

from .data import DatasetError
from .engine import run_benchmark
from .report import console_report, encode_json, html_report


SAMPLE_DATA = Path(__file__).resolve().parent / "data" / "subscriptions.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="modelbench", description="Compare small tabular ML baselines locally.")
    parser.add_argument("dataset", nargs="?", type=Path, default=SAMPLE_DATA, help=f"CSV file (default: bundled demo dataset {SAMPLE_DATA.name})")
    parser.add_argument("--target", default="churned", help="Column to predict (default: churned)")
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction between 0.1 and 0.5 (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split and estimators")
    parser.add_argument("--cv", type=int, choices=(0, 3, 5, 10), default=0, help="Optional cross-validation folds on training data (default: off)")
    parser.add_argument("--format", choices=("console", "html", "json"), default="console")
    parser.add_argument("--output", type=Path, help="Write an HTML or JSON report to this file")
    args = parser.parse_args(argv)
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")
    if args.output and args.format == "console":
        parser.error("Use --format html or --format json with --output")
    try:
        result = run_benchmark(args.dataset, args.target, args.task, args.test_size, args.seed, args.cv)
    except DatasetError as exc:
        parser.error(str(exc))
    output = html_report(result) if args.format == "html" else encode_json(result) if args.format == "json" else None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}")
    elif output is not None:
        print(output)
    else:
        console_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
