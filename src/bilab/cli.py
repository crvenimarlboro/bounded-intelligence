"""Command-line interface for the bounded-intelligence laboratory."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from bilab.benchmarks import BenchmarkError, normalize_sources, render_summary
from bilab.config import load_config, repository_root
from bilab.doctor import collect_environment, render_environment
from bilab.manifest import ManifestError, load_manifest
from bilab.smoke import run_smoke


def _paths(inputs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(
                sorted(
                    child
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() in {".csv", ".jsonl"}
                )
            )
        else:
            paths.append(path)
    return paths


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bilab")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="inspect the environment without mutating it")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--json", action="store_true", help="emit structured JSON")

    benchmark = subcommands.add_parser("bench", help="normalize llama-bench results")
    benchmark_commands = benchmark.add_subparsers(dest="bench_command", required=True)
    ingest = benchmark_commands.add_parser("ingest", help="ingest JSONL/CSV sources")
    ingest.add_argument("inputs", nargs="+")
    ingest.add_argument("--output", type=Path, help="write canonical JSON")
    ingest.add_argument("--summary-output", type=Path, help="write the human summary")

    manifest = subcommands.add_parser("manifest", help="validate an experiment manifest")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    validate = manifest_commands.add_parser("validate")
    validate.add_argument("path", type=Path)

    smoke = subcommands.add_parser("smoke", help="run the disposable pipeline experiment")
    smoke.add_argument(
        "--manifest",
        type=Path,
        default=repository_root() / "experiments" / "smoke" / "manifest.json",
    )
    smoke.add_argument("--output", type=Path, default=repository_root() / "results" / "smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            config = load_config(config_path=args.config)
            report = collect_environment(config)
            print(json.dumps(report, indent=2) if args.json else render_environment(report))
        elif args.command == "bench":
            document = normalize_sources(_paths(args.inputs))
            summary = render_summary(document)
            if args.output:
                _write(args.output, json.dumps(document, indent=2, sort_keys=True) + "\n")
            if args.summary_output:
                _write(args.summary_output, summary + "\n")
            print(summary)
        elif args.command == "manifest":
            manifest = load_manifest(args.path)
            print(f"valid manifest: {manifest['experiment_id']} ({manifest['status']})")
        elif args.command == "smoke":
            result = run_smoke(repository_root(), args.manifest, args.output)
            decisions = result["decision_demonstrations"]
            print(
                "smoke complete: "
                f"primary={decisions['succeeded']['status']}, "
                f"failure_demo={decisions['failed']['status']}, "
                f"rejection_demo={decisions['rejected_incomparable']['status']}"
            )
    except (BenchmarkError, ManifestError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
