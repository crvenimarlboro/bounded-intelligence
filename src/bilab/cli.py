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
from bilab.environments.adaptation_ladder import exhaustive_oracle_validation
from bilab.manifest import ManifestError, load_manifest
from bilab.smoke import run_smoke
from bilab.training.commands import (
    ablate_checkpoint_file,
    evaluate_checkpoint_file,
    regenerate_report,
    validate_worlds,
)
from bilab.training.commands import (
    run_smoke as run_core_smoke,
)
from bilab.training.experiment import load_experiment_config, run_experiment
from bilab.training.v1_experiment import (
    run_compression_pilot,
    run_level5_pilot,
    run_level7_pilot,
    run_level8_pilot,
    run_level9_pilot,
    run_overfit_suite,
    run_pilot,
)
from bilab.training.v1_final import (
    compare_v1_result_files,
    diagnose_v1_checkpoint_file,
    evaluate_v1_checkpoint_file,
    refresh_result_output_bytes,
    refresh_temporal_credit_results,
    regenerate_v1_report,
    run_confirmatory,
)
from bilab.v2.final import evaluate_v2_checkpoint, run_v2_final
from bilab.v2.runner import run_v2_overfit, run_v2_pilot
from bilab.v2.scaffold_audit import run_v1_scaffold_audit


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

    core = subcommands.add_parser("core", help="train and evaluate Cognitive Core v0")
    core_commands = core.add_subparsers(dest="core_command", required=True)
    core_validate = core_commands.add_parser("validate-worlds")
    core_validate.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v0/configs/final.json",
    )
    core_smoke = core_commands.add_parser("smoke")
    core_smoke.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v0/configs/pilot.json",
    )
    core_smoke.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v0/smoke",
    )
    core_run = core_commands.add_parser("run")
    core_run.add_argument("--config", type=Path, required=True)
    core_run.add_argument("--output", type=Path, required=True)
    core_evaluate = core_commands.add_parser("evaluate")
    core_evaluate.add_argument("--checkpoint", type=Path, required=True)
    core_evaluate.add_argument("--output", type=Path, required=True)
    core_ablate = core_commands.add_parser("ablate")
    core_ablate.add_argument("--checkpoint", type=Path, required=True)
    core_ablate.add_argument("--output", type=Path, required=True)
    core_report = core_commands.add_parser("report")
    core_report.add_argument("--results", type=Path, required=True)
    core_report.add_argument("--output", type=Path, required=True)

    v1 = subcommands.add_parser("v1", help="run the Cognitive Core v1 adaptation ladder")
    v1_commands = v1.add_subparsers(dest="v1_command", required=True)
    v1_commands.add_parser("validate", help="exhaustively validate the minimal oracle")
    v1_overfit = v1_commands.add_parser("overfit", help="run the learnability ladder")
    v1_overfit.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/development/overfit.json",
    )
    v1_pilot = v1_commands.add_parser("pilot", help="run reserved-pilot candidate selection")
    v1_pilot.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/pilot.json",
    )
    v1_pilot.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/pilot",
    )
    v1_level5 = v1_commands.add_parser(
        "pilot-level5", help="test two independent rules on reserved pilot seeds"
    )
    v1_level5.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/level5_pilot.json",
    )
    v1_level5.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/level5-pilot",
    )
    v1_compression = v1_commands.add_parser(
        "pilot-compression", help="sweep Level-5 state width and quantization"
    )
    v1_compression.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/compression_pilot.json",
    )
    v1_compression.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/compression-pilot",
    )
    v1_level7 = v1_commands.add_parser("pilot-level7", help="test composition of two learned rules")
    v1_level7.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/level7_pilot.json",
    )
    v1_level7.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/level7-pilot",
    )
    v1_level8 = v1_commands.add_parser(
        "pilot-level8", help="test delayed retention across marked non-events"
    )
    v1_level8.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/level8_pilot.json",
    )
    v1_level8.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/level8-pilot",
    )
    v1_level9 = v1_commands.add_parser(
        "pilot-level9", help="test unmarked rule replacement and retention"
    )
    v1_level9.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/level9_pilot.json",
    )
    v1_level9.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/level9-pilot",
    )
    v1_final = v1_commands.add_parser("final", help="run the frozen v1 confirmatory protocol")
    v1_final.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/final.json",
    )
    v1_final.add_argument(
        "--manifest",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/manifest.json",
    )
    v1_final.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/final-v1.0",
    )
    v1_evaluate = v1_commands.add_parser(
        "evaluate", help="independently evaluate a saved v1 checkpoint"
    )
    v1_evaluate.add_argument("--checkpoint", type=Path, required=True)
    v1_evaluate.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/final.json",
    )
    v1_evaluate.add_argument("--output", type=Path, required=True)
    for command, help_text in (
        ("probe", "reproduce state probes and temporal-gradient diagnostics"),
        ("intervene", "reproduce causal state interventions"),
        ("ablate", "reproduce state, compression, and recurrence ablations"),
    ):
        diagnostic = v1_commands.add_parser(command, help=help_text)
        diagnostic.add_argument("--checkpoint", type=Path, required=True)
        diagnostic.add_argument(
            "--config",
            type=Path,
            default=repository_root() / "experiments/cognitive_core_v1/configs/final.json",
        )
        diagnostic.add_argument("--output", type=Path, required=True)
    v1_report = v1_commands.add_parser(
        "report", help="regenerate the v1 report from normalized results"
    )
    v1_report.add_argument("--results", type=Path, required=True)
    v1_report.add_argument("--output", type=Path, required=True)
    v1_report.add_argument("--summary-output", type=Path)
    v1_report.add_argument("--reproduction-comparison", type=Path)
    v1_report.add_argument(
        "--refresh-resource-accounting",
        action="store_true",
        help="include results.json itself in the generated-byte measurement",
    )
    v1_report.add_argument(
        "--refresh-temporal-credit",
        action="store_true",
        help="rerun amended temporal-credit probes from core checkpoints",
    )
    v1_report.add_argument("--checkpoint-root", type=Path)
    v1_compare = v1_commands.add_parser(
        "compare", help="compare deterministic evidence and model tensors across two runs"
    )
    v1_compare.add_argument("--left", type=Path, required=True)
    v1_compare.add_argument("--right", type=Path, required=True)
    v1_compare.add_argument("--output", type=Path, required=True)

    v2 = subcommands.add_parser("v2", help="run Cognitive Core v2 scaffold-removal experiments")
    v2_commands = v2.add_subparsers(dest="v2_command", required=True)
    v2_audit = v2_commands.add_parser(
        "audit-v1", help="measure dependence of frozen v1 checkpoints on engineered scaffolds"
    )
    v2_audit.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v1/configs/final.json",
    )
    v2_audit.add_argument(
        "--checkpoint-root",
        type=Path,
        default=repository_root() / "results/cognitive_core_v1/final-v1.0/checkpoints",
    )
    v2_audit.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v2/v1-scaffold-audit/results.json",
    )
    v2_overfit = v2_commands.add_parser("overfit", help="run the v2 fixed-data learnability ladder")
    v2_overfit.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v2/overfit/results.json",
    )
    v2_pilot = v2_commands.add_parser(
        "pilot", help="train reserved v2 scaffold-removal pilot candidates"
    )
    v2_pilot.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v2/configs/pilot.json",
    )
    v2_pilot.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v2/pilot-v1.0",
    )
    v2_final = v2_commands.add_parser(
        "final", help="run the frozen v2 confirmatory protocol and controls"
    )
    v2_final.add_argument(
        "--config",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v2/configs/final.json",
    )
    v2_final.add_argument(
        "--manifest",
        type=Path,
        default=repository_root() / "experiments/cognitive_core_v2/manifest.json",
    )
    v2_final.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "results/cognitive_core_v2/final-v1.0",
    )
    v2_evaluate = v2_commands.add_parser(
        "evaluate", help="independently evaluate a saved v2 checkpoint"
    )
    v2_evaluate.add_argument("--checkpoint", type=Path, required=True)
    v2_evaluate.add_argument("--output", type=Path, required=True)
    v2_evaluate.add_argument("--seed-base", type=int, default=420_000)
    v2_evaluate.add_argument("--groups", type=int, default=64)
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
        elif args.command == "core":
            if args.core_command == "validate-worlds":
                report = validate_worlds(load_experiment_config(args.config))
                print(json.dumps(report, indent=2, sort_keys=True))
            elif args.core_command == "smoke":
                result = run_core_smoke(repository_root(), args.config, args.output)
                print(
                    f"core smoke complete: supported={result['assessment']['supported']}, "
                    f"wall_seconds={result['resources']['total_wall_seconds']:.2f}"
                )
            elif args.core_command == "run":
                result = run_experiment(repository_root(), args.config, args.output)
                print(
                    f"core run complete: supported={result['assessment']['supported']}, "
                    f"wall_seconds={result['resources']['total_wall_seconds']:.2f}"
                )
            elif args.core_command == "evaluate":
                document = evaluate_checkpoint_file(args.checkpoint, args.output)
                accuracy = document["evaluation"]["metrics"]["next_outcome_accuracy"]
                print(f"checkpoint reproduced: next_outcome_accuracy={accuracy:.6f}")
            elif args.core_command == "ablate":
                document = ablate_checkpoint_file(args.checkpoint, args.output)
                print(f"ablations complete: {', '.join(document['evaluations'])}")
            elif args.core_command == "report":
                regenerate_report(args.results, args.output)
                print(f"report written: {args.output}")
        elif args.command == "v1":
            if args.v1_command == "validate":
                print(json.dumps(exhaustive_oracle_validation(), indent=2, sort_keys=True))
            elif args.v1_command == "overfit":
                result = run_overfit_suite()
                _write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
                print(
                    "v1 overfit complete: "
                    f"sequence={result['one_sequence']['accuracy']:.3f}, "
                    f"world={result['one_world']['accuracy']:.3f}, "
                    f"explicit={result['explicit_correct_state']['accuracy']:.3f}"
                )
            elif args.v1_command == "pilot":
                result = run_pilot(repository_root(), args.config, args.output)
                print(
                    "v1 pilot complete: "
                    f"selected={result['selected_candidate']}, "
                    f"wall_seconds={result['pilot_wall_seconds']:.2f}"
                )
            elif args.v1_command == "pilot-level5":
                result = run_level5_pilot(repository_root(), args.config, args.output)
                print(
                    "v1 level-5 pilot complete: "
                    f"all_seeds_pass={result['all_seeds_pass']}, "
                    f"wall_seconds={result['pilot_wall_seconds']:.2f}"
                )
            elif args.v1_command == "pilot-compression":
                result = run_compression_pilot(repository_root(), args.config, args.output)
                print(
                    "v1 compression pilot complete: "
                    f"selected_state_dimension={result['selected_state_dimension']}, "
                    f"wall_seconds={result['pilot_wall_seconds']:.2f}"
                )
            elif args.v1_command == "pilot-level7":
                result = run_level7_pilot(repository_root(), args.config, args.output)
                print(
                    "v1 level-7 pilot complete: "
                    f"all_seeds_pass={result['all_seeds_pass']}, "
                    f"wall_seconds={result['pilot_wall_seconds']:.2f}"
                )
            elif args.v1_command == "pilot-level8":
                result = run_level8_pilot(repository_root(), args.config, args.output)
                print(
                    "v1 level-8 pilot complete: "
                    f"all_seeds_pass={result['all_seeds_pass']}, "
                    f"wall_seconds={result['pilot_wall_seconds']:.2f}"
                )
            elif args.v1_command == "pilot-level9":
                result = run_level9_pilot(repository_root(), args.config, args.output)
                print(
                    "v1 level-9 pilot complete: "
                    f"all_seeds_pass={result['all_seeds_pass']}, "
                    f"wall_seconds={result['pilot_wall_seconds']:.2f}"
                )
            elif args.v1_command == "final":
                result = run_confirmatory(
                    repository_root(), args.config, args.manifest, args.output
                )
                print(
                    "v1 final complete: "
                    f"conclusion={result['assessment']['conclusion_class']}, "
                    f"wall_seconds={result['resources']['total_wall_seconds']:.2f}"
                )
            elif args.v1_command == "evaluate":
                result = evaluate_v1_checkpoint_file(args.checkpoint, args.config, args.output)
                delay = result["evaluation"]["delay"]["fully_informed_accuracy"]
                print(
                    f"v1 checkpoint reproduced: delay_accuracy={delay:.6f}, "
                    f"weights_unchanged={result['weights_unchanged']}"
                )
            elif args.v1_command in {"probe", "intervene", "ablate"}:
                result = diagnose_v1_checkpoint_file(
                    args.checkpoint,
                    args.config,
                    args.output,
                    section=args.v1_command,
                )
                print(
                    f"v1 {args.v1_command} complete: "
                    f"weights_unchanged={result['weights_unchanged']}"
                )
            elif args.v1_command == "report":
                if args.refresh_resource_accounting:
                    refresh_result_output_bytes(args.results)
                if args.refresh_temporal_credit:
                    if args.checkpoint_root is None:
                        raise ValueError("--refresh-temporal-credit requires --checkpoint-root")
                    refresh_temporal_credit_results(args.results, args.checkpoint_root)
                summary = regenerate_v1_report(
                    args.results,
                    args.output,
                    summary_output=args.summary_output,
                    reproduction_comparison=args.reproduction_comparison,
                )
                print(
                    "v1 report written: "
                    f"conclusion={summary['assessment']['conclusion_class']}, "
                    f"path={args.output}"
                )
            elif args.v1_command == "compare":
                comparison = compare_v1_result_files(args.left, args.right, args.output)
                print(
                    "v1 comparison complete: "
                    f"stable_rows_equal={comparison['stable_rows_equal']}, "
                    "checkpoint_digests_equal="
                    f"{comparison['all_checkpoint_model_digests_equal']}"
                )
        elif args.command == "v2":
            if args.v2_command == "audit-v1":
                result = run_v1_scaffold_audit(args.config, args.checkpoint_root, args.output)
                native = result["aggregate"]["native"]
                print(
                    "v1 scaffold audit complete: "
                    f"delay={native['delay_accuracy']['mean']:.3f}, "
                    f"composition={native['composition_accuracy']['mean']:.3f}, "
                    f"wall_seconds={result['wall_seconds']:.2f}"
                )
            elif args.v2_command == "overfit":
                result = run_v2_overfit(args.output)
                print(
                    "v2 learnability ladder complete: "
                    f"all_primary_overfit_pass={result['all_primary_overfit_pass']}"
                )
            elif args.v2_command == "pilot":
                result = run_v2_pilot(repository_root(), args.config, args.output)
                passed = [
                    name for name, summary in result["summary"].items() if summary["all_seeds_pass"]
                ]
                print(
                    "v2 pilot complete: "
                    f"passing_candidates={','.join(passed) or 'none'}, "
                    f"wall_seconds={result['wall_seconds']:.2f}"
                )
            elif args.v2_command == "final":
                result = run_v2_final(repository_root(), args.config, args.manifest, args.output)
                print(
                    "v2 final complete: "
                    f"all_primary_reproduced={result['all_primary_reproduced']}, "
                    f"wall_seconds={result['wall_seconds']:.2f}"
                )
            elif args.v2_command == "evaluate":
                result = evaluate_v2_checkpoint(
                    args.checkpoint,
                    args.output,
                    seed_base=args.seed_base,
                    groups=args.groups,
                )
                diagnostics = result["diagnostics"]
                print(
                    "v2 checkpoint evaluation complete: "
                    f"delay={diagnostics['delay']['fully_informed_accuracy']:.3f}"
                )
    except (BenchmarkError, ManifestError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
