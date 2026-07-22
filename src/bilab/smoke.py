"""Disposable pipeline smoke experiment; it does not measure intelligence."""

from __future__ import annotations

import json
import random
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

from bilab.manifest import load_manifest, validate_manifest
from bilab.resources import deep_size, directory_bytes, measure_call


def _observations(seed: int, count: int) -> tuple[list[int], int]:
    generator = random.Random(seed)
    values = [generator.randrange(0, 10_000) for _ in range(count)]
    target = 1 + sum(values) % 996
    return values, target


def _decision(
    baseline: dict[str, Any],
    intervention: dict[str, Any],
    expected_gain: float,
) -> dict[str, Any]:
    mismatches = [
        key
        for key in ("seeds", "observations_consumed", "condition_fingerprint")
        if baseline[key] != intervention[key]
    ]
    if mismatches:
        return {"status": "rejected_incomparable", "mismatches": mismatches, "score_gain": None}
    gain = intervention["mean_score"] - baseline["mean_score"]
    return {
        "status": "succeeded" if gain >= expected_gain else "failed",
        "mismatches": [],
        "score_gain": gain,
    }


def run_smoke(repo: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    """Run deterministic baseline/intervention trials and emit compact telemetry."""

    manifest = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    condition = {"task": "modular-checksum-smoke", "observations_per_seed": 64}
    fingerprint = json.dumps(condition, sort_keys=True, separators=(",", ":"))
    trials: list[dict[str, Any]] = []

    for seed in manifest["seeds"]:
        values, target = _observations(seed, condition["observations_per_seed"])
        temporary_bytes = deep_size(values)
        algorithms = {
            "baseline_constant_zero": (lambda: 0, "one constant emission"),
            "intervention_modular_sum": (
                lambda values=values: 1 + sum(values) % 996,
                "64 additions",
            ),
        }
        for algorithm, (function, proxy) in algorithms.items():
            result, resources = measure_call(
                function,
                experiment_id=manifest["experiment_id"],
                seed=seed,
                config={**condition, "algorithm": algorithm},
                repo=repo,
                observations=len(values),
                score_of=lambda prediction, target=target: float(prediction == target),
                persistent_bytes=0,
                temporary_bytes=temporary_bytes,
                model_parameters=0,
                compute_proxy=proxy,
            )
            trials.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "prediction": result,
                    "target": target,
                    "condition_fingerprint": fingerprint,
                    "resources": resources.to_dict(),
                }
            )

    aggregates: dict[str, dict[str, Any]] = {}
    for algorithm in {trial["algorithm"] for trial in trials}:
        members = [trial for trial in trials if trial["algorithm"] == algorithm]
        aggregates[algorithm] = {
            "seeds": [member["seed"] for member in members],
            "observations_consumed": sum(
                member["resources"]["observations_consumed"] for member in members
            ),
            "condition_fingerprint": fingerprint,
            "mean_score": statistics.fmean(member["resources"]["score"] for member in members),
            "elapsed_seconds": sum(member["resources"]["elapsed_seconds"] for member in members),
            "persistent_bytes": sum(member["resources"]["persistent_bytes"] for member in members),
        }
    baseline = aggregates["baseline_constant_zero"]
    intervention = aggregates["intervention_modular_sum"]
    succeeded = _decision(baseline, intervention, expected_gain=0.5)
    budgets = manifest["resource_budgets"]
    budget_violations: list[str] = []
    total_elapsed = baseline["elapsed_seconds"] + intervention["elapsed_seconds"]
    if total_elapsed > budgets["max_wall_seconds"]:
        budget_violations.append("max_wall_seconds")
    if any(
        trial["resources"]["observations_consumed"] > budgets["max_observations_per_repetition"]
        for trial in trials
    ):
        budget_violations.append("max_observations_per_repetition")
    if any(
        trial["resources"]["persistent_bytes"] > budgets["max_persistent_cognitive_bytes"]
        for trial in trials
    ):
        budget_violations.append("max_persistent_cognitive_bytes")
    if any(
        trial["resources"]["model_parameters"] != budgets["model_parameters"] for trial in trials
    ):
        budget_violations.append("model_parameters")
    succeeded["budget_violations"] = budget_violations
    if budget_violations:
        succeeded["status"] = "failed"
    failed = _decision(baseline, intervention, expected_gain=2.0)
    incomparable_intervention = deepcopy(intervention)
    incomparable_intervention["observations_consumed"] += 1
    rejected = _decision(baseline, incomparable_intervention, expected_gain=0.5)

    document = {
        "schema_version": "1.0",
        "purpose": "pipeline smoke test only; not an intelligence measurement",
        "manifest_id": manifest["experiment_id"],
        "trials": trials,
        "aggregates": aggregates,
        "decision_demonstrations": {
            "succeeded": succeeded,
            "failed": failed,
            "rejected_incomparable": rejected,
        },
    }
    results_path = output / "results.json"
    results_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed_manifest = dict(manifest)
    completed_manifest["status"] = "completed"
    completed_manifest["final_conclusion"] = (
        "Established only that the laboratory can validate a manifest, execute deterministic "
        "baseline/intervention trials, account resources, and classify comparisons."
    )
    completed_manifest["evidence_classification"] = "established"
    validate_manifest(completed_manifest)
    (output / "manifest.completed.json").write_text(
        json.dumps(completed_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    telemetry_before_report = directory_bytes(output)
    report = "\n".join(
        [
            "# Disposable smoke experiment report",
            "",
            "This experiment validates infrastructure only; it does not measure intelligence.",
            "",
            f"- Repetitions: {manifest['repetition_count']} with seeds {manifest['seeds']}",
            f"- Baseline mean score: {baseline['mean_score']:.3f}",
            f"- Intervention mean score: {intervention['mean_score']:.3f}",
            f"- Primary decision: {succeeded['status']}",
            f"- Primary budget violations: {budget_violations or 'none'}",
            f"- Deliberate unmet-threshold decision: {failed['status']}",
            f"- Deliberate condition-mismatch decision: {rejected['status']}",
            f"- Runtime persistent cognitive bytes: {intervention['persistent_bytes']}",
            f"- Research telemetry bytes before this report: {telemetry_before_report}",
            "- Peak VRAM: unavailable (CPU-only smoke experiment)",
            "",
            "Conclusion: pipeline behavior is established; no architecture claim was tested.",
        ]
    )
    (output / "report.md").write_text(report + "\n", encoding="utf-8")
    return document
