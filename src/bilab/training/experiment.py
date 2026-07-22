"""End-to-end training, ablation, checkpoint, reproduction, and report orchestration."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bilab.environments.rule_worlds import make_evaluation_episodes
from bilab.models.factory import build_model, count_parameters
from bilab.resources import configuration_hash
from bilab.training.checkpoints import load_checkpoint, save_checkpoint
from bilab.training.evaluation import comparison_differences, evaluate_model
from bilab.training.reporting import (
    _aggregate_conditions,
    assess_success,
    write_curve_svg,
    write_json,
    write_learning_curves,
    write_report,
)
from bilab.training.trainer import train_model

PRIMARY_VARIANTS = ("no_memory", "episodic", "cognitive_core")
CORE_MODES = (
    "workspace_disabled",
    "workspace_frozen",
    "no_prediction_error",
    "recurrence_k1",
    "full",
)


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "workspace_bytes",
        "environment",
        "model",
        "training",
        "evaluation",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"experiment config missing fields: {sorted(missing)}")
    return config


def _persistent_bytes(model: object, variant: str) -> int:
    if variant == "no_memory":
        return 0
    state = model.initial_state(1)
    return int(model.state_nbytes(state))


def _metric_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    values: list[float] = []
    for category in left["category_accuracy"]:
        values.append(
            abs(
                float(left["category_accuracy"][category])
                - float(right["category_accuracy"][category])
            )
        )
    return max(values, default=0.0)


def _validate_parameter_match(config: dict[str, Any]) -> dict[str, int]:
    counts = {
        variant: count_parameters(build_model(variant, config)) for variant in PRIMARY_VARIANTS
    }
    difference = (max(counts.values()) - min(counts.values())) / max(counts.values())
    if difference > float(config["success_criteria"]["maximum_parameter_difference_fraction"]):
        raise ValueError(f"parameter match exceeds preregistered tolerance: {counts}")
    return counts


def run_experiment(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    output = output.resolve() if output.is_absolute() else (repo / output).resolve()
    if not output.is_relative_to(repo.resolve()):
        raise ValueError("experiment output must remain inside the repository")
    output.mkdir(parents=True, exist_ok=True)
    parameter_counts = _validate_parameter_match(config)
    evaluation_episodes = make_evaluation_episodes(config)
    config_hash = configuration_hash(config)
    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    raw_path = output / "raw_metrics.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    checkpoint_total = 0
    reproduction_observations = 0
    reproduction_seconds = 0.0

    for seed in config["training"]["seeds"]:
        seed_primary: dict[str, dict[str, Any]] = {}
        for variant in PRIMARY_VARIANTS:
            training = train_model(config, variant, int(seed))
            persistent_bytes = _persistent_bytes(training.model, variant)
            checkpoint_path = output / "checkpoints" / f"{variant}-seed{seed}.pt"
            checkpoint = save_checkpoint(
                checkpoint_path,
                training.model,
                config,
                repo=repo,
                variant=variant,
                seed=int(seed),
                persistent_bytes=persistent_bytes,
                training_step=training.final_step,
                validation_score=training.validation_accuracy,
            )
            checkpoint_total += int(checkpoint["checkpoint_bytes"])
            modes = CORE_MODES if variant == "cognitive_core" else ("full",)
            for mode in modes:
                evaluation = evaluate_model(
                    training.model,
                    evaluation_episodes,
                    variant=variant,
                    seed=int(seed),
                    mode=mode,
                    batch_size=int(config["training"]["batch_size"]),
                    adaptation_checkpoints=list(config["evaluation"]["adaptation_checkpoints"]),
                    recovery_windows=list(config["evaluation"]["recovery_windows"]),
                )
                row = asdict(evaluation)
                row.update(
                    {
                        "configuration_hash": config_hash,
                        "training_seconds": training.training_seconds,
                        "training_cpu_seconds": training.cpu_seconds,
                        "training_peak_ram_bytes": training.peak_ram_bytes,
                        "training_observations": training.observations_consumed,
                        "training_data_fingerprint": training.training_data_fingerprint,
                        "checkpoint": str(checkpoint_path.relative_to(repo)),
                        "checkpoint_bytes": checkpoint["checkpoint_bytes"],
                        "checkpoint_reproduction_error": 0.0,
                    }
                )
                rows.append(row)
                if mode == "full":
                    seed_primary[variant] = row
            for curve in training.curves:
                training_rows.append({"variant": variant, "seed": int(seed), **curve})

            reloaded, reloaded_config, _ = load_checkpoint(checkpoint_path)
            reproduction = evaluate_model(
                reloaded,
                make_evaluation_episodes(reloaded_config),
                variant=variant,
                seed=int(seed),
                mode="full",
                batch_size=int(config["training"]["batch_size"]),
                adaptation_checkpoints=list(config["evaluation"]["adaptation_checkpoints"]),
                recovery_windows=list(config["evaluation"]["recovery_windows"]),
            )
            reproduction_error = _metric_error(
                seed_primary[variant]["metrics"], reproduction.metrics
            )
            seed_primary[variant]["checkpoint_reproduction_error"] = reproduction_error
            reproduction_observations += reproduction.observations_consumed
            reproduction_seconds += reproduction.evaluation_seconds

        for baseline in ("no_memory", "episodic"):
            differences = comparison_differences(
                seed_primary["cognitive_core"], seed_primary[baseline]
            )
            if differences:
                raise ValueError(
                    f"incompatible comparison core vs {baseline}, seed {seed}: {differences}"
                )
        if time.perf_counter() - start_time > float(config["budgets"]["max_total_wall_seconds"]):
            raise RuntimeError("experiment exceeded hard wall-time budget")

    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    aggregates = _aggregate_conditions(rows)
    assessment = assess_success(rows, aggregates, config)
    total_wall = time.perf_counter() - start_time
    peak_ram = max(int(row["peak_ram_bytes"]) for row in rows)
    resources = {
        "total_wall_seconds": total_wall,
        "peak_ram_bytes": peak_ram,
        "checkpoint_bytes": checkpoint_total,
        "training_observations": sum(
            int(row["training_observations"]) for row in rows if row["mode"] == "full"
        ),
        "evaluation_observations": sum(int(row["observations_consumed"]) for row in rows),
        "checkpoint_reproduction_observations": reproduction_observations,
        "checkpoint_reproduction_seconds": reproduction_seconds,
        "parameter_counts": parameter_counts,
    }
    resources["evaluation_observations"] += reproduction_observations
    if peak_ram > int(config["budgets"]["max_peak_ram_bytes"]):
        raise RuntimeError("experiment exceeded peak RAM budget")
    if checkpoint_total > int(config["budgets"]["max_checkpoint_bytes"]):
        raise RuntimeError("experiment exceeded checkpoint byte budget")
    results = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "configuration_hash": config_hash,
        "rows": rows,
        "aggregates": aggregates,
        "assessment": assessment,
        "resources": resources,
    }
    write_json(output / "results.json", results)
    write_learning_curves(output / "learning_curves.csv", training_rows)
    write_curve_svg(output / "learning_curves.svg", training_rows)
    write_report(output / "report.md", results, title=config["experiment_id"])
    return results


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["experiment_id"] = "cognitive-core-v0-smoke"
    value["protocol_version"] = "smoke"
    value["environment"].update(
        {
            "train_worlds": 4,
            "train_episode_length": 8,
            "validation_worlds": 2,
            "evaluation_worlds_per_category": 2,
            "evaluation_episode_length": 10,
            "rule_change_episode_length": 12,
            "rule_change_step": 5,
        }
    )
    value["training"].update({"seeds": [7], "epochs": 1, "batch_size": 2})
    value["evaluation"]["adaptation_checkpoints"] = [0, 1, 2, 4, 8]
    value["evaluation"]["recovery_windows"] = [[0, 1], [2, 3], [4, 6]]
    value["success_criteria"]["minimum_positive_seeds"] = 1
    value["budgets"]["max_total_wall_seconds"] = 300
    return value
