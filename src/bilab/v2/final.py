"""Confirmatory execution and independent checkpoint evaluation for v2."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from torch import nn

from bilab.environments.adaptation_ladder import (
    balanced_composition_episodes,
    balanced_delayed_episodes,
    balanced_reversal_episodes,
    random_delayed_episodes,
)
from bilab.training.v1 import evaluate_model, evaluate_rule_change, state_dict_digest
from bilab.v2.checkpoints import load_v2_checkpoint, save_v2_checkpoint
from bilab.v2.models import BaseV2Core
from bilab.v2.runner import _metric, _passes
from bilab.v2.training import (
    V2TrainConfig,
    assert_v2_ranges_disjoint,
    diagnose_v2_candidate,
    long_sequence_stability,
    train_v2_candidate,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _variant_config(document: dict[str, Any], variant: dict[str, Any], seed: int) -> V2TrainConfig:
    training = document["training"]
    config = V2TrainConfig(
        family=variant["family"],
        seed=seed,
        world_seed_start=training["world_seed_start"],
        validation_seed_start=training["validation_seed_start"],
        optimizer_steps=training["optimizer_steps"],
        batch_groups=training["batch_groups"],
        episode_steps=training["episode_steps"],
        hidden_dim=variant["hidden_dim"],
        state_dim=variant.get("state_dim", 2),
        thought_cycles=1,
        learning_rate=training["learning_rate"],
        gradient_clip=training["gradient_clip"],
        validation_groups=training["validation_groups"],
        validation_interval=training["validation_interval"],
        torch_threads=training["torch_threads"],
        budget_bytes=8,
        environment=training["environment"],
        checkpoint_selection=training["checkpoint_selection"],
        quantization_bits=variant.get("quantization_bits", 32),
    )
    assert_v2_ranges_disjoint(
        config,
        additional={
            "final_evaluation": (
                document["evaluation_seed_start"],
                document["evaluation_seed_start"] + 4_999,
            )
        },
    )
    return config


def _control_diagnostics(
    model: nn.Module, *, seed: int, seed_base: int, groups: int
) -> dict[str, Any]:
    """Behavioral controls without assuming a neural writer trace or float state."""

    delay = balanced_delayed_episodes(
        seed_start=seed_base + 500, groups=groups, delay_steps=8, query_steps=6
    )
    composition = balanced_composition_episodes(
        seed_start=seed_base + 1_000, groups=groups, steps=11
    )
    reversal = balanced_reversal_episodes(seed_start=seed_base, groups=groups)
    relabelled = balanced_delayed_episodes(
        seed_start=seed_base + 2_000,
        groups=groups,
        delay_steps=8,
        query_steps=6,
        relabel=True,
    )
    random_control = random_delayed_episodes(
        seed_start=seed_base + 2_500,
        count=groups * 4,
        delay_steps=8,
        query_steps=6,
        relabel=True,
    )
    before = state_dict_digest(model)
    result = {
        "delay": evaluate_model(model, delay),
        "composition": evaluate_model(model, composition),
        "reversal": evaluate_rule_change(model, reversal),
        "surface_relabelled_delay": evaluate_model(model, relabelled),
        "random_control": evaluate_model(model, random_control),
        "state_interventions": {
            name: evaluate_model(model, delay, intervention=name, random_seed=seed)
            for name in ("reset", "frozen")
        },
    }
    result["weights_unchanged"] = before == state_dict_digest(model)
    return result


def evaluate_v2_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    seed_base: int,
    groups: int = 64,
) -> dict[str, Any]:
    model, config, metadata = load_v2_checkpoint(checkpoint)
    if not isinstance(model, BaseV2Core):
        diagnostics = _control_diagnostics(
            model, seed=config.seed, seed_base=seed_base, groups=groups
        )
    else:
        diagnostics = diagnose_v2_candidate(
            model, seed=config.seed, seed_base=seed_base, groups=groups
        )
    result = {
        "schema_version": "1.0",
        "checkpoint": str(checkpoint),
        "metadata": metadata,
        "diagnostics": diagnostics,
        "model_digest": state_dict_digest(model),
    }
    _write_json(output, result)
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    names = sorted({row["variant"] for row in rows})
    for name in names:
        selected = [row for row in rows if row["variant"] == name]
        metric_values: dict[str, list[float]] = {}
        for metric, path in {
            "delay": ("delay", "fully_informed_accuracy"),
            "composition": ("composition", "composition_accuracy"),
            "recovery": ("reversal", "post_feedback_recovery_accuracy"),
            "retention": ("reversal", "unrelated_rule_retention_accuracy"),
            "surface": ("surface_relabelled_delay", "fully_informed_accuracy"),
            "random": ("random_control", "fully_informed_accuracy"),
        }.items():
            values: list[float] = []
            for row in selected:
                value: Any = row["diagnostics"]
                for key in path:
                    value = value[key]
                values.append(float(value))
            metric_values[metric] = values
        aggregate[name] = {
            "family": selected[0]["family"],
            "parameter_count": selected[0]["training"]["parameter_count"],
            "persistent_state_bytes": selected[0]["training"]["persistent_state_bytes"],
            "metrics": {
                metric: {
                    "mean": statistics.mean(values),
                    "standard_deviation": statistics.pstdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for metric, values in metric_values.items()
            },
            "training_wall_seconds": sum(
                row["training"]["training_wall_seconds"] for row in selected
            ),
            "checkpoint_bytes": sum(row["checkpoint"]["checkpoint_bytes"] for row in selected),
        }
        if selected[0]["primary"]:
            aggregate[name]["all_seeds_pass"] = all(row["passes_stage_gates"] for row in selected)
            for metric in ("donor", "rule_probe", "relation_probe", "route_separation"):
                values = [_metric(row, metric) for row in selected]
                aggregate[name]["metrics"][metric] = {
                    "mean": statistics.mean(values),
                    "standard_deviation": statistics.pstdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
    return aggregate


def _write_curves(output_directory: Path, rows: list[dict[str, Any]]) -> None:
    with (output_directory / "learning_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "variant",
                "seed",
                "optimizer_step",
                "training_observations",
                "train_accuracy",
                "validation_accuracy",
            )
        )
        for row in rows:
            for point in row["training"]["history"]:
                writer.writerow(
                    (
                        row["variant"],
                        row["seed"],
                        point["optimizer_step"],
                        point["training_observations"],
                        point["train"]["fully_informed_accuracy"],
                        point["validation"]["fully_informed_accuracy"],
                    )
                )
    with (output_directory / "adaptation_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("variant", "seed", "prior_observations", "accuracy"))
        for row in rows:
            for prior, accuracy in row["diagnostics"]["delay"]["adaptation_curve"].items():
                writer.writerow((row["variant"], row["seed"], prior, accuracy))


def run_v2_final(
    repo: Path,
    config_path: Path,
    manifest_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Run every frozen final seed and declared equal-resource control."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "planned":
        raise ValueError("v2 final manifest must be planned before execution")
    if manifest["seeds"] != document["seeds"]:
        raise ValueError("v2 final config and manifest seeds differ")
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    partial = output_directory / "results.json"
    for variant_name, variant in document["variants"].items():
        for seed in document["seeds"]:
            config = _variant_config(document, variant, seed)
            run = train_v2_candidate(config)
            checkpoint_path = output_directory / "checkpoints" / variant_name / f"seed-{seed}.pt"
            checkpoint = save_v2_checkpoint(
                checkpoint_path,
                run.model,
                config,
                repo=repo,
                experiment_id=document["experiment_id"],
                training_step=run.metrics["best_step"],
                validation_score=run.metrics["best_validation_accuracy"],
            )
            primary = bool(variant.get("primary", False))
            if primary:
                if not isinstance(run.model, BaseV2Core):
                    raise ValueError("primary v2 model must expose generic diagnostics")
                diagnostics = diagnose_v2_candidate(
                    run.model,
                    seed=seed,
                    seed_base=document["evaluation_seed_start"],
                    groups=document["evaluation_groups"],
                )
                stability = long_sequence_stability(
                    run.model, lengths=tuple(document["long_sequence_lengths"])
                )
            else:
                diagnostics = _control_diagnostics(
                    run.model,
                    seed=seed,
                    seed_base=document["evaluation_seed_start"],
                    groups=document["evaluation_groups"],
                )
                stability = None
            row = {
                "variant": variant_name,
                "family": variant["family"],
                "stage": variant["stage"],
                "primary": primary,
                "seed": seed,
                "training": run.metrics,
                "diagnostics": diagnostics,
                "long_sequence_stability": stability,
                "checkpoint": checkpoint,
            }
            row["passes_stage_gates"] = _passes(row, document["stage_gates"]) if primary else None
            rows.append(row)
            _write_json(
                partial,
                {
                    "schema_version": "1.0",
                    "experiment_id": document["experiment_id"],
                    "status": "running",
                    "rows": rows,
                },
            )
    primary_rows = [row for row in rows if row["primary"]]
    reproduction: list[dict[str, Any]] = []
    for row in primary_rows:
        loaded, config, metadata = load_v2_checkpoint(Path(row["checkpoint"]["path"]))
        if not isinstance(loaded, BaseV2Core):
            raise ValueError("reloaded primary checkpoint lost its v2 model type")
        diagnostics = diagnose_v2_candidate(
            loaded,
            seed=config.seed,
            seed_base=document["evaluation_seed_start"],
            groups=document["evaluation_groups"],
        )
        reproduction.append(
            {
                "seed": config.seed,
                "metadata": metadata,
                "model_digest_equal": state_dict_digest(loaded)
                == row["checkpoint"]["model_digest"],
                "diagnostics_equal": _json_ready(diagnostics) == _json_ready(row["diagnostics"]),
            }
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "status": "completed",
        "config": str(config_path),
        "manifest": str(manifest_path),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "reproduction": reproduction,
        "all_primary_reproduced": all(
            item["model_digest_equal"] and item["diagnostics_equal"] for item in reproduction
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(partial, result)
    _write_curves(output_directory, rows)
    return result
