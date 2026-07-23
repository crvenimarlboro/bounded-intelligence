"""Development and reserved-pilot execution for Cognitive Core v2."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from bilab.environments.adaptation_ladder import (
    ContextBatch,
    balanced_context_episodes,
    batch_adaptation_episodes,
    make_context_episode,
)
from bilab.v2.checkpoints import save_v2_checkpoint
from bilab.v2.models import BaseV2Core, V2ModelConfig, build_v2_model
from bilab.v2.training import (
    V2TrainConfig,
    diagnose_v2_candidate,
    seed_v2,
    train_v2_candidate,
    v2_episode_objective,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fit_fixed_batch(
    family: str,
    batch: ContextBatch,
    *,
    seed: int,
    steps: int,
    relation_auxiliary: float = 0.0,
    routing_auxiliary: float = 0.0,
) -> dict[str, Any]:
    seed_v2(seed, threads=1)
    config = V2ModelConfig(family=family, hidden_dim=24)
    model = build_v2_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history: list[dict[str, float]] = []
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = v2_episode_objective(
            model,
            batch,
            relation_auxiliary_weight=relation_auxiliary,
            routing_auxiliary_weight=routing_auxiliary,
        )
        loss.backward()
        optimizer.step()
        if step in {0, 9, 49, 99, steps - 1}:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "accuracy": metrics["accuracy"],
                    "fully_informed_accuracy": metrics["fully_informed_accuracy"],
                }
            )
    model.eval()
    with torch.no_grad():
        _, final = v2_episode_objective(model, batch)
    return {
        "family": family,
        "steps": steps,
        "episodes": len(batch.public),
        "history": history,
        "accuracy": final["accuracy"],
        "fully_informed_accuracy": final["fully_informed_accuracy"],
    }


def run_v2_overfit(output_path: Path) -> dict[str, Any]:
    """Execute the fixed-data learnability ladder before procedural training."""

    single_sequence = ContextBatch.from_episodes(
        [make_context_episode(seed=100_001, steps=3, rules=(0, 1), surface_flip=0)]
    )
    one_world = ContextBatch.from_episodes(
        [make_context_episode(seed=100_002, steps=12, rules=(1, 0), surface_flip=0)]
    )
    multiple_worlds = batch_adaptation_episodes(
        balanced_context_episodes(seed_start=100_100, groups=2, steps=12)
    )
    results: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": "cognitive-core-v2-learnability",
        "one_sequence": _fit_fixed_batch("raw_router", single_sequence, seed=2101, steps=300),
        "one_world": _fit_fixed_batch("raw_router", one_world, seed=2102, steps=300),
        "multiple_worlds": _fit_fixed_batch("raw_router", multiple_worlds, seed=2103, steps=500),
        "explicit_relation_diagnostic": _fit_fixed_batch(
            "relation_router",
            multiple_worlds,
            seed=2104,
            steps=300,
            routing_auxiliary=0.25,
        ),
        "relation_supervised_raw_diagnostic": _fit_fixed_batch(
            "raw_router",
            multiple_worlds,
            seed=2105,
            steps=300,
            relation_auxiliary=0.25,
            routing_auxiliary=0.25,
        ),
    }
    results["all_primary_overfit_pass"] = all(
        results[name]["fully_informed_accuracy"] >= 0.99
        for name in ("one_sequence", "one_world", "multiple_worlds")
    )
    _write_json(output_path, results)
    return results


def _train_config(document: dict[str, Any], candidate: dict[str, Any], seed: int) -> V2TrainConfig:
    training = document["training"]
    return V2TrainConfig(
        family=candidate["family"],
        seed=seed,
        world_seed_start=training["world_seed_start"],
        validation_seed_start=training["validation_seed_start"],
        optimizer_steps=training["optimizer_steps"],
        batch_groups=training["batch_groups"],
        episode_steps=training["episode_steps"],
        hidden_dim=candidate["hidden_dim"],
        state_dim=candidate.get("state_dim", 2),
        thought_cycles=1,
        learning_rate=candidate.get("learning_rate", training["learning_rate"]),
        gradient_clip=training["gradient_clip"],
        relation_auxiliary_start=candidate.get("relation_auxiliary_start", 0.0),
        relation_auxiliary_end=candidate.get("relation_auxiliary_end", 0.0),
        routing_auxiliary_start=candidate.get("routing_auxiliary_start", 0.0),
        routing_auxiliary_end=candidate.get("routing_auxiliary_end", 0.0),
        validation_groups=training["validation_groups"],
        validation_interval=training["validation_interval"],
        torch_threads=training["torch_threads"],
        budget_bytes=8,
        environment=training["environment"],
        checkpoint_selection=training["checkpoint_selection"],
        quantization_bits=candidate.get("quantization_bits", 32),
    )


def _metric(row: dict[str, Any], name: str) -> float:
    diagnostic = row["diagnostics"]
    paths: dict[str, tuple[str, ...]] = {
        "delay": ("delay", "fully_informed_accuracy"),
        "composition": ("composition", "composition_accuracy"),
        "recovery": ("reversal", "post_feedback_recovery_accuracy"),
        "retention": ("reversal", "unrelated_rule_retention_accuracy"),
        "surface": ("surface_relabelled_delay", "fully_informed_accuracy"),
        "random": ("random_control", "fully_informed_accuracy"),
        "donor": ("composition_donor_swap", "donor_rule_consistency"),
        "rule_probe": ("rule_probe", "held_out_accuracy"),
        "relation_probe": ("relation_probe", "held_out_accuracy"),
        "route_separation": ("routing", "primitive_route_separation"),
        "reset": ("state_interventions", "reset", "fully_informed_accuracy"),
        "frozen": ("state_interventions", "frozen", "fully_informed_accuracy"),
        "uniform": (
            "routing_interventions",
            "uniform",
            "fully_informed_accuracy",
        ),
        "writer_disabled": (
            "routing_interventions",
            "writer_disabled",
            "fully_informed_accuracy",
        ),
    }
    value: Any = diagnostic
    for key in paths[name]:
        value = value[key]
    return float(value)


def _passes(row: dict[str, Any], gates: dict[str, Any]) -> bool:
    stage = row["stage"]
    thresholds = gates[stage]
    delay = _metric(row, "delay")
    required = all(
        (
            delay >= thresholds["minimum_delay"],
            _metric(row, "composition") >= thresholds["minimum_composition"],
            _metric(row, "recovery") >= thresholds["minimum_recovery"],
            _metric(row, "retention") >= thresholds["minimum_retention"],
            _metric(row, "donor") >= thresholds["minimum_donor"],
            abs(_metric(row, "random") - 0.5) <= thresholds["maximum_random_deviation"],
            delay - _metric(row, "reset") >= thresholds["minimum_state_drop"],
            delay - _metric(row, "frozen") >= thresholds["minimum_state_drop"],
        )
    )
    if stage in {"v2b", "v2c"}:
        required = required and (
            _metric(row, "route_separation") >= thresholds["minimum_route_separation"]
        )
        if "minimum_routing_drop" in thresholds:
            required = required and (
                delay - _metric(row, "uniform") >= thresholds["minimum_routing_drop"]
            )
    if stage == "v2c":
        required = required and _metric(row, "rule_probe") >= thresholds["minimum_rule_probe"]
    return required


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in sorted({row["candidate"] for row in rows}):
        selected = [row for row in rows if row["candidate"] == candidate]
        metrics = {
            name: [float(_metric(row, name)) for row in selected]
            for name in (
                "delay",
                "composition",
                "recovery",
                "retention",
                "surface",
                "random",
                "donor",
                "rule_probe",
                "relation_probe",
                "route_separation",
                "reset",
                "frozen",
                "uniform",
                "writer_disabled",
            )
        }
        result[candidate] = {
            "stage": selected[0]["stage"],
            "family": selected[0]["family"],
            "parameter_count": selected[0]["training"]["parameter_count"],
            "persistent_state_bytes": selected[0]["training"]["persistent_state_bytes"],
            "all_seeds_pass": all(row["passes_stage_gates"] for row in selected),
            "metrics": {
                name: {
                    "mean": statistics.mean(values),
                    "standard_deviation": statistics.pstdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in metrics.items()
            },
            "training_wall_seconds": {
                "mean": statistics.mean(
                    row["training"]["training_wall_seconds"] for row in selected
                )
            },
        }
    return result


def run_v2_pilot(repo: Path, config_path: Path, output_directory: Path) -> dict[str, Any]:
    """Train every declared isolated and joint candidate on reserved pilot seeds."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    partial_path = output_directory / "pilot_results.json"
    for candidate_name, candidate in document["candidates"].items():
        for seed in document["seeds"]:
            config = _train_config(document, candidate, seed)
            run = train_v2_candidate(config)
            if not isinstance(run.model, BaseV2Core):
                raise ValueError("v2 pilot candidates must expose bounded neural state")
            diagnostics = diagnose_v2_candidate(
                run.model,
                seed=seed,
                seed_base=document["evaluation_seed_start"],
                groups=document["evaluation_groups"],
            )
            metadata = save_v2_checkpoint(
                output_directory / "checkpoints" / candidate_name / f"seed-{seed}.pt",
                run.model,
                config,
                repo=repo,
                experiment_id=document["experiment_id"],
                training_step=run.metrics["best_step"],
                validation_score=run.metrics["best_validation_accuracy"],
            )
            row = {
                "candidate": candidate_name,
                "stage": candidate["stage"],
                "family": candidate["family"],
                "seed": seed,
                "training": run.metrics,
                "diagnostics": diagnostics,
                "checkpoint": metadata,
            }
            row["passes_stage_gates"] = _passes(row, document["stage_gates"])
            rows.append(row)
            _write_json(
                partial_path,
                {
                    "schema_version": "1.0",
                    "experiment_id": document["experiment_id"],
                    "status": "in_progress",
                    "rows": rows,
                },
            )
    result = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "status": "completed",
        "config": str(config_path),
        "seeds": document["seeds"],
        "candidate_selection_rule": document["candidate_selection_rule"],
        "rows": rows,
        "summary": _summary(rows),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(partial_path, result)
    return result
