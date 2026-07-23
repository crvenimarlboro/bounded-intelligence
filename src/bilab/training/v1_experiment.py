"""Executable pilot and diagnostic stages for Cognitive Core v1."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bilab.environments.adaptation_ladder import (
    BinaryBatch,
    WorldKind,
    balanced_composition_episodes,
    balanced_context_episodes,
    balanced_delayed_episodes,
    balanced_reversal_episodes,
    exhaustive_oracle_validation,
    make_binary_episode,
    paired_structured_episodes,
    random_composition_episodes,
    random_context_episodes,
    random_delayed_episodes,
)
from bilab.models.v1 import (
    AdaptiveCore,
    FactorizedStateCore,
    NoMemoryControl,
    V1ModelConfig,
)
from bilab.training.v1 import (
    V1TrainConfig,
    assert_generation_splits_disjoint,
    composition_donor_state_swap,
    context_donor_state_swap,
    delayed_donor_state_swap,
    donor_state_swap,
    episode_objective,
    evaluate_model,
    evaluate_rule_change,
    linear_reversal_probe,
    linear_rule_probe,
    quantized_state_evaluation,
    seed_everything,
    state_component_ablation,
    temporal_credit_audit,
    train_candidate,
)
from bilab.training.v1_checkpoints import save_v1_checkpoint


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _fit_fixed_batch(
    model: nn.Module,
    batch: BinaryBatch,
    *,
    optimizer_steps: int,
    learning_rate: float = 0.01,
    auxiliary_rule_weight: float = 0.0,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    curve: list[float] = []
    for step in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = episode_objective(model, batch, auxiliary_rule_weight=auxiliary_rule_weight)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == optimizer_steps:
            curve.append(metrics["accuracy"])
    return curve


def _explicit_state_solver(seed: int = 104) -> dict[str, Any]:
    seed_everything(seed, 2)
    model = FactorizedStateCore(V1ModelConfig(hidden_dim=24, state_dim=4))
    episodes = paired_structured_episodes(seed_start=10_400, pairs=32, steps=6)
    batch = BinaryBatch.from_episodes(episodes)
    state = model.initial_state(len(episodes))
    state[:, 0] = torch.where(batch.hidden_rules == 0, 1.0, -1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    curve: list[float] = []
    for step in range(250):
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        correct = 0
        count = 0
        for time_index in range(batch.public.shape[1]):
            prediction = model.predict(batch.public[:, time_index], state)
            target = batch.outcomes[:, time_index]
            losses.append(F.cross_entropy(prediction.logits, target))
            correct += int((prediction.logits.argmax(-1) == target).sum())
            count += len(episodes)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0:
            curve.append(correct / count)
    with torch.no_grad():
        correct = 0
        count = 0
        for time_index in range(batch.public.shape[1]):
            prediction = model.predict(batch.public[:, time_index], state)
            correct += int((prediction.logits.argmax(-1) == batch.outcomes[:, time_index]).sum())
            count += len(episodes)
    return {"accuracy": correct / count, "learning_curve": curve, "optimizer_steps": 250}


def run_overfit_suite() -> dict[str, Any]:
    """Execute the mandatory learnability ladder before meta-learning claims."""

    start = time.perf_counter()
    seed_everything(101, 2)
    sequence = make_binary_episode(seed=10_101, steps=8, rule=1)
    sequence_model = NoMemoryControl(hidden_dim=16)
    sequence_curve = _fit_fixed_batch(
        sequence_model, BinaryBatch.from_episodes([sequence]), optimizer_steps=150
    )
    sequence_eval = evaluate_model(sequence_model, [sequence])

    seed_everything(102, 2)
    world = make_binary_episode(seed=10_102, steps=8, rule=0)
    world_model = FactorizedStateCore(V1ModelConfig(hidden_dim=16, state_dim=4))
    world_curve = _fit_fixed_batch(
        world_model, BinaryBatch.from_episodes([world]), optimizer_steps=150
    )
    world_eval = evaluate_model(world_model, [world])

    seed_everything(103, 2)
    several = paired_structured_episodes(seed_start=10_200, pairs=4, steps=8)
    several_model = FactorizedStateCore(V1ModelConfig(hidden_dim=16, state_dim=4))
    several_curve = _fit_fixed_batch(
        several_model, BinaryBatch.from_episodes(several), optimizer_steps=200
    )
    several_eval = evaluate_model(several_model, several)

    explicit = _explicit_state_solver()
    return {
        "one_sequence": {
            "accuracy": sequence_eval["overall_accuracy"],
            "learning_curve": sequence_curve,
            "optimizer_steps": 150,
        },
        "one_world": {
            "accuracy": world_eval["overall_accuracy"],
            "post_evidence_accuracy": world_eval["post_evidence_accuracy"],
            "learning_curve": world_curve,
            "optimizer_steps": 150,
        },
        "several_worlds": {
            "post_evidence_accuracy": several_eval["post_evidence_accuracy"],
            "learning_curve": several_curve,
            "optimizer_steps": 200,
        },
        "explicit_correct_state": explicit,
        "all_finite": all(
            math.isfinite(value)
            for curve in (sequence_curve, world_curve, several_curve)
            for value in curve
        ),
        "wall_seconds": time.perf_counter() - start,
    }


def _random_episodes(seed_start: int, count: int, steps: int) -> list:
    return [
        make_binary_episode(
            seed=seed_start + index,
            steps=steps,
            rule=None,
            kind=WorldKind.RANDOM,
            surface_flip=index % 2,
        )
        for index in range(count)
    ]


def _candidate_config(
    document: dict[str, Any], candidate: dict[str, Any], seed: int
) -> V1TrainConfig:
    return V1TrainConfig(
        family=candidate["family"],
        seed=seed,
        world_seed_start=document["training_world_seed_start"],
        validation_seed_start=document["validation_world_seed_start"],
        optimizer_steps=document["optimizer_steps"],
        batch_pairs=document["batch_pairs"],
        episode_steps=document["episode_steps"],
        hidden_dim=candidate["hidden_dim"],
        state_dim=document["state_dim"],
        thought_cycles=document["thought_cycles"],
        feedback_mode=candidate["feedback_mode"],
        learning_rate=document["learning_rate"],
        predictive_weight=candidate["predictive_weight"],
        validation_pairs=document["validation_pairs"],
        validation_interval=50,
        torch_threads=document["torch_threads"],
        budget_bytes=document["state_bytes"],
    )


def _diagnose_candidate(model: nn.Module, document: dict[str, Any]) -> dict[str, Any]:
    seed_start = document["evaluation_world_seed_start"]
    pairs = document["evaluation_pairs"]
    steps = document["episode_steps"]
    structured = paired_structured_episodes(seed_start=seed_start, pairs=pairs, steps=steps)
    relabelled = paired_structured_episodes(
        seed_start=seed_start, pairs=pairs, steps=steps, relabel=True
    )
    random_episodes = _random_episodes(document["random_world_seed_start"], pairs * 2, steps)
    diagnosis: dict[str, Any] = {
        "structured": evaluate_model(model, structured),
        "surface_relabelled": evaluate_model(model, relabelled),
        "random_control": evaluate_model(model, random_episodes),
        "reset": evaluate_model(model, structured, intervention="reset"),
        "frozen": evaluate_model(model, structured, intervention="frozen"),
        "random_state": evaluate_model(model, structured, intervention="random", random_seed=91),
        "shuffled_state": evaluate_model(model, structured, intervention="shuffled"),
    }
    if isinstance(model, AdaptiveCore):
        probe_train = paired_structured_episodes(seed_start=36_000, pairs=128, steps=steps)
        diagnosis.update(
            {
                "rule_probe": linear_rule_probe(model, probe_train, structured, seed=83),
                "donor_swap": donor_state_swap(model, structured),
                "component_ablation": state_component_ablation(model, structured),
                "temporal_credit": temporal_credit_audit(model, gap_steps=4),
            }
        )
    return diagnosis


def _passes_selection_gates(result: dict[str, Any], gates: dict[str, float]) -> bool:
    diagnosis = result["diagnostics"]
    structured = diagnosis["structured"]
    return all(
        (
            structured["post_evidence_accuracy"] >= gates["minimum_post_evidence_accuracy"],
            structured["post_evidence_accuracy"] - structured["pre_evidence_accuracy"]
            >= gates["minimum_adaptation_gain"],
            structured["post_evidence_accuracy"] - diagnosis["reset"]["post_evidence_accuracy"]
            >= gates["minimum_reset_drop"],
            diagnosis["rule_probe"]["held_out_accuracy"] >= gates["minimum_rule_probe_accuracy"],
            diagnosis["donor_swap"]["donor_rule_consistency"] >= gates["minimum_donor_consistency"],
        )
    )


def run_pilot(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Run development gates and reserved-pilot candidate selection, never final seeds."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    oracle = exhaustive_oracle_validation()
    overfit = run_overfit_suite()

    supervision: dict[str, Any] = {}
    for condition in document["supervision_conditions"]:
        config = V1TrainConfig(
            family="factorized",
            seed=document["pilot_model_seeds"][0],
            world_seed_start=document["training_world_seed_start"],
            validation_seed_start=document["validation_world_seed_start"],
            optimizer_steps=150,
            batch_pairs=8,
            episode_steps=document["episode_steps"],
            hidden_dim=32,
            state_dim=document["state_dim"],
            auxiliary_rule_start=condition["start"],
            auxiliary_rule_end=condition["end"],
            validation_pairs=32,
            validation_interval=50,
            torch_threads=document["torch_threads"],
        )
        run = train_candidate(config)
        evaluation = evaluate_model(
            run.model,
            paired_structured_episodes(
                seed_start=document["evaluation_world_seed_start"],
                pairs=64,
                steps=document["episode_steps"],
            ),
        )
        supervision[condition["name"]] = {"training": run.metrics, "evaluation": evaluation}

    candidates: dict[str, list[dict[str, Any]]] = {}
    retained_models: dict[tuple[str, int], tuple[nn.Module, V1TrainConfig]] = {}
    for candidate in document["candidates"]:
        family = candidate["family"]
        candidates[family] = []
        for seed in document["pilot_model_seeds"]:
            config = _candidate_config(document, candidate, seed)
            run = train_candidate(config)
            diagnostics = _diagnose_candidate(run.model, document)
            checkpoint_path = output / "checkpoints" / f"{family}-seed-{seed}.pt"
            metadata = save_v1_checkpoint(
                checkpoint_path,
                run.model,
                config,
                repo=repo,
                experiment_id=document["experiment_id"],
                training_step=run.metrics["best_step"],
                validation_score=run.metrics["best_validation_post_evidence_accuracy"],
            )
            result = {
                "seed": seed,
                "training": run.metrics,
                "diagnostics": diagnostics,
                "checkpoint": metadata,
                "passes_selection_gates": False,
            }
            result["passes_selection_gates"] = _passes_selection_gates(
                result, document["selection_gates"]
            )
            candidates[family].append(result)
            retained_models[(family, seed)] = (run.model, config)

    eligible = [
        family
        for family, seed_results in candidates.items()
        if all(result["passes_selection_gates"] for result in seed_results)
    ]
    if eligible:

        def selection_key(family: str) -> tuple[float, int, int, float]:
            values = candidates[family]
            mean_accuracy = sum(
                item["diagnostics"]["structured"]["post_evidence_accuracy"] for item in values
            ) / len(values)
            first = values[0]["training"]
            mean_wall = sum(item["training"]["training_wall_seconds"] for item in values) / len(
                values
            )
            return (
                -mean_accuracy,
                first["persistent_state_bytes"],
                first["parameter_count"],
                mean_wall,
            )

        selected = min(eligible, key=selection_key)
    else:
        selected = None
    result_document = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "phase": "pilot",
        "oracle": oracle,
        "overfit": overfit,
        "supervision_ladder": supervision,
        "candidates": candidates,
        "eligible_candidates": eligible,
        "selected_candidate": selected,
        "selection_rule": (
            "all seeds pass fixed gates; then post-evidence accuracy, persistent bytes, "
            "parameters, and wall time"
        ),
        "pilot_wall_seconds": time.perf_counter() - started,
        "final_seed_range_opened": False,
    }
    _write_json(output / "pilot_results.json", result_document)
    return result_document


def run_level5_pilot(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Test two independent context rules on reserved pilot seeds."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    structured = balanced_context_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
        steps=document["episode_steps"],
    )
    relabelled = balanced_context_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
        steps=document["episode_steps"],
        relabel=True,
    )
    random_episodes = random_context_episodes(
        seed_start=document["random_world_seed_start"],
        count=document["evaluation_groups"] * 4,
        steps=document["episode_steps"],
        relabel=True,
    )
    probe_train = balanced_context_episodes(
        seed_start=36_000, groups=128, steps=document["episode_steps"]
    )
    seed_results: list[dict[str, Any]] = []
    for seed in document["seeds"]:
        config = V1TrainConfig(
            family=document["family"],
            seed=seed,
            world_seed_start=document["training_world_seed_start"],
            validation_seed_start=document["validation_world_seed_start"],
            optimizer_steps=document["optimizer_steps"],
            batch_pairs=document["batch_groups"],
            episode_steps=document["episode_steps"],
            hidden_dim=document["hidden_dim"],
            state_dim=document["state_dim"],
            thought_cycles=document["thought_cycles"],
            feedback_mode=document["feedback_mode"],
            learning_rate=document["learning_rate"],
            validation_pairs=document["validation_groups"],
            validation_interval=50,
            torch_threads=document["torch_threads"],
            budget_bytes=document["state_bytes"],
            context_count=2,
        )
        run = train_candidate(config)
        full = evaluate_model(run.model, structured)
        reset = evaluate_model(run.model, structured, intervention="reset")
        random_evaluation = evaluate_model(run.model, random_episodes)
        diagnostics = {
            "structured": full,
            "surface_relabelled": evaluate_model(run.model, relabelled),
            "random_control": random_evaluation,
            "reset": reset,
            "frozen": evaluate_model(run.model, structured, intervention="frozen"),
            "random_state": evaluate_model(
                run.model, structured, intervention="random", random_seed=97
            ),
            "shuffled_state": evaluate_model(run.model, structured, intervention="shuffled"),
            "rule_probe": linear_rule_probe(run.model, probe_train, structured, seed=89),
            "donor_swap": context_donor_state_swap(run.model, structured),
            "component_ablation": state_component_ablation(run.model, structured),
            "temporal_credit": temporal_credit_audit(run.model, gap_steps=4),
        }
        gates = document["gates"]
        passes = all(
            (
                full["fully_informed_accuracy"] >= gates["minimum_fully_informed_accuracy"],
                full["fully_informed_accuracy"] - reset["fully_informed_accuracy"]
                >= gates["minimum_reset_drop"],
                diagnostics["rule_probe"]["held_out_accuracy"]
                >= gates["minimum_rule_probe_accuracy"],
                diagnostics["donor_swap"]["donor_rule_consistency"]
                >= gates["minimum_donor_consistency"],
                random_evaluation["fully_informed_accuracy"] - 0.5
                <= gates["maximum_random_advantage"],
            )
        )
        metadata = save_v1_checkpoint(
            output / "checkpoints" / f"factorized-context-seed-{seed}.pt",
            run.model,
            config,
            repo=repo,
            experiment_id=document["experiment_id"],
            training_step=run.metrics["best_step"],
            validation_score=run.metrics["best_validation_post_evidence_accuracy"],
        )
        seed_results.append(
            {
                "seed": seed,
                "passes": passes,
                "training": run.metrics,
                "diagnostics": diagnostics,
                "checkpoint": metadata,
            }
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "phase": "pilot",
        "level": 5,
        "environment": "two independent context-conditioned COPY/FLIP rules",
        "seed_results": seed_results,
        "all_seeds_pass": all(item["passes"] for item in seed_results),
        "pilot_wall_seconds": time.perf_counter() - started,
        "final_seed_range_opened": False,
    }
    _write_json(output / "level5_pilot_results.json", result)
    return result


def run_compression_pilot(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Sweep learned state width and post-update quantization on Level 5."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    evaluation = balanced_context_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
        steps=document["episode_steps"],
    )
    probe_train = balanced_context_episodes(
        seed_start=36_000, groups=128, steps=document["episode_steps"]
    )
    widths: dict[str, list[dict[str, Any]]] = {}
    for state_dim in document["state_dimensions"]:
        width_results: list[dict[str, Any]] = []
        for seed in document["seeds"]:
            config = V1TrainConfig(
                family="factorized",
                seed=seed,
                world_seed_start=document["training_world_seed_start"],
                validation_seed_start=document["validation_world_seed_start"],
                optimizer_steps=document["optimizer_steps"],
                batch_pairs=document["batch_groups"],
                episode_steps=document["episode_steps"],
                hidden_dim=document["hidden_dim"],
                state_dim=state_dim,
                learning_rate=document["learning_rate"],
                validation_pairs=document["validation_groups"],
                validation_interval=50,
                torch_threads=document["torch_threads"],
                budget_bytes=state_dim * 4,
                context_count=2,
            )
            run = train_candidate(config)
            full = evaluate_model(run.model, evaluation)
            probe = linear_rule_probe(run.model, probe_train, evaluation, seed=107)
            swap = context_donor_state_swap(run.model, evaluation)
            quantization = {
                str(bits): quantized_state_evaluation(run.model, evaluation, bits)
                for bits in document["quantization_bits"]
            }
            passes = all(
                (
                    full["fully_informed_accuracy"] >= document["minimum_fully_informed_accuracy"],
                    swap["donor_rule_consistency"] >= document["minimum_donor_consistency"],
                    probe["held_out_accuracy"] >= document["minimum_rule_probe_accuracy"],
                )
            )
            metadata = save_v1_checkpoint(
                output / "checkpoints" / f"state-{state_dim}-seed-{seed}.pt",
                run.model,
                config,
                repo=repo,
                experiment_id=document["experiment_id"],
                training_step=run.metrics["best_step"],
                validation_score=run.metrics["best_validation_post_evidence_accuracy"],
            )
            width_results.append(
                {
                    "seed": seed,
                    "passes": passes,
                    "training": run.metrics,
                    "evaluation": full,
                    "rule_probe": probe,
                    "donor_swap": swap,
                    "quantization": quantization,
                    "checkpoint": metadata,
                }
            )
        widths[str(state_dim)] = width_results
    passing_widths = [
        int(state_dim)
        for state_dim, results in widths.items()
        if all(item["passes"] for item in results)
    ]
    selected = min(passing_widths) if passing_widths else None
    result = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "phase": "pilot",
        "widths": widths,
        "passing_state_dimensions": passing_widths,
        "selected_state_dimension": selected,
        "selected_float32_bytes": selected * 4 if selected is not None else None,
        "selection_rule": (
            "all seeds pass accuracy, donor swap, and linear rule-probe gates; "
            "choose smallest width"
        ),
        "pilot_wall_seconds": time.perf_counter() - started,
        "final_seed_range_opened": False,
    }
    _write_json(output / "compression_pilot_results.json", result)
    return result


def run_level7_pilot(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Test composition of two learned rule bits on reserved pilot seeds."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    structured = balanced_composition_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
        steps=document["episode_steps"],
    )
    relabelled = balanced_composition_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
        steps=document["episode_steps"],
        relabel=True,
    )
    random_episodes = random_composition_episodes(
        seed_start=document["random_world_seed_start"],
        count=document["evaluation_groups"] * 4,
        steps=document["episode_steps"],
        relabel=True,
    )
    probe_train = balanced_composition_episodes(
        seed_start=36_000, groups=128, steps=document["episode_steps"]
    )
    seed_results: list[dict[str, Any]] = []
    for seed in document["seeds"]:
        config = V1TrainConfig(
            family=document["family"],
            seed=seed,
            world_seed_start=document["training_world_seed_start"],
            validation_seed_start=document["validation_world_seed_start"],
            optimizer_steps=document["optimizer_steps"],
            batch_pairs=document["batch_groups"],
            episode_steps=document["episode_steps"],
            hidden_dim=document["hidden_dim"],
            state_dim=document["state_dim"],
            learning_rate=document["learning_rate"],
            validation_pairs=document["validation_groups"],
            validation_interval=50,
            torch_threads=document["torch_threads"],
            budget_bytes=document["state_bytes"],
            context_count=3,
        )
        run = train_candidate(config)
        full = evaluate_model(run.model, structured)
        reset = evaluate_model(run.model, structured, intervention="reset")
        random_evaluation = evaluate_model(run.model, random_episodes)
        diagnostics = {
            "structured": full,
            "surface_relabelled": evaluate_model(run.model, relabelled),
            "random_control": random_evaluation,
            "reset": reset,
            "frozen": evaluate_model(run.model, structured, intervention="frozen"),
            "random_state": evaluate_model(
                run.model, structured, intervention="random", random_seed=101
            ),
            "shuffled_state": evaluate_model(run.model, structured, intervention="shuffled"),
            "rule_probe": linear_rule_probe(run.model, probe_train, structured, seed=109),
            "donor_swap": composition_donor_state_swap(run.model, structured),
            "component_ablation": state_component_ablation(run.model, structured),
            "temporal_credit": temporal_credit_audit(run.model, gap_steps=4),
            "quantization": {
                str(bits): quantized_state_evaluation(run.model, structured, bits)
                for bits in (1, 2, 4, 8, 16)
            },
        }
        gates = document["gates"]
        passes = all(
            (
                full["fully_informed_accuracy"] >= gates["minimum_fully_informed_accuracy"],
                full["composition_accuracy"] >= gates["minimum_composition_accuracy"],
                full["fully_informed_accuracy"] - reset["fully_informed_accuracy"]
                >= gates["minimum_reset_drop"],
                diagnostics["rule_probe"]["held_out_accuracy"]
                >= gates["minimum_rule_probe_accuracy"],
                diagnostics["donor_swap"]["donor_rule_consistency"]
                >= gates["minimum_donor_consistency"],
                random_evaluation["fully_informed_accuracy"] - 0.5
                <= gates["maximum_random_advantage"],
            )
        )
        metadata = save_v1_checkpoint(
            output / "checkpoints" / f"factorized-composition-seed-{seed}.pt",
            run.model,
            config,
            repo=repo,
            experiment_id=document["experiment_id"],
            training_step=run.metrics["best_step"],
            validation_score=run.metrics["best_validation_post_evidence_accuracy"],
        )
        seed_results.append(
            {
                "seed": seed,
                "passes": passes,
                "training": run.metrics,
                "diagnostics": diagnostics,
                "checkpoint": metadata,
            }
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "phase": "pilot",
        "level": 7,
        "environment": "two rule bits with held-out XOR composition queries",
        "seed_results": seed_results,
        "all_seeds_pass": all(item["passes"] for item in seed_results),
        "pilot_wall_seconds": time.perf_counter() - started,
        "final_seed_range_opened": False,
    }
    _write_json(output / "level7_pilot_results.json", result)
    return result


def run_level8_pilot(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Test two-bit retention across a fixed interval of marked non-events."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment_arguments = {
        "delay_steps": document["delay_steps"],
        "query_steps": document["query_steps"],
    }
    structured = balanced_delayed_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
        **environment_arguments,
    )
    relabelled = balanced_delayed_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
        relabel=True,
        **environment_arguments,
    )
    random_episodes = random_delayed_episodes(
        seed_start=document["random_world_seed_start"],
        count=document["evaluation_groups"] * 4,
        relabel=True,
        **environment_arguments,
    )
    probe_train = balanced_delayed_episodes(seed_start=36_000, groups=128, **environment_arguments)
    seed_results: list[dict[str, Any]] = []
    for seed in document["seeds"]:
        config = V1TrainConfig(
            family=document["family"],
            seed=seed,
            world_seed_start=document["training_world_seed_start"],
            validation_seed_start=document["validation_world_seed_start"],
            optimizer_steps=document["optimizer_steps"],
            batch_pairs=document["batch_groups"],
            episode_steps=document["episode_steps"],
            hidden_dim=document["hidden_dim"],
            state_dim=document["state_dim"],
            learning_rate=document["learning_rate"],
            validation_pairs=document["validation_groups"],
            validation_interval=50,
            torch_threads=document["torch_threads"],
            budget_bytes=document["state_bytes"],
            context_count=4,
            delay_steps=document["delay_steps"],
            query_steps=document["query_steps"],
        )
        run = train_candidate(config)
        full = evaluate_model(run.model, structured)
        reset = evaluate_model(run.model, structured, intervention="reset")
        random_evaluation = evaluate_model(run.model, random_episodes)
        diagnostics = {
            "structured": full,
            "surface_relabelled": evaluate_model(run.model, relabelled),
            "random_control": random_evaluation,
            "reset": reset,
            "frozen": evaluate_model(run.model, structured, intervention="frozen"),
            "random_state": evaluate_model(
                run.model, structured, intervention="random", random_seed=113
            ),
            "shuffled_state": evaluate_model(run.model, structured, intervention="shuffled"),
            "rule_probe": linear_rule_probe(run.model, probe_train, structured, seed=127),
            "donor_swap": delayed_donor_state_swap(run.model, structured),
            "component_ablation": state_component_ablation(run.model, structured),
            "temporal_credit": temporal_credit_audit(run.model, gap_steps=8),
            "quantization": {
                str(bits): quantized_state_evaluation(run.model, structured, bits)
                for bits in (1, 2, 4, 8, 16)
            },
        }
        gates = document["gates"]
        passes = all(
            (
                full["fully_informed_accuracy"] >= gates["minimum_query_accuracy"],
                full["composition_accuracy"] >= gates["minimum_composition_accuracy"],
                full["fully_informed_accuracy"] - reset["fully_informed_accuracy"]
                >= gates["minimum_reset_drop"],
                diagnostics["rule_probe"]["held_out_accuracy"]
                >= gates["minimum_rule_probe_accuracy"],
                diagnostics["donor_swap"]["donor_rule_consistency"]
                >= gates["minimum_donor_consistency"],
                random_evaluation["fully_informed_accuracy"] - 0.5
                <= gates["maximum_random_advantage"],
                full["distractor_state_drift_max"] <= gates["maximum_distractor_state_drift"],
            )
        )
        metadata = save_v1_checkpoint(
            output / "checkpoints" / f"factorized-delay-seed-{seed}.pt",
            run.model,
            config,
            repo=repo,
            experiment_id=document["experiment_id"],
            training_step=run.metrics["best_step"],
            validation_score=run.metrics["best_validation_post_evidence_accuracy"],
        )
        seed_results.append(
            {
                "seed": seed,
                "passes": passes,
                "training": run.metrics,
                "diagnostics": diagnostics,
                "checkpoint": metadata,
            }
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "phase": "pilot",
        "level": 8,
        "environment": "two rule bits retained across eight marked non-events",
        "seed_results": seed_results,
        "all_seeds_pass": all(item["passes"] for item in seed_results),
        "pilot_wall_seconds": time.perf_counter() - started,
        "final_seed_range_opened": False,
    }
    _write_json(output / "level8_pilot_results.json", result)
    return result


def run_level9_pilot(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Test unmarked replacement of one rule while retaining the other."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    structured = balanced_reversal_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
    )
    relabelled = balanced_reversal_episodes(
        seed_start=document["evaluation_world_seed_start"],
        groups=document["evaluation_groups"],
        relabel=True,
    )
    probe_train = balanced_reversal_episodes(seed_start=36_000, groups=128)
    seed_results: list[dict[str, Any]] = []
    for seed in document["seeds"]:
        config = V1TrainConfig(
            family=document["family"],
            seed=seed,
            world_seed_start=document["training_world_seed_start"],
            validation_seed_start=document["validation_world_seed_start"],
            optimizer_steps=document["optimizer_steps"],
            batch_pairs=document["batch_groups"],
            episode_steps=document["episode_steps"],
            hidden_dim=document["hidden_dim"],
            state_dim=document["state_dim"],
            learning_rate=document["learning_rate"],
            validation_pairs=document["validation_groups"],
            validation_interval=50,
            torch_threads=document["torch_threads"],
            budget_bytes=document["state_bytes"],
            context_count=document["context_count"],
            environment_variant="reversal",
        )
        evaluation_start = document["evaluation_world_seed_start"]
        assert_generation_splits_disjoint(
            config,
            evaluation_ranges={
                "pilot_evaluation": (
                    evaluation_start,
                    evaluation_start + document["evaluation_groups"] - 1,
                )
            },
        )
        run = train_candidate(config)
        reversal = evaluate_rule_change(run.model, structured)
        diagnostics = {
            "reversal": reversal,
            "surface_relabelled_reversal": evaluate_rule_change(run.model, relabelled),
            "aggregate": evaluate_model(run.model, structured),
            "reset": evaluate_model(run.model, structured, intervention="reset"),
            "frozen": evaluate_model(run.model, structured, intervention="frozen"),
            "rule_probe": linear_reversal_probe(run.model, probe_train, structured, seed=131),
            "temporal_credit": temporal_credit_audit(run.model, gap_steps=4),
        }
        gates = document["gates"]
        passes = all(
            (
                reversal["change_step_accuracy"] <= gates["maximum_change_step_accuracy"],
                reversal["post_feedback_recovery_accuracy"] >= gates["minimum_recovery_accuracy"],
                reversal["recovery_speed_steps"] <= gates["maximum_recovery_steps"],
                reversal["unrelated_rule_retention_accuracy"]
                >= gates["minimum_retention_accuracy"],
                diagnostics["rule_probe"]["held_out_accuracy"]
                >= gates["minimum_rule_probe_accuracy"],
            )
        )
        metadata = save_v1_checkpoint(
            output / "checkpoints" / f"factorized-reversal-seed-{seed}.pt",
            run.model,
            config,
            repo=repo,
            experiment_id=document["experiment_id"],
            training_step=run.metrics["best_step"],
            validation_score=run.metrics["best_validation_post_evidence_accuracy"],
        )
        seed_results.append(
            {
                "seed": seed,
                "passes": passes,
                "training": run.metrics,
                "diagnostics": diagnostics,
                "checkpoint": metadata,
            }
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "phase": "pilot",
        "level": 9,
        "environment": "one unmarked rule reversal with one unrelated retained rule",
        "seed_results": seed_results,
        "all_seeds_pass": all(item["passes"] for item in seed_results),
        "pilot_wall_seconds": time.perf_counter() - started,
        "final_seed_range_opened": False,
    }
    _write_json(output / "level9_pilot_results.json", result)
    return result
