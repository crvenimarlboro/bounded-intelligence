"""Development, failure-audit, and reserved-pilot execution for V3."""

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
    exhaustive_oracle_validation,
    make_context_episode,
)
from bilab.v2.checkpoints import load_v2_checkpoint
from bilab.v3.checkpoints import load_v3_checkpoint, save_v3_checkpoint
from bilab.v3.environments import balance_evidence_inputs
from bilab.v3.models import V3ModelConfig, build_v3_model
from bilab.v3.training import (
    V3TrainConfig,
    compose_v3c_staged_state_dict,
    diagnose_v3_candidate,
    preservation_stress,
    seed_v3,
    train_v3_candidate,
    v3_episode_objective,
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
    predictive_state_training: bool = False,
) -> dict[str, Any]:
    seed_v3(seed, threads=1)
    model = build_v3_model(V3ModelConfig(family=family, hidden_dim=24))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history: list[dict[str, float]] = []
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = v3_episode_objective(
            model,
            batch,
            predictive_state_training=predictive_state_training,
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
        _, final = v3_episode_objective(
            model,
            batch,
            predictive_state_training=predictive_state_training,
        )
    return {
        "family": family,
        "steps": steps,
        "episodes": len(batch.public),
        "history": history,
        "accuracy": final["accuracy"],
        "fully_informed_accuracy": final["fully_informed_accuracy"],
    }


def run_v3_overfit(output_path: Path) -> dict[str, Any]:
    """Run fixed-data learnability for both serious raw-relation mechanisms."""

    single = ContextBatch.from_episodes(
        [make_context_episode(seed=500_001, steps=3, rules=(0, 1), surface_flip=0)]
    )
    one_world = ContextBatch.from_episodes(
        [make_context_episode(seed=500_002, steps=12, rules=(1, 0), surface_flip=0)]
    )
    multiple = batch_adaptation_episodes(
        balance_evidence_inputs(balanced_context_episodes(seed_start=500_100, groups=1, steps=12))
    )
    results: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": "cognitive-core-v3-learnability",
        "raw_bilinear": {
            "one_sequence": _fit_fixed_batch(
                "raw_bilinear_overwrite",
                single,
                seed=3101,
                steps=300,
                predictive_state_training=True,
            ),
            "one_world": _fit_fixed_batch(
                "raw_bilinear_overwrite",
                one_world,
                seed=3102,
                steps=300,
                predictive_state_training=True,
            ),
            "multiple_worlds": _fit_fixed_batch(
                "raw_bilinear_overwrite",
                multiple,
                seed=3103,
                steps=500,
                predictive_state_training=True,
            ),
        },
        "raw_discrete": {
            "one_sequence": _fit_fixed_batch(
                "raw_discrete_overwrite", single, seed=3111, steps=300
            ),
            "one_world": _fit_fixed_batch(
                "raw_discrete_overwrite", one_world, seed=3112, steps=300
            ),
            "multiple_worlds": _fit_fixed_batch(
                "raw_discrete_overwrite", multiple, seed=3113, steps=500
            ),
        },
    }
    results["all_overfit_pass"] = all(
        condition["fully_informed_accuracy"] >= 0.99
        for family in ("raw_bilinear", "raw_discrete")
        for condition in results[family].values()
    )
    _write_json(output_path, results)
    return results


def validate_v3_protocol(contract_path: Path) -> dict[str, Any]:
    """Validate the raw-field contract and inherited exhaustive binary oracle."""

    document = json.loads(contract_path.read_text(encoding="utf-8"))
    prohibited = set(document["prohibited"])
    required_prohibitions = {
        "input_xor_outcome",
        "input_equals_outcome",
        "signed_rule_relation",
        "hidden_rule",
        "correct_state_slot",
        "semantic_write_or_no_write_target",
        "future_observation",
        "generation_seed",
        "complete_history",
        "probe_output",
        "research_telemetry",
    }
    if not required_prohibitions <= prohibited:
        missing = sorted(required_prohibitions - prohibited)
        raise ValueError(f"V3 raw contract lacks prohibitions: {missing}")
    oracle = exhaustive_oracle_validation()
    if oracle["post_evidence_accuracy"] < 0.99:
        raise ValueError("inherited binary oracle failed")
    if oracle["pre_evidence_accuracy"] > 0.55:
        raise ValueError("inherited no-memory ambiguity limit failed")
    return {
        "schema_version": "1.0",
        "contract_valid": True,
        "contract": str(contract_path),
        "oracle": oracle,
    }


def _resolve_repo_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _read_pilot_result(repo: Path, value: str) -> tuple[Path, dict[str, Any]]:
    path = _resolve_repo_path(repo, value)
    if not path.is_file():
        raise FileNotFoundError(f"required V3 pilot result does not exist: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _rows_for_candidate(document: dict[str, Any], candidate: str) -> list[dict[str, Any]]:
    return [row for row in document.get("rows", []) if row.get("candidate") == candidate]


def _validate_v3c_prerequisites(repo: Path, document: dict[str, Any]) -> list[dict[str, Any]]:
    """Require passing isolated V3A and V3B evidence before a joint pilot can run."""

    prerequisites = document.get("prerequisites", [])
    if len(prerequisites) < 2:
        raise ValueError("V3C configuration must declare passing V3A and V3B prerequisites")
    validated: list[dict[str, Any]] = []
    stages: set[str] = set()
    for specification in prerequisites:
        path, result = _read_pilot_result(repo, specification["results"])
        candidate = specification["candidate"]
        rows = _rows_for_candidate(result, candidate)
        if not rows or not all(row.get("passes_stage_gates") is True for row in rows):
            raise ValueError(f"V3C prerequisite did not pass all recorded seeds: {candidate}")
        stage = rows[0]["stage"]
        if any(row["stage"] != stage for row in rows):
            raise ValueError(f"V3C prerequisite mixes stages: {candidate}")
        stages.add(stage)
        validated.append(
            {
                "results": str(path),
                "candidate": candidate,
                "stage": stage,
                "seeds": [row["seed"] for row in rows],
            }
        )
    if not {"v3a", "v3b"} <= stages:
        raise ValueError("V3C prerequisites must include one passing V3A and one passing V3B")
    return validated


def _load_staged_source(
    repo: Path,
    specification: dict[str, Any],
    target_seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    result_path, result = _read_pilot_result(repo, specification["results"])
    seed_map = specification.get("source_seed_map", {})
    source_seed = int(seed_map.get(str(target_seed), target_seed))
    candidate = specification["candidate"]
    rows = [
        row for row in _rows_for_candidate(result, candidate) if int(row["seed"]) == source_seed
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one staged source row for {candidate} seed {source_seed}, found {len(rows)}"
        )
    row = rows[0]
    if row.get("passes_stage_gates") is not True:
        raise ValueError(
            f"staged source did not pass its stage gates: {candidate} seed {source_seed}"
        )
    checkpoint_path = _resolve_repo_path(repo, row["checkpoint"]["path"])
    model, config, metadata = load_v3_checkpoint(checkpoint_path)
    expected_family = specification.get("expected_family")
    if expected_family is not None and metadata["family"] != expected_family:
        raise ValueError(
            f"staged source family mismatch: expected {expected_family}, got {metadata['family']}"
        )
    return model, {
        "results": str(result_path),
        "candidate": candidate,
        "source_seed": source_seed,
        "checkpoint": str(checkpoint_path),
        "family": metadata["family"],
        "stage": config.stage,
        "model_digest": metadata["model_digest"],
    }


def _prepare_v3c_staged_initialization(
    repo: Path,
    candidate: dict[str, Any],
    config: V3TrainConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    sources = candidate.get("staged_sources")
    if not isinstance(sources, dict):
        raise ValueError("staged V3C candidate must declare staged_sources")
    raw_model, raw_metadata = _load_staged_source(repo, sources["raw_relation"], config.seed)
    preservation_model, preservation_metadata = _load_staged_source(
        repo, sources["preservation"], config.seed
    )
    target = build_v3_model(config.model_config())
    state_dict, composition = compose_v3c_staged_state_dict(target, raw_model, preservation_model)
    return state_dict, {
        "raw_relation_source": raw_metadata,
        "preservation_source": preservation_metadata,
        "composition": composition,
    }


def _diagnose_staged_initialization(
    config: V3TrainConfig,
    initial_state_dict: dict[str, torch.Tensor],
    *,
    seed_base: int,
    groups: int,
    long_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Evaluate the composed system before joint optimization on identical audit worlds."""

    seed_v3(config.seed, config.torch_threads)
    model = build_v3_model(config.model_config())
    own = model.state_dict()
    compatible = {
        name: tensor
        for name, tensor in initial_state_dict.items()
        if name in own and own[name].shape == tensor.shape
    }
    model.load_state_dict(compatible, strict=False)
    diagnostics = diagnose_v3_candidate(
        model,
        seed=config.seed,
        seed_base=seed_base,
        groups=groups,
        long_lengths=long_lengths,
    )
    row = {"stage": "v3c", "diagnostics": diagnostics}
    return {
        "diagnostics": diagnostics,
        "passes_behavioral_stage_gates": _passes_v3c_behavioral_gates(row),
        "passes_exact_preservation": _passes_exact_preservation(diagnostics),
        "passes_stage_gates": _passes_stage(row),
    }


def _train_config(
    document: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
) -> V3TrainConfig:
    training = document["training"]
    return V3TrainConfig(
        family=candidate["family"],
        stage=candidate["stage"],
        seed=seed,
        world_seed_start=training["world_seed_start"],
        validation_seed_start=training["validation_seed_start"],
        optimizer_steps=training["optimizer_steps"],
        batch_groups=training["batch_groups"],
        episode_steps=training["episode_steps"],
        hidden_dim=candidate["hidden_dim"],
        learning_rate=candidate.get("learning_rate", training["learning_rate"]),
        gradient_clip=training["gradient_clip"],
        write_cost=candidate.get("write_cost", 0.0),
        predictive_state_training=candidate.get("predictive_state_training", False),
        validation_groups=training["validation_groups"],
        validation_interval=training["validation_interval"],
        torch_threads=training["torch_threads"],
        initialization=candidate.get("initialization", "random"),
    )


def _metric(row: dict[str, Any], name: str) -> float:
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
        "writer_disabled": (
            "routing_interventions",
            "writer_disabled",
            "fully_informed_accuracy",
        ),
        "relation_zero": (
            "relation_interventions",
            "zero",
            "fully_informed_accuracy",
        ),
        "long_10": ("preservation", "measurements", "10", "retained_rule_accuracy"),
        "long_100": ("preservation", "measurements", "100", "retained_rule_accuracy"),
        "long_1000": ("preservation", "measurements", "1000", "retained_rule_accuracy"),
        "long_10000": ("preservation", "measurements", "10000", "retained_rule_accuracy"),
        "long_100000": (
            "preservation",
            "measurements",
            "100000",
            "retained_rule_accuracy",
        ),
        "long_recovery": ("preservation", "post_long_span_one_feedback_recovery"),
        "long_retention": ("preservation", "post_long_span_unrelated_rule_retention"),
    }
    value: Any = row["diagnostics"]
    for key in paths[name]:
        value = value[key]
    return float(value)


def _passes_common_behavioral_gates(row: dict[str, Any]) -> bool:
    delay = _metric(row, "delay")
    return all(
        (
            delay >= 0.95,
            _metric(row, "composition") >= 0.95,
            _metric(row, "recovery") >= 0.95,
            _metric(row, "retention") >= 0.95,
            _metric(row, "surface") >= 0.95,
            _metric(row, "donor") >= 0.95,
            _metric(row, "relation_probe") >= 0.95,
            abs(_metric(row, "random") - 0.5) <= 0.05,
            delay - _metric(row, "reset") >= 0.30,
            delay - _metric(row, "frozen") >= 0.30,
            delay - _metric(row, "writer_disabled") >= 0.30,
        )
    )


def _passes_v3c_behavioral_gates(row: dict[str, Any]) -> bool:
    delay = _metric(row, "delay")
    return _passes_common_behavioral_gates(row) and all(
        (
            delay - _metric(row, "relation_zero") >= 0.20,
            _metric(row, "long_100000") >= 0.90,
            _metric(row, "long_recovery") >= 0.90,
            _metric(row, "long_retention") >= 0.90,
        )
    )


def _passes_exact_preservation(diagnostics: dict[str, Any]) -> bool:
    preservation = diagnostics["preservation"]
    measurements = preservation["measurements"].values()
    return all(
        (
            preservation["state_size_constant"],
            preservation["total_changed_state_events"] == 0,
            preservation["total_code_transitions"] == 0,
            preservation["total_nonzero_distractor_write_events"] == 0,
            all(
                measurement["drift_norm"] == 0.0
                and measurement["predictions_bit_identical_to_reference"] is True
                and measurement["cumulative_changed_state_events"] == 0
                and measurement["cumulative_nonzero_write_events"] == 0
                for measurement in measurements
            ),
        )
    )


def _passes_stage(row: dict[str, Any]) -> bool:
    delay = _metric(row, "delay")
    common = _passes_common_behavioral_gates(row)
    if row["stage"] == "v3a":
        return common and delay - _metric(row, "relation_zero") >= 0.20
    if row["stage"] == "v3b":
        preservation = row["diagnostics"]["preservation"]
        return common and all(
            (
                _metric(row, "long_10") >= 0.99,
                _metric(row, "long_100") >= 0.99,
                _metric(row, "long_1000") >= 0.98,
                _metric(row, "long_10000") >= 0.97,
                _metric(row, "long_100000") >= 0.95,
                _metric(row, "long_recovery") >= 0.95,
                _metric(row, "long_retention") >= 0.95,
                preservation["state_size_constant"],
            )
        )
    return _passes_v3c_behavioral_gates(row) and _passes_exact_preservation(row["diagnostics"])


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for candidate in sorted({row["candidate"] for row in rows}):
        selected = [row for row in rows if row["candidate"] == candidate]
        metric_names = (
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
            "writer_disabled",
            "relation_zero",
            "long_10",
            "long_100",
            "long_1000",
            "long_10000",
            "long_100000",
            "long_recovery",
            "long_retention",
        )
        metrics = {name: [_metric(row, name) for row in selected] for name in metric_names}
        summary[candidate] = {
            "stage": selected[0]["stage"],
            "family": selected[0]["family"],
            "parameter_count": selected[0]["training"]["parameter_count"],
            "persistent_state_bytes": selected[0]["training"]["persistent_state_bytes"],
            "initialization": selected[0]["training"]["initialization"],
            "all_seeds_pass": all(row["passes_stage_gates"] for row in selected),
            "all_seeds_pass_behavioral": all(
                row.get("passes_behavioral_stage_gates", row["passes_stage_gates"])
                for row in selected
            ),
            "all_seeds_pass_exact_preservation": all(
                row.get("passes_exact_preservation", True) for row in selected
            ),
            "metrics": {
                name: {
                    "mean": statistics.mean(values),
                    "standard_deviation": statistics.pstdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in metrics.items()
            },
            "training_wall_seconds": sum(
                row["training"]["training_wall_seconds"] for row in selected
            ),
        }
    return summary


def run_v3_pilot(
    repo: Path,
    config_path: Path,
    output_directory: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Train every declared candidate for one isolated or joint V3 stage."""

    if stage not in {"v3a", "v3b", "v3c"}:
        raise ValueError("pilot stage must be v3a, v3b, or v3c")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    validated_prerequisites = _validate_v3c_prerequisites(repo, document) if stage == "v3c" else []
    candidates = {
        name: value for name, value in document["candidates"].items() if value["stage"] == stage
    }
    if len(candidates) < 2:
        raise ValueError(f"V3 {stage} pilot requires at least two serious candidates")
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    partial_path = output_directory / "pilot_results.json"
    lengths = tuple(document["long_sequence_lengths"])
    for candidate_name, candidate in candidates.items():
        for seed in document["seeds"]:
            config = _train_config(document, candidate, seed)
            initial_state_dict: dict[str, torch.Tensor] | None = None
            initialization_metadata: dict[str, Any] | None = None
            initial_evaluation: dict[str, Any] | None = None
            if config.initialization == "staged":
                if stage != "v3c":
                    raise ValueError("staged initialization is reserved for V3C")
                initial_state_dict, initialization_metadata = _prepare_v3c_staged_initialization(
                    repo, candidate, config
                )
                if candidate.get("diagnose_initialization", False):
                    initial_evaluation = _diagnose_staged_initialization(
                        config,
                        initial_state_dict,
                        seed_base=document["evaluation_seed_start"],
                        groups=document["evaluation_groups"],
                        long_lengths=lengths,
                    )
            run = train_v3_candidate(
                config,
                initial_state_dict=initial_state_dict,
                initialization_metadata=initialization_metadata,
            )
            diagnostics = diagnose_v3_candidate(
                run.model,
                seed=seed,
                seed_base=document["evaluation_seed_start"],
                groups=document["evaluation_groups"],
                long_lengths=lengths,
            )
            metadata = save_v3_checkpoint(
                output_directory / "checkpoints" / candidate_name / f"seed-{seed}.pt",
                run.model,
                config,
                repo=repo,
                experiment_id=document["experiment_id"],
                training_step=config.optimizer_steps,
                validation_score=run.metrics["final_validation_accuracy"],
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
            if initial_evaluation is not None:
                row["initial_evaluation"] = initial_evaluation
            if stage == "v3c":
                row["passes_behavioral_stage_gates"] = _passes_v3c_behavioral_gates(row)
                row["passes_exact_preservation"] = _passes_exact_preservation(diagnostics)
            row["passes_stage_gates"] = _passes_stage(row)
            rows.append(row)
            _write_json(
                partial_path,
                {
                    "schema_version": "1.0",
                    "experiment_id": document["experiment_id"],
                    "status": "in_progress",
                    "stage": stage,
                    "rows": rows,
                },
            )
    result = {
        "schema_version": "1.0",
        "experiment_id": document["experiment_id"],
        "status": "completed",
        "stage": stage,
        "config": str(config_path),
        "seeds": document["seeds"],
        "candidate_selection_rule": document["candidate_selection_rule"],
        "validated_prerequisites": validated_prerequisites,
        "rows": rows,
        "summary": _summary(rows),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(partial_path, result)
    return result


def run_v2_failure_audit(
    reproduction_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Localize the committed V2 raw-writer failures from reproduced pilot checkpoints."""

    sources = {
        "equal_budget": reproduction_root / "pilot-v1.0/pilot_results.json",
        "bilinear": reproduction_root / "pilot-bilinear-v1.0/pilot_results.json",
        "double_budget": reproduction_root / "pilot-learning-curve-v1.0/pilot_results.json",
    }
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": "cognitive-core-v3-v2-failure-localization",
        "sources": {name: str(path) for name, path in sources.items()},
        "candidates": {},
    }
    for source_name, path in sources.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document["rows"]:
            checkpoint = Path(row["checkpoint"]["path"])
            model, _, metadata = load_v2_checkpoint(checkpoint)
            pair_candidates: dict[str, float] = {}
            pair_gates: dict[str, list[float]] = {}
            model.eval()
            with torch.no_grad():
                for input_value in range(2):
                    for outcome_value in range(2):
                        public = torch.tensor([[float(input_value), 0.0, 0.0]])
                        outcome = torch.tensor([outcome_value])
                        state = model.initial_state(1)
                        update = model.update(
                            public,
                            state,
                            outcome,
                            model.predict(public, state),
                        )
                        key = f"{input_value}{outcome_value}"
                        pair_candidates[key] = float(update.candidate[0, 0])
                        pair_gates[key] = update.gate[0].tolist()
            key = f"{source_name}:{row['candidate']}:seed-{row['seed']}"
            result["candidates"][key] = {
                "family": row["family"],
                "parameter_count": metadata["parameter_count"],
                "relation_probe": row["diagnostics"]["relation_probe"]["held_out_accuracy"],
                "delay": row["diagnostics"]["delay"]["fully_informed_accuracy"],
                "recovery": row["diagnostics"]["reversal"]["post_feedback_recovery_accuracy"],
                "retention": row["diagnostics"]["reversal"]["unrelated_rule_retention_accuracy"],
                "route_separation": row["diagnostics"]["routing"]["primitive_route_separation"],
                "raw_pair_candidate": pair_candidates,
                "raw_pair_gate": pair_gates,
                "failure_localization": (
                    "relation present but overwrite/deployment unstable"
                    if row["diagnostics"]["relation_probe"]["held_out_accuracy"] >= 0.95
                    and row["diagnostics"]["reversal"]["post_feedback_recovery_accuracy"] < 0.95
                    else "routing/retention instability"
                ),
            }
    result["conclusion"] = (
        "All reproduced raw candidates encoded the relation, but at least one required behavioral "
        "gate failed; doubled observations did not remove the failure."
    )
    _write_json(output_path, result)
    return result


def evaluate_v3_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    seed_base: int,
    groups: int = 64,
) -> dict[str, Any]:
    """Independently regenerate all deterministic V3 checkpoint diagnostics."""

    model, config, metadata = load_v3_checkpoint(checkpoint)
    diagnostics = diagnose_v3_candidate(
        model,
        seed=config.seed,
        seed_base=seed_base,
        groups=groups,
    )
    result = {
        "schema_version": "1.0",
        "checkpoint": str(checkpoint),
        "metadata": metadata,
        "model_digest": metadata["model_digest"],
        "diagnostics": diagnostics,
    }
    _write_json(output, result)
    return result


def diagnose_v3_checkpoint_section(
    checkpoint: Path,
    output: Path,
    *,
    seed_base: int,
    section: str,
    groups: int = 64,
) -> dict[str, Any]:
    """Regenerate one researcher-facing diagnostic section from a frozen checkpoint."""

    if section == "stress":
        model, config, metadata = load_v3_checkpoint(checkpoint)
        result = {
            "schema_version": "1.0",
            "checkpoint": str(checkpoint),
            "metadata": metadata,
            "section": section,
            "diagnostics": preservation_stress(model),
            "seed": config.seed,
        }
        _write_json(output, result)
        return result
    complete = evaluate_v3_checkpoint(
        checkpoint,
        output,
        seed_base=seed_base,
        groups=groups,
    )
    selections = {
        "probe": (
            "relation_probe",
            "rule_probe",
            "surface_probe",
            "relation_combinations",
            "temporal_gradient",
        ),
        "intervene": (
            "relation_interventions",
            "routing_interventions",
            "state_interventions",
            "component_ablation",
            "composition_donor_swap",
            "slot_permutation",
        ),
    }
    if section not in selections:
        raise ValueError(f"unknown V3 checkpoint diagnostic section: {section}")
    result = {
        "schema_version": "1.0",
        "checkpoint": str(checkpoint),
        "metadata": complete["metadata"],
        "section": section,
        "diagnostics": {name: complete["diagnostics"][name] for name in selections[section]},
    }
    _write_json(output, result)
    return result
