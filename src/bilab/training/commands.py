"""Human-facing Cognitive Core experiment commands."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from bilab.environments.rule_worlds import (
    OBSERVATION_FIELDS,
    Episode,
    make_evaluation_episodes,
    make_training_episodes,
    make_validation_episodes,
)
from bilab.models.cognitive_core import CognitiveCore
from bilab.models.factory import build_model, count_parameters
from bilab.training.checkpoints import load_checkpoint
from bilab.training.evaluation import evaluate_model
from bilab.training.experiment import load_experiment_config, run_experiment, smoke_config
from bilab.training.reporting import write_json, write_report


def validate_worlds(config: dict[str, Any]) -> dict[str, Any]:
    train = make_training_episodes(config)
    validation = make_validation_episodes(config)
    evaluation = make_evaluation_episodes(config)
    train_seeds = {episode.generation_seed for episode in train}
    validation_seeds = {episode.generation_seed for episode in validation}
    test_seeds = {
        episode.generation_seed for episodes in evaluation.values() for episode in episodes
    }
    if train_seeds & validation_seeds or train_seeds & test_seeds or validation_seeds & test_seeds:
        raise ValueError("Rule Worlds generation-seed splits overlap")
    if set(Episode.__dataclass_fields__) & {"hidden_rules", "rule_values", "future"}:
        raise ValueError("Episode exposes privileged simulator state")
    models = {
        variant: build_model(variant, config)
        for variant in ("no_memory", "episodic", "cognitive_core")
    }
    parameter_counts = {variant: count_parameters(model) for variant, model in models.items()}
    public_bounds = {
        "observation_fields": list(OBSERVATION_FIELDS),
        "train_worlds": len(train),
        "validation_worlds": len(validation),
        "evaluation_worlds": {key: len(value) for key, value in evaluation.items()},
        "split_seed_intersections": 0,
        "hidden_rules_exposed": False,
        "future_observations_exposed": False,
        "parameter_counts": parameter_counts,
        "persistent_state_bytes": {
            "no_memory": 0,
            "episodic": models["episodic"].state_nbytes(models["episodic"].initial_state(1)),
            "cognitive_core": models["cognitive_core"].state_nbytes(
                models["cognitive_core"].initial_state(1)
            ),
        },
    }
    return public_bounds


def run_smoke(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    config = smoke_config(load_experiment_config(config_path))
    generated_config = output / "smoke-config.json"
    write_json(generated_config, config)
    return run_experiment(repo, generated_config, output)


def evaluate_checkpoint_file(checkpoint: Path, output: Path) -> dict[str, Any]:
    model, config, metadata = load_checkpoint(checkpoint)
    evaluation = evaluate_model(
        model,
        make_evaluation_episodes(config),
        variant=str(metadata["variant"]),
        seed=int(metadata["seed"]),
        mode="full",
        batch_size=int(config["training"]["batch_size"]),
        adaptation_checkpoints=list(config["evaluation"]["adaptation_checkpoints"]),
        recovery_windows=list(config["evaluation"]["recovery_windows"]),
    )
    document = {
        "checkpoint": str(checkpoint),
        "metadata": metadata,
        "evaluation": asdict(evaluation),
    }
    write_json(output, document)
    return document


def ablate_checkpoint_file(checkpoint: Path, output: Path) -> dict[str, Any]:
    model, config, metadata = load_checkpoint(checkpoint)
    if not isinstance(model, CognitiveCore):
        raise ValueError("ablation command requires a cognitive_core checkpoint")
    evaluations = {}
    episodes = make_evaluation_episodes(config)
    for mode in config["evaluation"]["core_ablations"]:
        evaluations[mode] = asdict(
            evaluate_model(
                model,
                episodes,
                variant="cognitive_core",
                seed=int(metadata["seed"]),
                mode=mode,
                batch_size=int(config["training"]["batch_size"]),
                adaptation_checkpoints=list(config["evaluation"]["adaptation_checkpoints"]),
                recovery_windows=list(config["evaluation"]["recovery_windows"]),
            )
        )
    document = {"checkpoint": str(checkpoint), "evaluations": evaluations}
    write_json(output, document)
    return document


def regenerate_report(results_path: Path, output: Path) -> None:
    import json

    results = json.loads(results_path.read_text(encoding="utf-8"))
    write_report(output, results, title=results["experiment_id"])
