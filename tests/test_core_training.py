from pathlib import Path

import torch

from bilab.environments.rule_worlds import make_evaluation_episodes
from bilab.training.checkpoints import load_checkpoint, save_checkpoint
from bilab.training.evaluation import (
    comparison_differences,
    evaluate_model,
    mean_and_sample_std,
)
from bilab.training.trainer import train_model


def test_tiny_training_is_deterministic_and_checkpoint_reproduces(
    tiny_core_config: dict, tmp_path: Path
) -> None:
    first = train_model(tiny_core_config, "cognitive_core", 3)
    second = train_model(tiny_core_config, "cognitive_core", 3)
    assert first.curves == second.curves
    for name, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[name])

    checkpoint = tmp_path / "core.pt"
    save_checkpoint(
        checkpoint,
        first.model,
        tiny_core_config,
        repo=tmp_path,
        variant="cognitive_core",
        seed=3,
        persistent_bytes=first.model.state_nbytes(first.model.initial_state(1)),
        training_step=first.final_step,
        validation_score=first.validation_accuracy,
    )
    reloaded, config, _ = load_checkpoint(checkpoint)
    episodes = make_evaluation_episodes(config)
    arguments = {
        "variant": "cognitive_core",
        "seed": 3,
        "mode": "full",
        "batch_size": 2,
        "adaptation_checkpoints": config["evaluation"]["adaptation_checkpoints"],
        "recovery_windows": config["evaluation"]["recovery_windows"],
    }
    original = evaluate_model(first.model, episodes, **arguments)
    reproduced = evaluate_model(reloaded, episodes, **arguments)
    assert original.metrics == reproduced.metrics
    assert all(
        torch.equal(first.model.state_dict()[name], reloaded.state_dict()[name])
        for name in first.model.state_dict()
    )


def test_evaluation_freezes_weights(tiny_core_config: dict) -> None:
    training = train_model(tiny_core_config, "cognitive_core", 3)
    before = {name: value.clone() for name, value in training.model.state_dict().items()}
    evaluate_model(
        training.model,
        make_evaluation_episodes(tiny_core_config),
        variant="cognitive_core",
        seed=3,
        mode="full",
        batch_size=2,
        adaptation_checkpoints=tiny_core_config["evaluation"]["adaptation_checkpoints"],
        recovery_windows=tiny_core_config["evaluation"]["recovery_windows"],
    )
    assert all(
        torch.equal(value, training.model.state_dict()[name]) for name, value in before.items()
    )


def test_incompatible_core_comparisons_are_rejected() -> None:
    common = {
        "seed": 1,
        "configuration_hash": "a",
        "training_data_fingerprint": "b",
        "evaluation_data_fingerprint": "c",
        "training_observations": 10,
        "evaluation_observations": 20,
    }
    assert comparison_differences(common, dict(common)) == []
    changed = dict(common, evaluation_observations=21)
    assert comparison_differences(common, changed) == ["evaluation_observations"]


def test_single_seed_statistics_have_zero_not_undefined_variation() -> None:
    assert mean_and_sample_std([0.5]) == {"mean": 0.5, "sample_std": 0.0}
