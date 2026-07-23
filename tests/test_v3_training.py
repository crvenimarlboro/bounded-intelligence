from pathlib import Path

import torch

from bilab.environments.adaptation_ladder import (
    balanced_context_episodes,
    balanced_reversal_episodes,
)
from bilab.training.v1 import evaluate_model
from bilab.v3.checkpoints import load_v3_checkpoint, save_v3_checkpoint
from bilab.v3.environments import (
    balance_evidence_inputs,
    raw_evidence_pair_counts,
    validate_balanced_evidence,
)
from bilab.v3.models import RawHardRouterCore, V3ModelConfig
from bilab.v3.training import (
    V3TrainConfig,
    assert_v3_ranges_disjoint,
    seed_v3,
    temporal_gradient_audit_v3,
    train_v3_candidate,
)


def _tiny_config(seed: int = 41) -> V3TrainConfig:
    return V3TrainConfig(
        family="raw_hard_router",
        stage="v3c",
        seed=seed,
        world_seed_start=560_000,
        validation_seed_start=561_000,
        optimizer_steps=2,
        batch_groups=4,
        episode_steps=12,
        hidden_dim=16,
        validation_groups=1,
        validation_interval=1,
        torch_threads=1,
    )


def test_all_raw_input_outcome_combinations_are_balanced() -> None:
    episodes = balance_evidence_inputs(
        balanced_context_episodes(seed_start=562_000, groups=4, steps=12)
    )
    validate_balanced_evidence(episodes)
    assert len(set(raw_evidence_pair_counts(episodes).values())) == 1


def test_current_public_observation_is_history_ambiguous() -> None:
    episodes = balanced_reversal_episodes(seed_start=563_000, groups=8)
    targets_by_public: dict[tuple[int, int, int], set[int]] = {}
    for episode in episodes:
        for public, outcome in zip(episode.public, episode.outcomes, strict=True):
            key = (int(public[0]), int(public[1]), int(public[2]))
            targets_by_public.setdefault(key, set()).add(int(outcome))
    assert any(targets == {0, 1} for targets in targets_by_public.values())


def test_delayed_loss_reaches_raw_relation_gate_router_and_reader() -> None:
    model = RawHardRouterCore(V3ModelConfig(family="raw_hard_router", hidden_dim=16))
    audit = temporal_gradient_audit_v3(model, gap_steps=8)
    modules = audit["module_gradient_norms"]
    assert audit["minimum_state_gradient_norm"] > 0
    assert modules["writer_encoder"] > 0
    assert modules["value_candidate"] > 0
    assert modules["write_controller"] > 0
    assert modules["router"] > 0
    assert modules["state_reader"] > 0
    assert modules["observation_encoder"] > 0
    assert audit["nan_or_inf"] is False


def test_training_is_deterministic_and_evaluation_freezes_weights() -> None:
    first = train_v3_candidate(_tiny_config())
    second = train_v3_candidate(_tiny_config())
    first_state = first.model.state_dict()
    second_state = second.model.state_dict()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    worlds = balanced_reversal_episodes(seed_start=564_000, groups=2)
    before = {name: value.clone() for name, value in first.model.state_dict().items()}
    evaluation = evaluate_model(first.model, worlds)
    assert evaluation["weights_unchanged"] is True
    assert evaluation["autograd_history_retained"] is False
    assert all(torch.equal(before[name], value) for name, value in first.model.state_dict().items())


def test_checkpoint_reload_reproduces_model_and_metadata(tmp_path: Path) -> None:
    config = _tiny_config(seed=42)
    run = train_v3_candidate(config)
    checkpoint = tmp_path / "v3.pt"
    metadata = save_v3_checkpoint(
        checkpoint,
        run.model,
        config,
        repo=Path(__file__).parents[1],
        experiment_id="test-v3",
        training_step=2,
        validation_score=run.metrics["final_validation_accuracy"],
    )
    loaded, loaded_config, loaded_metadata = load_v3_checkpoint(checkpoint)
    assert loaded_config == config
    assert loaded_metadata["configuration_hash"] == metadata["configuration_hash"]
    assert all(
        torch.equal(run.model.state_dict()[name], loaded.state_dict()[name])
        for name in run.model.state_dict()
    )


def test_v3_seed_ranges_are_disjoint_and_overlap_is_rejected() -> None:
    config = _tiny_config()
    assert_v3_ranges_disjoint(config, additional={"evaluation": (570_000, 570_999)})
    try:
        assert_v3_ranges_disjoint(config, additional={"evaluation": (560_001, 560_005)})
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping V3 generation ranges were accepted")


def test_diagnostics_are_not_model_inputs() -> None:
    seed_v3(44, threads=1)
    model = RawHardRouterCore(V3ModelConfig(family="raw_hard_router", hidden_dim=16))
    names = {name for name, _ in model.named_modules()}
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("probe_output", "telemetry", "history_cache")
    )
