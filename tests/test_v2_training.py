from pathlib import Path

from bilab.environments.adaptation_ladder import balanced_reversal_episodes
from bilab.training.v1 import evaluate_model
from bilab.v2.checkpoints import load_v2_checkpoint, save_v2_checkpoint
from bilab.v2.models import RawRouterCore, V2ModelConfig
from bilab.v2.training import (
    V2TrainConfig,
    assert_v2_ranges_disjoint,
    slot_permutation_equivariance,
    state_digest,
    temporal_gradient_audit,
    train_v2_candidate,
)


def _tiny_config(seed: int = 31) -> V2TrainConfig:
    return V2TrainConfig(
        family="raw_router",
        seed=seed,
        world_seed_start=410_000,
        validation_seed_start=420_000,
        optimizer_steps=2,
        batch_groups=1,
        episode_steps=12,
        hidden_dim=16,
        validation_groups=1,
        validation_interval=1,
        torch_threads=1,
    )


def test_delayed_query_gradient_reaches_raw_writer_router_and_reader() -> None:
    model = RawRouterCore(V2ModelConfig(family="raw_router", hidden_dim=16))
    audit = temporal_gradient_audit(model, gap_steps=4)
    assert audit["minimum_state_gradient_norm"] > 0
    assert audit["raw_writer_reached"]
    assert audit["routing_controller_reached"]
    assert audit["writer_candidate_reached"]
    assert audit["writer_gate_reached"]
    assert audit["reader_reached"]
    assert audit["observation_encoder_reached"]


def test_training_is_deterministic_and_evaluation_freezes_weights() -> None:
    first = train_v2_candidate(_tiny_config())
    second = train_v2_candidate(_tiny_config())
    assert state_digest(first.model) == state_digest(second.model)
    worlds = balanced_reversal_episodes(seed_start=430_000, groups=2)
    before = state_digest(first.model)
    evaluation = evaluate_model(first.model, worlds)
    assert evaluation["weights_unchanged"] is True
    assert evaluation["autograd_history_retained"] is False
    assert state_digest(first.model) == before


def test_corresponding_slot_and_parameter_permutation_preserves_behavior() -> None:
    run = train_v2_candidate(_tiny_config(seed=35))
    worlds = balanced_reversal_episodes(seed_start=431_000, groups=2)
    result = slot_permutation_equivariance(run.model, worlds)
    assert result["applicable"] is True
    assert result["behavior_preserved"] is True
    assert (
        result["original"]["adaptation_curve"]
        == result["corresponding_permutation"]["adaptation_curve"]
    )
    assert result["observed_loss_difference"] <= 1e-7


def test_checkpoint_reload_reproduces_model_and_metadata(tmp_path: Path) -> None:
    config = _tiny_config(seed=32)
    run = train_v2_candidate(config)
    checkpoint = tmp_path / "model.pt"
    metadata = save_v2_checkpoint(
        checkpoint,
        run.model,
        config,
        repo=Path(__file__).parents[1],
        experiment_id="test-v2",
        training_step=2,
        validation_score=run.metrics["best_validation_accuracy"],
    )
    loaded, loaded_config, loaded_metadata = load_v2_checkpoint(checkpoint)
    assert loaded_config == config
    assert loaded_metadata["configuration_hash"] == metadata["configuration_hash"]
    assert state_digest(loaded) == state_digest(run.model)


def test_resource_comparison_rejects_overlapping_ranges() -> None:
    config = _tiny_config()
    assert_v2_ranges_disjoint(config, additional={"evaluation": (430_000, 430_100)})
    try:
        assert_v2_ranges_disjoint(config, additional={"evaluation": (410_001, 410_010)})
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping v2 seed ranges were accepted")


def test_mixed_curriculum_introduces_one_factor_at_a_time_with_equal_lengths() -> None:
    config = _tiny_config()
    mixed = V2TrainConfig(
        **{
            **config.__dict__,
            "environment": "mixed",
        }
    )
    assert mixed.episode_steps == 12
