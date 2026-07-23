from dataclasses import replace
from pathlib import Path

import pytest
import torch

from bilab.environments.adaptation_ladder import paired_structured_episodes
from bilab.models.v1 import FactorizedStateCore, V1ModelConfig
from bilab.training.v1 import (
    V1TrainConfig,
    assert_generation_splits_disjoint,
    compare_compatible,
    donor_state_swap,
    evaluate_model,
    generation_seed_ranges,
    linear_rule_probe,
    state_dict_digest,
    temporal_credit_audit,
    train_candidate,
)
from bilab.training.v1_checkpoints import load_v1_checkpoint, save_v1_checkpoint


def _micro_config() -> V1TrainConfig:
    return V1TrainConfig(
        family="factorized",
        seed=17,
        world_seed_start=10_000,
        validation_seed_start=18_000,
        optimizer_steps=40,
        batch_pairs=4,
        validation_pairs=8,
        validation_interval=20,
        episode_steps=6,
        hidden_dim=16,
        state_dim=4,
        torch_threads=1,
    )


def test_deterministic_micro_runs_repeat_exactly() -> None:
    first = train_candidate(_micro_config())
    second = train_candidate(_micro_config())
    assert state_dict_digest(first.model) == state_dict_digest(second.model)
    assert first.metrics["history"] == second.metrics["history"]


def test_checkpoint_reload_reproduces_frozen_evaluation(tmp_path: Path) -> None:
    run = train_candidate(_micro_config())
    episodes = paired_structured_episodes(seed_start=30_000, pairs=8, steps=6)
    expected = evaluate_model(run.model, episodes)
    checkpoint = tmp_path / "model.pt"
    metadata = save_v1_checkpoint(
        checkpoint,
        run.model,
        _micro_config(),
        repo=Path(__file__).parents[1],
        experiment_id="test-v1",
        training_step=run.metrics["best_step"],
        validation_score=run.metrics["best_validation_post_evidence_accuracy"],
    )
    loaded, config, loaded_metadata = load_v1_checkpoint(checkpoint)
    reproduced = evaluate_model(loaded, episodes)
    assert config == _micro_config()
    assert loaded_metadata["configuration_hash"] == metadata["configuration_hash"]
    assert reproduced == expected


def test_frozen_online_evaluation_is_stateless_in_autograd_and_weights() -> None:
    run = train_candidate(_micro_config())
    digest = state_dict_digest(run.model)
    evaluation = evaluate_model(
        run.model, paired_structured_episodes(seed_start=31_000, pairs=4, steps=6)
    )
    assert evaluation["weights_unchanged"]
    assert not evaluation["autograd_history_retained"]
    assert state_dict_digest(run.model) == digest


def test_temporal_credit_probe_and_donor_intervention_are_substantive() -> None:
    run = train_candidate(_micro_config())
    assert isinstance(run.model, FactorizedStateCore)
    audit = temporal_credit_audit(run.model, gap_steps=4)
    assert audit["early_state_gradient_norm"] > 0
    assert audit["gap_updates_applied"] == 4
    assert len(audit["intermediate_state_gradient_norms"]) == 4
    assert all(norm > 0 for norm in audit["intermediate_state_gradient_norms"])
    assert audit["writer_reached"] and audit["reader_reached"]
    train = paired_structured_episodes(seed_start=12_000, pairs=16, steps=6)
    test = paired_structured_episodes(seed_start=32_000, pairs=16, steps=6)
    probe = linear_rule_probe(run.model, train, test, steps=50)
    assert 0 <= probe["held_out_accuracy"] <= 1
    swap = donor_state_swap(run.model, test)
    assert swap["state_pairs"] == 16
    assert swap["donor_rule_consistency"] != swap["recipient_rule_consistency"]


def test_incompatible_resource_comparison_is_rejected() -> None:
    left = {
        "evaluation_seed_start": 1,
        "episode_steps": 6,
        "evaluation_pairs": 4,
        "training_observations": 100,
        "post_evidence_accuracy": 0.8,
    }
    right = dict(left, training_observations=101, post_evidence_accuracy=0.5)
    with pytest.raises(ValueError, match="training_observations"):
        compare_compatible(left, right)
    right["training_observations"] = 100
    assert compare_compatible(left, right)["post_evidence_accuracy_difference"] == pytest.approx(
        0.3
    )


def test_actual_generation_seed_consumption_cannot_overlap_splits() -> None:
    clean = _micro_config()
    assert generation_seed_ranges(clean)["training"] == (10_000, 10_159)
    assert_generation_splits_disjoint(clean, evaluation_ranges={"final": (30_000, 30_100)})
    with pytest.raises(ValueError, match="training and validation"):
        assert_generation_splits_disjoint(replace(clean, validation_seed_start=10_100))


def test_diagnostics_are_not_model_inputs() -> None:
    model = FactorizedStateCore(V1ModelConfig(hidden_dim=16, state_dim=4))
    public = torch.tensor([[0.0, 0.0]])
    state = model.initial_state(1)
    prediction = model.predict(public, state)
    with pytest.raises(TypeError):
        model.predict(public, state, diagnostics={"hidden_rule": 1})
    assert prediction.logits.shape == (1, 2)


def test_temporal_credit_audit_supports_composition_observations() -> None:
    model = FactorizedStateCore(
        V1ModelConfig(hidden_dim=16, state_dim=2, context_count=3, rule_count=2)
    )
    audit = temporal_credit_audit(model, gap_steps=3)
    assert audit["writer_reached"]
    assert audit["reader_reached"]
