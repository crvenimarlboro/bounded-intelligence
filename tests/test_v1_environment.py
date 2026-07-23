import itertools

import pytest
import torch

from bilab.environments.adaptation_ladder import (
    BinaryBatch,
    WorldKind,
    assert_disjoint_seed_ranges,
    balanced_composition_episodes,
    balanced_context_episodes,
    balanced_delayed_episodes,
    balanced_reversal_episodes,
    evaluate_composition_oracle,
    evaluate_context_oracle,
    evaluate_delay_oracle,
    evaluate_oracle,
    evaluate_reversal_oracle,
    exhaustive_oracle_validation,
    make_binary_episode,
    make_reversal_episode,
    paired_structured_episodes,
)


def test_exhaustive_oracle_and_theoretical_state() -> None:
    result = exhaustive_oracle_validation(max_steps=5)
    assert result["cases"] == sum(2**steps * 4 for steps in range(2, 6))
    assert result["pre_evidence_accuracy"] == 0.5
    assert result["post_evidence_accuracy"] == 1.0
    assert result["persistent_bits"] == 1


def test_identical_current_observations_require_opposite_targets() -> None:
    episodes = paired_structured_episodes(seed_start=10_000, pairs=8, steps=9)
    for copy, flip in itertools.batched(episodes, 2):
        assert torch.equal(copy.public, flip.public)
        assert torch.equal(copy.outcomes, 1 - flip.outcomes)
    curve = evaluate_oracle(episodes)
    assert curve["0"] == 0.5
    assert all(accuracy == 1.0 for prior, accuracy in curve.items() if prior != "0")


def test_hidden_rule_and_future_are_not_model_inputs() -> None:
    episode = make_binary_episode(seed=12, steps=6, rule=1)
    public, outcome = episode.model_step(2)
    assert public.shape == (2,)
    assert public.tolist() == episode.public[2].tolist()
    assert outcome.ndim == 0
    mutated_future = episode.outcomes.clone()
    mutated_future[3:] = 1 - mutated_future[3:]
    assert torch.equal(public, episode.model_step(2)[0])


def test_surface_relabelling_preserves_relation() -> None:
    plain = make_binary_episode(seed=33, steps=10, rule=1, surface_flip=0)
    relabelled = make_binary_episode(seed=33, steps=10, rule=1, surface_flip=1)
    assert torch.equal(relabelled.public[:, 0], 1 - plain.public[:, 0])
    assert torch.equal(relabelled.outcomes, 1 - plain.outcomes)
    for episode in (plain, relabelled):
        inferred = episode.public[:, 0].long() ^ episode.outcomes
        assert torch.all(inferred == 1)


def test_random_control_has_no_hidden_rule_and_is_deterministic() -> None:
    first = make_binary_episode(seed=44, steps=32, rule=None, kind=WorldKind.RANDOM, surface_flip=1)
    second = make_binary_episode(
        seed=44, steps=32, rule=None, kind=WorldKind.RANDOM, surface_flip=1
    )
    assert first.hidden_rule is None
    assert torch.equal(first.public, second.public)
    assert torch.equal(first.outcomes, second.outcomes)


def test_batch_separates_public_data_from_research_truth() -> None:
    episodes = paired_structured_episodes(seed_start=100, pairs=2, steps=5)
    batch = BinaryBatch.from_episodes(episodes)
    assert batch.public.shape == (4, 5, 2)
    assert batch.outcomes.shape == (4, 5)
    assert batch.hidden_rules.tolist() == [0, 1, 0, 1]
    assert batch.public.shape[-1] == 2


def test_seed_classes_are_disjoint_and_overlap_is_rejected() -> None:
    assert_disjoint_seed_ranges(
        {
            "development": range(10_000, 20_000),
            "pilot": range(30_000, 40_000),
            "final": range(70_000, 80_000),
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_seed_ranges({"development": range(1, 4), "pilot": range(3, 6)})


def test_episode_validation_rejects_malformed_world_requests() -> None:
    with pytest.raises(ValueError, match="structured"):
        make_binary_episode(seed=1, steps=4, rule=None)
    with pytest.raises(ValueError, match="random worlds"):
        make_binary_episode(seed=1, steps=4, rule=1, kind=WorldKind.RANDOM)


def test_two_context_oracle_learns_one_independent_bit_per_context() -> None:
    episodes = balanced_context_episodes(seed_start=20_000, groups=32, steps=9)
    curve = evaluate_context_oracle(episodes)
    assert curve["0"] == 0.5
    assert curve["1"] == 0.5
    assert all(value == 1.0 for index, value in curve.items() if int(index) >= 2)
    for group_start in range(0, len(episodes), 4):
        public = episodes[group_start].public
        assert all(
            torch.equal(public, episode.public)
            for episode in episodes[group_start : group_start + 4]
        )


def test_two_bit_oracle_solves_composed_queries_after_both_evidence_events() -> None:
    episodes = balanced_composition_episodes(seed_start=21_000, groups=16, steps=11)
    curve = evaluate_composition_oracle(episodes)
    assert curve["0"] == curve["1"] == 0.5
    assert all(value == 1.0 for index, value in curve.items() if int(index) >= 2)
    assert any(int(value.item()) == 2 for value in episodes[0].public[:, 1])


def test_two_bit_oracle_retains_rules_across_marked_non_events() -> None:
    episodes = balanced_delayed_episodes(
        seed_start=22_000, groups=8, delay_steps=100, query_steps=6
    )
    result = evaluate_delay_oracle(episodes)
    assert result["query_accuracy_after_delay"] == 1.0
    assert result["maximum_distractor_steps"] == 100
    assert result["persistent_bits"] == 2


def test_oracle_makes_one_error_then_recovers_after_unmarked_reversal() -> None:
    episodes = balanced_reversal_episodes(seed_start=23_000, groups=16)
    result = evaluate_reversal_oracle(episodes)
    assert result["change_step_accuracy"] == 0.0
    assert result["post_feedback_recovery_accuracy"] == 1.0
    assert result["unrelated_rule_retention_accuracy"] == 1.0


def test_reversal_timing_is_not_deterministically_revealed_by_public_history() -> None:
    early = make_reversal_episode(seed=24_000, rules=(0, 1), change_step=4)
    late = make_reversal_episode(seed=24_000, rules=(0, 1), change_step=8)
    assert torch.equal(early.public, late.public)
    assert torch.equal(early.outcomes[:4], late.outcomes[:4])
    assert early.outcomes[4] != late.outcomes[4]
    generated = balanced_reversal_episodes(seed_start=24_000, groups=64)
    assert {episode.rule_change_step for episode in generated} == {4, 5, 8, 11}
    mixed = balanced_reversal_episodes(seed_start=24_000, groups=64, include_no_change=True)
    assert None in {episode.rule_change_step for episode in mixed}
