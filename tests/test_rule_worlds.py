from dataclasses import fields

from bilab.environments.rule_worlds import (
    Episode,
    PublicObservation,
    generate_episode,
    make_evaluation_episodes,
    make_training_episodes,
    make_validation_episodes,
    public_tensor_values,
)


def test_generation_splits_are_disjoint(tiny_core_config: dict) -> None:
    train = {episode.generation_seed for episode in make_training_episodes(tiny_core_config)}
    validation = {episode.generation_seed for episode in make_validation_episodes(tiny_core_config)}
    test = {
        episode.generation_seed
        for episodes in make_evaluation_episodes(tiny_core_config).values()
        for episode in episodes
    }
    assert not train & validation
    assert not train & test
    assert not validation & test


def test_model_facing_values_exclude_rules_seed_targets_and_future() -> None:
    episode = generate_episode(generation_seed=42, category="structured", length=12)
    assert [field.name for field in fields(PublicObservation)] == [
        "entity_a",
        "entity_b",
        "operation",
        "context",
        "kind",
        "noise",
        "marker",
    ]
    assert "target" not in {field.name for field in fields(PublicObservation)}
    assert "hidden_rules" not in Episode.__dataclass_fields__
    public = public_tensor_values(episode)
    assert all(len(step) == 7 for step in public)


def test_future_generation_cannot_change_existing_prefix() -> None:
    short = generate_episode(generation_seed=99, category="structured", length=10)
    long = generate_episode(generation_seed=99, category="structured", length=20)
    assert public_tensor_values(short) == public_tensor_values(long)[:10]
    assert [step.target for step in short.steps] == [step.target for step in long.steps[:10]]


def test_each_complete_query_block_has_the_declared_mix() -> None:
    episode = generate_episode(generation_seed=91, category="structured", length=30)
    expected = ["direct"] * 5 + ["composed"] * 2 + ["counterfactual"] + ["delayed"] * 2
    orders: list[tuple[str, ...]] = []
    for start in range(0, 30, 10):
        block = tuple(step.query_kind for step in episode.steps[start : start + 10])
        assert sorted(block) == sorted(expected)
        orders.append(block)
    assert len(set(orders)) > 1


def test_rule_change_and_random_controls_are_annotated_without_leakage() -> None:
    changed = generate_episode(
        generation_seed=7, category="rule_change", length=20, rule_change_step=8
    )
    random_control = generate_episode(generation_seed=8, category="random", length=20)
    assert sum(step.observation.marker for step in changed.steps) == 1
    assert any(step.affected_by_change for step in changed.steps[8:])
    assert any(step.retention_probe for step in changed.steps[8:])
    assert all(step.phase == "pre_change" for step in random_control.steps)


def test_surface_relabelled_worlds_use_new_public_permutations(tiny_core_config: dict) -> None:
    worlds = make_evaluation_episodes(tiny_core_config)
    structured = {episode.surface_signature for episode in worlds["structured"]}
    relabelled = {episode.surface_signature for episode in worlds["surface_relabelled"]}
    assert structured.isdisjoint(relabelled)
