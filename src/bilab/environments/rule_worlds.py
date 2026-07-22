"""Rule Worlds v0: inspectable sequential worlds without privileged model inputs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

WorldCategory = Literal["structured", "surface_relabelled", "rule_change", "random"]
KIND_NAMES = ("direct", "composed", "counterfactual", "delayed")
OBSERVATION_FIELDS = ("entity_a", "entity_b", "operation", "context", "kind", "noise", "marker")


@dataclass(frozen=True)
class PublicObservation:
    """The complete information visible to a model at one step."""

    entity_a: int
    entity_b: int
    operation: int
    context: int
    kind: int
    noise: int
    marker: int

    def values(self) -> tuple[int, ...]:
        return (
            self.entity_a,
            self.entity_b,
            self.operation,
            self.context,
            self.kind,
            self.noise,
            self.marker,
        )


@dataclass(frozen=True)
class Step:
    observation: PublicObservation
    target: int
    query_kind: str
    phase: str
    affected_by_change: bool
    retention_probe: bool
    steps_since_change: int | None


@dataclass(frozen=True)
class Episode:
    """Public sequence plus researcher-only evaluation annotations, never hidden rules."""

    steps: tuple[Step, ...]
    category: WorldCategory
    generation_seed: int
    surface_signature: tuple[int, ...]


def _structured_outcome(
    kind: int,
    entity_a: int,
    entity_b: int,
    operation: int,
    context: int,
    previous_entity_a: int,
    values: list[int],
    biases: list[int],
) -> int:
    if kind == 0:
        return (values[entity_a] + values[entity_b] + biases[operation]) % 4
    if kind == 1:
        return (values[entity_a] + values[entity_b] + biases[operation] + biases[context]) % 4
    if kind == 2:
        return (values[entity_a] + values[entity_b] + biases[context]) % 4
    return (values[previous_entity_a] + values[entity_b] + biases[operation]) % 4


def generate_episode(
    *,
    generation_seed: int,
    category: WorldCategory,
    length: int,
    entity_count: int = 8,
    operation_count: int = 4,
    noise_values: int = 16,
    rule_change_step: int | None = None,
    latent_seed: int | None = None,
) -> Episode:
    """Generate one episode; hidden rules are local variables and are never returned."""

    if length < 2:
        raise ValueError("Rule Worlds episodes require at least two steps")
    if category == "rule_change" and rule_change_step is None:
        raise ValueError("rule_change episodes require rule_change_step")
    if rule_change_step is not None and not 1 <= rule_change_step < length:
        raise ValueError("rule_change_step must be within the episode")

    generator = random.Random(generation_seed)
    latent_generator = random.Random(generation_seed if latent_seed is None else latent_seed)
    values = [latent_generator.randrange(4) for _ in range(entity_count)]
    biases = [latent_generator.randrange(4) for _ in range(operation_count)]
    permutation = list(range(entity_count))
    generator.shuffle(permutation)
    changed_operation = latent_generator.randrange(operation_count)
    replacement_delta = latent_generator.randrange(1, 4)
    random_table: dict[tuple[int, ...], int] = {}
    previous_entity_a = latent_generator.randrange(entity_count)
    steps: list[Step] = []
    query_block: list[int] = []

    for step_index in range(length):
        if category == "rule_change" and step_index == rule_change_step:
            biases[changed_operation] = (biases[changed_operation] + replacement_delta) % 4

        block_index = step_index % 10
        if block_index == 0:
            query_block = [0, 0, 0, 0, 0, 1, 1, 2, 3, 3]
            generator.shuffle(query_block)
            if step_index == 0 and query_block[0] == 3:
                first_non_delayed = next(
                    index for index, value in enumerate(query_block[1:], start=1) if value != 3
                )
                query_block[0], query_block[first_non_delayed] = (
                    query_block[first_non_delayed],
                    query_block[0],
                )
        kind = query_block[block_index]
        entity_a = generator.randrange(entity_count)
        entity_b = generator.randrange(entity_count)
        operation = generator.randrange(operation_count)
        context = operation_count
        if kind in {1, 2}:
            context = generator.randrange(operation_count)
        marker = int(category == "rule_change" and step_index == rule_change_step)
        visible_a = entity_count if kind == 3 else permutation[entity_a]
        observation = PublicObservation(
            entity_a=visible_a,
            entity_b=permutation[entity_b],
            operation=operation,
            context=context,
            kind=kind,
            noise=generator.randrange(noise_values),
            marker=marker,
        )

        if category == "random":
            key = (
                kind,
                entity_a,
                entity_b,
                operation,
                context,
                previous_entity_a if kind == 3 else -1,
            )
            target = random_table.setdefault(key, latent_generator.randrange(4))
        else:
            target = _structured_outcome(
                kind,
                entity_a,
                entity_b,
                operation,
                context,
                previous_entity_a,
                values,
                biases,
            )

        post_change = category == "rule_change" and step_index >= int(rule_change_step or 0)
        affected = post_change and (
            operation == changed_operation or (kind in {1, 2} and context == changed_operation)
        )
        steps.append(
            Step(
                observation=observation,
                target=target,
                query_kind=KIND_NAMES[kind],
                phase="post_change" if post_change else "pre_change",
                affected_by_change=affected,
                retention_probe=post_change and not affected,
                steps_since_change=(
                    step_index - int(rule_change_step) if post_change and rule_change_step else None
                ),
            )
        )
        previous_entity_a = entity_a

    return Episode(
        steps=tuple(steps),
        category=category,
        generation_seed=generation_seed,
        surface_signature=tuple(permutation),
    )


def make_training_episodes(config: dict[str, Any]) -> list[Episode]:
    environment = config["environment"]
    training = config["training"]
    count = int(environment["train_worlds"])
    structured_count = round(count * float(training["structured_fraction"]))
    start = int(environment["train_generation_seed_start"])
    episodes: list[Episode] = []
    for index in range(count):
        category: WorldCategory = "structured" if index < structured_count else "rule_change"
        episodes.append(
            generate_episode(
                generation_seed=start + index,
                category=category,
                length=int(environment["train_episode_length"]),
                entity_count=int(environment["entity_count"]),
                operation_count=int(environment["operation_count"]),
                noise_values=int(environment["noise_values"]),
                rule_change_step=(
                    int(environment["train_episode_length"]) // 2
                    if category == "rule_change"
                    else None
                ),
            )
        )
    return episodes


def make_validation_episodes(config: dict[str, Any]) -> list[Episode]:
    environment = config["environment"]
    start = int(environment["validation_generation_seed_start"])
    return [
        generate_episode(
            generation_seed=start + index,
            category="structured",
            length=int(environment["evaluation_episode_length"]),
            entity_count=int(environment["entity_count"]),
            operation_count=int(environment["operation_count"]),
            noise_values=int(environment["noise_values"]),
        )
        for index in range(int(environment["validation_worlds"]))
    ]


def make_evaluation_episodes(config: dict[str, Any]) -> dict[str, list[Episode]]:
    environment = config["environment"]
    start = int(environment["test_generation_seed_start"])
    count = int(environment["evaluation_worlds_per_category"])
    common = {
        "entity_count": int(environment["entity_count"]),
        "operation_count": int(environment["operation_count"]),
        "noise_values": int(environment["noise_values"]),
    }
    result: dict[str, list[Episode]] = {}
    for block, category in enumerate(("structured", "surface_relabelled", "rule_change", "random")):
        episodes: list[Episode] = []
        for index in range(count):
            generation_seed = start + block * 10_000 + index
            episodes.append(
                generate_episode(
                    generation_seed=generation_seed,
                    latent_seed=start + index if category == "surface_relabelled" else None,
                    category=category,
                    length=(
                        int(environment["rule_change_episode_length"])
                        if category == "rule_change"
                        else int(environment["evaluation_episode_length"])
                    ),
                    rule_change_step=(
                        int(environment["rule_change_step"]) if category == "rule_change" else None
                    ),
                    **common,
                )
            )
        result[category] = episodes
    return result


def public_tensor_values(episode: Episode) -> tuple[tuple[int, ...], ...]:
    """Return only values that the model is allowed to receive."""

    return tuple(step.observation.values() for step in episode.steps)
