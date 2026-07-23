"""Minimal oracle-backed environments for bounded online adaptation research.

The model receives only ``public`` observations one timestep at a time. Hidden rule and seed
metadata live in the research-only episode object and are never accepted by model APIs.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from enum import IntEnum

import torch


class WorldKind(IntEnum):
    """Procedural condition, represented publicly only through generated events."""

    STRUCTURED = 0
    RANDOM = 1


@dataclass(frozen=True)
class BinaryEpisode:
    """One research episode with public events separated from privileged truth."""

    public: torch.Tensor
    outcomes: torch.Tensor
    hidden_rule: int | None
    generation_seed: int
    kind: WorldKind
    surface_flip: int

    def __post_init__(self) -> None:
        if self.public.ndim != 2 or self.public.shape[1] != 2:
            raise ValueError("public observations must have shape [steps, 2]")
        if self.outcomes.shape != (self.public.shape[0],):
            raise ValueError("outcomes must have one label per observation")
        if self.public.dtype != torch.float32:
            raise ValueError("public observations must use float32")
        if self.outcomes.dtype != torch.long:
            raise ValueError("outcomes must use integer class labels")
        if self.public.shape[0] < 2:
            raise ValueError("an episode needs evidence and at least one later query")
        if not torch.all((self.public == 0) | (self.public == 1)):
            raise ValueError("public fields must be binary")
        expected_phase = torch.ones(self.public.shape[0], dtype=torch.float32)
        expected_phase[0] = 0
        if not torch.equal(self.public[:, 1], expected_phase):
            raise ValueError("phase must mark zero prior outcomes only on the first step")

    @property
    def steps(self) -> int:
        return int(self.public.shape[0])

    def model_step(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only the current public observation and current outcome feedback."""

        return self.public[index], self.outcomes[index]


@dataclass(frozen=True)
class BinaryBatch:
    """A tensor batch that keeps privileged labels outside the public model input."""

    public: torch.Tensor
    outcomes: torch.Tensor
    hidden_rules: torch.Tensor
    seeds: tuple[int, ...]
    kinds: tuple[WorldKind, ...]

    @classmethod
    def from_episodes(cls, episodes: list[BinaryEpisode]) -> BinaryBatch:
        if not episodes:
            raise ValueError("cannot batch zero episodes")
        lengths = {episode.steps for episode in episodes}
        if len(lengths) != 1:
            raise ValueError("batched episodes must have equal lengths")
        rules = [
            episode.hidden_rule if episode.hidden_rule is not None else -1 for episode in episodes
        ]
        return cls(
            public=torch.stack([episode.public for episode in episodes]),
            outcomes=torch.stack([episode.outcomes for episode in episodes]),
            hidden_rules=torch.tensor(rules, dtype=torch.long),
            seeds=tuple(episode.generation_seed for episode in episodes),
            kinds=tuple(episode.kind for episode in episodes),
        )


@dataclass(frozen=True)
class ContextEpisode:
    """Two-context world with two independent hidden rule bits."""

    public: torch.Tensor
    outcomes: torch.Tensor
    hidden_rules: tuple[int, int] | None
    generation_seed: int
    surface_flip: int
    rule_change_step: int | None = None
    changed_context: int | None = None
    final_rules: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.public.ndim != 2 or self.public.shape[1] != 3:
            raise ValueError("context observations must have shape [steps, 3]")
        if self.outcomes.shape != (self.public.shape[0],):
            raise ValueError("context outcomes must have one label per observation")
        if self.public.dtype != torch.float32 or self.outcomes.dtype != torch.long:
            raise ValueError("context episode tensor dtypes are invalid")
        if self.public.shape[0] < 3:
            raise ValueError("context worlds require two evidence events and one query")
        if self.hidden_rules is not None and self.hidden_rules not in {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }:
            raise ValueError("context rules must be two binary values")
        if self.rule_change_step is not None:
            if not 0 <= self.rule_change_step < self.public.shape[0]:
                raise ValueError("rule change step is outside the episode")
            if self.changed_context not in {0, 1} or self.final_rules is None:
                raise ValueError("rule change metadata is incomplete")

    @property
    def steps(self) -> int:
        return int(self.public.shape[0])

    def model_step(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.public[index], self.outcomes[index]


@dataclass(frozen=True)
class ContextBatch:
    public: torch.Tensor
    outcomes: torch.Tensor
    hidden_rules: torch.Tensor
    seeds: tuple[int, ...]

    @classmethod
    def from_episodes(cls, episodes: list[ContextEpisode]) -> ContextBatch:
        if not episodes:
            raise ValueError("cannot batch zero context episodes")
        if len({episode.steps for episode in episodes}) != 1:
            raise ValueError("context episodes must have equal lengths")
        labels = [
            episode.hidden_rules[0] * 2 + episode.hidden_rules[1]
            if episode.hidden_rules is not None
            else -1
            for episode in episodes
        ]
        return cls(
            public=torch.stack([episode.public for episode in episodes]),
            outcomes=torch.stack([episode.outcomes for episode in episodes]),
            hidden_rules=torch.tensor(labels, dtype=torch.long),
            seeds=tuple(episode.generation_seed for episode in episodes),
        )


def _input_schedule(seed: int, steps: int) -> list[int]:
    generator = random.Random(seed)
    first = generator.randrange(2)
    pairs = [[0, 1] if generator.randrange(2) == 0 else [1, 0] for _ in range((steps + 1) // 2)]
    values = [first, *(value for pair in pairs for value in pair)]
    return values[:steps]


def make_binary_episode(
    *,
    seed: int,
    steps: int,
    rule: int | None,
    kind: WorldKind = WorldKind.STRUCTURED,
    surface_flip: int = 0,
) -> BinaryEpisode:
    """Generate one COPY/FLIP or incompressible random-control episode.

    Public fields are ``[surface_input_bit, has_prior_feedback]``. The first field and outcome are
    relabelled together, preserving the hidden relation without exposing the relabelling.
    """

    if steps < 2:
        raise ValueError("steps must be at least two")
    if surface_flip not in (0, 1):
        raise ValueError("surface_flip must be binary")
    if kind is WorldKind.STRUCTURED and rule not in (0, 1):
        raise ValueError("structured worlds require a binary hidden rule")
    if kind is WorldKind.RANDOM and rule is not None:
        raise ValueError("random worlds have no hidden rule")

    inputs = _input_schedule(seed, steps)
    if kind is WorldKind.STRUCTURED:
        semantic_outcomes = [value ^ int(rule) for value in inputs]
    else:
        outcome_generator = random.Random(seed ^ 0x5EEDC0DE)
        semantic_outcomes = [outcome_generator.randrange(2) for _ in inputs]
    surface_inputs = [value ^ surface_flip for value in inputs]
    surface_outcomes = [value ^ surface_flip for value in semantic_outcomes]
    phase = [0, *([1] * (steps - 1))]
    public = torch.tensor(list(zip(surface_inputs, phase, strict=True)), dtype=torch.float32)
    outcomes = torch.tensor(surface_outcomes, dtype=torch.long)
    return BinaryEpisode(
        public=public,
        outcomes=outcomes,
        hidden_rule=rule,
        generation_seed=seed,
        kind=kind,
        surface_flip=surface_flip,
    )


def paired_structured_episodes(
    *, seed_start: int, pairs: int, steps: int, relabel: bool = False
) -> list[BinaryEpisode]:
    """Generate paired histories with identical observations and opposite correct outcomes."""

    if pairs < 1:
        raise ValueError("pairs must be positive")
    episodes: list[BinaryEpisode] = []
    for offset in range(pairs):
        seed = seed_start + offset
        surface_flip = random.Random(seed ^ 0x51FACE).randrange(2) if relabel else 0
        episodes.extend(
            make_binary_episode(
                seed=seed,
                steps=steps,
                rule=rule,
                surface_flip=surface_flip,
            )
            for rule in (0, 1)
        )
    return episodes


def make_context_episode(
    *,
    seed: int,
    steps: int,
    rules: tuple[int, int] | None,
    surface_flip: int = 0,
) -> ContextEpisode:
    """Generate a two-rule world; contexts 0 and 1 appear once before held-out queries."""

    if steps < 3:
        raise ValueError("two-context episodes need at least three steps")
    if rules is not None and rules not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("rules must contain two binary values")
    generator = random.Random(seed)
    contexts = [0, 1, *[generator.randrange(2) for _ in range(steps - 2)]]
    inputs = _input_schedule(seed ^ 0xC07E57, steps)
    if rules is None:
        outcome_generator = random.Random(seed ^ 0x5EEDC0DE)
        outcomes = [outcome_generator.randrange(2) for _ in inputs]
    else:
        outcomes = [value ^ rules[context] for value, context in zip(inputs, contexts, strict=True)]
    surface_inputs = [value ^ surface_flip for value in inputs]
    surface_outcomes = [value ^ surface_flip for value in outcomes]
    phase = [0, *([1] * (steps - 1))]
    public = torch.tensor(
        list(zip(surface_inputs, contexts, phase, strict=True)), dtype=torch.float32
    )
    return ContextEpisode(
        public=public,
        outcomes=torch.tensor(surface_outcomes, dtype=torch.long),
        hidden_rules=rules,
        generation_seed=seed,
        surface_flip=surface_flip,
    )


def balanced_context_episodes(
    *, seed_start: int, groups: int, steps: int, relabel: bool = False
) -> list[ContextEpisode]:
    """Generate all four rule pairs over identical public input/context schedules."""

    if groups < 1:
        raise ValueError("groups must be positive")
    episodes: list[ContextEpisode] = []
    for offset in range(groups):
        seed = seed_start + offset
        surface_flip = random.Random(seed ^ 0x51FACE).randrange(2) if relabel else 0
        episodes.extend(
            make_context_episode(seed=seed, steps=steps, rules=rules, surface_flip=surface_flip)
            for rules in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
    return episodes


def random_context_episodes(
    *, seed_start: int, count: int, steps: int, relabel: bool = False
) -> list[ContextEpisode]:
    if count < 1:
        raise ValueError("count must be positive")
    return [
        make_context_episode(
            seed=seed_start + index,
            steps=steps,
            rules=None,
            surface_flip=(index % 2 if relabel else 0),
        )
        for index in range(count)
    ]


def make_composition_episode(
    *,
    seed: int,
    steps: int,
    rules: tuple[int, int] | None,
    surface_flip: int = 0,
) -> ContextEpisode:
    """Generate two evidence rules plus direct and composed held-out queries."""

    if steps < 5:
        raise ValueError("composition episodes need at least five steps")
    if rules is not None and rules not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("composition rules must contain two binary values")
    generator = random.Random(seed)
    query_pattern = [2, 0, 1]
    contexts = [0, 1] + [query_pattern[index % len(query_pattern)] for index in range(steps - 2)]
    if generator.randrange(2):
        contexts[2:] = [2 if value == 2 else 1 - value for value in contexts[2:]]
    inputs = _input_schedule(seed ^ 0xC0A905E, steps)
    if rules is None:
        outcome_generator = random.Random(seed ^ 0x5EEDC0DE)
        outcomes = [outcome_generator.randrange(2) for _ in inputs]
    else:
        outcomes = [
            value ^ (rules[0] ^ rules[1] if context == 2 else rules[context])
            for value, context in zip(inputs, contexts, strict=True)
        ]
    surface_inputs = [value ^ surface_flip for value in inputs]
    surface_outcomes = [value ^ surface_flip for value in outcomes]
    phase = [0, *([1] * (steps - 1))]
    return ContextEpisode(
        public=torch.tensor(
            list(zip(surface_inputs, contexts, phase, strict=True)), dtype=torch.float32
        ),
        outcomes=torch.tensor(surface_outcomes, dtype=torch.long),
        hidden_rules=rules,
        generation_seed=seed,
        surface_flip=surface_flip,
    )


def balanced_composition_episodes(
    *, seed_start: int, groups: int, steps: int, relabel: bool = False
) -> list[ContextEpisode]:
    if groups < 1:
        raise ValueError("groups must be positive")
    episodes: list[ContextEpisode] = []
    for offset in range(groups):
        seed = seed_start + offset
        surface_flip = random.Random(seed ^ 0x51FACE).randrange(2) if relabel else 0
        episodes.extend(
            make_composition_episode(seed=seed, steps=steps, rules=rules, surface_flip=surface_flip)
            for rules in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
    return episodes


def random_composition_episodes(
    *, seed_start: int, count: int, steps: int, relabel: bool = False
) -> list[ContextEpisode]:
    if count < 1:
        raise ValueError("count must be positive")
    return [
        make_composition_episode(
            seed=seed_start + index,
            steps=steps,
            rules=None,
            surface_flip=(index % 2 if relabel else 0),
        )
        for index in range(count)
    ]


def make_delayed_episode(
    *,
    seed: int,
    rules: tuple[int, int] | None,
    delay_steps: int = 8,
    query_steps: int = 6,
    surface_flip: int = 0,
) -> ContextEpisode:
    """Insert publicly marked non-events between evidence and compositional queries."""

    if delay_steps < 1 or query_steps < 1:
        raise ValueError("delay and query counts must be positive")
    if rules is not None and rules not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("delayed rules must contain two binary values")
    generator = random.Random(seed)
    query_pattern = [2, 0, 1]
    contexts = [0, 1, *([3] * delay_steps)] + [
        query_pattern[index % 3] for index in range(query_steps)
    ]
    inputs = _input_schedule(seed ^ 0xDE1A7, len(contexts))
    outcomes: list[int] = []
    for value, context in zip(inputs, contexts, strict=True):
        if rules is None or context == 3:
            outcomes.append(generator.randrange(2))
        elif context == 2:
            outcomes.append(value ^ rules[0] ^ rules[1])
        else:
            outcomes.append(value ^ rules[context])
    surface_inputs = [value ^ surface_flip for value in inputs]
    surface_outcomes = [value ^ surface_flip for value in outcomes]
    phase = [0, *([1] * (len(contexts) - 1))]
    return ContextEpisode(
        public=torch.tensor(
            list(zip(surface_inputs, contexts, phase, strict=True)), dtype=torch.float32
        ),
        outcomes=torch.tensor(surface_outcomes, dtype=torch.long),
        hidden_rules=rules,
        generation_seed=seed,
        surface_flip=surface_flip,
    )


def balanced_delayed_episodes(
    *,
    seed_start: int,
    groups: int,
    delay_steps: int = 8,
    query_steps: int = 6,
    relabel: bool = False,
) -> list[ContextEpisode]:
    if groups < 1:
        raise ValueError("groups must be positive")
    episodes: list[ContextEpisode] = []
    for offset in range(groups):
        seed = seed_start + offset
        surface_flip = random.Random(seed ^ 0x51FACE).randrange(2) if relabel else 0
        episodes.extend(
            make_delayed_episode(
                seed=seed,
                rules=rules,
                delay_steps=delay_steps,
                query_steps=query_steps,
                surface_flip=surface_flip,
            )
            for rules in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
    return episodes


def random_delayed_episodes(
    *,
    seed_start: int,
    count: int,
    delay_steps: int = 8,
    query_steps: int = 6,
    relabel: bool = False,
) -> list[ContextEpisode]:
    if count < 1:
        raise ValueError("count must be positive")
    return [
        make_delayed_episode(
            seed=seed_start + index,
            rules=None,
            delay_steps=delay_steps,
            query_steps=query_steps,
            surface_flip=(index % 2 if relabel else 0),
        )
        for index in range(count)
    ]


def evaluate_delay_oracle(episodes: list[ContextEpisode]) -> dict[str, float | int]:
    """Verify retained two-bit state after marked non-events."""

    query_correct = 0
    query_count = 0
    max_drift_events = 0
    for episode in episodes:
        oracle = ContextRuleOracle()
        distractors = 0
        for index in range(episode.steps):
            observation, outcome = episode.model_step(index)
            context = int(observation[1].item())
            if context == 3:
                distractors += 1
                continue
            if context == 2:
                prediction = int(observation[0].item()) ^ oracle.rule_bits[0] ^ oracle.rule_bits[1]
            else:
                prediction = oracle.predict(observation)
            if index >= 2:
                query_correct += int(prediction == int(outcome.item()))
                query_count += 1
            if context < 2:
                oracle.update(observation, outcome)
        max_drift_events = max(max_drift_events, distractors)
    return {
        "query_accuracy_after_delay": query_correct / query_count,
        "query_count": query_count,
        "maximum_distractor_steps": max_drift_events,
        "persistent_bits": 2,
    }


def make_reversal_episode(
    *,
    seed: int,
    rules: tuple[int, int],
    surface_flip: int = 0,
    change_step: int | None = None,
    include_no_change: bool = False,
) -> ContextEpisode:
    """Change context-0's rule without a marker and preserve context-1's rule."""

    if rules not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("reversal rules must contain two binary values")
    contexts = [0, 1, 2, 1, 0, 0, 1, 2, 0, 1, 2, 0]
    if change_step is None:
        candidates = (4, 5, 8, 11, -1) if include_no_change else (4, 5, 8, 11)
        change_step = candidates[random.Random(seed ^ 0xC4A96E).randrange(len(candidates))]
    if change_step not in {-1, 4, 5, 8, 11}:
        raise ValueError("reversal change_step must be a context-0 query or -1")
    actual_change_step = None if change_step == -1 else change_step
    final_rules = rules if actual_change_step is None else (1 - rules[0], rules[1])
    inputs = _input_schedule(seed ^ 0xAE7E25A1, len(contexts))
    outcomes: list[int] = []
    for index, (value, context) in enumerate(zip(inputs, contexts, strict=True)):
        active = (
            final_rules if actual_change_step is not None and index >= actual_change_step else rules
        )
        rule = active[0] ^ active[1] if context == 2 else active[context]
        outcomes.append(value ^ rule)
    surface_inputs = [value ^ surface_flip for value in inputs]
    surface_outcomes = [value ^ surface_flip for value in outcomes]
    phase = [0, *([1] * (len(contexts) - 1))]
    return ContextEpisode(
        public=torch.tensor(
            list(zip(surface_inputs, contexts, phase, strict=True)), dtype=torch.float32
        ),
        outcomes=torch.tensor(surface_outcomes, dtype=torch.long),
        hidden_rules=rules,
        generation_seed=seed,
        surface_flip=surface_flip,
        rule_change_step=actual_change_step,
        changed_context=0 if actual_change_step is not None else None,
        final_rules=final_rules if actual_change_step is not None else None,
    )


def balanced_reversal_episodes(
    *,
    seed_start: int,
    groups: int,
    relabel: bool = False,
    include_no_change: bool = False,
) -> list[ContextEpisode]:
    if groups < 1:
        raise ValueError("groups must be positive")
    episodes: list[ContextEpisode] = []
    for offset in range(groups):
        seed = seed_start + offset
        surface_flip = random.Random(seed ^ 0x51FACE).randrange(2) if relabel else 0
        episodes.extend(
            make_reversal_episode(
                seed=seed,
                rules=rules,
                surface_flip=surface_flip,
                include_no_change=include_no_change,
            )
            for rules in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
    return episodes


def evaluate_reversal_oracle(episodes: list[ContextEpisode]) -> dict[str, float]:
    """Return exact change-step, recovery, and unrelated-rule retention bounds."""

    change_correct = 0
    recovered_correct = 0
    recovered_count = 0
    retained_correct = 0
    retained_count = 0
    for episode in episodes:
        oracle = ContextRuleOracle()
        change_step = int(episode.rule_change_step)
        for index in range(episode.steps):
            observation, outcome = episode.model_step(index)
            context = int(observation[1].item())
            if context == 2:
                prediction = int(observation[0].item()) ^ oracle.rule_bits[0] ^ oracle.rule_bits[1]
            else:
                prediction = oracle.predict(observation)
            if index == change_step:
                change_correct += int(prediction == int(outcome.item()))
            elif index > change_step and context in {0, 2}:
                recovered_correct += int(prediction == int(outcome.item()))
                recovered_count += 1
            elif index > change_step and context == 1:
                retained_correct += int(prediction == int(outcome.item()))
                retained_count += 1
            if context < 2:
                oracle.update(observation, outcome)
    return {
        "change_step_accuracy": change_correct / len(episodes),
        "post_feedback_recovery_accuracy": recovered_correct / recovered_count,
        "unrelated_rule_retention_accuracy": retained_correct / retained_count,
    }


def evaluate_composition_oracle(episodes: list[ContextEpisode]) -> dict[str, float]:
    """Evaluate the same two-bit oracle on direct and XOR-composed queries."""

    correct = [0] * episodes[0].steps
    for episode in episodes:
        oracle = ContextRuleOracle()
        for index in range(episode.steps):
            observation, outcome = episode.model_step(index)
            context = int(observation[1].item())
            if context == 2:
                prediction = int(observation[0].item()) ^ oracle.rule_bits[0] ^ oracle.rule_bits[1]
            else:
                prediction = oracle.predict(observation)
            correct[index] += int(prediction == int(outcome.item()))
            if context < 2:
                oracle.update(observation, outcome)
    return {str(index): value / len(episodes) for index, value in enumerate(correct)}


@dataclass
class ContextRuleOracle:
    """Exact two-bit oracle; public context identifies which bit is queried."""

    rule_bits: list[int] = field(default_factory=lambda: [0, 0])

    @property
    def persistent_bits(self) -> int:
        return 2

    def predict(self, public_observation: torch.Tensor) -> int:
        x = int(public_observation[0].item())
        context = int(public_observation[1].item())
        # The deterministic schedule makes context 0 known from step 1 and both known from step 2.
        phase = int(public_observation[2].item())
        return x ^ (self.rule_bits[context] if phase else 0)

    def update(self, public_observation: torch.Tensor, outcome: torch.Tensor) -> None:
        x = int(public_observation[0].item())
        context = int(public_observation[1].item())
        self.rule_bits[context] = x ^ int(outcome.item())


def evaluate_context_oracle(episodes: list[ContextEpisode]) -> dict[str, float]:
    correct = [0] * episodes[0].steps
    for episode in episodes:
        oracle = ContextRuleOracle()
        for index in range(episode.steps):
            observation, outcome = episode.model_step(index)
            correct[index] += int(oracle.predict(observation) == int(outcome.item()))
            oracle.update(observation, outcome)
    return {str(index): value / len(episodes) for index, value in enumerate(correct)}


def batch_adaptation_episodes(
    episodes: list[BinaryEpisode] | list[ContextEpisode],
) -> BinaryBatch | ContextBatch:
    if not episodes:
        raise ValueError("cannot batch zero episodes")
    if isinstance(episodes[0], ContextEpisode):
        return ContextBatch.from_episodes(episodes)  # type: ignore[arg-type]
    return BinaryBatch.from_episodes(episodes)  # type: ignore[arg-type]


@dataclass
class BinaryRuleOracle:
    """Exact one-bit sufficient-state oracle after the first public outcome."""

    rule_bit: int = 0

    @property
    def persistent_bits(self) -> int:
        return 1

    def reset(self) -> None:
        self.rule_bit = 0

    def predict(self, public_observation: torch.Tensor) -> int:
        x = int(public_observation[0].item())
        has_feedback = int(public_observation[1].item())
        return x ^ (self.rule_bit if has_feedback else 0)

    def update(self, public_observation: torch.Tensor, outcome: torch.Tensor) -> None:
        self.rule_bit = int(public_observation[0].item()) ^ int(outcome.item())


def evaluate_oracle(episodes: list[BinaryEpisode]) -> dict[str, float]:
    """Evaluate prediction accuracy before and after public evidence."""

    correct_by_prior: dict[int, int] = {}
    count_by_prior: dict[int, int] = {}
    for episode in episodes:
        oracle = BinaryRuleOracle()
        for index in range(episode.steps):
            observation, outcome = episode.model_step(index)
            correct_by_prior[index] = correct_by_prior.get(index, 0) + int(
                oracle.predict(observation) == int(outcome.item())
            )
            count_by_prior[index] = count_by_prior.get(index, 0) + 1
            oracle.update(observation, outcome)
    return {
        str(prior): correct_by_prior[prior] / count_by_prior[prior]
        for prior in sorted(count_by_prior)
    }


def exhaustive_oracle_validation(max_steps: int = 6) -> dict[str, int | float]:
    """Check every binary input sequence, rule, and relabelling for short worlds."""

    if max_steps < 2:
        raise ValueError("max_steps must be at least two")
    cases = 0
    post_evidence_correct = 0
    post_evidence_total = 0
    pre_evidence_correct = 0
    pre_evidence_total = 0
    for steps in range(2, max_steps + 1):
        for inputs in itertools.product((0, 1), repeat=steps):
            for rule, surface_flip in itertools.product((0, 1), repeat=2):
                oracle = BinaryRuleOracle()
                for index, semantic_input in enumerate(inputs):
                    surface_input = semantic_input ^ surface_flip
                    outcome = (semantic_input ^ rule) ^ surface_flip
                    observation = torch.tensor([surface_input, int(index > 0)], dtype=torch.float32)
                    prediction = oracle.predict(observation)
                    if index == 0:
                        pre_evidence_correct += int(prediction == outcome)
                        pre_evidence_total += 1
                    else:
                        post_evidence_correct += int(prediction == outcome)
                        post_evidence_total += 1
                    oracle.update(observation, torch.tensor(outcome))
                cases += 1
    return {
        "cases": cases,
        "pre_evidence_accuracy": pre_evidence_correct / pre_evidence_total,
        "post_evidence_accuracy": post_evidence_correct / post_evidence_total,
        "persistent_bits": BinaryRuleOracle().persistent_bits,
    }


def assert_disjoint_seed_ranges(ranges: dict[str, range]) -> None:
    """Reject overlapping development, pilot, and confirmatory seed ranges."""

    names = list(ranges)
    materialized = {name: set(ranges[name]) for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if materialized[left] & materialized[right]:
                raise ValueError(f"seed ranges overlap: {left} and {right}")
