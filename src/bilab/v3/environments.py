"""V3 curriculum transforms that remove the fixed-evidence-input shortcut."""

from __future__ import annotations

from dataclasses import replace

import torch

from bilab.environments.adaptation_ladder import ContextEpisode
from bilab.v2.training import _episodes


def balance_evidence_inputs(episodes: list[ContextEpisode]) -> list[ContextEpisode]:
    """Balance all raw input/outcome pairs at the two initial public evidence events.

    Existing environment families place operation 0 then operation 1 in the first two positions,
    but their input schedule can make those evidence inputs predictable. V3 changes only the public
    input and corresponding public outcome at those two evidence positions. Hidden rules, later
    events, episode length, surface convention, and all privileged metadata remain unchanged.
    """

    if len(episodes) % 4:
        raise ValueError("balanced V3 curriculum expects complete four-rule groups")
    balanced: list[ContextEpisode] = []
    input_patterns = ((0, 0), (0, 1), (1, 0), (1, 1))
    for group_index in range(0, len(episodes), 4):
        # Cross every four-rule world group with every evidence-input pattern while
        # holding its later public schedule fixed. Merely assigning one pattern to
        # each independently generated group leaves the pattern correlated with
        # later queries and permits an outcome-only shortcut.
        for pattern in input_patterns:
            for episode in episodes[group_index : group_index + 4]:
                if episode.hidden_rules is None:
                    raise ValueError("V3 training balance requires structured hidden rules")
                public = episode.public.clone()
                outcomes = episode.outcomes.clone()
                for time_index, operation in enumerate((0, 1)):
                    public_input = pattern[operation]
                    public[time_index, 0] = float(public_input)
                    outcomes[time_index] = public_input ^ episode.hidden_rules[operation]
                balanced.append(replace(episode, public=public, outcomes=outcomes))
    return balanced


def v3_training_episodes(
    *,
    seed_start: int,
    groups: int,
    episode_steps: int,
    mix_index: int | None,
) -> list[ContextEpisode]:
    """Generate one V3 mixed batch with balanced raw evidence combinations."""

    if groups % 4:
        raise ValueError("V3 batch groups must be divisible by four")
    episodes = _episodes(
        "mixed",
        seed_start=seed_start,
        groups=groups // 4,
        episode_steps=episode_steps,
        mix_index=mix_index,
    )
    return balance_evidence_inputs(episodes)


def raw_evidence_pair_counts(
    episodes: list[ContextEpisode],
) -> dict[tuple[int, int], int]:
    counts = {(input_value, outcome): 0 for input_value in range(2) for outcome in range(2)}
    for episode in episodes:
        for time_index in (0, 1):
            public_input = int(episode.public[time_index, 0])
            outcome = int(episode.outcomes[time_index])
            counts[(public_input, outcome)] += 1
    return counts


def validate_balanced_evidence(episodes: list[ContextEpisode]) -> None:
    counts = raw_evidence_pair_counts(episodes)
    if not all(value > 0 for value in counts.values()):
        raise ValueError("V3 evidence curriculum omitted a raw input/outcome pair")
    if max(counts.values()) != min(counts.values()):
        raise ValueError(f"V3 raw evidence pairs are imbalanced: {counts}")
    if any(not torch.isfinite(episode.public).all() for episode in episodes):
        raise ValueError("V3 evidence curriculum produced non-finite public fields")
