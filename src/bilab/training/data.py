"""Tensor conversion and provenance hashes for public Rule Worlds sequences."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Sequence

import torch

from bilab.environments.rule_worlds import Episode


def episode_tensors(
    episodes: Sequence[Episode], device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    if not episodes:
        raise ValueError("at least one episode is required")
    lengths = {len(episode.steps) for episode in episodes}
    if len(lengths) != 1:
        raise ValueError("episodes in one tensor batch must have equal length")
    observations = torch.tensor(
        [[step.observation.values() for step in episode.steps] for episode in episodes],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [[step.target for step in episode.steps] for episode in episodes],
        dtype=torch.long,
        device=device,
    )
    return observations, targets


def shuffled_batches[T](items: Sequence[T], batch_size: int, seed: int) -> Iterator[list[T]]:
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [items[index] for index in indices[start : start + batch_size]]


def episode_fingerprint(episodes: Sequence[Episode]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.category.encode())
        digest.update(episode.generation_seed.to_bytes(8, "little", signed=False))
        for step in episode.steps:
            digest.update(bytes(step.observation.values()))
            digest.update(bytes((step.target,)))
    return digest.hexdigest()
