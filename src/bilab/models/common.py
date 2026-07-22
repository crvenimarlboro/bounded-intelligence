"""Shared observation and recurrent-computation modules."""

from __future__ import annotations

import torch
from torch import nn


class ObservationEncoder(nn.Module):
    """Encode only the seven public Rule Worlds fields."""

    def __init__(self, environment: dict[str, int], model: dict[str, int]) -> None:
        super().__init__()
        self.entity_count = int(environment["entity_count"])
        self.operation_count = int(environment["operation_count"])
        self.symbol = nn.Embedding(self.entity_count + 1, int(model["symbol_embedding_dim"]))
        self.operation = nn.Embedding(
            self.operation_count + 1, int(model["operation_embedding_dim"])
        )
        self.kind = nn.Embedding(4, int(model["kind_embedding_dim"]))
        self.noise = nn.Embedding(
            int(environment["noise_values"]), int(model["noise_embedding_dim"])
        )
        self.marker = nn.Embedding(2, int(model["marker_embedding_dim"]))
        input_dim = (
            2 * int(model["symbol_embedding_dim"])
            + 2 * int(model["operation_embedding_dim"])
            + int(model["kind_embedding_dim"])
            + int(model["noise_embedding_dim"])
            + int(model["marker_embedding_dim"])
        )
        hidden = int(model["hidden_dim"])
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 7:
            raise ValueError("Rule Worlds observations must have exactly seven public fields")
        observation = observation.long()
        pieces = (
            self.symbol(observation[..., 0]),
            self.symbol(observation[..., 1]),
            self.operation(observation[..., 2]),
            self.operation(observation[..., 3]),
            self.kind(observation[..., 4]),
            self.noise(observation[..., 5]),
            self.marker(observation[..., 6]),
        )
        return self.projection(torch.cat(pieces, dim=-1))


class SharedThoughtBlock(nn.Module):
    """One gated residual block reused for every internal thought cycle."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        combined = hidden_dim * 3
        self.candidate = nn.Sequential(
            nn.Linear(combined, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gate = nn.Linear(combined, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden: torch.Tensor, memory_read: torch.Tensor, observation: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat((hidden, memory_read, observation), dim=-1)
        proposal = self.candidate(combined)
        gate = torch.sigmoid(self.gate(combined))
        return self.normalization(hidden + gate * proposal)


class ActiveCapacityMLP(nn.Module):
    """An active parameter/compute matcher for non-core neural baselines."""

    def __init__(self, hidden_dim: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


def apply_thought_cycles(
    block: SharedThoughtBlock,
    hidden: torch.Tensor,
    memory_read: torch.Tensor,
    observation: torch.Tensor,
    cycles: int,
) -> torch.Tensor:
    if cycles < 1:
        raise ValueError("thought cycles must be positive")
    for _ in range(cycles):
        hidden = block(hidden, memory_read, observation)
    return hidden
