"""Parameter-matched no-memory and exact-byte episodic neural baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from bilab.models.common import (
    ActiveCapacityMLP,
    ObservationEncoder,
    SharedThoughtBlock,
    apply_thought_cycles,
)

EPISODIC_RECORD_BYTES = 8
EPISODIC_METADATA_BYTES = 8


@dataclass
class EpisodicState:
    records: torch.Tensor
    write_index: torch.Tensor
    count: torch.Tensor


class NoMemoryModel(nn.Module):
    """A real neural predictor whose output cannot depend on prior steps."""

    persistent_bytes = 0

    def __init__(self, config: dict[str, Any], capacity_width: int) -> None:
        super().__init__()
        environment = config["environment"]
        model = config["model"]
        hidden = int(model["hidden_dim"])
        self.cycles = int(model["thought_cycles"])
        self.observation_encoder = ObservationEncoder(environment, model)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.memory_free_compute = ActiveCapacityMLP(hidden, capacity_width)
        self.thought_block = SharedThoughtBlock(hidden)
        self.output_head = nn.Linear(hidden, int(environment["outcome_classes"]))

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> None:
        del batch_size, device
        return None

    def step(
        self,
        observation: torch.Tensor,
        state: None,
        target: torch.Tensor | None,
        *,
        mode: str = "full",
    ) -> tuple[torch.Tensor, None, dict[str, torch.Tensor]]:
        del state, target, mode
        observation_hidden = self.observation_encoder(observation)
        hidden = self.memory_free_compute(self.input_projection(observation_hidden))
        zero_memory = torch.zeros_like(hidden)
        hidden = apply_thought_cycles(
            self.thought_block, hidden, zero_memory, observation_hidden, self.cycles
        )
        return self.output_head(hidden), None, {}


class EpisodicModel(nn.Module):
    """Neural reader over a deterministic recent-first 4096-byte ring buffer."""

    def __init__(self, config: dict[str, Any], capacity_width: int) -> None:
        super().__init__()
        environment = config["environment"]
        model = config["model"]
        hidden = int(model["hidden_dim"])
        self.persistent_bytes = int(config["workspace_bytes"])
        self.capacity = (self.persistent_bytes - EPISODIC_METADATA_BYTES) // EPISODIC_RECORD_BYTES
        if self.capacity * EPISODIC_RECORD_BYTES + EPISODIC_METADATA_BYTES != self.persistent_bytes:
            raise ValueError("episodic budget must exactly fit records plus metadata")
        self.retrieval_records = min(int(model["episodic_retrieval_records"]), self.capacity)
        self.cycles = int(model["thought_cycles"])
        self.observation_encoder = ObservationEncoder(environment, model)
        self.target_embedding = nn.Embedding(
            int(environment["outcome_classes"]), int(model["outcome_embedding_dim"])
        )
        self.record_encoder = nn.Sequential(
            nn.Linear(hidden + int(model["outcome_embedding_dim"]), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.query = nn.Linear(hidden, hidden)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.episodic_compute = ActiveCapacityMLP(hidden, capacity_width)
        self.thought_block = SharedThoughtBlock(hidden)
        self.output_head = nn.Linear(hidden * 2, int(environment["outcome_classes"]))

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> EpisodicState:
        return EpisodicState(
            records=torch.zeros(
                (batch_size, self.capacity, EPISODIC_RECORD_BYTES),
                dtype=torch.uint8,
                device=device,
            ),
            write_index=torch.zeros(batch_size, dtype=torch.int32, device=device),
            count=torch.zeros(batch_size, dtype=torch.int32, device=device),
        )

    def state_nbytes(self, state: EpisodicState) -> int:
        if state.records.shape[1:] != (self.capacity, EPISODIC_RECORD_BYTES):
            raise ValueError("invalid episodic record shape")
        per_world = (
            state.records[0].numel() * state.records.element_size()
            + state.write_index[0].numel() * state.write_index.element_size()
            + state.count[0].numel() * state.count.element_size()
        )
        return int(per_world)

    def _read(self, observation_hidden: torch.Tensor, state: EpisodicState) -> torch.Tensor:
        batch = observation_hidden.shape[0]
        offsets = torch.arange(self.retrieval_records, device=observation_hidden.device)
        indices = (state.write_index.long().unsqueeze(1) - 1 - offsets) % self.capacity
        gathered = state.records.gather(
            1, indices.unsqueeze(-1).expand(batch, self.retrieval_records, EPISODIC_RECORD_BYTES)
        )
        record_observations = gathered[..., :7].long()
        record_targets = gathered[..., 7].long()
        record_hidden = self.record_encoder(
            torch.cat(
                (
                    self.observation_encoder(record_observations),
                    self.target_embedding(record_targets),
                ),
                dim=-1,
            )
        )
        valid = offsets.unsqueeze(0) < state.count.long().unsqueeze(1)
        scores = (record_hidden * self.query(observation_hidden).unsqueeze(1)).sum(dim=-1)
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1) * valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return (record_hidden * weights.unsqueeze(-1)).sum(dim=1)

    def _write(
        self, observation: torch.Tensor, target: torch.Tensor, state: EpisodicState
    ) -> EpisodicState:
        records = state.records.clone()
        rows = torch.arange(observation.shape[0], device=observation.device)
        payload = torch.cat((observation.to(torch.uint8), target[:, None].to(torch.uint8)), dim=1)
        records[rows, state.write_index.long()] = payload
        return EpisodicState(
            records=records,
            write_index=((state.write_index + 1) % self.capacity).to(torch.int32),
            count=torch.clamp(state.count + 1, max=self.capacity).to(torch.int32),
        )

    def step(
        self,
        observation: torch.Tensor,
        state: EpisodicState,
        target: torch.Tensor | None,
        *,
        mode: str = "full",
    ) -> tuple[torch.Tensor, EpisodicState, dict[str, torch.Tensor]]:
        del mode
        observation_hidden = self.observation_encoder(observation)
        memory_read = self._read(observation_hidden, state)
        hidden = self.input_projection(torch.cat((observation_hidden, memory_read), dim=-1))
        hidden = self.episodic_compute(hidden)
        hidden = apply_thought_cycles(
            self.thought_block, hidden, memory_read, observation_hidden, self.cycles
        )
        logits = self.output_head(torch.cat((hidden, memory_read), dim=-1))
        next_state = self._write(observation, target, state) if target is not None else state
        return logits, next_state, {}
