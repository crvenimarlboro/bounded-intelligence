"""Cognitive Core v0 with bounded latent state and feedback-driven updates."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from bilab.models.common import ObservationEncoder, SharedThoughtBlock, apply_thought_cycles

CoreMode = Literal[
    "full",
    "workspace_disabled",
    "workspace_frozen",
    "no_prediction_error",
    "recurrence_k1",
]


class CognitiveCore(nn.Module):
    """Fixed-size workspace read, recurrent thought, prediction, then gated update."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        environment = config["environment"]
        model = config["model"]
        workspace_bytes = int(config["workspace_bytes"])
        if workspace_bytes % 4:
            raise ValueError("float32 workspace budget must be divisible by four")
        self.workspace_dim = workspace_bytes // 4
        self.workspace_bytes = workspace_bytes
        self.hidden_dim = int(model["hidden_dim"])
        self.default_cycles = int(model["thought_cycles"])
        self.outcome_classes = int(environment["outcome_classes"])

        self.observation_encoder = ObservationEncoder(environment, model)
        self.workspace_reader = nn.Sequential(
            nn.Linear(self.workspace_dim, self.hidden_dim), nn.Tanh()
        )
        self.input_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.thought_block = SharedThoughtBlock(self.hidden_dim)
        self.output_head = nn.Linear(self.hidden_dim * 2, self.outcome_classes)
        self.outcome_embedding = nn.Embedding(
            self.outcome_classes, int(model["outcome_embedding_dim"])
        )
        self.error_encoder = nn.Sequential(
            nn.Linear(self.outcome_classes, int(model["error_embedding_dim"])), nn.Tanh()
        )
        update_dim = (
            self.hidden_dim * 2
            + int(model["outcome_embedding_dim"])
            + int(model["error_embedding_dim"])
        )
        self.workspace_candidate = nn.Linear(update_dim, self.workspace_dim)
        self.workspace_gate = nn.Linear(update_dim, self.workspace_dim)
        self.initial_workspace = nn.Parameter(torch.zeros(self.workspace_dim, dtype=torch.float32))

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return self.initial_workspace.to(device).unsqueeze(0).expand(batch_size, -1).clone()

    def state_nbytes(self, state: torch.Tensor) -> int:
        if state.ndim != 2 or state.shape[1] != self.workspace_dim:
            raise ValueError("invalid cognitive workspace shape")
        return state.shape[1] * state.element_size()

    def step(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        target: torch.Tensor | None,
        *,
        mode: CoreMode = "full",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if mode not in {
            "full",
            "workspace_disabled",
            "workspace_frozen",
            "no_prediction_error",
            "recurrence_k1",
        }:
            raise ValueError(f"unknown core mode: {mode}")
        state_for_read = torch.zeros_like(state) if mode == "workspace_disabled" else state
        observation_hidden = self.observation_encoder(observation)
        memory_read = self.workspace_reader(state_for_read)
        hidden = self.input_projection(torch.cat((observation_hidden, memory_read), dim=-1))
        cycles = 1 if mode == "recurrence_k1" else self.default_cycles
        hidden = apply_thought_cycles(
            self.thought_block, hidden, memory_read, observation_hidden, cycles
        )
        logits = self.output_head(torch.cat((hidden, memory_read), dim=-1))

        if target is None or mode in {"workspace_disabled", "workspace_frozen"}:
            next_state = torch.zeros_like(state) if mode == "workspace_disabled" else state
            return logits, next_state, {"update_gate_mean": torch.zeros((), device=logits.device)}

        probabilities = torch.softmax(logits, dim=-1)
        desired = torch.nn.functional.one_hot(target.long(), self.outcome_classes).float()
        prediction_error = desired - probabilities.detach()
        if mode == "no_prediction_error":
            prediction_error = torch.zeros_like(prediction_error)
        error_hidden = self.error_encoder(prediction_error)
        outcome_hidden = self.outcome_embedding(target.long())
        update_input = torch.cat((hidden, observation_hidden, outcome_hidden, error_hidden), dim=-1)
        candidate = torch.tanh(self.workspace_candidate(update_input))
        gate = torch.sigmoid(self.workspace_gate(update_input))
        next_state = state + gate * (candidate - state)
        return logits, next_state, {"update_gate_mean": gate.mean()}
