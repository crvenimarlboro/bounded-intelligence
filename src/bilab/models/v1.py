"""Small bounded-state models for the Cognitive Core v1 adaptation ladder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class V1ModelConfig:
    """Replaceable model dimensions; runtime state is fixed before an episode."""

    hidden_dim: int = 64
    state_dim: int = 4
    thought_cycles: int = 1
    feedback_mode: str = "outcome_only"
    context_count: int = 1
    rule_count: int = 1

    def __post_init__(self) -> None:
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim must be at least eight")
        if self.state_dim < 1:
            raise ValueError("state_dim must be positive")
        if self.thought_cycles < 1:
            raise ValueError("thought_cycles must be positive")
        if self.context_count not in {1, 2, 3, 4}:
            raise ValueError("v1 supports one to four public operation contexts")
        if not 1 <= self.rule_count <= min(2, self.context_count):
            raise ValueError("rule_count must fit the public contexts")
        valid_modes = {
            "outcome_only",
            "detached_error",
            "differentiable_error",
            "surprise",
        }
        if self.feedback_mode not in valid_modes:
            raise ValueError(f"unknown feedback mode: {self.feedback_mode}")


@dataclass(frozen=True)
class StepPrediction:
    logits: torch.Tensor
    hidden: torch.Tensor
    thought_states: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class StateUpdate:
    state: torch.Tensor
    gate: torch.Tensor
    candidate: torch.Tensor
    error_signal: torch.Tensor


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class BinaryObservationEncoder(nn.Module):
    """Encode only current input and the public prior-feedback phase marker."""

    def __init__(self, hidden_dim: int, context_count: int = 1) -> None:
        super().__init__()
        self.context_count = context_count
        self.rule_count = min(2, context_count)
        input_dim = 3 if context_count == 1 else 3 + context_count
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, public: torch.Tensor) -> torch.Tensor:
        expected_fields = 2 if self.context_count == 1 else 3
        if public.ndim != 2 or public.shape[1] != expected_fields:
            raise ValueError(
                f"model input must contain current public fields only: [batch, {expected_fields}]"
            )
        x = public[:, :1].float()
        phase = public[:, -1:].float()
        pieces = [x, 1.0 - x, phase]
        if self.context_count > 1:
            context = F.one_hot(public[:, 1].long(), num_classes=self.context_count).float()
            pieces.append(context)
        return self.network(torch.cat(pieces, dim=-1))


class SharedGatedThought(nn.Module):
    """One gated residual transformation reused for every requested thought cycle."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        combined_dim = hidden_dim * 3
        self.candidate = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gate = nn.Linear(combined_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden: torch.Tensor, state_read: torch.Tensor, observation: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat((hidden, state_read, observation), dim=-1)
        proposal = self.candidate(combined)
        gate = torch.sigmoid(self.gate(combined))
        return self.normalization(hidden + gate * proposal)


class AdaptiveCore(nn.Module):
    """Common reader and shared recurrent computation for trainable float-state cores."""

    family = "abstract"

    def __init__(self, config: V1ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.observation_encoder = BinaryObservationEncoder(hidden_dim, config.context_count)
        self.state_reader = nn.Sequential(
            nn.Linear(config.state_dim, hidden_dim), nn.Tanh(), nn.LayerNorm(hidden_dim)
        )
        self.thought_block = SharedGatedThought(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + config.state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.rule_probe = nn.Linear(config.state_dim, 2**config.rule_count)

    @property
    def persistent_bytes(self) -> int:
        return self.config.state_dim * torch.tensor([], dtype=torch.float32).element_size()

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.config.state_dim, dtype=torch.float32, device=device)

    def predict(
        self, public: torch.Tensor, state: torch.Tensor, *, cycles: int | None = None
    ) -> StepPrediction:
        if state.shape != (public.shape[0], self.config.state_dim):
            raise ValueError("state shape does not match fixed model state dimension")
        observation = self.observation_encoder(public)
        state_read = self.state_reader(state)
        hidden = observation
        thought_states: list[torch.Tensor] = []
        for _ in range(self.config.thought_cycles if cycles is None else cycles):
            hidden = self.thought_block(hidden, state_read, observation)
            thought_states.append(hidden)
        logits = self.output_head(torch.cat((hidden, state), dim=-1))
        return StepPrediction(logits=logits, hidden=hidden, thought_states=tuple(thought_states))

    def probe(self, state: torch.Tensor) -> torch.Tensor:
        return self.rule_probe(state)

    def _error_signal(self, logits: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
        probability = logits.softmax(dim=-1)[:, 1:2]
        target = outcome.float().unsqueeze(1)
        if self.config.feedback_mode == "outcome_only":
            return torch.zeros_like(target)
        if self.config.feedback_mode == "detached_error":
            return (target - probability).detach()
        if self.config.feedback_mode == "differentiable_error":
            return target - probability
        selected = logits.log_softmax(dim=-1).gather(1, outcome.unsqueeze(1))
        return -selected.detach()

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> StateUpdate:
        raise NotImplementedError


class LearnedGRUWriter(nn.Module):
    """Inspectable GRU-style convex state update."""

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        combined_dim = input_dim + state_dim
        self.update_gate = nn.Linear(combined_dim, state_dim)
        self.reset_gate = nn.Linear(combined_dim, state_dim)
        self.candidate = nn.Linear(combined_dim, state_dim)
        nn.init.constant_(self.update_gate.bias, -1.0)

    def forward(self, feedback: torch.Tensor, state: torch.Tensor) -> StateUpdate:
        combined = torch.cat((feedback, state), dim=-1)
        gate = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        candidate_input = torch.cat((feedback, reset * state), dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input))
        updated = (1.0 - gate) * state + gate * candidate
        empty = torch.zeros(state.shape[0], 1, device=state.device, dtype=state.dtype)
        return StateUpdate(updated, gate, candidate, empty)


class GRUStateCore(AdaptiveCore):
    """General learned recurrent writer with no task-specific relation feature."""

    family = "gru"

    def __init__(self, config: V1ModelConfig) -> None:
        super().__init__(config)
        feedback_dim = (2 if config.context_count == 1 else 3) + 4
        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 32),
            nn.Tanh(),
        )
        self.writer = LearnedGRUWriter(32, config.state_dim)

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> StateUpdate:
        error = self._error_signal(prediction.logits, outcome)
        one_hot = F.one_hot(outcome, num_classes=2).float()
        features = torch.cat((public.float(), one_hot, error, 1 - error), dim=-1)
        update = self.writer(self.feedback_encoder(features), state)
        return StateUpdate(update.state, update.gate, update.candidate, error)


class PredictiveStateCore(GRUStateCore):
    """GRU state trained to predict both possible future query outcomes."""

    family = "predictive"

    def __init__(self, config: V1ModelConfig) -> None:
        super().__init__(config)
        self.future_head = nn.Sequential(
            nn.Linear(config.state_dim, 32), nn.SiLU(), nn.Linear(32, 4)
        )

    def predict_future_table(self, state: torch.Tensor) -> torch.Tensor:
        return self.future_head(state).reshape(state.shape[0], 2, 2)


class FactorizedStateCore(AdaptiveCore):
    """Evidence/confidence-biased writer using a public relational feature."""

    family = "factorized"

    def __init__(self, config: V1ModelConfig) -> None:
        super().__init__(config)
        relation_dim = 5 if config.context_count == 1 else 5 + config.context_count
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, 32), nn.Tanh(), nn.Linear(32, config.state_dim), nn.Tanh()
        )
        self.write_gate = nn.Linear(relation_dim + config.state_dim, config.state_dim)
        nn.init.constant_(self.write_gate.bias, -1.0)

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> StateUpdate:
        error = self._error_signal(prediction.logits, outcome)
        x_signed = public[:, :1].float() * 2.0 - 1.0
        y_signed = outcome.float().unsqueeze(1) * 2.0 - 1.0
        relation = x_signed * y_signed
        phase = public[:, -1:]
        feature_parts = [x_signed, y_signed, relation, phase, error]
        context_mask = None
        if self.config.context_count > 1:
            contexts = F.one_hot(public[:, 1].long(), num_classes=self.config.context_count).float()
            feature_parts.append(contexts)
            if (
                self.config.context_count >= 3
                and self.config.rule_count == 2
                and self.config.state_dim == 2
            ):
                context_mask = contexts[:, :2]
            elif self.config.state_dim == self.config.context_count:
                context_mask = contexts
            elif self.config.state_dim == self.config.context_count * 2:
                context_mask = torch.cat((contexts, contexts), dim=-1)
        features = torch.cat(feature_parts, dim=-1)
        candidate = self.relation_encoder(features)
        gate = torch.sigmoid(self.write_gate(torch.cat((features, state), dim=-1)))
        if context_mask is not None:
            gate = gate * context_mask
        updated = (1.0 - gate) * state + gate * candidate
        return StateUpdate(updated, gate, candidate, error)


class NoMemoryControl(nn.Module):
    """Current-observation-only neural control with no cross-step state."""

    family = "no_memory"
    persistent_bytes = 0

    def __init__(
        self, hidden_dim: int = 64, thought_cycles: int = 1, context_count: int = 1
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.thought_cycles = thought_cycles
        self.context_count = context_count
        self.rule_count = min(2, context_count)
        self.observation_encoder = BinaryObservationEncoder(hidden_dim, context_count)
        self.capacity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.thought_block = SharedGatedThought(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.empty(batch_size, 0, device=device)

    def predict(
        self, public: torch.Tensor, state: torch.Tensor, *, cycles: int | None = None
    ) -> StepPrediction:
        if state.shape != (public.shape[0], 0):
            raise ValueError("no-memory control state must be empty")
        observation = self.observation_encoder(public)
        hidden = observation + self.capacity(observation)
        zero_read = torch.zeros_like(hidden)
        thought_states: list[torch.Tensor] = []
        for _ in range(self.thought_cycles if cycles is None else cycles):
            hidden = self.thought_block(hidden, zero_read, observation)
            thought_states.append(hidden)
        return StepPrediction(self.output_head(hidden), hidden, tuple(thought_states))

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> StateUpdate:
        del public, outcome, prediction
        empty = torch.empty_like(state)
        return StateUpdate(state, empty, empty, torch.zeros(state.shape[0], 1, device=state.device))


class EpisodicControl(nn.Module):
    """Learned reader over an exact-byte deterministic ring of public event records."""

    family = "episodic"

    def __init__(
        self, *, budget_bytes: int = 16, hidden_dim: int = 64, context_count: int = 1
    ) -> None:
        super().__init__()
        if not 2 <= budget_bytes <= 16:
            raise ValueError("v1 packed episodic budget must be between 2 and 16 bytes")
        self.persistent_bytes = budget_bytes
        self.record_capacity = budget_bytes - 1
        self.hidden_dim = hidden_dim
        self.context_count = context_count
        self.rule_count = min(2, context_count)
        self.observation_encoder = BinaryObservationEncoder(hidden_dim, context_count)
        decoded_fields = 4 if context_count == 1 else 4 + context_count
        decoded_dim = self.record_capacity * decoded_fields + 2
        self.memory_reader = nn.Sequential(
            nn.Linear(decoded_dim, hidden_dim), nn.Tanh(), nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_head = nn.Linear(hidden_dim, 2)

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.persistent_bytes, dtype=torch.uint8, device=device)

    def _decode(self, state: torch.Tensor) -> torch.Tensor:
        header = state[:, 0].long()
        records = state[:, 1:].long()
        fields = [
            (records & 1).float(),
            ((records >> 1) & 1).float(),
            ((records >> 2) & 1).float(),
            ((records >> 3) & 1).float(),
        ]
        if self.context_count > 1:
            context_values = (records >> 4) & 3
            context_one_hot = F.one_hot(context_values, num_classes=self.context_count).float()
            fields.extend(context_one_hot.unbind(dim=-1))
        decoded = torch.stack(fields, dim=-1).flatten(1)
        denominator = max(1, self.record_capacity - 1)
        next_index = (header & 15).float().unsqueeze(1) / denominator
        count = ((header >> 4) & 15).float().unsqueeze(1) / self.record_capacity
        return torch.cat((decoded, next_index, count), dim=-1)

    def predict(self, public: torch.Tensor, state: torch.Tensor, **_: object) -> StepPrediction:
        if state.dtype != torch.uint8 or state.shape != (public.shape[0], self.persistent_bytes):
            raise ValueError("episodic state violates its canonical byte representation")
        observation = self.observation_encoder(public)
        memory = self.memory_reader(self._decode(state))
        hidden = self.fusion(torch.cat((observation, memory), dim=-1))
        return StepPrediction(self.output_head(hidden), hidden, (hidden,))

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> StateUpdate:
        del prediction
        updated = state.clone()
        for batch_index in range(state.shape[0]):
            if self.context_count == 4 and int(public[batch_index, 1].item()) == 3:
                continue
            header = int(state[batch_index, 0].item())
            next_index = header & 15
            count = (header >> 4) & 15
            record = (
                int(public[batch_index, 0].item())
                | (int(outcome[batch_index].item()) << 1)
                | (int(public[batch_index, -1].item()) << 2)
                | 8
            )
            if self.context_count > 1:
                record |= int(public[batch_index, 1].item()) << 4
            updated[batch_index, 1 + next_index] = record
            new_next = (next_index + 1) % self.record_capacity
            new_count = min(self.record_capacity, count + 1)
            updated[batch_index, 0] = (new_count << 4) | new_next
        empty = torch.empty(state.shape[0], 0, device=state.device)
        error = torch.zeros(state.shape[0], 1, device=state.device)
        return StateUpdate(updated, empty, empty, error)

    def canonical_bytes(self, state: torch.Tensor) -> bytes:
        if state.shape[0] != 1:
            raise ValueError("serialize one world's episodic state at a time")
        return bytes(state[0].cpu().tolist())


def build_v1_model(family: str, config: V1ModelConfig, *, budget_bytes: int = 16) -> nn.Module:
    if family == "gru":
        return GRUStateCore(config)
    if family == "predictive":
        return PredictiveStateCore(config)
    if family == "factorized":
        return FactorizedStateCore(config)
    if family == "no_memory":
        return NoMemoryControl(config.hidden_dim, config.thought_cycles, config.context_count)
    if family == "episodic":
        return EpisodicControl(
            budget_bytes=budget_bytes,
            hidden_dim=config.hidden_dim,
            context_count=config.context_count,
        )
    raise ValueError(f"unknown v1 model family: {family}")
