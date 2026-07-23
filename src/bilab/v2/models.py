"""Generic eight-byte bounded-state candidates for Cognitive Core v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bilab.models.v1 import (
    BinaryObservationEncoder,
    EpisodicControl,
    FactorizedStateCore,
    NoMemoryControl,
    SharedGatedThought,
    StepPrediction,
    V1ModelConfig,
)

RAW_WRITER_INPUT_CONTRACT: dict[str, Any] = {
    "schema_version": "1.0",
    "interface": "cognitive-core-v2-raw-feedback",
    "fields": [
        {
            "name": "current_input",
            "origin": "public observation",
            "observable": "before prediction",
            "dimensions": 1,
            "derived_by": "none",
        },
        {
            "name": "operation_one_hot",
            "origin": "public operation symbol",
            "observable": "before prediction",
            "dimensions": 4,
            "derived_by": "generic categorical encoding inside model",
        },
        {
            "name": "phase",
            "origin": "public episode phase marker",
            "observable": "before prediction",
            "dimensions": 1,
            "derived_by": "none",
        },
        {
            "name": "observed_outcome_one_hot",
            "origin": "public outcome feedback",
            "observable": "only after current prediction",
            "dimensions": 2,
            "derived_by": "generic categorical encoding inside model",
        },
    ],
    "prohibited": [
        "input_xor_outcome",
        "input_equals_outcome",
        "signed_rule_relation",
        "hidden_rule",
        "correct_state_slot",
        "write_or_no_write_target",
        "future_outcome",
        "generation_seed",
    ],
}

RAW_WRITER_FIELD_NAMES = (
    "current_input",
    "operation_0",
    "operation_1",
    "operation_2",
    "operation_3",
    "phase",
    "outcome_0",
    "outcome_1",
)


@dataclass(frozen=True)
class V2ModelConfig:
    family: str
    hidden_dim: int = 64
    state_dim: int = 2
    context_count: int = 4
    rule_count: int = 2
    thought_cycles: int = 1
    quantization_bits: int = 32

    def __post_init__(self) -> None:
        valid = {
            "raw_fixed",
            "relation_router",
            "raw_router",
            "raw_gru",
            "bilinear_fixed",
            "bilinear_router",
            "v1_scaffolded",
            "no_memory",
            "episodic",
        }
        if self.family not in valid:
            raise ValueError(f"unknown v2 family: {self.family}")
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim must be at least eight")
        if self.state_dim != 2 and self.family not in {"no_memory", "episodic"}:
            raise ValueError("initial v2 float candidates use exactly two state values")
        if self.context_count != 4:
            raise ValueError("v2 scaffold-removal ladder uses four public contexts")
        if self.rule_count != 2:
            raise ValueError("v2 scaffold-removal ladder has exactly two hidden rules")
        if self.thought_cycles != 1:
            raise ValueError("v2 begins with the v1-supported K=1 recurrence")
        if self.quantization_bits not in {1, 2, 4, 8, 32}:
            raise ValueError("quantization_bits must be 1, 2, 4, 8, or 32")


@dataclass(frozen=True)
class WriterTrace:
    writer_input: torch.Tensor
    field_names: tuple[str, ...]
    derived_fields: tuple[str, ...]
    routing_mode: str
    route: torch.Tensor
    write_strength: torch.Tensor
    writer_latent: torch.Tensor


@dataclass(frozen=True)
class V2StateUpdate:
    state: torch.Tensor
    gate: torch.Tensor
    candidate: torch.Tensor
    error_signal: torch.Tensor
    trace: WriterTrace


def count_v2_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def raw_writer_input(public: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """Encode raw public fields without computing their sufficient statistic."""

    if public.ndim != 2 or public.shape[1] != 3:
        raise ValueError("v2 public input must be [input, operation, phase]")
    if outcome.shape != (public.shape[0],):
        raise ValueError("v2 outcome must contain one public class per batch element")
    if not torch.all((public[:, 0] == 0) | (public[:, 0] == 1)):
        raise ValueError("public input must be binary")
    if not torch.all((public[:, 2] == 0) | (public[:, 2] == 1)):
        raise ValueError("public phase must be binary")
    operations = public[:, 1].long()
    if not torch.all((operations >= 0) & (operations < 4)):
        raise ValueError("public operation symbol is outside the declared vocabulary")
    if not torch.all((outcome == 0) | (outcome == 1)):
        raise ValueError("public outcome must be binary")
    return torch.cat(
        (
            public[:, :1].float(),
            F.one_hot(operations, num_classes=4).float(),
            public[:, 2:3].float(),
            F.one_hot(outcome, num_classes=2).float(),
        ),
        dim=-1,
    )


def validate_raw_writer_trace(
    trace: WriterTrace, public: torch.Tensor, outcome: torch.Tensor
) -> None:
    expected = raw_writer_input(public, outcome)
    if trace.field_names != RAW_WRITER_FIELD_NAMES:
        raise ValueError("writer field names violate the frozen raw-input contract")
    if trace.derived_fields:
        raise ValueError(f"prohibited derived writer fields: {trace.derived_fields}")
    if not torch.equal(trace.writer_input, expected):
        raise ValueError("captured writer input differs from raw public fields")


class BaseV2Core(nn.Module):
    """Shared current-observation reader and K=1 thought computation."""

    family = "abstract"
    relation_scaffold = False
    fixed_routing = False

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rule_count = 2
        self.observation_encoder = BinaryObservationEncoder(config.hidden_dim, config.context_count)
        self.state_reader = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.thought_block = SharedGatedThought(config.hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.state_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )
        self.rule_probe = nn.Linear(config.state_dim, 4)
        self.relation_probe = nn.Linear(32, 2)
        self.routing_intervention = "learned"
        self._intervention_step = 0
        self.quantization_events = 0

    @property
    def persistent_bytes(self) -> int:
        bits = self.config.state_dim * self.config.quantization_bits
        return math.ceil(bits / 8)

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        self._intervention_step = 0
        self.quantization_events = 0
        return torch.zeros(batch_size, self.config.state_dim, dtype=torch.float32, device=device)

    def predict(
        self, public: torch.Tensor, state: torch.Tensor, *, cycles: int | None = None
    ) -> StepPrediction:
        if state.shape != (public.shape[0], self.config.state_dim):
            raise ValueError("v2 state shape violates the fixed-size contract")
        state_float = state.float()
        observation = self.observation_encoder(public)
        state_read = self.state_reader(state_float)
        hidden = observation
        thought_states: list[torch.Tensor] = []
        count = self.config.thought_cycles if cycles is None else cycles
        for _ in range(count):
            hidden = self.thought_block(hidden, state_read, observation)
            thought_states.append(hidden)
        logits = self.output_head(torch.cat((hidden, state_float), dim=-1))
        return StepPrediction(logits, hidden, tuple(thought_states))

    def probe(self, state: torch.Tensor) -> torch.Tensor:
        return self.rule_probe(state.float())

    def _quantize(self, state: torch.Tensor) -> torch.Tensor:
        bits = self.config.quantization_bits
        if bits == 32:
            return state
        levels = 2**bits
        clipped = state.clamp(-1.0, 1.0)
        if bits == 1:
            quantized = torch.where(
                clipped >= 0, torch.ones_like(clipped), -torch.ones_like(clipped)
            )
        else:
            quantized = torch.round((clipped + 1.0) * (levels - 1) / 2.0) * 2.0 / (levels - 1) - 1.0
        self.quantization_events += 1
        if self.training:
            return clipped + (quantized - clipped).detach()
        return quantized

    def canonical_state_bytes(self, state: torch.Tensor) -> bytes:
        """Serialize one world's declared quantized state with no hidden metadata."""

        if state.shape != (1, self.config.state_dim):
            raise ValueError("serialize exactly one fixed-size v2 state")
        bits = self.config.quantization_bits
        if bits == 32:
            return state.detach().cpu().to(torch.float32).numpy().tobytes()
        levels = 2**bits
        clipped = state.detach().cpu().clamp(-1, 1)
        codes = torch.round((clipped + 1) * (levels - 1) / 2).to(torch.int64).flatten()
        packed = 0
        offset = 0
        result = bytearray()
        for code in codes.tolist():
            packed |= int(code) << offset
            offset += bits
            while offset >= 8:
                result.append(packed & 255)
                packed >>= 8
                offset -= 8
        if offset:
            result.append(packed & 255)
        return bytes(result)

    def set_routing_intervention(self, intervention: str) -> None:
        valid = {
            "learned",
            "uniform",
            "random",
            "swapped",
            "writer_disabled",
        }
        if intervention not in valid:
            raise ValueError(f"unknown v2 routing intervention: {intervention}")
        self.routing_intervention = intervention
        self._intervention_step = 0

    def _intervene_route(
        self, route: torch.Tensor, strength: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.routing_intervention == "uniform":
            route = torch.full_like(route, 1.0 / route.shape[1])
        elif self.routing_intervention == "random":
            indices = (
                torch.arange(len(route), device=route.device) + self._intervention_step
            ) % route.shape[1]
            route = F.one_hot(indices, num_classes=route.shape[1]).float()
        elif self.routing_intervention == "swapped":
            route = route.flip(1)
        elif self.routing_intervention == "writer_disabled":
            strength = torch.zeros_like(strength)
        self._intervention_step += 1
        return route, strength


class RawFixedRoutingCore(BaseV2Core):
    """V2A: learn the relation from raw fields while retaining v1's fixed route."""

    family = "raw_fixed"
    fixed_routing = True

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__(config)
        self.writer_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 32),
            nn.Tanh(),
        )
        self.writer_candidate = nn.Linear(32, config.state_dim)
        self.writer_gate = nn.Linear(32 + config.state_dim, config.state_dim)
        nn.init.constant_(self.writer_gate.bias, -1.0)

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> V2StateUpdate:
        del prediction
        writer_input = raw_writer_input(public, outcome)
        latent = self.writer_encoder(writer_input)
        candidate = torch.tanh(self.writer_candidate(latent))
        learned_gate = torch.sigmoid(self.writer_gate(torch.cat((latent, state.float()), dim=-1)))
        contexts = F.one_hot(public[:, 1].long(), num_classes=4).float()
        route = contexts[:, :2]
        strength = torch.ones(len(public), 1, device=public.device)
        gate = learned_gate * route
        updated = self._quantize((1.0 - gate) * state.float() + gate * candidate)
        return V2StateUpdate(
            updated,
            gate,
            candidate,
            torch.zeros(len(public), 1, device=public.device),
            WriterTrace(
                writer_input,
                RAW_WRITER_FIELD_NAMES,
                (),
                "fixed_context_mask",
                route,
                strength,
                latent,
            ),
        )


class _SoftRouterCore(BaseV2Core):
    """Common learned address/write controller; subclasses choose value evidence."""

    def __init__(self, config: V2ModelConfig, *, relation_scaffold: bool) -> None:
        super().__init__(config)
        self.relation_scaffold = relation_scaffold
        value_input_dim = 9 if relation_scaffold else 8
        self.writer_encoder = nn.Sequential(
            nn.Linear(value_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 32),
            nn.Tanh(),
        )
        self.value_candidate = nn.Sequential(
            nn.Linear(32 + config.state_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        controller_dim = 8 + config.state_dim
        self.router = nn.Sequential(
            nn.Linear(controller_dim, 32),
            nn.Tanh(),
            nn.Linear(32, config.state_dim),
        )
        self.write_controller = nn.Sequential(
            nn.Linear(controller_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.write_controller[-1].bias, -1.0)

    def _value_input(
        self, writer_input: torch.Tensor, public: torch.Tensor, outcome: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        del public, outcome
        return writer_input, ()

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> V2StateUpdate:
        del prediction
        writer_input = raw_writer_input(public, outcome)
        value_input, derived = self._value_input(writer_input, public, outcome)
        latent = self.writer_encoder(value_input)
        controller_input = torch.cat((writer_input, state.float()), dim=-1)
        route = torch.softmax(self.router(controller_input), dim=-1)
        strength = torch.sigmoid(self.write_controller(controller_input))
        route, strength = self._intervene_route(route, strength)
        candidate_scalar = self.value_candidate(torch.cat((latent, state.float()), dim=-1))
        candidate = candidate_scalar.expand_as(state)
        gate = route * strength
        updated = self._quantize((1.0 - gate) * state.float() + gate * candidate)
        return V2StateUpdate(
            updated,
            gate,
            candidate,
            torch.zeros(len(public), 1, device=public.device),
            WriterTrace(
                writer_input,
                RAW_WRITER_FIELD_NAMES,
                derived,
                "learned_soft_router",
                route,
                strength,
                latent,
            ),
        )


class RelationRouterCore(_SoftRouterCore):
    """V2B diagnostic: relation supplied, but all addressing and non-write behavior learned."""

    family = "relation_router"
    relation_scaffold = True

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__(config, relation_scaffold=True)

    def _value_input(
        self, writer_input: torch.Tensor, public: torch.Tensor, outcome: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        x_signed = public[:, :1].float() * 2.0 - 1.0
        y_signed = outcome.float().unsqueeze(1) * 2.0 - 1.0
        relation = x_signed * y_signed
        return torch.cat((writer_input, relation), dim=-1), ("signed_rule_relation",)


class RawRouterCore(_SoftRouterCore):
    """V2C: relation and routing are both learned from raw public fields."""

    family = "raw_router"

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__(config, relation_scaffold=False)


class LearnedBilinearEncoder(nn.Module):
    """Learn raw input/outcome interactions without an external rule transform."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, 16)
        self.outcome_projection = nn.Linear(2, 16)
        self.public_projection = nn.Linear(8, 16)
        self.output = nn.Sequential(
            nn.Linear(48, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
        )

    def forward(self, writer_input: torch.Tensor) -> torch.Tensor:
        input_features = torch.tanh(self.input_projection(writer_input[:, :1]))
        outcome_features = torch.tanh(self.outcome_projection(writer_input[:, 6:8]))
        interaction = input_features * outcome_features
        public_features = torch.tanh(self.public_projection(writer_input))
        return self.output(
            torch.cat((input_features, outcome_features, interaction + public_features), dim=-1)
        )


class BilinearFixedRoutingCore(_SoftRouterCore):
    """V2A revision: learned multiplicative relation features with fixed routing."""

    family = "bilinear_fixed"
    fixed_routing = True

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__(config, relation_scaffold=False)
        self.writer_encoder = LearnedBilinearEncoder(config.hidden_dim)

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> V2StateUpdate:
        del prediction
        writer_input = raw_writer_input(public, outcome)
        latent = self.writer_encoder(writer_input)
        controller_input = torch.cat((writer_input, state.float()), dim=-1)
        strength = torch.sigmoid(self.write_controller(controller_input))
        route = F.one_hot(public[:, 1].long(), num_classes=4).float()[:, :2]
        candidate_scalar = self.value_candidate(torch.cat((latent, state.float()), dim=-1))
        candidate = candidate_scalar.expand_as(state)
        gate = route * strength
        updated = self._quantize((1.0 - gate) * state.float() + gate * candidate)
        return V2StateUpdate(
            updated,
            gate,
            candidate,
            torch.zeros(len(public), 1, device=public.device),
            WriterTrace(
                writer_input,
                RAW_WRITER_FIELD_NAMES,
                (),
                "fixed_context_mask",
                route,
                strength,
                latent,
            ),
        )


class BilinearRouterCore(_SoftRouterCore):
    """V2C revision: learned raw interactions plus learned addressing."""

    family = "bilinear_router"

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__(config, relation_scaffold=False)
        self.writer_encoder = LearnedBilinearEncoder(config.hidden_dim)


class RawGRUCore(BaseV2Core):
    """V2C simplicity control: dense raw-feedback GRU writer with no slot controller."""

    family = "raw_gru"

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__(config)
        self.writer_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 32),
            nn.Tanh(),
        )
        combined_dim = 32 + config.state_dim
        self.update_gate = nn.Linear(combined_dim, config.state_dim)
        self.reset_gate = nn.Linear(combined_dim, config.state_dim)
        self.writer_candidate = nn.Linear(combined_dim, config.state_dim)
        nn.init.constant_(self.update_gate.bias, -1.0)

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> V2StateUpdate:
        del prediction
        writer_input = raw_writer_input(public, outcome)
        latent = self.writer_encoder(writer_input)
        combined = torch.cat((latent, state.float()), dim=-1)
        gate = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        candidate = torch.tanh(
            self.writer_candidate(torch.cat((latent, reset * state.float()), dim=-1))
        )
        strength = gate.mean(dim=-1, keepdim=True)
        route = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        route, strength = self._intervene_route(route, strength)
        effective_gate = route * strength * self.config.state_dim
        effective_gate = effective_gate.clamp(0.0, 1.0)
        updated = self._quantize(
            (1.0 - effective_gate) * state.float() + effective_gate * candidate
        )
        return V2StateUpdate(
            updated,
            effective_gate,
            candidate,
            torch.zeros(len(public), 1, device=public.device),
            WriterTrace(
                writer_input,
                RAW_WRITER_FIELD_NAMES,
                (),
                "dense_gru_gate",
                route,
                strength,
                latent,
            ),
        )


def build_v2_model(config: V2ModelConfig, *, budget_bytes: int = 8) -> nn.Module:
    if config.family == "raw_fixed":
        return RawFixedRoutingCore(config)
    if config.family == "relation_router":
        return RelationRouterCore(config)
    if config.family == "raw_router":
        return RawRouterCore(config)
    if config.family == "raw_gru":
        return RawGRUCore(config)
    if config.family == "bilinear_fixed":
        return BilinearFixedRoutingCore(config)
    if config.family == "bilinear_router":
        return BilinearRouterCore(config)
    if config.family == "v1_scaffolded":
        return FactorizedStateCore(
            V1ModelConfig(
                hidden_dim=config.hidden_dim,
                state_dim=config.state_dim,
                thought_cycles=config.thought_cycles,
                context_count=config.context_count,
                rule_count=config.rule_count,
            )
        )
    if config.family == "no_memory":
        return NoMemoryControl(config.hidden_dim, config.thought_cycles, config.context_count)
    if config.family == "episodic":
        return EpisodicControl(
            budget_bytes=budget_bytes,
            hidden_dim=config.hidden_dim,
            context_count=config.context_count,
        )
    raise ValueError(f"unknown v2 family: {config.family}")
