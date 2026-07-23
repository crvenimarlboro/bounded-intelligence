"""Bounded V3 mechanisms for raw relation discovery and exact preservation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bilab.models.v1 import StepPrediction
from bilab.v2.models import (
    RAW_WRITER_FIELD_NAMES,
    BaseV2Core,
    V2ModelConfig,
    raw_writer_input,
)

V3_RAW_INPUT_CONTRACT: dict[str, Any] = {
    "schema_version": "1.0",
    "interface": "cognitive-core-v3-raw-public-feedback",
    "fields": (
        "current_input",
        "operation_one_hot",
        "phase",
        "observed_outcome_one_hot",
        "previous_state",
    ),
    "prohibited": (
        "input_xor_outcome",
        "input_equals_outcome",
        "signed_rule_relation",
        "hidden_rule",
        "correct_state_slot",
        "semantic_write_or_no_write_target",
        "future_observation",
        "generation_seed",
        "complete_history",
        "probe_output",
        "research_telemetry",
    ),
}


@dataclass(frozen=True)
class V3ModelConfig:
    family: str
    hidden_dim: int = 64
    state_dim: int = 2
    thought_cycles: int = 1

    def __post_init__(self) -> None:
        valid = {
            "raw_bilinear_overwrite",
            "raw_discrete_overwrite",
            "relation_hard_skip",
            "relation_hard_skip_softtrain",
            "relation_attractor",
            "relation_attractor_softtrain",
            "raw_hard_router",
            "raw_discrete_router",
        }
        if self.family not in valid:
            raise ValueError(f"unknown v3 family: {self.family}")
        if self.hidden_dim < 8:
            raise ValueError("v3 hidden_dim must be at least eight")
        if self.state_dim != 2:
            raise ValueError("v3 uses exactly two float32 state values")
        if self.thought_cycles != 1:
            raise ValueError("v3 retains the supported K=1 thought cycle")


@dataclass(frozen=True)
class V3WriterTrace:
    writer_input: torch.Tensor
    previous_state: torch.Tensor
    field_names: tuple[str, ...]
    derived_fields: tuple[str, ...]
    routing_mode: str
    route: torch.Tensor
    write_probability: torch.Tensor
    write_strength: torch.Tensor
    writer_latent: torch.Tensor
    exact_skip: torch.Tensor
    code_indices: torch.Tensor | None


@dataclass(frozen=True)
class V3StateUpdate:
    state: torch.Tensor
    gate: torch.Tensor
    candidate: torch.Tensor
    error_signal: torch.Tensor
    trace: V3WriterTrace


def count_v3_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _hard_binary(probability: torch.Tensor, *, training: bool) -> torch.Tensor:
    hard = (probability >= 0.5).to(probability.dtype)
    if training:
        return probability + (hard - probability).detach()
    return hard


def _hard_categorical(probability: torch.Tensor, *, training: bool) -> torch.Tensor:
    hard = F.one_hot(probability.argmax(dim=-1), num_classes=probability.shape[-1]).to(
        probability.dtype
    )
    if training:
        return probability + (hard - probability).detach()
    return hard


class LearnedRawRelationEncoder(nn.Module):
    """Generic learned input/outcome interaction with no external sufficient statistic."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, 16)
        self.outcome_projection = nn.Linear(2, 16)
        self.context_projection = nn.Linear(5, 16)
        self.bilinear = nn.Bilinear(2, 2, 16)
        self.output = nn.Sequential(
            nn.Linear(80, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
        )

    def forward(self, writer_input: torch.Tensor) -> torch.Tensor:
        input_one_hot = torch.cat(
            (1.0 - writer_input[:, :1], writer_input[:, :1]),
            dim=-1,
        )
        input_features = torch.tanh(self.input_projection(input_one_hot))
        outcome_features = torch.tanh(self.outcome_projection(writer_input[:, 6:8]))
        context_features = torch.tanh(self.context_projection(writer_input[:, 1:6]))
        interaction = input_features * outcome_features
        bilinear = torch.tanh(self.bilinear(input_one_hot, writer_input[:, 6:8]))
        return self.output(
            torch.cat(
                (
                    input_features,
                    outcome_features,
                    interaction,
                    bilinear,
                    context_features,
                ),
                dim=-1,
            )
        )


class BaseV3Core(BaseV2Core):
    """V2-compatible reader plus V3 trace and causal-intervention controls."""

    family = "abstract"
    stage = "abstract"
    relation_scaffold = False
    fixed_routing = False
    hard_preservation = False

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__(
            V2ModelConfig(
                family="raw_router",
                hidden_dim=config.hidden_dim,
                state_dim=config.state_dim,
                thought_cycles=config.thought_cycles,
            )
        )
        self.v3_config = config
        self.relation_intervention = "normal"

    def set_relation_intervention(self, intervention: str) -> None:
        if intervention not in {"normal", "zero", "shuffle", "negate"}:
            raise ValueError(f"unknown relation-path intervention: {intervention}")
        self.relation_intervention = intervention

    def _intervene_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.relation_intervention == "zero":
            return torch.zeros_like(latent)
        if self.relation_intervention == "shuffle":
            return latent.roll(1, dims=0)
        if self.relation_intervention == "negate":
            return -latent
        return latent

    def _trace(
        self,
        *,
        writer_input: torch.Tensor,
        previous_state: torch.Tensor,
        derived_fields: tuple[str, ...],
        routing_mode: str,
        route: torch.Tensor,
        write_probability: torch.Tensor,
        write_strength: torch.Tensor,
        latent: torch.Tensor,
        code_indices: torch.Tensor | None = None,
    ) -> V3WriterTrace:
        return V3WriterTrace(
            writer_input=writer_input,
            previous_state=previous_state,
            field_names=RAW_WRITER_FIELD_NAMES,
            derived_fields=derived_fields,
            routing_mode=routing_mode,
            route=route,
            write_probability=write_probability,
            write_strength=write_strength,
            writer_latent=latent,
            exact_skip=write_strength == 0,
            code_indices=code_indices,
        )

    def clone(self) -> BaseV3Core:
        return copy.deepcopy(self)


def validate_v3_raw_trace(
    trace: V3WriterTrace,
    public: torch.Tensor,
    outcome: torch.Tensor,
    previous_state: torch.Tensor,
) -> None:
    expected = raw_writer_input(public, outcome)
    if trace.field_names != RAW_WRITER_FIELD_NAMES:
        raise ValueError("V3 writer field names violate the raw contract")
    if trace.derived_fields:
        raise ValueError(
            f"V3 raw writer received prohibited derived fields: {trace.derived_fields}"
        )
    if not torch.equal(trace.writer_input, expected):
        raise ValueError("V3 captured writer input differs from raw public fields")
    if not torch.equal(trace.previous_state, previous_state):
        raise ValueError("V3 captured previous state differs from bounded runtime state")


class _FixedResearchRouteCore(BaseV3Core):
    """Explicit V3A isolation scaffold: fixed route and exact primitive overwrite."""

    stage = "v3a"
    fixed_routing = True

    def _fixed_route(self, public: torch.Tensor) -> torch.Tensor:
        return F.one_hot(public[:, 1].long(), num_classes=4).float()[:, :2]

    def _candidate(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> V3StateUpdate:
        del prediction
        writer_input = raw_writer_input(public, outcome)
        latent = self._intervene_latent(self.writer_encoder(writer_input))
        candidate_scalar, codes = self._candidate(latent)
        candidate = candidate_scalar.expand_as(state)
        route = self._fixed_route(public)
        strength = route.max(dim=-1, keepdim=True).values
        route, strength = self._intervene_route(route, strength)
        gate = route * strength
        updated = (1.0 - gate) * state.float() + gate * candidate
        return V3StateUpdate(
            state=updated,
            gate=gate,
            candidate=candidate,
            error_signal=torch.zeros(len(public), 1, device=public.device),
            trace=self._trace(
                writer_input=writer_input,
                previous_state=state,
                derived_fields=(),
                routing_mode="fixed_research_route",
                route=route,
                write_probability=strength,
                write_strength=strength,
                latent=latent,
                code_indices=codes,
            ),
        )


class RawBilinearOverwriteCore(_FixedResearchRouteCore):
    """V3A candidate: learned raw relation followed by an exact routed overwrite."""

    family = "raw_bilinear_overwrite"

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__(config)
        self.writer_encoder = LearnedRawRelationEncoder(config.hidden_dim)
        self.value_candidate = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def _candidate(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.value_candidate(latent), None


class RawDiscreteOverwriteCore(_FixedResearchRouteCore):
    """V3A candidate: raw evidence selects learned discrete relation codes."""

    family = "raw_discrete_overwrite"

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__(config)
        self.writer_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 32),
            nn.Tanh(),
        )
        self.code_logits = nn.Linear(32, 4)
        self.codebook = nn.Parameter(torch.tensor([-0.9, -0.3, 0.3, 0.9]))

    def _candidate(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        probability = torch.softmax(self.code_logits(latent), dim=-1)
        selection = _hard_categorical(probability, training=self.training)
        values = torch.tanh(self.codebook)
        return (selection * values).sum(dim=-1, keepdim=True), selection.argmax(dim=-1)


class _LearnedRouterCore(BaseV3Core):
    """Common learned route and value writer; subclasses define evidence and preservation."""

    def __init__(self, config: V3ModelConfig, *, relation_scaffold: bool) -> None:
        super().__init__(config)
        self.relation_scaffold = relation_scaffold
        self.writer_encoder: nn.Module
        if relation_scaffold:
            self.writer_encoder = nn.Sequential(
                nn.Linear(9, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 32),
                nn.Tanh(),
            )
        else:
            self.writer_encoder = LearnedRawRelationEncoder(config.hidden_dim)
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

    def _value_input(
        self, writer_input: torch.Tensor, public: torch.Tensor, outcome: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        if not self.relation_scaffold:
            return writer_input, ()
        x_signed = public[:, :1].float() * 2.0 - 1.0
        y_signed = outcome.float().unsqueeze(1) * 2.0 - 1.0
        relation = x_signed * y_signed
        return torch.cat((writer_input, relation), dim=-1), ("signed_rule_relation",)

    def _route(self, controller_input: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.router(controller_input), dim=-1)

    def _strength(self, controller_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(self.write_controller(controller_input))
        return probability, probability

    def _candidate(
        self, latent: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.value_candidate(torch.cat((latent, state.float()), dim=-1)), None

    def _project_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return state, None

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: StepPrediction,
    ) -> V3StateUpdate:
        del prediction
        writer_input = raw_writer_input(public, outcome)
        value_input, derived = self._value_input(writer_input, public, outcome)
        latent = self._intervene_latent(self.writer_encoder(value_input))
        controller_input = torch.cat((writer_input, state.float()), dim=-1)
        route = self._route(controller_input)
        probability, strength = self._strength(controller_input)
        route, strength = self._intervene_route(route, strength)
        candidate_scalar, codes = self._candidate(latent, state)
        candidate = candidate_scalar.expand_as(state)
        gate = route * strength
        proposed = (1.0 - gate) * state.float() + gate * candidate
        updated, projected_codes = self._project_state(proposed)
        if projected_codes is not None:
            codes = projected_codes
        return V3StateUpdate(
            state=updated,
            gate=gate,
            candidate=candidate,
            error_signal=torch.zeros(len(public), 1, device=public.device),
            trace=self._trace(
                writer_input=writer_input,
                previous_state=state,
                derived_fields=derived,
                routing_mode=(
                    "learned_hard_router"
                    if torch.all((route == 0) | (route == 1))
                    else "learned_soft_router"
                ),
                route=route,
                write_probability=probability,
                write_strength=strength,
                latent=latent,
                code_indices=codes,
            ),
        )


class RelationHardSkipCore(_LearnedRouterCore):
    """V3B candidate: learned exact write/skip with supplied relation."""

    family = "relation_hard_skip"
    stage = "v3b"
    hard_preservation = True

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__(config, relation_scaffold=True)
        nn.init.constant_(self.write_controller[-1].bias, 0.5)

    def _strength(self, controller_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(self.write_controller(controller_input))
        return probability, _hard_binary(probability, training=self.training)


class RelationAttractorCore(_LearnedRouterCore):
    """V3B candidate: learned state prototypes repair small unwanted updates."""

    family = "relation_attractor"
    stage = "v3b"
    hard_preservation = True

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__(config, relation_scaffold=True)
        self.state_codebook = nn.Parameter(torch.tensor([-1.5, 0.0, 1.5]))
        nn.init.constant_(self.write_controller[-1].bias, -1.0)

    def _project_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        values = torch.tanh(self.state_codebook)
        distance = (state.unsqueeze(-1) - values.view(1, 1, -1)).square()
        probability = torch.softmax(-8.0 * distance, dim=-1)
        selection = _hard_categorical(probability.flatten(0, 1), training=self.training).view_as(
            probability
        )
        projected = (selection * values.view(1, 1, -1)).sum(dim=-1)
        return projected, selection.argmax(dim=-1)


class RelationHardSkipSoftTrainCore(RelationHardSkipCore):
    """Hard-skip revision: soft training strength and exact deterministic evaluation."""

    family = "relation_hard_skip_softtrain"

    def _strength(self, controller_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(self.write_controller(controller_input))
        if self.training:
            return probability, probability
        return probability, _hard_binary(probability, training=False)


class RelationAttractorSoftTrainCore(RelationAttractorCore):
    """Attractor revision: soft projection for training and hard projection at evaluation."""

    family = "relation_attractor_softtrain"

    def _project_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        values = torch.tanh(self.state_codebook)
        distance = (state.unsqueeze(-1) - values.view(1, 1, -1)).square()
        probability = torch.softmax(-8.0 * distance, dim=-1)
        if self.training:
            projected = (probability * values.view(1, 1, -1)).sum(dim=-1)
            return projected, probability.argmax(dim=-1)
        selection = _hard_categorical(probability.flatten(0, 1), training=False).view_as(
            probability
        )
        projected = (selection * values.view(1, 1, -1)).sum(dim=-1)
        return projected, selection.argmax(dim=-1)


class RawHardRouterCore(_LearnedRouterCore):
    """V3C candidate: raw relation, learned hard address, and exact learned skip."""

    family = "raw_hard_router"
    stage = "v3c"
    hard_preservation = True

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__(config, relation_scaffold=False)
        nn.init.constant_(self.write_controller[-1].bias, 0.5)

    def _route(self, controller_input: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(self.router(controller_input), dim=-1)
        return _hard_categorical(probability, training=self.training)

    def _strength(self, controller_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(self.write_controller(controller_input))
        return probability, _hard_binary(probability, training=self.training)


class RawDiscreteRouterCore(RawHardRouterCore):
    """V3C alternative: learned raw evidence code plus hard learned state transitions."""

    family = "raw_discrete_router"

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__(config)
        self.code_logits = nn.Linear(32, 4)
        self.codebook = nn.Parameter(torch.tensor([-0.9, -0.3, 0.3, 0.9]))

    def _candidate(
        self, latent: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del state
        probability = torch.softmax(self.code_logits(latent), dim=-1)
        selection = _hard_categorical(probability, training=self.training)
        values = torch.tanh(self.codebook)
        return (selection * values).sum(dim=-1, keepdim=True), selection.argmax(dim=-1)


def build_v3_model(config: V3ModelConfig) -> BaseV3Core:
    families: dict[str, type[BaseV3Core]] = {
        "raw_bilinear_overwrite": RawBilinearOverwriteCore,
        "raw_discrete_overwrite": RawDiscreteOverwriteCore,
        "relation_hard_skip": RelationHardSkipCore,
        "relation_hard_skip_softtrain": RelationHardSkipSoftTrainCore,
        "relation_attractor": RelationAttractorCore,
        "relation_attractor_softtrain": RelationAttractorSoftTrainCore,
        "raw_hard_router": RawHardRouterCore,
        "raw_discrete_router": RawDiscreteRouterCore,
    }
    return families[config.family](config)
