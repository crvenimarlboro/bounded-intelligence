"""Construct parameter-matched Cognitive Core v0 variants."""

from __future__ import annotations

from typing import Any, Literal

from torch import nn

from bilab.models.baselines import EpisodicModel, NoMemoryModel
from bilab.models.cognitive_core import CognitiveCore

ModelVariant = Literal["no_memory", "episodic", "cognitive_core"]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _matched_width(base_factory, target_parameters: int, hidden_dim: int) -> int:
    base = base_factory(1)
    base_count = count_parameters(base)
    per_width = 2 * hidden_dim + 1
    width = max(1, round(1 + (target_parameters - base_count) / per_width))
    return width


def build_model(variant: ModelVariant, config: dict[str, Any]) -> nn.Module:
    if variant == "cognitive_core":
        return CognitiveCore(config)
    target = count_parameters(CognitiveCore(config))
    hidden = int(config["model"]["hidden_dim"])
    if variant == "no_memory":

        def factory(width: int) -> NoMemoryModel:
            return NoMemoryModel(config, width)

        return factory(_matched_width(factory, target, hidden))
    if variant == "episodic":

        def factory(width: int) -> EpisodicModel:
            return EpisodicModel(config, width)

        return factory(_matched_width(factory, target, hidden))
    raise ValueError(f"unknown model variant: {variant}")
