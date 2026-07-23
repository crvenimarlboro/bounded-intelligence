"""Small trainable systems used by Cognitive Core experiments."""

from bilab.models.baselines import EpisodicModel, NoMemoryModel
from bilab.models.cognitive_core import CognitiveCore
from bilab.models.factory import build_model, count_parameters
from bilab.models.v1 import (
    EpisodicControl,
    FactorizedStateCore,
    GRUStateCore,
    NoMemoryControl,
    PredictiveStateCore,
    V1ModelConfig,
    build_v1_model,
    count_trainable_parameters,
)

__all__ = [
    "CognitiveCore",
    "EpisodicControl",
    "EpisodicModel",
    "FactorizedStateCore",
    "GRUStateCore",
    "NoMemoryControl",
    "NoMemoryModel",
    "PredictiveStateCore",
    "V1ModelConfig",
    "build_model",
    "build_v1_model",
    "count_parameters",
    "count_trainable_parameters",
]
