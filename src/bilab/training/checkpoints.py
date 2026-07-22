"""Compact final-model checkpoints with complete evaluation provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from bilab.models.factory import build_model, count_parameters
from bilab.resources import configuration_hash, git_revision


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict[str, Any],
    *,
    repo: Path,
    variant: str,
    seed: int,
    persistent_bytes: int,
    training_step: int,
    validation_score: float,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "git_revision": git_revision(repo),
        "variant": variant,
        "seed": seed,
        "parameter_count": count_parameters(model),
        "persistent_state_bytes": persistent_bytes,
        "training_step": training_step,
        "validation_score": validation_score,
        "configuration_hash": configuration_hash(config),
    }
    torch.save(
        {"metadata": metadata, "configuration": config, "model_state": model.state_dict()}, path
    )
    metadata["checkpoint_bytes"] = path.stat().st_size
    return metadata


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    config = payload["configuration"]
    model = build_model(metadata["variant"], config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if count_parameters(model) != metadata["parameter_count"]:
        raise ValueError("checkpoint parameter count does not match reconstructed model")
    if configuration_hash(config) != metadata["configuration_hash"]:
        raise ValueError("checkpoint configuration hash mismatch")
    return model, config, metadata
