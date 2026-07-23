"""Compact model-only checkpoints for Cognitive Core v2."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bilab.resources import configuration_hash, git_revision
from bilab.training.v1 import state_dict_digest
from bilab.v2.models import build_v2_model, count_v2_parameters
from bilab.v2.training import V2TrainConfig


def save_v2_checkpoint(
    path: Path,
    model: nn.Module,
    config: V2TrainConfig,
    *,
    repo: Path,
    experiment_id: str,
    training_step: int,
    validation_score: float,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    configuration = asdict(config)
    metadata = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "git_revision": git_revision(repo),
        "family": config.family,
        "seed": config.seed,
        "parameter_count": count_v2_parameters(model),
        "persistent_state_bytes": int(getattr(model, "persistent_bytes", 0)),
        "training_step": training_step,
        "validation_score": validation_score,
        "configuration_hash": configuration_hash(configuration),
        "model_digest": state_dict_digest(model),
    }
    torch.save(
        {
            "metadata": metadata,
            "configuration": configuration,
            "model_state": model.state_dict(),
        },
        path,
    )
    metadata["path"] = str(path)
    metadata["checkpoint_bytes"] = path.stat().st_size
    return metadata


def load_v2_checkpoint(path: Path) -> tuple[nn.Module, V2TrainConfig, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    configuration = payload["configuration"]
    if configuration_hash(configuration) != metadata["configuration_hash"]:
        raise ValueError("v2 checkpoint configuration hash mismatch")
    config = V2TrainConfig.from_dict(configuration)
    model = build_v2_model(config.model_config(), budget_bytes=config.budget_bytes)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if count_v2_parameters(model) != metadata["parameter_count"]:
        raise ValueError("v2 checkpoint parameter count mismatch")
    if int(getattr(model, "persistent_bytes", 0)) != metadata["persistent_state_bytes"]:
        raise ValueError("v2 checkpoint persistent-state budget mismatch")
    return model, config, metadata
