"""Compact model-only checkpoints for Cognitive Core v3."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from bilab.resources import configuration_hash, git_revision
from bilab.training.v1 import state_dict_digest
from bilab.v3.models import build_v3_model, count_v3_parameters
from bilab.v3.training import V3TrainConfig


def save_v3_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: V3TrainConfig,
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
        "stage": config.stage,
        "initialization": config.initialization,
        "seed": config.seed,
        "parameter_count": count_v3_parameters(model),
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


def load_v3_checkpoint(
    path: Path,
) -> tuple[torch.nn.Module, V3TrainConfig, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    configuration = payload["configuration"]
    if configuration_hash(configuration) != metadata["configuration_hash"]:
        raise ValueError("V3 checkpoint configuration hash mismatch")
    config = V3TrainConfig.from_dict(configuration)
    model = build_v3_model(config.model_config())
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if count_v3_parameters(model) != metadata["parameter_count"]:
        raise ValueError("V3 checkpoint parameter count mismatch")
    if model.persistent_bytes != metadata["persistent_state_bytes"]:
        raise ValueError("V3 checkpoint persistent-state byte mismatch")
    if state_dict_digest(model) != metadata["model_digest"]:
        raise ValueError("V3 checkpoint model digest mismatch")
    return model, config, metadata
