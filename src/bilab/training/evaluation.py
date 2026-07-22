"""Held-out online evaluation with frozen weights and bounded state only."""

from __future__ import annotations

import math
import resource
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from bilab.environments.rule_worlds import Episode
from bilab.models.cognitive_core import CoreMode
from bilab.models.factory import count_parameters
from bilab.training.data import episode_fingerprint, episode_tensors


@dataclass
class EvaluationResult:
    variant: str
    mode: str
    seed: int
    metrics: dict[str, Any]
    evaluation_seconds: float
    cpu_seconds: float
    peak_ram_bytes: int
    observations_consumed: int
    parameter_count: int
    persistent_bytes: int
    evaluation_data_fingerprint: str


def _state_bytes(model: nn.Module, state: object) -> int:
    if state is None:
        return 0
    return int(model.state_nbytes(state))


def evaluate_model(
    model: nn.Module,
    episodes_by_category: dict[str, list[Episode]],
    *,
    variant: str,
    seed: int,
    mode: CoreMode | str = "full",
    batch_size: int = 8,
    adaptation_checkpoints: list[int],
    recovery_windows: list[list[int]],
) -> EvaluationResult:
    model.eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    counters: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    adaptation: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    recovery: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    losses: list[float] = []
    observations_consumed = 0
    observed_state_bytes: set[int] = set()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    with torch.no_grad():
        for category, episodes in episodes_by_category.items():
            for start in range(0, len(episodes), batch_size):
                batch = episodes[start : start + batch_size]
                observations, targets = episode_tensors(batch)
                state = model.initial_state(len(batch))
                observed_state_bytes.add(_state_bytes(model, state))
                for step_index in range(observations.shape[1]):
                    logits, state, _ = model.step(
                        observations[:, step_index],
                        state,
                        targets[:, step_index],
                        mode=mode,
                    )
                    observed_state_bytes.add(_state_bytes(model, state))
                    predictions = logits.argmax(dim=-1)
                    loss = torch.nn.functional.cross_entropy(
                        logits, targets[:, step_index], reduction="sum"
                    )
                    losses.append(float(loss))
                    for row, episode in enumerate(batch):
                        step = episode.steps[step_index]
                        correct = int(predictions[row].item() == step.target)

                        def add(key: str, result: int = correct) -> None:
                            counters[key][0] += result
                            counters[key][1] += 1

                        add(f"category:{category}")
                        add(f"kind:{category}:{step.query_kind}")
                        if category in {"structured", "surface_relabelled"}:
                            add("structured_all")
                            if step.query_kind == "direct":
                                add("hidden_rule_query")
                            if step.query_kind in {"composed", "counterfactual"}:
                                add("composed_counterfactual")
                        if step.retention_probe:
                            add("rule_change_retention")
                        if step.affected_by_change and step.steps_since_change is not None:
                            for lower, upper in recovery_windows:
                                if lower <= step.steps_since_change <= upper:
                                    key = f"{lower}-{upper}"
                                    recovery[key][0] += correct
                                    recovery[key][1] += 1
                        if step_index in adaptation_checkpoints:
                            point = adaptation[category][step_index]
                            point[0] += correct
                            point[1] += 1
                    observations_consumed += len(batch)

    after = model.state_dict()
    if any(not torch.equal(before[name], after[name]) for name in before):
        raise RuntimeError("model weights changed during held-out online evaluation")
    if len(observed_state_bytes) != 1:
        raise RuntimeError(
            f"persistent state size changed during evaluation: {observed_state_bytes}"
        )

    def accuracy(key: str) -> float | None:
        correct, total = counters[key]
        return correct / total if total else None

    category_accuracy = {
        category: accuracy(f"category:{category}") for category in episodes_by_category
    }
    adaptation_curves = {
        category: {
            str(point): values[0] / values[1] if values[1] else None
            for point, values in sorted(points.items())
        }
        for category, points in adaptation.items()
    }
    recovery_curve = {
        key: values[0] / values[1] if values[1] else None for key, values in recovery.items()
    }
    persistent_bytes = next(iter(observed_state_bytes))
    structured = category_accuracy.get("structured")
    metrics = {
        "category_accuracy": category_accuracy,
        "mean_cross_entropy": sum(losses) / observations_consumed,
        "hidden_rule_query_accuracy": accuracy("hidden_rule_query"),
        "composed_counterfactual_accuracy": accuracy("composed_counterfactual"),
        "surface_relabel_accuracy": category_accuracy.get("surface_relabelled"),
        "random_control_accuracy": category_accuracy.get("random"),
        "rule_change_retention": accuracy("rule_change_retention"),
        "recovery_curve": recovery_curve,
        "adaptation_curves": adaptation_curves,
        "capability_per_persistent_byte": (
            structured / persistent_bytes if structured is not None and persistent_bytes else None
        ),
        "state_bytes_observed": sorted(observed_state_bytes),
    }
    total_correct = sum(counters[f"category:{category}"][0] for category in episodes_by_category)
    total_count = sum(counters[f"category:{category}"][1] for category in episodes_by_category)
    metrics["next_outcome_accuracy"] = total_correct / total_count
    evaluation_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    all_episodes = [episode for episodes in episodes_by_category.values() for episode in episodes]
    return EvaluationResult(
        variant=variant,
        mode=str(mode),
        seed=seed,
        metrics=metrics,
        evaluation_seconds=evaluation_seconds,
        cpu_seconds=cpu_seconds,
        peak_ram_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        observations_consumed=observations_consumed,
        parameter_count=count_parameters(model),
        persistent_bytes=persistent_bytes,
        evaluation_data_fingerprint=episode_fingerprint(all_episodes),
    )


def comparison_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    controlled = (
        "seed",
        "configuration_hash",
        "training_data_fingerprint",
        "evaluation_data_fingerprint",
        "training_observations",
        "evaluation_observations",
    )
    return [key for key in controlled if left.get(key) != right.get(key)]


def mean_and_sample_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "sample_std": math.nan}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "sample_std": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "sample_std": math.sqrt(variance)}
