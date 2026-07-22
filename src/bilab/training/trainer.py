"""CPU training loop shared by all neural variants."""

from __future__ import annotations

import random
import resource
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from bilab.environments.rule_worlds import Episode, make_training_episodes, make_validation_episodes
from bilab.models.factory import ModelVariant, build_model, count_parameters
from bilab.training.data import episode_fingerprint, episode_tensors, shuffled_batches


@dataclass
class TrainingResult:
    model: nn.Module
    variant: str
    seed: int
    curves: list[dict[str, float | int]]
    training_seconds: float
    cpu_seconds: float
    peak_ram_bytes: int
    observations_consumed: int
    parameter_count: int
    training_data_fingerprint: str
    validation_accuracy: float
    final_step: int


def configure_determinism(seed: int, threads: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def _run_episode_batch(
    model: nn.Module,
    episodes: list[Episode],
    *,
    train: bool,
) -> tuple[torch.Tensor, int, int]:
    observations, targets = episode_tensors(episodes)
    state = model.initial_state(len(episodes))
    losses: list[torch.Tensor] = []
    correct = 0
    total = 0
    for step_index in range(observations.shape[1]):
        logits, state, _ = model.step(
            observations[:, step_index], state, targets[:, step_index], mode="full"
        )
        losses.append(torch.nn.functional.cross_entropy(logits, targets[:, step_index]))
        correct += int((logits.argmax(dim=-1) == targets[:, step_index]).sum().item())
        total += len(episodes)
    loss = torch.stack(losses).mean()
    if not train:
        loss = loss.detach()
    return loss, correct, total


def evaluate_accuracy(model: nn.Module, episodes: list[Episode], batch_size: int) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            _, batch_correct, batch_total = _run_episode_batch(
                model, episodes[start : start + batch_size], train=False
            )
            correct += batch_correct
            total += batch_total
    return correct / total


def train_model(config: dict[str, Any], variant: ModelVariant, seed: int) -> TrainingResult:
    training = config["training"]
    configure_determinism(seed, int(training["torch_threads"]))
    model = build_model(variant, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_episodes = make_training_episodes(config)
    validation_episodes = make_validation_episodes(config)
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    curves: list[dict[str, float | int]] = []
    observations = 0
    step = 0
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batches = 0
        for batch in shuffled_batches(train_episodes, batch_size, seed + epoch * 1009):
            optimizer.zero_grad(set_to_none=True)
            loss, correct, total = _run_episode_batch(model, batch, train=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss for {variant}, seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_correct += correct
            epoch_total += total
            observations += total
            step += 1
            batches += 1
        validation_accuracy = evaluate_accuracy(model, validation_episodes, batch_size)
        curves.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": step,
                "train_loss": epoch_loss / batches,
                "train_accuracy": epoch_correct / epoch_total,
                "validation_accuracy": validation_accuracy,
                "observations_consumed": observations,
            }
        )

    cpu_seconds = time.process_time() - cpu_start
    training_seconds = time.perf_counter() - wall_start
    peak_ram_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return TrainingResult(
        model=model,
        variant=variant,
        seed=seed,
        curves=curves,
        training_seconds=training_seconds,
        cpu_seconds=cpu_seconds,
        peak_ram_bytes=peak_ram_bytes,
        observations_consumed=observations,
        parameter_count=count_parameters(model),
        training_data_fingerprint=episode_fingerprint(train_episodes),
        validation_accuracy=float(curves[-1]["validation_accuracy"]),
        final_step=step,
    )
