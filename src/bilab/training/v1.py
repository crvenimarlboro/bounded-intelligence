"""Training, evaluation, and mechanistic diagnostics for Cognitive Core v1."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import resource
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bilab.environments.adaptation_ladder import (
    BinaryBatch,
    BinaryEpisode,
    ContextEpisode,
    balanced_composition_episodes,
    balanced_context_episodes,
    balanced_delayed_episodes,
    balanced_reversal_episodes,
    batch_adaptation_episodes,
    paired_structured_episodes,
)
from bilab.models.v1 import (
    AdaptiveCore,
    PredictiveStateCore,
    V1ModelConfig,
    build_v1_model,
    count_trainable_parameters,
)


@dataclass(frozen=True)
class V1TrainConfig:
    family: str
    seed: int
    world_seed_start: int
    validation_seed_start: int
    optimizer_steps: int = 400
    batch_pairs: int = 16
    episode_steps: int = 9
    hidden_dim: int = 64
    state_dim: int = 4
    thought_cycles: int = 1
    feedback_mode: str = "outcome_only"
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    auxiliary_rule_start: float = 0.0
    auxiliary_rule_end: float = 0.0
    predictive_weight: float = 0.0
    validation_pairs: int = 64
    validation_interval: int = 50
    torch_threads: int = 6
    budget_bytes: int = 16
    context_count: int = 1
    delay_steps: int = 8
    query_steps: int = 6
    environment_variant: str = "standard"
    checkpoint_selection: str = "best_validation"

    def __post_init__(self) -> None:
        if self.family not in {"gru", "predictive", "factorized", "no_memory", "episodic"}:
            raise ValueError(f"unknown family: {self.family}")
        if self.optimizer_steps < 1 or self.batch_pairs < 1 or self.episode_steps < 2:
            raise ValueError("training steps, pairs, and episode length must be positive")
        if self.torch_threads < 1 or self.torch_threads > 6:
            raise ValueError("v1 uses one to six PyTorch threads")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        if self.context_count not in {1, 2, 3, 4}:
            raise ValueError("context_count must be one through four")
        if (
            self.context_count == 4
            and self.environment_variant != "reversal"
            and self.episode_steps != 2 + self.delay_steps + self.query_steps
        ):
            raise ValueError(
                "delayed episode_steps must equal two evidence, delay, and query steps"
            )
        if self.environment_variant not in {"standard", "reversal"}:
            raise ValueError("unknown environment_variant")
        if self.checkpoint_selection not in {"best_validation", "final_step"}:
            raise ValueError("unknown checkpoint_selection")
        if self.environment_variant == "reversal" and (
            self.context_count not in {3, 4} or self.episode_steps != 12
        ):
            raise ValueError("reversal uses three or four operation contexts and twelve steps")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> V1TrainConfig:
        return cls(**value)

    def model_config(self) -> V1ModelConfig:
        return V1ModelConfig(
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            thought_cycles=self.thought_cycles,
            feedback_mode=self.feedback_mode,
            context_count=self.context_count,
            rule_count=min(2, self.context_count),
        )


@dataclass
class TrainRun:
    model: nn.Module
    metrics: dict[str, Any]


def seed_everything(seed: int, threads: int = 6) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def generation_seed_ranges(config: V1TrainConfig) -> dict[str, tuple[int, int]]:
    """Return inclusive procedural seed ranges actually consumed by training and validation."""

    return {
        "training": (
            config.world_seed_start,
            config.world_seed_start + config.optimizer_steps * config.batch_pairs - 1,
        ),
        "validation": (
            config.validation_seed_start,
            config.validation_seed_start + config.validation_pairs - 1,
        ),
    }


def assert_generation_splits_disjoint(
    config: V1TrainConfig, *, evaluation_ranges: dict[str, tuple[int, int]] | None = None
) -> None:
    """Reject any inclusive train/validation/evaluation generation-seed overlap."""

    ranges = generation_seed_ranges(config)
    ranges.update(evaluation_ranges or {})
    names = list(ranges)
    for left_index, left_name in enumerate(names):
        left_start, left_end = ranges[left_name]
        if left_start > left_end:
            raise ValueError(f"invalid seed range: {left_name}")
        for right_name in names[left_index + 1 :]:
            right_start, right_end = ranges[right_name]
            if max(left_start, right_start) <= min(left_end, right_end):
                raise ValueError(f"generation seed ranges overlap: {left_name} and {right_name}")


def _auxiliary_weight(config: V1TrainConfig, step: int) -> float:
    if config.optimizer_steps == 1:
        return config.auxiliary_rule_end
    fraction = step / (config.optimizer_steps - 1)
    return config.auxiliary_rule_start + fraction * (
        config.auxiliary_rule_end - config.auxiliary_rule_start
    )


def _public_predictive_targets(public: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """Derive future x=0/x=1 outcomes from feedback, without privileged rule labels."""

    relation = public[:, 0].long() ^ outcome
    return torch.stack((relation, 1 - relation), dim=1)


def _batch_for_step(config: V1TrainConfig, step: int) -> Any:
    seed_start = config.world_seed_start + step * config.batch_pairs
    if config.environment_variant == "reversal":
        return batch_adaptation_episodes(
            balanced_reversal_episodes(
                seed_start=seed_start,
                groups=config.batch_pairs,
                include_no_change=True,
            )
        )
    if config.context_count == 3:
        return batch_adaptation_episodes(
            balanced_composition_episodes(
                seed_start=seed_start,
                groups=config.batch_pairs,
                steps=config.episode_steps,
            )
        )
    if config.context_count == 4:
        return batch_adaptation_episodes(
            balanced_delayed_episodes(
                seed_start=seed_start,
                groups=config.batch_pairs,
                delay_steps=config.delay_steps,
                query_steps=config.query_steps,
            )
        )
    if config.context_count == 2:
        return batch_adaptation_episodes(
            balanced_context_episodes(
                seed_start=seed_start,
                groups=config.batch_pairs,
                steps=config.episode_steps,
            )
        )
    episodes = paired_structured_episodes(
        seed_start=seed_start,
        pairs=config.batch_pairs,
        steps=config.episode_steps,
    )
    return BinaryBatch.from_episodes(episodes)


def episode_objective(
    model: nn.Module,
    batch: BinaryBatch,
    *,
    auxiliary_rule_weight: float = 0.0,
    predictive_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run full BPTT across one batch of worlds; state is never detached inside the episode."""

    batch_size, episode_steps, _ = batch.public.shape
    state = model.initial_state(batch_size, batch.public.device)
    primary_losses: list[torch.Tensor] = []
    auxiliary_losses: list[torch.Tensor] = []
    predictive_losses: list[torch.Tensor] = []
    correct = 0
    prediction_count = 0
    post_correct = 0
    post_count = 0
    gate_values: list[torch.Tensor] = []
    update_norms: list[torch.Tensor] = []
    for time_index in range(episode_steps):
        public = batch.public[:, time_index]
        outcome = batch.outcomes[:, time_index]
        prediction = model.predict(public, state)
        predicted = prediction.logits.argmax(dim=-1)
        valid = torch.ones(batch_size, dtype=torch.bool, device=public.device)
        if public.shape[1] == 3:
            valid = public[:, 1] != 3
        if valid.any():
            primary_losses.append(F.cross_entropy(prediction.logits[valid], outcome[valid]))
            correct += int((predicted[valid] == outcome[valid]).sum().item())
            prediction_count += int(valid.sum().item())
            if time_index > 0:
                post_correct += int((predicted[valid] == outcome[valid]).sum().item())
                post_count += int(valid.sum().item())
        old_state = state
        update = model.update(public, state, outcome, prediction)
        state = update.state
        if update.gate.numel():
            gate_values.append(update.gate)
            update_norms.append((state.float() - old_state.float()).norm(dim=-1))
        if auxiliary_rule_weight and isinstance(model, AdaptiveCore):
            auxiliary_losses.append(F.cross_entropy(model.probe(state), batch.hidden_rules))
        if predictive_weight and isinstance(model, PredictiveStateCore):
            table = model.predict_future_table(state)
            targets = _public_predictive_targets(public, outcome)
            predictive_losses.append(
                (
                    F.cross_entropy(table[:, 0], targets[:, 0])
                    + F.cross_entropy(table[:, 1], targets[:, 1])
                )
                / 2
            )
    primary = torch.stack(primary_losses).mean()
    auxiliary = torch.stack(auxiliary_losses).mean() if auxiliary_losses else primary.new_zeros(())
    predictive = (
        torch.stack(predictive_losses).mean() if predictive_losses else primary.new_zeros(())
    )
    total = primary + auxiliary_rule_weight * auxiliary + predictive_weight * predictive
    gate = torch.cat([value.flatten() for value in gate_values]) if gate_values else None
    update_norm = torch.cat(update_norms) if update_norms else None
    diagnostics = {
        "primary_loss": float(primary.detach().item()),
        "auxiliary_loss": float(auxiliary.detach().item()),
        "predictive_loss": float(predictive.detach().item()),
        "accuracy": correct / prediction_count,
        "post_evidence_accuracy": post_correct / post_count,
        "gate_mean": float(gate.mean().detach().item()) if gate is not None else math.nan,
        "gate_std": float(gate.std(unbiased=False).detach().item())
        if gate is not None
        else math.nan,
        "gate_below_005": float((gate < 0.05).float().mean().item())
        if gate is not None
        else math.nan,
        "gate_above_095": float((gate > 0.95).float().mean().item())
        if gate is not None
        else math.nan,
        "update_norm": float(update_norm.mean().detach().item())
        if update_norm is not None
        else math.nan,
        "final_state_norm": float(state.float().norm(dim=-1).mean().detach().item()),
    }
    return total, diagnostics


def _gradient_diagnostics(model: nn.Module, clip_threshold: float) -> dict[str, Any]:
    modules: dict[str, dict[str, float]] = {}
    all_squared = 0.0
    for module_name, module in model.named_children():
        gradients = [
            parameter.grad.detach().flatten()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            continue
        values = torch.cat(gradients)
        modules[module_name] = {
            "norm": float(values.norm().item()),
            "zero_fraction": float((values == 0).float().mean().item()),
            "maximum_absolute": float(values.abs().max().item()),
        }
        all_squared += float(values.square().sum().item())
    total_norm = math.sqrt(all_squared)
    return {
        "modules": modules,
        "total_norm": total_norm,
        "would_clip": total_norm > clip_threshold,
        "nan_or_inf": any(
            not math.isfinite(metric) for values in modules.values() for metric in values.values()
        ),
    }


def _peak_ram_bytes() -> int | None:
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (AttributeError, OSError):
        return None


def state_dict_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_candidate(config: V1TrainConfig) -> TrainRun:
    """Train one candidate and select the best checkpoint on fixed validation worlds."""

    assert_generation_splits_disjoint(config)
    seed_everything(config.seed, config.torch_threads)
    model = build_v1_model(config.family, config.model_config(), budget_bytes=config.budget_bytes)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if config.environment_variant == "reversal":
        validation_episodes = balanced_reversal_episodes(
            seed_start=config.validation_seed_start,
            groups=config.validation_pairs,
            include_no_change=True,
        )
    elif config.context_count == 4:
        validation_episodes = balanced_delayed_episodes(
            seed_start=config.validation_seed_start,
            groups=config.validation_pairs,
            delay_steps=config.delay_steps,
            query_steps=config.query_steps,
        )
    elif config.context_count == 3:
        validation_episodes = balanced_composition_episodes(
            seed_start=config.validation_seed_start,
            groups=config.validation_pairs,
            steps=config.episode_steps,
        )
    elif config.context_count == 2:
        validation_episodes = balanced_context_episodes(
            seed_start=config.validation_seed_start,
            groups=config.validation_pairs,
            steps=config.episode_steps,
        )
    else:
        validation_episodes = paired_structured_episodes(
            seed_start=config.validation_seed_start,
            pairs=config.validation_pairs,
            steps=config.episode_steps,
        )
    start = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    latest_gradient: dict[str, Any] = {}
    clipping_count = 0
    for step in range(config.optimizer_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = _batch_for_step(config, step)
        auxiliary_weight = _auxiliary_weight(config, step)
        loss, training_metrics = episode_objective(
            model,
            batch,
            auxiliary_rule_weight=auxiliary_weight,
            predictive_weight=config.predictive_weight,
        )
        loss.backward()
        latest_gradient = _gradient_diagnostics(model, config.gradient_clip)
        clipping_count += int(latest_gradient["would_clip"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        current_step = step + 1
        should_validate = (
            current_step == 1
            or current_step % config.validation_interval == 0
            or current_step == config.optimizer_steps
        )
        if should_validate:
            validation = evaluate_model(model, validation_episodes)
            score = validation["fully_informed_accuracy"]
            history.append(
                {
                    "optimizer_step": current_step,
                    "training_observations": current_step
                    * config.batch_pairs
                    * (4 if config.context_count > 1 else 2)
                    * config.episode_steps,
                    "learning_rate": config.learning_rate,
                    "auxiliary_rule_weight": auxiliary_weight,
                    "train": training_metrics,
                    "validation": validation,
                }
            )
            if score > best_score:
                best_score = score
                best_step = current_step
                best_state = copy.deepcopy(model.state_dict())
    if config.checkpoint_selection == "best_validation":
        model.load_state_dict(best_state)
    else:
        best_step = config.optimizer_steps
        best_score = float(history[-1]["validation"]["fully_informed_accuracy"])
    elapsed = time.perf_counter() - start
    observations = (
        config.optimizer_steps
        * config.batch_pairs
        * (4 if config.context_count > 1 else 2)
        * config.episode_steps
    )
    metrics = {
        "configuration": asdict(config),
        "parameter_count": count_trainable_parameters(model),
        "persistent_state_bytes": int(getattr(model, "persistent_bytes", 0)),
        "optimizer_steps": config.optimizer_steps,
        "training_observations": observations,
        "best_step": best_step,
        "best_validation_post_evidence_accuracy": best_score,
        "checkpoint_selection_metric": "fully_informed_accuracy",
        "history": history,
        "gradient_diagnostics": latest_gradient,
        "gradient_clipping_fraction": clipping_count / config.optimizer_steps,
        "training_wall_seconds": elapsed,
        "peak_ram_bytes": _peak_ram_bytes(),
        "bptt": {
            "detach_inside_episode": False,
            "detach_at_world_boundary": True,
            "window_steps": config.episode_steps,
            "graph_depth_updates": config.episode_steps,
        },
    }
    return TrainRun(model=model, metrics=metrics)


def evaluate_model(
    model: nn.Module,
    episodes: list[BinaryEpisode] | list[ContextEpisode],
    *,
    intervention: str = "full",
    random_seed: int = 0,
    cycles: int | None = None,
    noise_std: float = 0.1,
) -> dict[str, Any]:
    """Evaluate frozen weights while only the bounded runtime state changes."""

    if intervention not in {"full", "reset", "frozen", "random", "shuffled", "noise"}:
        raise ValueError(f"unknown intervention: {intervention}")
    if not episodes:
        raise ValueError("evaluation requires episodes")
    before = state_dict_digest(model)
    was_training = model.training
    model.eval()
    batch = batch_adaptation_episodes(episodes)
    batch_size, episode_steps, _ = batch.public.shape
    state = model.initial_state(batch_size)
    noise_generator = torch.Generator().manual_seed(random_seed)
    if intervention == "random" and state.dtype.is_floating_point:
        generator = torch.Generator().manual_seed(random_seed)
        state = torch.randn(state.shape, generator=generator)
    correct = [0 for _ in range(episode_steps)]
    task_correct = [0 for _ in range(episode_steps)]
    task_counts = [0 for _ in range(episode_steps)]
    losses = [0.0 for _ in range(episode_steps)]
    gates: list[torch.Tensor] = []
    state_norms: list[float] = []
    update_norms: list[float] = []
    composition_correct = 0
    composition_count = 0
    direct_correct = 0
    direct_count = 0
    distractor_correct = 0
    distractor_count = 0
    distractor_drifts: list[float] = []
    with torch.no_grad():
        for time_index in range(episode_steps):
            if intervention == "reset":
                state = model.initial_state(batch_size)
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            prediction = model.predict(public, state, cycles=cycles)
            predicted = prediction.logits.argmax(dim=-1)
            correct[time_index] = int((predicted == outcome).sum())
            valid = torch.ones(batch_size, dtype=torch.bool)
            if public.shape[1] == 3:
                composed_mask = public[:, 1] == 2
                distractor_mask = public[:, 1] == 3
                valid = ~distractor_mask
                direct_mask = valid & ~composed_mask
                composition_correct += int(
                    (predicted[composed_mask] == outcome[composed_mask]).sum()
                )
                composition_count += int(composed_mask.sum())
                direct_correct += int((predicted[direct_mask] == outcome[direct_mask]).sum())
                direct_count += int(direct_mask.sum())
                distractor_correct += int(
                    (predicted[distractor_mask] == outcome[distractor_mask]).sum()
                )
                distractor_count += int(distractor_mask.sum())
            task_correct[time_index] = int((predicted[valid] == outcome[valid]).sum())
            task_counts[time_index] = int(valid.sum())
            if valid.any():
                losses[time_index] = float(
                    F.cross_entropy(
                        prediction.logits[valid], outcome[valid], reduction="sum"
                    ).item()
                )
            old_state = state
            if intervention not in {"frozen", "reset", "random"}:
                update = model.update(public, state, outcome, prediction)
                state = update.state
                if intervention == "shuffled" and batch_size > 1:
                    state = state.roll(1, dims=0)
                elif intervention == "noise" and state.dtype.is_floating_point:
                    state = state + noise_std * torch.randn(state.shape, generator=noise_generator)
                if update.gate.numel():
                    gates.append(update.gate.flatten())
            state_norms.append(float(state.float().norm(dim=-1).mean().item()))
            drift = float((state.float() - old_state.float()).norm(dim=-1).mean().item())
            update_norms.append(drift)
            if public.shape[1] == 3 and torch.all(public[:, 1] == 3):
                distractor_drifts.append(drift)
    if was_training:
        model.train()
    after = state_dict_digest(model)
    if before != after:
        raise RuntimeError("held-out online evaluation changed learned weights")
    counts = [len(episodes)] * episode_steps
    curve = {str(index): correct[index] / counts[index] for index in range(episode_steps)}
    task_curve = {
        str(index): task_correct[index] / task_counts[index] if task_counts[index] else None
        for index in range(episode_steps)
    }
    gate = torch.cat(gates) if gates else None
    evidence_steps = int(
        getattr(getattr(model, "config", None), "rule_count", getattr(model, "rule_count", 1))
    )
    return {
        "intervention": intervention,
        "adaptation_curve": curve,
        "task_adaptation_curve": task_curve,
        "pre_evidence_accuracy": curve["0"],
        "post_evidence_accuracy": sum(task_correct[1:]) / sum(task_counts[1:]),
        "fully_informed_accuracy": sum(task_correct[evidence_steps:])
        / sum(task_counts[evidence_steps:]),
        "overall_accuracy": sum(task_correct) / sum(task_counts),
        "composition_accuracy": (
            composition_correct / composition_count if composition_count else None
        ),
        "direct_accuracy": direct_correct / direct_count if direct_count else None,
        "distractor_accuracy": (
            distractor_correct / distractor_count if distractor_count else None
        ),
        "distractor_state_drift_mean": (
            sum(distractor_drifts) / len(distractor_drifts) if distractor_drifts else None
        ),
        "distractor_state_drift_max": max(distractor_drifts) if distractor_drifts else None,
        "mean_loss": sum(losses) / sum(task_counts),
        "observations": len(episodes) * episode_steps,
        "state_norm_by_prior": state_norms,
        "update_norm_by_prior": update_norms,
        "gate": {
            "mean": float(gate.mean().item()) if gate is not None else None,
            "std": float(gate.std(unbiased=False).item()) if gate is not None else None,
            "below_0_05": float((gate < 0.05).float().mean().item()) if gate is not None else None,
            "above_0_95": float((gate > 0.95).float().mean().item()) if gate is not None else None,
        },
        "weights_unchanged": before == after,
        "autograd_history_retained": bool(state.grad_fn is not None),
    }


def temporal_credit_audit(model: AdaptiveCore, gap_steps: int = 4) -> dict[str, Any]:
    """Prove a delayed query loss crosses intervening updates to the first write."""

    model.zero_grad(set_to_none=True)
    model.train()
    state = model.initial_state(1)
    if model.config.context_count > 1:
        evidence = torch.tensor([[0.0, 0.0, 0.0]])
    else:
        evidence = torch.tensor([[0.0, 0.0]])

    def query_observation(index: int) -> torch.Tensor:
        if model.config.context_count > 1:
            return torch.tensor([[float(index % 2), 0.0, 1.0]])
        return torch.tensor([[float(index % 2), 1.0]])

    prediction = model.predict(evidence, state)
    written = model.update(evidence, state, torch.tensor([1]), prediction).state
    written.retain_grad()
    current = written
    intermediate_states: list[torch.Tensor] = []
    for index in range(gap_steps):
        if model.config.context_count == 4:
            gap_observation = torch.tensor([[float(index % 2), 3.0, 1.0]])
        else:
            gap_observation = query_observation(index)
        gap_prediction = model.predict(gap_observation, current)
        current = model.update(
            gap_observation,
            current,
            torch.tensor([index % 2]),
            gap_prediction,
        ).state
        current.retain_grad()
        intermediate_states.append(current)
    query = model.predict(query_observation(1), current)
    loss = F.cross_entropy(query.logits, torch.tensor([0]))
    loss.backward()
    module_norms: dict[str, float] = {}
    for name in (
        "observation_encoder",
        "state_reader",
        "thought_block",
        "output_head",
        "feedback_encoder",
        "writer",
        "relation_encoder",
        "write_gate",
    ):
        module = getattr(model, name, None)
        if module is None:
            continue
        squared = sum(
            float(parameter.grad.detach().square().sum().item())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        module_norms[name] = math.sqrt(squared)
    return {
        "gap_steps": gap_steps,
        "gap_updates_applied": len(intermediate_states),
        "early_state_gradient_norm": float(written.grad.norm().item()),
        "intermediate_state_gradient_norms": [
            float(state.grad.norm().item()) for state in intermediate_states
        ],
        "module_gradient_norms": module_norms,
        "writer_reached": any(
            module_norms.get(name, 0.0) > 0 for name in ("writer", "relation_encoder", "write_gate")
        ),
        "reader_reached": module_norms.get("state_reader", 0.0) > 0,
        "thought_block_reached": module_norms.get("thought_block", 0.0) > 0,
        "observation_encoder_reached": module_norms.get("observation_encoder", 0.0) > 0,
    }


def _collect_states(
    model: AdaptiveCore, episodes: list[BinaryEpisode] | list[ContextEpisode]
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = batch_adaptation_episodes(episodes)
    state = model.initial_state(len(episodes))
    states: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for time_index in range(batch.public.shape[1]):
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            prediction = model.predict(public, state)
            state = model.update(public, state, outcome, prediction).state
            states.append(state.clone())
            labels.append(batch.hidden_rules.clone())
    return torch.cat(states), torch.cat(labels)


def linear_rule_probe(
    model: AdaptiveCore,
    train_episodes: list[BinaryEpisode] | list[ContextEpisode],
    test_episodes: list[BinaryEpisode] | list[ContextEpisode],
    *,
    seed: int = 0,
    steps: int = 300,
) -> dict[str, float | int]:
    """Train a research-only linear probe on frozen state samples."""

    train_states, train_labels = _collect_states(model, train_episodes)
    test_states, test_labels = _collect_states(model, test_episodes)
    torch.manual_seed(seed)
    probe = nn.Linear(train_states.shape[1], int(train_labels.max().item()) + 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_states), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(train_states).argmax(-1) == train_labels).float().mean())
        test_accuracy = float((probe(test_states).argmax(-1) == test_labels).float().mean())
    return {
        "train_accuracy": train_accuracy,
        "held_out_accuracy": test_accuracy,
        "training_samples": len(train_states),
        "held_out_samples": len(test_states),
        "probe_steps": steps,
    }


def donor_state_swap(model: AdaptiveCore, episodes: list[BinaryEpisode]) -> dict[str, float]:
    """Swap one-evidence states between paired opposite-rule worlds and test donor behavior."""

    if len(episodes) % 2:
        raise ValueError("donor swaps require paired episodes")
    batch = BinaryBatch.from_episodes(episodes)
    if any(
        batch.hidden_rules[index] == batch.hidden_rules[index + 1]
        for index in range(0, len(episodes), 2)
    ):
        raise ValueError("adjacent donor pairs must have opposite rules")
    state = model.initial_state(len(episodes))
    model.eval()
    with torch.no_grad():
        public = batch.public[:, 0]
        outcome = batch.outcomes[:, 0]
        state = model.update(public, state, outcome, model.predict(public, state)).state
        donor_indices = torch.arange(len(episodes)).reshape(-1, 2).flip(1).flatten()
        donor_state = state[donor_indices]
        query_public = batch.public[:, 1]
        predictions = model.predict(query_public, donor_state).logits.argmax(-1)
        donor_targets = batch.outcomes[donor_indices, 1]
        recipient_targets = batch.outcomes[:, 1]
    return {
        "donor_rule_consistency": float((predictions == donor_targets).float().mean()),
        "recipient_rule_consistency": float((predictions == recipient_targets).float().mean()),
        "state_pairs": len(episodes) // 2,
    }


def context_donor_state_swap(
    model: AdaptiveCore, episodes: list[ContextEpisode]
) -> dict[str, float]:
    """Swap fully informed two-rule states with the opposite rule table in each group."""

    if len(episodes) % 4:
        raise ValueError("context donor swaps require balanced groups of four")
    batch = batch_adaptation_episodes(episodes)
    state = model.initial_state(len(episodes))
    model.eval()
    with torch.no_grad():
        for time_index in range(2):
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            state = model.update(public, state, outcome, model.predict(public, state)).state
        donors = torch.arange(len(episodes)).reshape(-1, 4).flip(1).flatten()
        query_public = batch.public[:, 2]
        predictions = model.predict(query_public, state[donors]).logits.argmax(-1)
        donor_targets = batch.outcomes[donors, 2]
        recipient_targets = batch.outcomes[:, 2]
    return {
        "donor_rule_consistency": float((predictions == donor_targets).float().mean()),
        "recipient_rule_consistency": float((predictions == recipient_targets).float().mean()),
        "state_groups": len(episodes) // 4,
    }


def composition_donor_state_swap(
    model: AdaptiveCore, episodes: list[ContextEpisode]
) -> dict[str, float]:
    """Swap states that differ in exactly one rule so composed predictions must flip."""

    if len(episodes) % 4:
        raise ValueError("composition donor swaps require balanced groups of four")
    batch = batch_adaptation_episodes(episodes)
    state = model.initial_state(len(episodes))
    model.eval()
    with torch.no_grad():
        for time_index in range(2):
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            state = model.update(public, state, outcome, model.predict(public, state)).state
        donors = (torch.arange(len(episodes)).reshape(-1, 4) ^ 1).flatten()
        query_public = batch.public[:, 2]
        if not torch.all(query_public[:, 1] == 2):
            raise ValueError("first post-evidence query must test composition")
        predictions = model.predict(query_public, state[donors]).logits.argmax(-1)
        donor_targets = batch.outcomes[donors, 2]
        recipient_targets = batch.outcomes[:, 2]
    return {
        "donor_rule_consistency": float((predictions == donor_targets).float().mean()),
        "recipient_rule_consistency": float((predictions == recipient_targets).float().mean()),
        "state_groups": len(episodes) // 4,
    }


def delayed_donor_state_swap(
    model: AdaptiveCore, episodes: list[ContextEpisode]
) -> dict[str, float]:
    """Swap one-rule-different states after the complete distractor interval."""

    if len(episodes) % 4:
        raise ValueError("delayed donor swaps require balanced groups of four")
    batch = batch_adaptation_episodes(episodes)
    state = model.initial_state(len(episodes))
    query_index = next(
        index
        for index in range(2, batch.public.shape[1])
        if int(batch.public[0, index, 1].item()) != 3
    )
    model.eval()
    with torch.no_grad():
        for time_index in range(query_index):
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            state = model.update(public, state, outcome, model.predict(public, state)).state
        donors = (torch.arange(len(episodes)).reshape(-1, 4) ^ 1).flatten()
        query_public = batch.public[:, query_index]
        predictions = model.predict(query_public, state[donors]).logits.argmax(-1)
        donor_targets = batch.outcomes[donors, query_index]
        recipient_targets = batch.outcomes[:, query_index]
    return {
        "donor_rule_consistency": float((predictions == donor_targets).float().mean()),
        "recipient_rule_consistency": float((predictions == recipient_targets).float().mean()),
        "state_groups": len(episodes) // 4,
        "delay_steps": query_index - 2,
    }


def state_component_ablation(
    model: AdaptiveCore, episodes: list[BinaryEpisode] | list[ContextEpisode]
) -> dict[str, float]:
    """Zero each state component after one evidence event and measure subsequent accuracy."""

    batch = batch_adaptation_episodes(episodes)
    model.eval()
    results: dict[str, float] = {}
    with torch.no_grad():
        state = model.initial_state(len(episodes))
        evidence_steps = model.config.rule_count
        for evidence_index in range(evidence_steps):
            evidence_public = batch.public[:, evidence_index]
            evidence_outcome = batch.outcomes[:, evidence_index]
            state = model.update(
                evidence_public,
                state,
                evidence_outcome,
                model.predict(evidence_public, state),
            ).state
        for component in range(state.shape[1]):
            ablated = state.clone()
            ablated[:, component] = 0
            correct = 0
            count = 0
            for time_index in range(evidence_steps, batch.public.shape[1]):
                public = batch.public[:, time_index]
                outcome = batch.outcomes[:, time_index]
                prediction = model.predict(public, ablated)
                valid = (
                    public[:, 1] != 3
                    if public.shape[1] == 3
                    else torch.ones(len(episodes), dtype=torch.bool)
                )
                correct += int((prediction.logits.argmax(-1)[valid] == outcome[valid]).sum())
                count += int(valid.sum())
            results[str(component)] = correct / count
    return results


def quantized_state_evaluation(
    model: AdaptiveCore, episodes: list[BinaryEpisode] | list[ContextEpisode], bits: int
) -> dict[str, Any]:
    """Quantize the runtime state after each update without changing learned weights."""

    if bits not in {1, 2, 4, 8, 16}:
        raise ValueError("supported state quantization is 1, 2, 4, 8, or 16 bits")
    batch = batch_adaptation_episodes(episodes)
    state = model.initial_state(len(episodes))
    correct = [0] * batch.public.shape[1]
    task_correct = [0] * batch.public.shape[1]
    task_counts = [0] * batch.public.shape[1]
    levels = 2**bits
    model.eval()
    with torch.no_grad():
        for time_index in range(batch.public.shape[1]):
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            prediction = model.predict(public, state)
            predicted = prediction.logits.argmax(-1)
            correct[time_index] = int((predicted == outcome).sum())
            valid = (
                public[:, 1] != 3
                if public.shape[1] == 3
                else torch.ones(len(episodes), dtype=torch.bool)
            )
            task_correct[time_index] = int((predicted[valid] == outcome[valid]).sum())
            task_counts[time_index] = int(valid.sum())
            state = model.update(public, state, outcome, prediction).state.clamp(-1, 1)
            if bits == 1:
                state = torch.where(state >= 0, torch.ones_like(state), -torch.ones_like(state))
            else:
                state = torch.round((state + 1) * (levels - 1) / 2) * 2 / (levels - 1) - 1
    curve = {str(index): value / len(episodes) for index, value in enumerate(correct)}
    evidence_steps = model.config.rule_count
    return {
        "bits_per_value": bits,
        "canonical_state_bits": bits * model.config.state_dim,
        "canonical_state_bytes_ceiling": math.ceil(bits * model.config.state_dim / 8),
        "adaptation_curve": curve,
        "post_evidence_accuracy": sum(correct[1:]) / (len(episodes) * (len(correct) - 1)),
        "fully_informed_accuracy": sum(task_correct[evidence_steps:])
        / sum(task_counts[evidence_steps:]),
    }


def compare_compatible(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    """Reject comparisons whose public conditions or resource budgets differ."""

    keys = ("evaluation_seed_start", "episode_steps", "evaluation_pairs", "training_observations")
    mismatches = [key for key in keys if left.get(key) != right.get(key)]
    if mismatches:
        raise ValueError(f"incompatible comparison fields: {', '.join(mismatches)}")
    return {
        "post_evidence_accuracy_difference": float(left["post_evidence_accuracy"])
        - float(right["post_evidence_accuracy"])
    }


def evaluate_rule_change(model: nn.Module, episodes: list[ContextEpisode]) -> dict[str, Any]:
    """Measure recovery of the changed rule and retention of the unrelated rule."""

    batch = batch_adaptation_episodes(episodes)
    if any(episode.rule_change_step is None for episode in episodes):
        raise ValueError("rule-change metadata is missing")
    change_steps = torch.tensor([int(episode.rule_change_step) for episode in episodes])
    state = model.initial_state(len(episodes))
    relative_correct: dict[int, int] = {}
    relative_count: dict[int, int] = {}
    change_correct = 0
    recovery_correct = 0
    recovery_count = 0
    recovery_relative_correct: dict[int, int] = {}
    recovery_relative_count: dict[int, int] = {}
    retention_correct = 0
    retention_count = 0
    change_update_norms: list[float] = []
    model.eval()
    before = state_dict_digest(model)
    with torch.no_grad():
        for time_index in range(batch.public.shape[1]):
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            prediction = model.predict(public, state)
            predicted = prediction.logits.argmax(-1)
            contexts = public[:, 1].long()
            relative = time_index - change_steps
            for offset in relative.unique().tolist():
                mask = relative == offset
                relative_correct[offset] = relative_correct.get(offset, 0) + int(
                    (predicted[mask] == outcome[mask]).sum()
                )
                relative_count[offset] = relative_count.get(offset, 0) + int(mask.sum())
            change_mask = relative == 0
            change_correct += int((predicted[change_mask] == outcome[change_mask]).sum())
            recovery_mask = (relative > 0) & ((contexts == 0) | (contexts == 2))
            retention_mask = (relative > 0) & (contexts == 1)
            recovery_correct += int((predicted[recovery_mask] == outcome[recovery_mask]).sum())
            recovery_count += int(recovery_mask.sum())
            for offset in relative[recovery_mask].unique().tolist():
                mask = recovery_mask & (relative == offset)
                recovery_relative_correct[offset] = recovery_relative_correct.get(offset, 0) + int(
                    (predicted[mask] == outcome[mask]).sum()
                )
                recovery_relative_count[offset] = recovery_relative_count.get(offset, 0) + int(
                    mask.sum()
                )
            retention_correct += int((predicted[retention_mask] == outcome[retention_mask]).sum())
            retention_count += int(retention_mask.sum())
            old_state = state
            state = model.update(public, state, outcome, prediction).state
            if change_mask.any():
                change_update_norms.extend(
                    (state.float() - old_state.float()).norm(dim=-1)[change_mask].tolist()
                )
    if before != state_dict_digest(model):
        raise RuntimeError("rule-change evaluation modified weights")
    relative_accuracy = {
        str(offset): relative_correct[offset] / relative_count[offset]
        for offset in sorted(relative_count)
    }
    recovery_speed = next(
        (
            offset
            for offset in sorted(recovery_relative_count)
            if recovery_relative_correct[offset] / recovery_relative_count[offset] >= 0.85
        ),
        None,
    )
    return {
        "change_steps": sorted(set(change_steps.tolist())),
        "accuracy_by_relative_step": relative_accuracy,
        "change_step_accuracy": change_correct / len(episodes),
        "post_feedback_recovery_accuracy": recovery_correct / recovery_count,
        "unrelated_rule_retention_accuracy": retention_correct / retention_count,
        "recovery_speed_steps": recovery_speed,
        "change_update_norm": sum(change_update_norms) / len(change_update_norms),
        "weights_unchanged": True,
    }


def linear_reversal_probe(
    model: AdaptiveCore,
    train_episodes: list[ContextEpisode],
    test_episodes: list[ContextEpisode],
    *,
    seed: int = 0,
    steps: int = 300,
) -> dict[str, float | int]:
    """Decode the currently valid rule pair before and after reversal."""

    def collect(episodes: list[ContextEpisode]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch_adaptation_episodes(episodes)
        state = model.initial_state(len(episodes))
        states: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for time_index in range(batch.public.shape[1]):
                public = batch.public[:, time_index]
                outcome = batch.outcomes[:, time_index]
                prediction = model.predict(public, state)
                state = model.update(public, state, outcome, prediction).state
                current_labels = []
                for episode in episodes:
                    rules = (
                        episode.final_rules
                        if time_index >= int(episode.rule_change_step)
                        else episode.hidden_rules
                    )
                    current_labels.append(rules[0] * 2 + rules[1])
                states.append(state.clone())
                labels.append(torch.tensor(current_labels))
        return torch.cat(states), torch.cat(labels)

    train_states, train_labels = collect(train_episodes)
    test_states, test_labels = collect(test_episodes)
    torch.manual_seed(seed)
    probe = nn.Linear(train_states.shape[1], 4)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_states), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(train_states).argmax(-1) == train_labels).float().mean())
        test_accuracy = float((probe(test_states).argmax(-1) == test_labels).float().mean())
    return {
        "train_accuracy": train_accuracy,
        "held_out_accuracy": test_accuracy,
        "training_samples": len(train_states),
        "held_out_samples": len(test_states),
        "probe_steps": steps,
    }


def state_geometry(model: AdaptiveCore, episodes: list[ContextEpisode]) -> dict[str, float | int]:
    """Summarize separation, rank, and within-world stability of bounded state."""

    batch = batch_adaptation_episodes(episodes)
    state = model.initial_state(len(episodes))
    after_evidence: torch.Tensor | None = None
    before_first_query: torch.Tensor | None = None
    model.eval()
    with torch.no_grad():
        for time_index in range(batch.public.shape[1]):
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            if (
                time_index >= model.config.rule_count
                and before_first_query is None
                and int(public[0, 1].item()) != 3
            ):
                before_first_query = state.clone()
            state = model.update(public, state, outcome, model.predict(public, state)).state
            if time_index == model.config.rule_count - 1:
                after_evidence = state.clone()
    if after_evidence is None:
        raise ValueError("episodes ended before complete evidence")
    if before_first_query is None:
        before_first_query = after_evidence
    centered = after_evidence - after_evidence.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    tolerance = max(centered.shape) * torch.finfo(centered.dtype).eps * float(singular.max())
    effective_rank = int((singular > tolerance).sum())
    labels = batch.hidden_rules
    means = torch.stack([after_evidence[labels == label].mean(dim=0) for label in range(4)])
    distances = torch.cdist(means, means)
    nonzero_distances = distances[~torch.eye(4, dtype=torch.bool)]
    cosine = F.cosine_similarity(after_evidence, before_first_query, dim=-1)
    return {
        "state_variance": float(after_evidence.var(dim=0, unbiased=False).mean()),
        "effective_rank": effective_rank,
        "minimum_rule_centroid_distance": float(nonzero_distances.min()),
        "mean_rule_centroid_distance": float(nonzero_distances.mean()),
        "delay_drift_norm": float((before_first_query - after_evidence).norm(dim=-1).mean()),
        "delay_cosine_similarity": float(cosine.mean()),
    }


def linear_surface_probe(
    model: AdaptiveCore,
    train_episodes: list[ContextEpisode],
    test_episodes: list[ContextEpisode],
    *,
    seed: int = 0,
    steps: int = 300,
) -> dict[str, float | int]:
    """Test whether state retains the nuisance surface-bit relabelling."""

    def final_states(episodes: list[ContextEpisode]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch_adaptation_episodes(episodes)
        state = model.initial_state(len(episodes))
        model.eval()
        with torch.no_grad():
            for time_index in range(batch.public.shape[1]):
                public = batch.public[:, time_index]
                outcome = batch.outcomes[:, time_index]
                state = model.update(public, state, outcome, model.predict(public, state)).state
        labels = torch.tensor([episode.surface_flip for episode in episodes])
        return state, labels

    train_states, train_labels = final_states(train_episodes)
    test_states, test_labels = final_states(test_episodes)
    torch.manual_seed(seed)
    probe = nn.Linear(train_states.shape[1], 2)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_states), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(train_states).argmax(-1) == train_labels).float().mean())
        test_accuracy = float((probe(test_states).argmax(-1) == test_labels).float().mean())
    return {
        "train_accuracy": train_accuracy,
        "held_out_accuracy": test_accuracy,
        "training_samples": len(train_states),
        "held_out_samples": len(test_states),
        "probe_steps": steps,
    }
