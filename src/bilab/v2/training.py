"""Training and mechanistic diagnostics for generic Cognitive Core v2 candidates."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bilab.environments.adaptation_ladder import (
    ContextBatch,
    ContextEpisode,
    balanced_composition_episodes,
    balanced_context_episodes,
    balanced_delayed_episodes,
    balanced_reversal_episodes,
    batch_adaptation_episodes,
    random_delayed_episodes,
)
from bilab.models.v1 import EpisodicControl
from bilab.training.v1 import (
    composition_donor_state_swap,
    evaluate_model,
    evaluate_rule_change,
    linear_reversal_probe,
    linear_surface_probe,
    state_component_ablation,
    state_geometry,
)
from bilab.v2.models import (
    BaseV2Core,
    V2ModelConfig,
    build_v2_model,
    count_v2_parameters,
    validate_raw_writer_trace,
)


@dataclass(frozen=True)
class V2TrainConfig:
    family: str
    seed: int
    world_seed_start: int
    validation_seed_start: int
    optimizer_steps: int = 400
    batch_groups: int = 8
    episode_steps: int = 12
    hidden_dim: int = 64
    state_dim: int = 2
    thought_cycles: int = 1
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    relation_auxiliary_start: float = 0.0
    relation_auxiliary_end: float = 0.0
    routing_auxiliary_start: float = 0.0
    routing_auxiliary_end: float = 0.0
    validation_groups: int = 32
    validation_interval: int = 50
    torch_threads: int = 6
    budget_bytes: int = 8
    environment: str = "reversal"
    checkpoint_selection: str = "final_step"
    quantization_bits: int = 32

    def __post_init__(self) -> None:
        valid_families = {
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
        if self.family not in valid_families:
            raise ValueError(f"unknown v2 family: {self.family}")
        if self.optimizer_steps < 1 or self.batch_groups < 1:
            raise ValueError("optimizer steps and batch groups must be positive")
        if not 1 <= self.torch_threads <= 6:
            raise ValueError("v2 permits one through six PyTorch threads")
        if self.environment not in {
            "context",
            "composition",
            "delay",
            "reversal",
            "mixed",
        }:
            raise ValueError("unknown v2 training environment")
        expected_steps = {
            "context": self.episode_steps,
            "composition": self.episode_steps,
            "delay": 16,
            "reversal": 12,
            "mixed": 12,
        }[self.environment]
        if self.episode_steps != expected_steps:
            raise ValueError(f"{self.environment} training requires {expected_steps} episode steps")
        if self.checkpoint_selection not in {"best_validation", "final_step"}:
            raise ValueError("unknown checkpoint selection")
        if self.quantization_bits != 32 and self.family not in {
            "raw_router",
            "raw_gru",
            "bilinear_router",
        }:
            raise ValueError("v2 quantization is tested only on generic joint candidates")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> V2TrainConfig:
        return cls(**value)

    def model_config(self) -> V2ModelConfig:
        return V2ModelConfig(
            family=self.family,
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            thought_cycles=self.thought_cycles,
            quantization_bits=self.quantization_bits,
        )


@dataclass
class V2TrainRun:
    model: nn.Module
    metrics: dict[str, Any]


def seed_v2(seed: int, threads: int = 6) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def v2_generation_ranges(config: V2TrainConfig) -> dict[str, tuple[int, int]]:
    return {
        "training": (
            config.world_seed_start,
            config.world_seed_start + config.optimizer_steps * config.batch_groups - 1,
        ),
        "validation": (
            config.validation_seed_start,
            config.validation_seed_start + config.validation_groups - 1,
        ),
    }


def assert_v2_ranges_disjoint(
    config: V2TrainConfig,
    *,
    additional: dict[str, tuple[int, int]] | None = None,
) -> None:
    ranges = v2_generation_ranges(config)
    ranges.update(additional or {})
    names = list(ranges)
    for left_index, left in enumerate(names):
        left_start, left_end = ranges[left]
        if left_start > left_end:
            raise ValueError(f"invalid seed range: {left}")
        for right in names[left_index + 1 :]:
            right_start, right_end = ranges[right]
            if max(left_start, right_start) <= min(left_end, right_end):
                raise ValueError(f"v2 generation seed ranges overlap: {left} and {right}")


def _episodes(
    environment: str,
    *,
    seed_start: int,
    groups: int,
    episode_steps: int,
    mix_index: int | None = None,
) -> list[ContextEpisode]:
    if environment == "mixed":
        kinds = ("context", "composition", "delay_mixed", "reversal")
        if mix_index is not None:
            return _episodes(
                kinds[mix_index % len(kinds)],
                seed_start=seed_start,
                groups=groups,
                episode_steps=episode_steps,
            )
        groups_per_kind = max(1, groups // len(kinds))
        episodes: list[ContextEpisode] = []
        for index, kind in enumerate(kinds):
            episodes.extend(
                _episodes(
                    kind,
                    seed_start=seed_start + index * groups_per_kind,
                    groups=groups_per_kind,
                    episode_steps=episode_steps,
                )
            )
        return episodes
    if environment == "context":
        return balanced_context_episodes(seed_start=seed_start, groups=groups, steps=episode_steps)
    if environment == "composition":
        return balanced_composition_episodes(
            seed_start=seed_start, groups=groups, steps=episode_steps
        )
    if environment == "delay":
        return balanced_delayed_episodes(
            seed_start=seed_start, groups=groups, delay_steps=8, query_steps=6
        )
    if environment == "delay_mixed":
        return balanced_delayed_episodes(
            seed_start=seed_start, groups=groups, delay_steps=4, query_steps=6
        )
    return balanced_reversal_episodes(seed_start=seed_start, groups=groups, include_no_change=True)


def _linear_schedule(start: float, end: float, step: int, total: int) -> float:
    if total == 1:
        return end
    return start + (end - start) * step / (total - 1)


def v2_episode_objective(
    model: nn.Module,
    batch: ContextBatch,
    *,
    relation_auxiliary_weight: float = 0.0,
    routing_auxiliary_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply full BPTT; final evidence uses outcome loss only when both weights are zero."""

    batch_size, episode_steps, _ = batch.public.shape
    state = model.initial_state(batch_size, batch.public.device)
    primary_losses: list[torch.Tensor] = []
    relation_losses: list[torch.Tensor] = []
    routing_losses: list[torch.Tensor] = []
    correct = 0
    valid_count = 0
    fully_informed_correct = 0
    fully_informed_count = 0
    gates: list[torch.Tensor] = []
    routes: list[torch.Tensor] = []
    strengths: list[torch.Tensor] = []
    for time_index in range(episode_steps):
        public = batch.public[:, time_index]
        outcome = batch.outcomes[:, time_index]
        prediction = model.predict(public, state)
        valid = public[:, 1] != 3
        if valid.any():
            primary_losses.append(F.cross_entropy(prediction.logits[valid], outcome[valid]))
            predicted = prediction.logits.argmax(-1)
            correct += int((predicted[valid] == outcome[valid]).sum())
            valid_count += int(valid.sum())
            if time_index >= 2:
                fully_informed_correct += int((predicted[valid] == outcome[valid]).sum())
                fully_informed_count += int(valid.sum())
        update = model.update(public, state, outcome, prediction)
        state = update.state
        if update.gate.numel():
            gates.append(update.gate.flatten())
        trace = getattr(update, "trace", None)
        if trace is not None:
            if not getattr(model, "relation_scaffold", False):
                validate_raw_writer_trace(trace, public, outcome)
            routes.append(trace.route)
            strengths.append(trace.write_strength)
            if relation_auxiliary_weight:
                relation_target = public[:, 0].long() ^ outcome
                relation_losses.append(
                    F.cross_entropy(model.relation_probe(trace.writer_latent), relation_target)
                )
            if routing_auxiliary_weight:
                direct = public[:, 1].long() < 2
                if direct.any():
                    selected = trace.route[direct].clamp_min(1e-8)
                    target = public[direct, 1].long()
                    routing_losses.append(
                        F.nll_loss(selected.log(), target)
                        + F.binary_cross_entropy(
                            trace.write_strength[direct],
                            torch.ones_like(trace.write_strength[direct]),
                        )
                    )
                if (~direct).any():
                    routing_losses.append(
                        F.binary_cross_entropy(
                            trace.write_strength[~direct],
                            torch.zeros_like(trace.write_strength[~direct]),
                        )
                    )
    primary = torch.stack(primary_losses).mean()
    relation_loss = (
        torch.stack(relation_losses).mean() if relation_losses else primary.new_zeros(())
    )
    routing_loss = torch.stack(routing_losses).mean() if routing_losses else primary.new_zeros(())
    total = (
        primary
        + relation_auxiliary_weight * relation_loss
        + routing_auxiliary_weight * routing_loss
    )
    gate = torch.cat(gates) if gates else None
    route = torch.cat(routes) if routes else None
    strength = torch.cat(strengths) if strengths else None
    entropy = (
        -(route.clamp_min(1e-8) * route.clamp_min(1e-8).log()).sum(dim=-1)
        if route is not None
        else None
    )
    return total, {
        "primary_loss": float(primary.detach()),
        "relation_auxiliary_loss": float(relation_loss.detach()),
        "routing_auxiliary_loss": float(routing_loss.detach()),
        "accuracy": correct / valid_count,
        "fully_informed_accuracy": fully_informed_correct / fully_informed_count,
        "gate_mean": float(gate.mean().detach()) if gate is not None else math.nan,
        "route_entropy": float(entropy.mean().detach()) if entropy is not None else math.nan,
        "write_strength_mean": float(strength.mean().detach())
        if strength is not None
        else math.nan,
        "final_state_norm": float(state.float().norm(dim=-1).mean().detach()),
    }


def _gradient_metrics(model: nn.Module) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name, module in model.named_children():
        values = [
            parameter.grad.detach().flatten()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not values:
            continue
        combined = torch.cat(values)
        modules[name] = {
            "norm": float(combined.norm()),
            "zero_fraction": float((combined == 0).float().mean()),
            "maximum_absolute": float(combined.abs().max()),
            "nan_or_inf": bool((~torch.isfinite(combined)).any()),
        }
    return {"modules": modules}


def _peak_ram_bytes() -> int | None:
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (AttributeError, OSError):
        return None


def train_v2_candidate(config: V2TrainConfig) -> V2TrainRun:
    """Train one candidate on procedurally fresh balanced worlds."""

    assert_v2_ranges_disjoint(config)
    seed_v2(config.seed, config.torch_threads)
    model = build_v2_model(config.model_config(), budget_bytes=config.budget_bytes)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    validation = _episodes(
        config.environment,
        seed_start=config.validation_seed_start,
        groups=config.validation_groups,
        episode_steps=config.episode_steps,
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    latest_gradient: dict[str, Any] = {}
    clipping_count = 0
    for step in range(config.optimizer_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        episodes = _episodes(
            config.environment,
            seed_start=config.world_seed_start + step * config.batch_groups,
            groups=config.batch_groups,
            episode_steps=config.episode_steps,
            mix_index=step,
        )
        batch = batch_adaptation_episodes(episodes)
        relation_weight = _linear_schedule(
            config.relation_auxiliary_start,
            config.relation_auxiliary_end,
            step,
            config.optimizer_steps,
        )
        routing_weight = _linear_schedule(
            config.routing_auxiliary_start,
            config.routing_auxiliary_end,
            step,
            config.optimizer_steps,
        )
        loss, train_metrics = v2_episode_objective(
            model,
            batch,
            relation_auxiliary_weight=relation_weight,
            routing_auxiliary_weight=routing_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite v2 loss at optimizer step {step + 1}")
        loss.backward()
        latest_gradient = _gradient_metrics(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        clipping_count += int(float(norm) > config.gradient_clip)
        optimizer.step()
        current = step + 1
        if (
            current == 1
            or current % config.validation_interval == 0
            or current == config.optimizer_steps
        ):
            validation_metrics = evaluate_model(model, validation)
            score = validation_metrics["fully_informed_accuracy"]
            history.append(
                {
                    "optimizer_step": current,
                    "training_observations": current
                    * config.batch_groups
                    * 4
                    * config.episode_steps,
                    "relation_auxiliary_weight": relation_weight,
                    "routing_auxiliary_weight": routing_weight,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            if score > best_score:
                best_score = score
                best_step = current
                best_state = copy.deepcopy(model.state_dict())
    if config.checkpoint_selection == "best_validation":
        model.load_state_dict(best_state)
    else:
        best_step = config.optimizer_steps
        best_score = float(history[-1]["validation"]["fully_informed_accuracy"])
    elapsed = time.perf_counter() - started
    observations = config.optimizer_steps * config.batch_groups * 4 * config.episode_steps
    return V2TrainRun(
        model=model,
        metrics={
            "configuration": asdict(config),
            "parameter_count": count_v2_parameters(model),
            "persistent_state_bytes": int(getattr(model, "persistent_bytes", 0)),
            "optimizer_steps": config.optimizer_steps,
            "training_observations": observations,
            "best_step": best_step,
            "best_validation_accuracy": best_score,
            "checkpoint_selection": config.checkpoint_selection,
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
        },
    )


def temporal_gradient_audit(model: BaseV2Core, gap_steps: int = 6) -> dict[str, Any]:
    """Prove delayed loss reaches raw relation and learned routing pathways."""

    model.zero_grad(set_to_none=True)
    model.train()
    state = model.initial_state(1)
    evidence = (
        (torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([1])),
        (torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([1])),
    )
    retained: list[torch.Tensor] = []
    for public, outcome in evidence:
        prediction = model.predict(public, state)
        state = model.update(public, state, outcome, prediction).state
        state.retain_grad()
        retained.append(state)
    for index in range(gap_steps):
        public = torch.tensor([[float(index % 2), 3.0, 1.0]])
        outcome = torch.tensor([index % 2])
        prediction = model.predict(public, state)
        state = model.update(public, state, outcome, prediction).state
        state.retain_grad()
        retained.append(state)
    query_public = torch.tensor([[1.0, 2.0, 1.0]])
    query = model.predict(query_public, state)
    loss = F.cross_entropy(query.logits, torch.tensor([1]))
    loss.backward()
    state_gradients = [
        float(item.grad.norm()) if item.grad is not None else 0.0 for item in retained
    ]
    modules: dict[str, float] = {}
    for name in (
        "observation_encoder",
        "state_reader",
        "thought_block",
        "output_head",
        "writer_encoder",
        "value_candidate",
        "router",
        "write_controller",
        "update_gate",
        "reset_gate",
        "writer_candidate",
    ):
        module = getattr(model, name, None)
        if module is None:
            continue
        squared = sum(
            float(parameter.grad.detach().square().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        modules[name] = math.sqrt(squared)
    return {
        "gap_steps": gap_steps,
        "updates_crossed": len(retained),
        "state_gradient_norms": state_gradients,
        "minimum_state_gradient_norm": min(state_gradients),
        "median_state_gradient_norm": statistics.median(state_gradients),
        "module_gradient_norms": modules,
        "raw_writer_reached": modules.get("writer_encoder", 0.0) > 0,
        "routing_controller_reached": modules.get("router", 0.0) > 0,
        "writer_candidate_reached": max(
            modules.get("value_candidate", 0.0),
            modules.get("writer_candidate", 0.0),
        )
        > 0,
        "writer_gate_reached": max(
            modules.get("write_controller", 0.0),
            modules.get("update_gate", 0.0),
        )
        > 0,
        "reader_reached": modules.get("state_reader", 0.0) > 0,
        "observation_encoder_reached": modules.get("observation_encoder", 0.0) > 0,
    }


def _fit_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    seed: int,
    steps: int = 250,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    probe = nn.Linear(train_features.shape[1], int(train_labels.max()) + 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_features), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return {
            "train_accuracy": float(
                (probe(train_features).argmax(-1) == train_labels).float().mean()
            ),
            "held_out_accuracy": float(
                (probe(test_features).argmax(-1) == test_labels).float().mean()
            ),
            "train_samples": len(train_labels),
            "held_out_samples": len(test_labels),
            "steps": steps,
        }


def relation_feature_probe(
    model: BaseV2Core,
    train_episodes: list[ContextEpisode],
    test_episodes: list[ContextEpisode],
    *,
    seed: int,
) -> dict[str, Any]:
    """Decode the sufficient relation from internal writer features without training the model."""

    def collect(episodes: list[ContextEpisode]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch_adaptation_episodes(episodes)
        state = model.initial_state(len(episodes))
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for time_index in range(batch.public.shape[1]):
                public = batch.public[:, time_index]
                outcome = batch.outcomes[:, time_index]
                prediction = model.predict(public, state)
                update = model.update(public, state, outcome, prediction)
                state = update.state
                direct = public[:, 1] < 2
                if direct.any():
                    features.append(update.trace.writer_latent[direct])
                    labels.append(public[direct, 0].long() ^ outcome[direct])
        return torch.cat(features), torch.cat(labels)

    train_features, train_labels = collect(train_episodes)
    test_features, test_labels = collect(test_episodes)
    return _fit_probe(
        train_features,
        train_labels,
        test_features,
        test_labels,
        seed=seed,
    )


def routing_diagnostics(model: BaseV2Core, episodes: list[ContextEpisode]) -> dict[str, Any]:
    """Summarize learned address and write behavior for each public context."""

    batch = batch_adaptation_episodes(episodes)
    state = model.initial_state(len(episodes))
    routes: dict[int, list[torch.Tensor]] = {index: [] for index in range(4)}
    strengths: dict[int, list[torch.Tensor]] = {index: [] for index in range(4)}
    model.eval()
    with torch.no_grad():
        for time_index in range(batch.public.shape[1]):
            public = batch.public[:, time_index]
            outcome = batch.outcomes[:, time_index]
            prediction = model.predict(public, state)
            update = model.update(public, state, outcome, prediction)
            state = update.state
            for context in range(4):
                mask = public[:, 1].long() == context
                if mask.any():
                    routes[context].append(update.trace.route[mask])
                    strengths[context].append(update.trace.write_strength[mask])
    result: dict[str, Any] = {}
    for context in range(4):
        route = torch.cat(routes[context])
        strength = torch.cat(strengths[context])
        entropy = -(route.clamp_min(1e-8) * route.clamp_min(1e-8).log()).sum(dim=-1)
        result[str(context)] = {
            "mean_route": route.mean(dim=0).tolist(),
            "route_entropy": float(entropy.mean()),
            "argmax_slot": int(route.mean(dim=0).argmax()),
            "argmax_consistency": float(
                (route.argmax(dim=-1) == route.mean(dim=0).argmax()).float().mean()
            ),
            "write_strength_mean": float(strength.mean()),
            "write_strength_std": float(strength.std(unbiased=False)),
        }
    result["primitive_route_separation"] = float(
        torch.tensor(result["0"]["mean_route"])
        .sub(torch.tensor(result["1"]["mean_route"]))
        .abs()
        .sum()
        / 2
    )
    return result


def routing_interventions(model: BaseV2Core, episodes: list[ContextEpisode]) -> dict[str, Any]:
    """Causally modify only the learned route or writer strength."""

    before = state_digest(model)
    original = model.routing_intervention
    results: dict[str, Any] = {}
    try:
        for intervention in (
            "learned",
            "uniform",
            "random",
            "swapped",
            "writer_disabled",
        ):
            model.set_routing_intervention(intervention)
            results[intervention] = evaluate_model(model, episodes)
    finally:
        model.set_routing_intervention(original)
    if before != state_digest(model):
        raise RuntimeError("routing interventions modified learned weights")
    return results


def slot_permutation_equivariance(
    model: BaseV2Core, episodes: list[ContextEpisode]
) -> dict[str, Any]:
    """Permute memory coordinates and every learned address/read coordinate together."""

    required = ("router", "write_controller", "value_candidate")
    if not all(hasattr(model, name) for name in required):
        return {"applicable": False}
    permuted = copy.deepcopy(model)
    permutation = torch.tensor([1, 0])
    state_columns = slice(-2, None)
    with torch.no_grad():
        permuted.state_reader[0].weight.copy_(permuted.state_reader[0].weight[:, permutation])
        for module in (
            permuted.output_head[0],
            permuted.value_candidate[0],
            permuted.router[0],
            permuted.write_controller[0],
        ):
            module.weight[:, state_columns].copy_(module.weight[:, state_columns][:, permutation])
        permuted.router[-1].weight.copy_(permuted.router[-1].weight[permutation])
        permuted.router[-1].bias.copy_(permuted.router[-1].bias[permutation])
        permuted.rule_probe.weight.copy_(permuted.rule_probe.weight[:, permutation])
    original = evaluate_model(model, episodes)
    corresponding = evaluate_model(permuted, episodes)
    stable_keys = (
        "overall_accuracy",
        "fully_informed_accuracy",
        "mean_loss",
        "adaptation_curve",
        "composition_accuracy",
    )
    return {
        "applicable": True,
        "permutation": [1, 0],
        "original": {key: original[key] for key in stable_keys},
        "corresponding_permutation": {key: corresponding[key] for key in stable_keys},
        "behavior_preserved": all(original[key] == corresponding[key] for key in stable_keys),
    }


def long_sequence_stability(
    model: BaseV2Core,
    *,
    lengths: tuple[int, ...] = (10, 100, 1_000, 10_000, 100_000),
) -> dict[str, Any]:
    """Apply marked random feedback far beyond training while state remains fixed-size."""

    model.eval()
    state = model.initial_state(4)
    evidence_public = (
        torch.tensor([[0.0, 0.0, 0.0]]).repeat(4, 1),
        torch.tensor([[1.0, 1.0, 1.0]]).repeat(4, 1),
    )
    evidence_outcomes = (torch.tensor([0, 1, 0, 1]), torch.tensor([1, 1, 0, 0]))
    rule_targets = (
        evidence_outcomes[0],
        torch.ones(4, dtype=torch.long) ^ evidence_outcomes[1],
    )
    with torch.no_grad():
        for public, outcome in zip(evidence_public, evidence_outcomes, strict=True):
            state = model.update(public, state, outcome, model.predict(public, state)).state
        reference = state.clone()
        measurements: dict[str, Any] = {}
        target_index = 0
        for step in range(1, max(lengths) + 1):
            public = torch.tensor([[float(step % 2), 3.0, 1.0]]).repeat(4, 1)
            outcome = torch.tensor([(step + index) % 2 for index in range(4)])
            state = model.update(public, state, outcome, model.predict(public, state)).state
            if step == lengths[target_index]:
                query_predictions = []
                for operation in range(2):
                    query = torch.tensor([[0.0, float(operation), 1.0]]).repeat(4, 1)
                    query_predictions.append(model.predict(query, state).logits.argmax(-1))
                retained_correct = sum(
                    int((prediction == target).sum())
                    for prediction, target in zip(query_predictions, rule_targets, strict=True)
                )
                measurements[str(step)] = {
                    "persistent_bytes": model.persistent_bytes,
                    "tensor_shape": list(state.shape),
                    "state_norm": float(state.float().norm(dim=-1).mean()),
                    "drift_norm": float((state.float() - reference.float()).norm(dim=-1).mean()),
                    "finite": bool(torch.isfinite(state.float()).all()),
                    "autograd_history_retained": state.grad_fn is not None,
                    "canonical_bytes_per_world": len(model.canonical_state_bytes(state[:1])),
                    "retained_rule_accuracy": retained_correct / 8,
                }
                target_index += 1
                if target_index == len(lengths):
                    break
    return {
        "lengths": list(lengths),
        "measurements": measurements,
        "state_size_constant": len(
            {
                (
                    tuple(item["tensor_shape"][1:]),
                    item["persistent_bytes"],
                    item["canonical_bytes_per_world"],
                )
                for item in measurements.values()
            }
        )
        == 1,
    }


def diagnose_v2_candidate(
    model: BaseV2Core,
    *,
    seed: int,
    seed_base: int,
    groups: int = 64,
) -> dict[str, Any]:
    """Run compact behavioral, probe, routing, gradient, and causal diagnostics."""

    reversal = balanced_reversal_episodes(seed_start=seed_base, groups=groups)
    delay = balanced_delayed_episodes(
        seed_start=seed_base + 500, groups=groups, delay_steps=8, query_steps=6
    )
    composition = balanced_composition_episodes(
        seed_start=seed_base + 1_000, groups=groups, steps=11
    )
    probe_train = balanced_reversal_episodes(seed_start=seed_base + 1_500, groups=groups)
    relabelled = balanced_delayed_episodes(
        seed_start=seed_base + 2_000,
        groups=groups,
        delay_steps=8,
        query_steps=6,
        relabel=True,
    )
    surface_probe_train = balanced_delayed_episodes(
        seed_start=seed_base + 3_000,
        groups=groups,
        delay_steps=8,
        query_steps=6,
        relabel=True,
    )
    random_control = random_delayed_episodes(
        seed_start=seed_base + 2_500,
        count=groups * 4,
        delay_steps=8,
        query_steps=6,
        relabel=True,
    )
    return {
        "delay": evaluate_model(model, delay),
        "composition": evaluate_model(model, composition),
        "reversal": evaluate_rule_change(model, reversal),
        "surface_relabelled_delay": evaluate_model(model, relabelled),
        "random_control": evaluate_model(model, random_control),
        "state_interventions": {
            name: evaluate_model(model, delay, intervention=name, random_seed=seed)
            for name in ("reset", "frozen", "random", "shuffled", "noise")
        },
        "composition_donor_swap": composition_donor_state_swap(model, composition),
        "component_ablation": state_component_ablation(model, delay),
        "rule_probe": linear_reversal_probe(model, probe_train, reversal, seed=seed),
        "surface_probe": linear_surface_probe(model, surface_probe_train, relabelled, seed=seed),
        "relation_probe": relation_feature_probe(model, probe_train, reversal, seed=seed),
        "routing": routing_diagnostics(model, delay),
        "routing_interventions": routing_interventions(model, delay),
        "slot_permutation": slot_permutation_equivariance(model, delay),
        "state_geometry": state_geometry(model, delay),
        "temporal_gradient": temporal_gradient_audit(model, gap_steps=6),
    }


def episodic_state_bytes(model: nn.Module, state: torch.Tensor) -> int:
    if not isinstance(model, EpisodicControl):
        raise ValueError("exact episodic accounting requires EpisodicControl")
    return len(model.canonical_bytes(state))
