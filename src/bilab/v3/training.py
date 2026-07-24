"""Training and mechanistic diagnostics for Cognitive Core v3."""

from __future__ import annotations

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
    balanced_delayed_episodes,
    balanced_reversal_episodes,
    batch_adaptation_episodes,
    random_delayed_episodes,
)
from bilab.training.v1 import (
    composition_donor_state_swap,
    evaluate_model,
    evaluate_rule_change,
    linear_reversal_probe,
    linear_surface_probe,
    state_component_ablation,
    state_dict_digest,
    state_geometry,
)
from bilab.v2.training import (
    _episodes,
    relation_feature_probe,
    routing_diagnostics,
    routing_interventions,
    slot_permutation_equivariance,
)
from bilab.v3.environments import v3_training_episodes
from bilab.v3.models import (
    BaseV3Core,
    V3ModelConfig,
    build_v3_model,
    count_v3_parameters,
    validate_v3_raw_trace,
)


@dataclass(frozen=True)
class V3TrainConfig:
    family: str
    stage: str
    seed: int
    world_seed_start: int
    validation_seed_start: int
    optimizer_steps: int = 800
    batch_groups: int = 8
    episode_steps: int = 12
    hidden_dim: int = 64
    state_dim: int = 2
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    write_cost: float = 0.0
    predictive_state_training: bool = False
    validation_groups: int = 64
    validation_interval: int = 50
    torch_threads: int = 6
    checkpoint_selection: str = "final_step"
    initialization: str = "random"

    def __post_init__(self) -> None:
        if self.stage not in {"v3a", "v3b", "v3c"}:
            raise ValueError(f"unknown v3 stage: {self.stage}")
        if self.optimizer_steps < 1 or self.batch_groups < 1:
            raise ValueError("optimizer_steps and batch_groups must be positive")
        if self.episode_steps != 12:
            raise ValueError("V3 equal-resource training uses full 12-step BPTT")
        if not 1 <= self.torch_threads <= 6:
            raise ValueError("V3 permits one through six PyTorch threads")
        if self.state_dim != 2:
            raise ValueError("V3 primary candidates use exactly two state values")
        if self.write_cost < 0:
            raise ValueError("write_cost cannot be negative")
        if self.checkpoint_selection != "final_step":
            raise ValueError("V3 preregisters final-step checkpoint selection")
        if self.initialization not in {"random", "staged"}:
            raise ValueError("V3 initialization must be random or staged")
        model = V3ModelConfig(
            family=self.family,
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
        )
        expected_stage = build_v3_model(model).stage
        if expected_stage != self.stage:
            raise ValueError(
                f"family {self.family} implements {expected_stage}, not declared {self.stage}"
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> V3TrainConfig:
        return cls(**value)

    def model_config(self) -> V3ModelConfig:
        return V3ModelConfig(
            family=self.family,
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
        )


@dataclass
class V3TrainRun:
    model: BaseV3Core
    metrics: dict[str, Any]


def compose_v3c_staged_state_dict(
    target_model: BaseV3Core,
    raw_relation_model: BaseV3Core,
    preservation_model: BaseV3Core,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compose a V3C initialization without silently mixing incompatible subsystem meanings.

    The raw V3A source contributes only the learned raw relation encoder. The V3B source
    contributes every target-compatible parameter except its scaffolded relation encoder, including
    the learned router, write controller, value writer, reader, thought block, and output pathway.
    The target architecture remains authoritative: only exact name-and-shape matches are
    transferred.
    """

    if target_model.stage != "v3c" or target_model.relation_scaffold:
        raise ValueError("staged composition target must be a raw V3C model")
    if raw_relation_model.stage != "v3a" or raw_relation_model.relation_scaffold:
        raise ValueError("raw staged source must be a non-scaffolded V3A model")
    if preservation_model.stage != "v3b" or not preservation_model.relation_scaffold:
        raise ValueError("preservation staged source must be a relation-scaffolded V3B model")
    if not preservation_model.hard_preservation:
        raise ValueError("preservation staged source must implement hard preservation")

    target = target_model.state_dict()
    raw = raw_relation_model.state_dict()
    preservation = preservation_model.state_dict()
    composed: dict[str, torch.Tensor] = {}
    provenance: dict[str, str] = {}

    for name, tensor in preservation.items():
        if name.startswith("writer_encoder."):
            continue
        if name in target and target[name].shape == tensor.shape:
            composed[name] = tensor.detach().clone()
            provenance[name] = "v3b_preservation"

    for name, tensor in raw.items():
        if not name.startswith("writer_encoder."):
            continue
        if name in target and target[name].shape == tensor.shape:
            composed[name] = tensor.detach().clone()
            provenance[name] = "v3a_raw_relation"

    required_prefixes = (
        "writer_encoder.",
        "value_candidate.",
        "router.",
        "write_controller.",
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in composed)
    ]
    if missing:
        raise ValueError(f"staged V3C initialization lacks required subsystems: {missing}")

    return composed, {
        "policy": "v3a-writer-encoder-plus-v3b-compatible-non-writer",
        "raw_relation_family": raw_relation_model.family,
        "preservation_family": preservation_model.family,
        "raw_relation_digest": state_dict_digest(raw_relation_model),
        "preservation_digest": state_dict_digest(preservation_model),
        "transferred_parameter_names": sorted(composed),
        "parameter_source": provenance,
    }


def seed_v3(seed: int, threads: int = 6) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def v3_generation_ranges(config: V3TrainConfig) -> dict[str, tuple[int, int]]:
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


def assert_v3_ranges_disjoint(
    config: V3TrainConfig,
    *,
    additional: dict[str, tuple[int, int]] | None = None,
) -> None:
    ranges = v3_generation_ranges(config)
    ranges.update(additional or {})
    names = list(ranges)
    for left_index, left in enumerate(names):
        left_start, left_end = ranges[left]
        if left_start > left_end:
            raise ValueError(f"invalid V3 seed range: {left}")
        for right in names[left_index + 1 :]:
            right_start, right_end = ranges[right]
            if max(left_start, right_start) <= min(left_end, right_end):
                raise ValueError(f"V3 generation seed ranges overlap: {left} and {right}")


def v3_episode_objective(
    model: BaseV3Core,
    batch: ContextBatch,
    *,
    write_cost: float = 0.0,
    predictive_state_training: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Outcome-only full BPTT plus an optional label-free global write cost."""

    batch_size, episode_steps, _ = batch.public.shape
    state = model.initial_state(batch_size, batch.public.device)
    primary_losses: list[torch.Tensor] = []
    write_probabilities: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    correct = 0
    valid_count = 0
    informed_correct = 0
    informed_count = 0
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
                informed_correct += int((predicted[valid] == outcome[valid]).sum())
                informed_count += int(valid.sum())
        update = model.update(public, state, outcome, prediction)
        if not model.relation_scaffold:
            validate_v3_raw_trace(update.trace, public, outcome, state)
        if predictive_state_training and time_index >= 2:
            # A V3A-only isolation scaffold: later public queries supervise predictions from the
            # same initial evidence state instead of allowing their feedback to become a shortcut.
            # The writer still sees no relation or hidden label, and held-out evaluation removes
            # this training-only freeze.
            pass
        else:
            state = update.state
        write_probabilities.append(update.trace.write_probability)
        gates.append(update.gate.flatten())
    primary = torch.stack(primary_losses).mean()
    mean_write_probability = torch.cat(write_probabilities).mean()
    total = primary + write_cost * mean_write_probability
    all_gates = torch.cat(gates)
    return total, {
        "primary_loss": float(primary.detach()),
        "write_cost_loss": float(mean_write_probability.detach()),
        "accuracy": correct / valid_count,
        "fully_informed_accuracy": informed_correct / informed_count,
        "gate_mean": float(all_gates.mean().detach()),
        "gate_zero_fraction": float((all_gates == 0).float().mean().detach()),
        "final_state_norm": float(state.float().norm(dim=-1).mean().detach()),
    }


def _gradient_metrics(model: nn.Module) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name, module in model.named_children():
        gradients = [
            parameter.grad.detach().flatten()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            continue
        combined = torch.cat(gradients)
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


def train_v3_candidate(
    config: V3TrainConfig,
    *,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    initialization_metadata: dict[str, Any] | None = None,
) -> V3TrainRun:
    """Train one V3 candidate on fresh mixed worlds under the equal V2 budget."""

    assert_v3_ranges_disjoint(config)
    seed_v3(config.seed, config.torch_threads)
    model = build_v3_model(config.model_config())
    transferred_parameters: list[str] = []
    if initial_state_dict is not None:
        own = model.state_dict()
        compatible = {
            name: tensor
            for name, tensor in initial_state_dict.items()
            if name in own and own[name].shape == tensor.shape
        }
        model.load_state_dict(compatible, strict=False)
        transferred_parameters = sorted(compatible)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    validation = _episodes(
        "mixed",
        seed_start=config.validation_seed_start,
        groups=config.validation_groups,
        episode_steps=config.episode_steps,
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    latest_gradient: dict[str, Any] = {}
    clipping_count = 0
    for step in range(config.optimizer_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        episodes = v3_training_episodes(
            seed_start=config.world_seed_start + step * config.batch_groups,
            groups=config.batch_groups,
            episode_steps=config.episode_steps,
            mix_index=step,
        )
        batch = batch_adaptation_episodes(episodes)
        loss, train_metrics = v3_episode_objective(
            model,
            batch,
            write_cost=config.write_cost,
            predictive_state_training=config.predictive_state_training,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite V3 loss at optimizer step {step + 1}")
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
            history.append(
                {
                    "optimizer_step": current,
                    "training_observations": current
                    * config.batch_groups
                    * 4
                    * config.episode_steps,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
    elapsed = time.perf_counter() - started
    observations = config.optimizer_steps * config.batch_groups * 4 * config.episode_steps
    return V3TrainRun(
        model=model,
        metrics={
            "configuration": asdict(config),
            "parameter_count": count_v3_parameters(model),
            "persistent_state_bytes": model.persistent_bytes,
            "optimizer_steps": config.optimizer_steps,
            "training_observations": observations,
            "checkpoint_selection": "final_step",
            "final_validation_accuracy": float(
                history[-1]["validation"]["fully_informed_accuracy"]
            ),
            "history": history,
            "gradient_diagnostics": latest_gradient,
            "gradient_clipping_fraction": clipping_count / config.optimizer_steps,
            "training_wall_seconds": elapsed,
            "peak_ram_bytes": _peak_ram_bytes(),
            "initialization": config.initialization,
            "initialization_metadata": initialization_metadata or {},
            "transferred_parameter_names": transferred_parameters,
            "bptt": {
                "detach_inside_episode": False,
                "detach_at_world_boundary": True,
                "window_steps": config.episode_steps,
                "graph_depth_updates": config.episode_steps,
            },
        },
    )


def temporal_gradient_audit_v3(
    model: BaseV3Core,
    *,
    gap_steps: int = 8,
) -> dict[str, Any]:
    """Backpropagate one delayed query through evidence and intervening updates."""

    model.zero_grad(set_to_none=True)
    model.train()
    state = model.initial_state(1)
    retained: list[torch.Tensor] = []
    evidence = (
        (torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([1])),
        (torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([1])),
    )
    for public, outcome in evidence:
        update = model.update(public, state, outcome, model.predict(public, state))
        state = update.state
        state.retain_grad()
        retained.append(state)
    for index in range(gap_steps):
        public = torch.tensor([[float(index % 2), 3.0, 1.0]])
        outcome = torch.tensor([index % 2])
        update = model.update(public, state, outcome, model.predict(public, state))
        state = update.state
        state.retain_grad()
        retained.append(state)
    query = torch.tensor([[1.0, 2.0, 1.0]])
    loss = F.cross_entropy(model.predict(query, state).logits, torch.tensor([1]))
    loss.backward()
    state_norms = [float(item.grad.norm()) if item.grad is not None else 0.0 for item in retained]
    modules: dict[str, float] = {}
    for name in (
        "writer_encoder",
        "value_candidate",
        "code_logits",
        "router",
        "write_controller",
        "state_reader",
        "observation_encoder",
        "thought_block",
        "output_head",
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
    finite = all(math.isfinite(value) for value in (*state_norms, *modules.values()))
    values = list(modules.values())
    return {
        "gap_steps": gap_steps,
        "updates_crossed": len(retained),
        "state_gradient_norms": state_norms,
        "minimum_state_gradient_norm": min(state_norms),
        "median_state_gradient_norm": statistics.median(state_norms),
        "module_gradient_norms": modules,
        "minimum_module_gradient_norm": min(values) if values else 0.0,
        "median_module_gradient_norm": statistics.median(values) if values else 0.0,
        "zero_module_fraction": (
            sum(value == 0 for value in values) / len(values) if values else 1.0
        ),
        "nan_or_inf": not finite,
        "temporal_distance_reached": (len(state_norms) if min(state_norms) > 0 and finite else 0),
    }


def relation_combination_diagnostics(model: BaseV3Core) -> dict[str, Any]:
    """Inspect latent codes and causal overwrite for every raw input/outcome pair."""

    model.eval()
    latent_by_pair: dict[str, list[float]] = {}
    candidate_by_pair: dict[str, float] = {}
    with torch.no_grad():
        for input_value in range(2):
            for outcome_value in range(2):
                public = torch.tensor([[float(input_value), 0.0, 0.0]])
                outcome = torch.tensor([outcome_value])
                state = model.initial_state(1)
                update = model.update(public, state, outcome, model.predict(public, state))
                key = f"{input_value}{outcome_value}"
                latent_by_pair[key] = update.trace.writer_latent[0].tolist()
                candidate_by_pair[key] = float(update.candidate[0, 0])
        pair_tensors = {key: torch.tensor(value) for key, value in latent_by_pair.items()}
        copy_distance = float((pair_tensors["00"] - pair_tensors["11"]).norm())
        flip_distance = float((pair_tensors["01"] - pair_tensors["10"]).norm())
        cross_distances = [
            float((pair_tensors[left] - pair_tensors[right]).norm())
            for left in ("00", "11")
            for right in ("01", "10")
        ]

        state = model.initial_state(1)
        first_public = torch.tensor([[0.0, 0.0, 0.0]])
        first = model.update(
            first_public,
            state,
            torch.tensor([0]),
            model.predict(first_public, state),
        ).state
        contradictory = model.update(
            first_public,
            first,
            torch.tensor([1]),
            model.predict(first_public, first),
        ).state
        query = torch.tensor([[0.0, 0.0, 1.0]])
        before_prediction = int(model.predict(query, first).logits.argmax(-1))
        after_prediction = int(model.predict(query, contradictory).logits.argmax(-1))
    return {
        "latent_by_raw_pair": latent_by_pair,
        "candidate_by_raw_pair": candidate_by_pair,
        "copy_pair_distance": copy_distance,
        "flip_pair_distance": flip_distance,
        "minimum_cross_relation_distance": min(cross_distances),
        "contradictory_state_change_norm": float((contradictory - first).norm()),
        "prediction_before_contradiction": before_prediction,
        "prediction_after_contradiction": after_prediction,
        "contradictory_feedback_changes_prediction": before_prediction != after_prediction,
    }


def relation_path_interventions(
    model: BaseV3Core,
    episodes: list[ContextEpisode],
) -> dict[str, Any]:
    """Ablate or shuffle only the learned evidence representation."""

    before = state_dict_digest(model)
    original = model.relation_intervention
    results: dict[str, Any] = {}
    try:
        for intervention in ("normal", "zero", "shuffle", "negate"):
            model.set_relation_intervention(intervention)
            results[intervention] = evaluate_model(model, episodes)
    finally:
        model.set_relation_intervention(original)
    if before != state_dict_digest(model):
        raise RuntimeError("relation intervention modified learned weights")
    return results


def preservation_stress(
    model: BaseV3Core,
    *,
    lengths: tuple[int, ...] = (10, 100, 1_000, 10_000, 100_000),
) -> dict[str, Any]:
    """Measure exact writes, drift, retained behavior, and revision after distractors."""

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
        nonzero_write_events = 0
        changed_state_events = 0
        code_transitions = 0
        target_index = 0
        previous_codes: torch.Tensor | None = None
        for step in range(1, max(lengths) + 1):
            public = torch.tensor([[float(step % 2), 3.0, 1.0]]).repeat(4, 1)
            outcome = torch.tensor([(step + index) % 2 for index in range(4)])
            previous = state.clone()
            update = model.update(public, state, outcome, model.predict(public, state))
            state = update.state
            nonzero_write_events += int((update.gate.abs().sum(dim=-1) > 0).sum())
            changed_state_events += int((state != previous).any(dim=-1).sum())
            if update.trace.code_indices is not None:
                codes = update.trace.code_indices
                if previous_codes is not None:
                    code_transitions += int((codes != previous_codes).sum())
                previous_codes = codes.clone()
            if step == lengths[target_index]:
                predictions: list[torch.Tensor] = []
                for operation in range(2):
                    query = torch.tensor([[0.0, float(operation), 1.0]]).repeat(4, 1)
                    predictions.append(model.predict(query, state).logits.argmax(-1))
                retained = sum(
                    int((prediction == target).sum())
                    for prediction, target in zip(predictions, rule_targets, strict=True)
                )
                measurements[str(step)] = {
                    "persistent_bytes": model.persistent_bytes,
                    "canonical_bytes_per_world": len(model.canonical_state_bytes(state[:1])),
                    "tensor_shape": list(state.shape),
                    "finite": bool(torch.isfinite(state).all()),
                    "autograd_history_retained": state.grad_fn is not None,
                    "state_norm": float(state.norm(dim=-1).mean()),
                    "drift_norm": float((state - reference).norm(dim=-1).mean()),
                    "retained_rule_accuracy": retained / 8,
                    "cumulative_nonzero_write_events": nonzero_write_events,
                    "cumulative_changed_state_events": changed_state_events,
                    "cumulative_code_transitions": code_transitions,
                    "predictions_bit_identical_to_reference": bool(torch.equal(state, reference)),
                }
                target_index += 1
                if target_index == len(lengths):
                    break

        contradictory_outcome = 1 - rule_targets[0]
        public = torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)
        revised = model.update(
            public,
            state,
            contradictory_outcome,
            model.predict(public, state),
        ).state
        changed_query = torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)
        retained_query = torch.tensor([[0.0, 1.0, 1.0]]).repeat(4, 1)
        recovery = float(
            (model.predict(changed_query, revised).logits.argmax(-1) == contradictory_outcome)
            .float()
            .mean()
        )
        unrelated = float(
            (model.predict(retained_query, revised).logits.argmax(-1) == rule_targets[1])
            .float()
            .mean()
        )
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
        "total_nonzero_distractor_write_events": nonzero_write_events,
        "total_changed_state_events": changed_state_events,
        "total_code_transitions": code_transitions,
        "post_long_span_one_feedback_recovery": recovery,
        "post_long_span_unrelated_rule_retention": unrelated,
    }


def diagnose_v3_candidate(
    model: BaseV3Core,
    *,
    seed: int,
    seed_base: int,
    groups: int = 64,
    long_lengths: tuple[int, ...] = (10, 100, 1_000, 10_000, 100_000),
) -> dict[str, Any]:
    """Run behavioral, causal, relation, routing, gradient, and preservation diagnostics."""

    reversal = balanced_reversal_episodes(seed_start=seed_base, groups=groups)
    delay = balanced_delayed_episodes(
        seed_start=seed_base + 500,
        groups=groups,
        delay_steps=8,
        query_steps=6,
    )
    composition = balanced_composition_episodes(
        seed_start=seed_base + 1_000,
        groups=groups,
        steps=11,
    )
    probe_train = balanced_reversal_episodes(
        seed_start=seed_base + 1_500,
        groups=groups,
    )
    relabelled = balanced_delayed_episodes(
        seed_start=seed_base + 2_000,
        groups=groups,
        delay_steps=8,
        query_steps=6,
        relabel=True,
    )
    surface_train = balanced_delayed_episodes(
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
        "surface_probe": linear_surface_probe(model, surface_train, relabelled, seed=seed),
        "relation_probe": relation_feature_probe(model, probe_train, reversal, seed=seed),
        "relation_combinations": relation_combination_diagnostics(model),
        "relation_interventions": relation_path_interventions(model, delay),
        "routing": routing_diagnostics(model, delay),
        "routing_interventions": routing_interventions(model, delay),
        "slot_permutation": slot_permutation_equivariance(model, delay),
        "state_geometry": state_geometry(model, delay),
        "temporal_gradient": temporal_gradient_audit_v3(model),
        "preservation": preservation_stress(model, lengths=long_lengths),
    }
