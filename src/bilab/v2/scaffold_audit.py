"""Post-hoc interventions that measure Cognitive Core v1 scaffold dependence.

This module loads immutable v1 checkpoints into an instrumented copy of the v1
factorized architecture. It never rewrites a checkpoint or v1 result.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from bilab.environments.adaptation_ladder import (
    balanced_composition_episodes,
    balanced_delayed_episodes,
    balanced_reversal_episodes,
)
from bilab.models.v1 import FactorizedStateCore, StateUpdate
from bilab.training.v1 import (
    composition_donor_state_swap,
    evaluate_model,
    evaluate_rule_change,
    linear_reversal_probe,
    state_dict_digest,
    state_geometry,
)
from bilab.training.v1_checkpoints import load_v1_checkpoint

SCAFFOLD_INTERVENTIONS = (
    "native",
    "relation_zero",
    "relation_random_balanced",
    "relation_incorrect",
    "routing_shuffled_by_world",
    "routing_swapped",
    "distractor_writes_enabled",
    "routing_disabled_uniform",
    "correct_relation_random_routing",
    "raw_input_outcome_no_relation",
)


def v1_scaffold_inventory() -> list[dict[str, Any]]:
    """Return the auditable inventory of task structure supplied to v1."""

    return [
        {
            "name": "signed_input_outcome_relation",
            "location": "bilab.models.v1.FactorizedStateCore.update",
            "mechanism": "x_signed * y_signed",
            "supplies": "the COPY/FLIP sufficient statistic",
            "final_v2_prohibited": True,
        },
        {
            "name": "fixed_context_to_slot_mask",
            "location": "bilab.models.v1.FactorizedStateCore.update",
            "mechanism": "context one-hot columns 0 and 1 multiply the two write gates",
            "supplies": "the correct state destination for primitive operation identities",
            "final_v2_prohibited": True,
        },
        {
            "name": "fixed_nonwrite_for_composition",
            "location": "bilab.models.v1.FactorizedStateCore.update",
            "mechanism": "composition context 2 maps to an all-zero two-slot mask",
            "supplies": "the decision not to overwrite primitive rules on composed feedback",
            "final_v2_prohibited": True,
        },
        {
            "name": "fixed_nonwrite_for_distractors",
            "location": "bilab.models.v1.FactorizedStateCore.update",
            "mechanism": "distractor context 3 maps to an all-zero two-slot mask",
            "supplies": "the decision not to write random marked non-events",
            "final_v2_prohibited": True,
        },
        {
            "name": "globally_fixed_operation_semantics",
            "location": "bilab.environments.adaptation_ladder",
            "mechanism": (
                "public contexts 0/1 are primitive rules, 2 is XOR composition, 3 is delay"
            ),
            "supplies": "stable operation identities across worlds",
            "final_v2d_prohibited": True,
        },
        {
            "name": "learned_composition_reader",
            "location": "bilab.models.v1.AdaptiveCore.predict",
            "mechanism": "generic observation/state reader and output MLP; no XOR branch",
            "supplies": "no hand-coded result; composition is learned from fixed context identity",
            "final_v2_prohibited": False,
        },
        {
            "name": "research_only_hidden_metadata",
            "location": "bilab.environments.adaptation_ladder.ContextEpisode",
            "mechanism": (
                "rules, generation seeds, relabelling, and change metadata stay off model APIs"
            ),
            "supplies": "evaluation labels only",
            "final_v2_prohibited": False,
        },
    ]


class V1ScaffoldAuditCore(FactorizedStateCore):
    """A state-dict-compatible v1 core with one controlled scaffold perturbation."""

    def __init__(self, config: Any, intervention: str) -> None:
        if intervention not in SCAFFOLD_INTERVENTIONS:
            raise ValueError(f"unknown scaffold intervention: {intervention}")
        super().__init__(config)
        self.intervention = intervention
        self._audit_step = 0

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        self._audit_step = 0
        return super().initial_state(batch_size, device)

    def _relation(self, x_signed: torch.Tensor, y_signed: torch.Tensor) -> torch.Tensor:
        native = x_signed * y_signed
        if self.intervention in {"relation_zero", "raw_input_outcome_no_relation"}:
            return torch.zeros_like(native)
        if self.intervention == "relation_incorrect":
            return -native
        if self.intervention == "relation_random_balanced":
            batch_indices = torch.arange(native.shape[0], device=native.device).unsqueeze(1)
            signs = ((batch_indices + self._audit_step) % 2).float() * 2.0 - 1.0
            return signs.to(native.dtype)
        return native

    def _routing(self, public: torch.Tensor) -> torch.Tensor:
        contexts = F.one_hot(public[:, 1].long(), num_classes=self.config.context_count).float()
        native = contexts[:, :2]
        if self.intervention == "routing_swapped":
            return native.flip(1)
        if self.intervention == "routing_shuffled_by_world":
            swap = (torch.arange(len(public), device=public.device) % 2).bool()
            shuffled = native.clone()
            shuffled[swap] = shuffled[swap].flip(1)
            return shuffled
        if self.intervention in {
            "correct_relation_random_routing",
        }:
            batch_indices = torch.arange(len(public), device=public.device)
            destinations = (batch_indices + self._audit_step) % 2
            return F.one_hot(destinations, num_classes=2).float()
        if self.intervention == "routing_disabled_uniform":
            return torch.ones_like(native)
        if self.intervention == "distractor_writes_enabled":
            distractor = (public[:, 1].long() == 3).unsqueeze(1)
            return torch.where(distractor, torch.ones_like(native), native)
        return native

    def update(
        self,
        public: torch.Tensor,
        state: torch.Tensor,
        outcome: torch.Tensor,
        prediction: Any,
    ) -> StateUpdate:
        error = self._error_signal(prediction.logits, outcome)
        x_signed = public[:, :1].float() * 2.0 - 1.0
        y_signed = outcome.float().unsqueeze(1) * 2.0 - 1.0
        relation = self._relation(x_signed, y_signed)
        phase = public[:, -1:]
        contexts = F.one_hot(public[:, 1].long(), num_classes=self.config.context_count).float()
        features = torch.cat((x_signed, y_signed, relation, phase, error, contexts), dim=-1)
        candidate = self.relation_encoder(features)
        gate = torch.sigmoid(self.write_gate(torch.cat((features, state), dim=-1)))
        gate = gate * self._routing(public)
        updated = (1.0 - gate) * state + gate * candidate
        self._audit_step += 1
        return StateUpdate(updated, gate, candidate, error)


def _mean(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _worlds(document: dict[str, Any]) -> dict[str, list]:
    evaluation = document["evaluation"]
    groups = evaluation["groups"]
    return {
        "delay": balanced_delayed_episodes(
            seed_start=evaluation["delay_seed_start"],
            groups=groups,
            delay_steps=evaluation["delay_steps"],
            query_steps=evaluation["delay_query_steps"],
        ),
        "composition": balanced_composition_episodes(
            seed_start=evaluation["composition_seed_start"],
            groups=groups,
            steps=evaluation["composition_steps"],
        ),
        "reversal": balanced_reversal_episodes(
            seed_start=evaluation["reversal_seed_start"], groups=groups
        ),
        "probe": balanced_reversal_episodes(
            seed_start=evaluation["rule_probe_seed_start"], groups=groups
        ),
    }


def _evaluate_intervention(
    source: FactorizedStateCore,
    intervention: str,
    worlds: dict[str, list],
    seed: int,
) -> dict[str, Any]:
    model = V1ScaffoldAuditCore(source.config, intervention)
    model.load_state_dict(source.state_dict(), strict=True)
    before = state_dict_digest(model)
    delay = evaluate_model(model, worlds["delay"])
    composition = evaluate_model(model, worlds["composition"])
    reversal = evaluate_rule_change(model, worlds["reversal"])
    donor = composition_donor_state_swap(model, worlds["composition"])
    probe = linear_reversal_probe(model, worlds["probe"], worlds["reversal"], seed=seed)
    geometry = state_geometry(model, worlds["delay"])
    return {
        "delay_accuracy": delay["fully_informed_accuracy"],
        "composition_accuracy": composition["composition_accuracy"],
        "recovery_accuracy": reversal["post_feedback_recovery_accuracy"],
        "retention_accuracy": reversal["unrelated_rule_retention_accuracy"],
        "donor_state_consistency": donor["donor_rule_consistency"],
        "rule_probe_accuracy": probe["held_out_accuracy"],
        "distractor_state_drift": delay["distractor_state_drift_mean"],
        "delay_state_drift": geometry["delay_drift_norm"],
        "weights_unchanged": before == state_dict_digest(model),
    }


def run_v1_scaffold_audit(
    config_path: Path, checkpoint_root: Path, output_path: Path
) -> dict[str, Any]:
    """Run every declared scaffold perturbation across the three v1 core checkpoints."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    worlds = _worlds(document)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in document["seeds"]:
        source, _, metadata = load_v1_checkpoint(checkpoint_root / "core" / f"seed-{seed}.pt")
        if not isinstance(source, FactorizedStateCore):
            raise ValueError("scaffold audit requires a v1 factorized-core checkpoint")
        for intervention in SCAFFOLD_INTERVENTIONS:
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_digest": state_dict_digest(source),
                    "checkpoint_git_revision": metadata["git_revision"],
                    "intervention": intervention,
                    "metrics": _evaluate_intervention(source, intervention, worlds, int(seed)),
                }
            )
    aggregate: dict[str, Any] = {}
    for intervention in SCAFFOLD_INTERVENTIONS:
        selected = [row["metrics"] for row in rows if row["intervention"] == intervention]
        aggregate[intervention] = {
            metric: _mean([float(row[metric]) for row in selected])
            for metric in (
                "delay_accuracy",
                "composition_accuracy",
                "recovery_accuracy",
                "retention_accuracy",
                "donor_state_consistency",
                "rule_probe_accuracy",
                "distractor_state_drift",
                "delay_state_drift",
            )
        }
        aggregate[intervention]["all_weights_unchanged"] = all(
            bool(row["weights_unchanged"]) for row in selected
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": "cognitive-core-v2-v1-scaffold-audit",
        "evidence_role": "post_hoc_v1_mechanistic_audit_not_v2_training_evidence",
        "source_experiment": document["experiment_id"],
        "source_checkpoints": str(checkpoint_root),
        "inventory": v1_scaffold_inventory(),
        "interventions": list(SCAFFOLD_INTERVENTIONS),
        "seeds": document["seeds"],
        "rows": rows,
        "aggregate": aggregate,
        "wall_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
