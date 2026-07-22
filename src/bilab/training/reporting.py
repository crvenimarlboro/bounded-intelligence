"""Compact machine-readable aggregation, tables, and dependency-free SVG curves."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bilab.training.evaluation import mean_and_sample_std


def _condition(row: dict[str, Any]) -> str:
    return f"{row['variant']}:{row['mode']}"


def _metric(row: dict[str, Any], category: str) -> float:
    return float(row["metrics"]["category_accuracy"][category])


def _aggregate_conditions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_condition(row)].append(row)
    result: dict[str, Any] = {}
    for condition, members in sorted(grouped.items()):
        categories = members[0]["metrics"]["category_accuracy"]
        category_stats = {
            category: mean_and_sample_std([_metric(member, category) for member in members])
            for category in categories
        }
        adaptation: dict[str, dict[str, dict[str, float]]] = {}
        for category in members[0]["metrics"]["adaptation_curves"]:
            adaptation[category] = {}
            for point in members[0]["metrics"]["adaptation_curves"][category]:
                values = [
                    float(member["metrics"]["adaptation_curves"][category][point])
                    for member in members
                ]
                adaptation[category][point] = mean_and_sample_std(values)
        result[condition] = {
            "seeds": [member["seed"] for member in members],
            "category_accuracy": category_stats,
            "adaptation_curves": adaptation,
            "parameter_count": members[0]["parameter_count"],
            "persistent_bytes": members[0]["persistent_bytes"],
            "evaluation_seconds": mean_and_sample_std(
                [float(member["evaluation_seconds"]) for member in members]
            ),
            "training_seconds": mean_and_sample_std(
                [float(member["training_seconds"]) for member in members]
            ),
            "checkpoint_bytes": sum(int(member["checkpoint_bytes"]) for member in members),
            "scalar_metrics": {
                key: mean_and_sample_std([float(member["metrics"][key]) for member in members])
                for key in (
                    "next_outcome_accuracy",
                    "hidden_rule_query_accuracy",
                    "composed_counterfactual_accuracy",
                    "rule_change_retention",
                )
            },
        }
    return result


def assess_success(
    rows: list[dict[str, Any]], aggregates: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    criteria = config["success_criteria"]
    full = aggregates["cognitive_core:full"]
    no_memory = aggregates["no_memory:full"]
    episodic = aggregates["episodic:full"]
    structured_advantage_no = (
        full["category_accuracy"]["structured"]["mean"]
        - no_memory["category_accuracy"]["structured"]["mean"]
    )
    structured_advantage_episode = (
        full["category_accuracy"]["structured"]["mean"]
        - episodic["category_accuracy"]["structured"]["mean"]
    )
    surface_advantage = (
        full["category_accuracy"]["surface_relabelled"]["mean"]
        - no_memory["category_accuracy"]["surface_relabelled"]["mean"]
    )
    random_advantage = (
        full["category_accuracy"]["random"]["mean"]
        - no_memory["category_accuracy"]["random"]["mean"]
    )
    curve = full["adaptation_curves"]["structured"]
    checkpoint_32 = "32" if "32" in curve else sorted(curve, key=int)[-1]
    adaptation_gain = curve[checkpoint_32]["mean"] - curve["0"]["mean"]
    ablation_drops = {
        mode: full["category_accuracy"]["structured"]["mean"]
        - aggregates[f"cognitive_core:{mode}"]["category_accuracy"]["structured"]["mean"]
        for mode in (
            "workspace_disabled",
            "workspace_frozen",
            "no_prediction_error",
            "recurrence_k1",
        )
    }
    primary_rows = {
        (row["variant"], row["seed"]): row
        for row in rows
        if row["mode"] == "full" and row["variant"] in {"cognitive_core", "no_memory"}
    }
    positive_seeds = sum(
        _metric(primary_rows[("cognitive_core", seed)], "structured")
        > _metric(primary_rows[("no_memory", seed)], "structured")
        for seed in config["training"]["seeds"]
    )
    counts = [
        aggregates[key]["parameter_count"]
        for key in ("cognitive_core:full", "no_memory:full", "episodic:full")
    ]
    parameter_difference = (max(counts) - min(counts)) / max(counts)
    eval_time_ratio = full["evaluation_seconds"]["mean"] / no_memory["evaluation_seconds"]["mean"]
    reproduction_error = max(float(row["checkpoint_reproduction_error"]) for row in rows)
    checks = {
        "structured_over_no_memory": structured_advantage_no
        >= float(criteria["structured_advantage_over_no_memory"]),
        "structured_over_episodic": structured_advantage_episode
        >= float(criteria["structured_advantage_over_episodic"]),
        "surface_transfer_advantage": surface_advantage
        >= float(criteria["surface_advantage_over_no_memory"]),
        "minimum_surface_accuracy": full["category_accuracy"]["surface_relabelled"]["mean"]
        >= float(criteria["minimum_surface_accuracy"]),
        "adaptation_gain": adaptation_gain >= float(criteria["minimum_adaptation_gain_0_to_32"]),
        "random_control_specificity": random_advantage
        <= float(criteria["maximum_random_advantage_over_no_memory"]),
        "ablation_weakens": max(ablation_drops.values())
        >= float(criteria["minimum_ablation_drop"]),
        "retention": full["scalar_metrics"]["rule_change_retention"]["mean"]
        >= float(criteria["minimum_rule_change_retention"]),
        "positive_seed_count": positive_seeds >= int(criteria["minimum_positive_seeds"]),
        "parameter_match": parameter_difference
        <= float(criteria["maximum_parameter_difference_fraction"]),
        "compute_not_extreme": eval_time_ratio <= float(criteria["maximum_core_eval_time_ratio"]),
        "checkpoint_reproduction": reproduction_error
        <= float(criteria["checkpoint_reproduction_tolerance"]),
        "fixed_persistent_bytes": full["persistent_bytes"] == config["workspace_bytes"]
        and episodic["persistent_bytes"] == config["workspace_bytes"],
    }
    return {
        "supported": all(checks.values()),
        "checks": checks,
        "observed": {
            "structured_advantage_over_no_memory": structured_advantage_no,
            "structured_advantage_over_episodic": structured_advantage_episode,
            "surface_advantage_over_no_memory": surface_advantage,
            "adaptation_gain_0_to_32": adaptation_gain,
            "random_advantage_over_no_memory": random_advantage,
            "rule_change_retention": full["scalar_metrics"]["rule_change_retention"]["mean"],
            "ablation_drops": ablation_drops,
            "positive_seeds": positive_seeds,
            "parameter_difference_fraction": parameter_difference,
            "core_eval_time_ratio": eval_time_ratio,
            "maximum_checkpoint_reproduction_error": reproduction_error,
        },
    }


def write_learning_curves(path: Path, training_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant",
                "seed",
                "epoch",
                "optimizer_step",
                "train_loss",
                "train_accuracy",
                "validation_accuracy",
                "observations_consumed",
            ),
        )
        writer.writeheader()
        writer.writerows(training_rows)


def write_curve_svg(path: Path, training_rows: list[dict[str, Any]]) -> None:
    width, height, pad = 800, 420, 50
    max_epoch = max(int(row["epoch"]) for row in training_rows)
    colors = {"no_memory": "#4c78a8", "episodic": "#f58518", "cognitive_core": "#54a24b"}
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in training_rows:
        grouped[str(row["variant"])][int(row["epoch"])].append(float(row["validation_accuracy"]))
    polylines: list[str] = []
    for variant, epochs in grouped.items():
        points = []
        for epoch, values in sorted(epochs.items()):
            x = pad + (width - 2 * pad) * (epoch - 1) / max(1, max_epoch - 1)
            mean = sum(values) / len(values)
            y = height - pad - (height - 2 * pad) * mean
            points.append(f"{x:.1f},{y:.1f}")
        polylines.append(
            f'<polyline fill="none" stroke="{colors[variant]}" stroke-width="3" '
            f'points="{" ".join(points)}"/><text x="{width - 180}" '
            f'y="{25 + 20 * len(polylines)}" fill="{colors[variant]}">{variant}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" '
        f'y2="{height - pad}" stroke="black"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="black"/>'
        '<text x="360" y="410">epoch</text><text x="8" y="30">validation accuracy</text>'
        + "".join(polylines)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def write_report(
    path: Path,
    results: dict[str, Any],
    *,
    title: str,
) -> None:
    aggregates = results["aggregates"]
    assessment = results["assessment"]
    conditions = (
        "no_memory:full",
        "episodic:full",
        "cognitive_core:full",
        "cognitive_core:workspace_disabled",
        "cognitive_core:workspace_frozen",
        "cognitive_core:no_prediction_error",
        "cognitive_core:recurrence_k1",
    )
    conclusion = "provisionally supported" if assessment["supported"] else "unsupported"
    lines = [
        f"# {title}",
        "",
        f"Conclusion: **{conclusion}**.",
        "",
        "| condition | structured | surface | random | rule-change | params | bytes | eval s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        value = aggregates[condition]
        category = value["category_accuracy"]
        lines.append(
            f"| {condition} | {category['structured']['mean']:.3f}"
            f"±{category['structured']['sample_std']:.3f} "
            f"| {category['surface_relabelled']['mean']:.3f} "
            f"| {category['random']['mean']:.3f} | {category['rule_change']['mean']:.3f} "
            f"| {value['parameter_count']} | {value['persistent_bytes']} "
            f"| {value['evaluation_seconds']['mean']:.2f} |"
        )
    lines.extend(["", "## Preregistered checks", ""])
    for name, passed in assessment["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Resource and integrity summary",
            "",
            f"- Total wall time: {results['resources']['total_wall_seconds']:.2f} seconds",
            f"- Peak RAM: {results['resources']['peak_ram_bytes']} bytes",
            f"- Checkpoints: {results['resources']['checkpoint_bytes']} bytes",
            f"- Training observations: {results['resources']['training_observations']}",
            f"- Evaluation observations: {results['resources']['evaluation_observations']}",
            "- Model weights were frozen during held-out online adaptation.",
            "- Random controls, all declared seeds, final-epoch checkpoints, and all required "
            "ablations are included.",
            "",
            "The result concerns this synthetic environment and resource setting only; it is not "
            "a general-intelligence metric.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
