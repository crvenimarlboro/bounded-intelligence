"""Validation for the versioned JSON experiment contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

EVIDENCE = {
    "established",
    "supported_but_uncertain",
    "project_hypothesis",
    "implementation_assumption",
    "speculative_possibility",
    "disproven_by_current_evidence",
}
STATUSES = {"planned", "running", "completed", "rejected", "failed"}
REQUIRED = {
    "schema_version",
    "experiment_id",
    "hypothesis",
    "baseline",
    "intervention",
    "controlled_variables",
    "independent_variables",
    "dependent_metrics",
    "resource_budgets",
    "expected_result",
    "falsification_condition",
    "repetition_count",
    "seeds",
    "artifact_locations",
    "stopping_rule",
    "status",
    "final_conclusion",
    "evidence_classification",
}


class ManifestError(ValueError):
    """A manifest violates the experiment contract."""


def _nonempty_string(data: dict[str, Any], key: str, errors: list[str]) -> None:
    if not isinstance(data.get(key), str) or not data[key].strip():
        errors.append(f"{key} must be a non-empty string")


def validate_manifest(data: object) -> dict[str, Any]:
    """Validate and return a manifest; raise one error containing every violation."""

    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    errors: list[str] = []
    missing = sorted(REQUIRED - data.keys())
    unknown = sorted(data.keys() - REQUIRED)
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown fields: {', '.join(unknown)}")
    if errors and missing:
        raise ManifestError("; ".join(errors))

    if data.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")
    experiment_id = data.get("experiment_id")
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{2,63}", experiment_id
    ):
        errors.append("experiment_id must be 3-64 lowercase identifier characters")
    for key in (
        "hypothesis",
        "baseline",
        "intervention",
        "expected_result",
        "falsification_condition",
        "stopping_rule",
    ):
        _nonempty_string(data, key, errors)
    for key in ("controlled_variables", "resource_budgets", "artifact_locations"):
        if not isinstance(data.get(key), dict) or not data[key]:
            errors.append(f"{key} must be a non-empty object")
    for key in ("independent_variables", "dependent_metrics"):
        value = data.get(key)
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            errors.append(f"{key} must be a non-empty string array")
    repetitions = data.get("repetition_count")
    seeds = data.get("seeds")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        errors.append("repetition_count must be a positive integer")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds
        )
    ):
        errors.append("seeds must be a non-empty array of non-negative integers")
    elif len(seeds) != len(set(seeds)):
        errors.append("seeds must be unique")
    if isinstance(repetitions, int) and isinstance(seeds, list) and repetitions != len(seeds):
        errors.append("repetition_count must equal the number of seeds")
    status = data.get("status")
    if status not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}")
    conclusion = data.get("final_conclusion")
    if status in {"completed", "rejected", "failed"} and (
        not isinstance(conclusion, str) or not conclusion.strip()
    ):
        errors.append("terminal status requires a non-empty final_conclusion")
    if status in {"planned", "running"} and conclusion is not None:
        errors.append("non-terminal status requires final_conclusion to be null")
    if data.get("evidence_classification") not in EVIDENCE:
        errors.append(f"evidence_classification must be one of {sorted(EVIDENCE)}")
    if errors:
        raise ManifestError("; ".join(errors))
    return data


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error
    return validate_manifest(data)
