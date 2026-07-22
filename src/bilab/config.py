"""Configuration loading for machine-specific, read-only asset paths."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabConfig:
    """Resolved paths used by inspection and ingestion commands."""

    repository: Path
    llama_bench: Path | None
    model: Path | None
    external_benchmarks: Path | None
    repository_benchmarks: Path
    source: Path | None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _optional_path(value: object, base: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(repo: Path | None = None, config_path: Path | None = None) -> LabConfig:
    """Load local override, falling back to the checked-in example and environment values."""

    root = (repo or repository_root()).resolve()
    selected = config_path
    if selected is None:
        local = root / "config" / "lab.local.json"
        example = root / "config" / "lab.example.json"
        selected = local if local.exists() else example

    raw: dict[str, Any] = {}
    if selected.exists():
        raw = json.loads(selected.read_text(encoding="utf-8"))

    values = raw.get("paths", {})
    env_map = {
        "llama_bench": "BILAB_LLAMA_BENCH",
        "model": "BILAB_MODEL",
        "external_benchmarks": "BILAB_EXTERNAL_BENCHMARKS",
        "repository_benchmarks": "BILAB_REPOSITORY_BENCHMARKS",
    }
    resolved = {key: os.environ.get(env_name, values.get(key)) for key, env_name in env_map.items()}
    repo_bench = _optional_path(resolved["repository_benchmarks"], root)
    return LabConfig(
        repository=root,
        llama_bench=_optional_path(resolved["llama_bench"], root),
        model=_optional_path(resolved["model"], root),
        external_benchmarks=_optional_path(resolved["external_benchmarks"], root),
        repository_benchmarks=repo_bench or root / "benchmarks" / "hardware" / "raw",
        source=selected if selected.exists() else None,
    )
