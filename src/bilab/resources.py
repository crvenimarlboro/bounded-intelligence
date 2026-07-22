"""Research telemetry primitives; these are not an intelligence representation."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def configuration_hash(config: object) -> str:
    """Return a stable SHA-256 identity for JSON-compatible configuration."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def git_revision(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def directory_bytes(path: Path, excluded_names: set[str] | None = None) -> int:
    """Count regular-file bytes without following symlinks."""

    excluded = excluded_names or set()
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if name not in excluded]
        root_path = Path(root)
        for name in files:
            candidate = root_path / name
            if not candidate.is_symlink():
                total += candidate.stat().st_size
    return total


def deep_size(value: object) -> int:
    """Estimate the live Python object graph size without double counting."""

    import sys

    seen: set[int] = set()

    def visit(item: object) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if isinstance(item, dict):
            size += sum(visit(key) + visit(val) for key, val in item.items())
        elif isinstance(item, (list, tuple, set, frozenset)):
            size += sum(visit(child) for child in item)
        return size

    return visit(value)


@dataclass(frozen=True)
class ResourceReport:
    experiment_id: str
    seed: int
    git_revision: str | None
    configuration_hash: str
    persistent_bytes: int
    temporary_bytes: int
    peak_ram_bytes: int | None
    peak_vram_bytes: int | None
    elapsed_seconds: float
    cpu_seconds: float | None
    model_parameters: int
    observations_consumed: int
    score: float
    compute_proxy: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _peak_ram_bytes() -> int | None:
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes. This laboratory currently targets Linux/WSL.
    return int(peak * 1024) if os.name == "posix" else None


def measure_call[T](
    function: Callable[[], T],
    *,
    experiment_id: str,
    seed: int,
    config: object,
    repo: Path,
    observations: int,
    score_of: Callable[[T], float],
    persistent_bytes: int = 0,
    temporary_bytes: int = 0,
    model_parameters: int = 0,
    compute_proxy: str | None = None,
) -> tuple[T, ResourceReport]:
    """Measure one deterministic call with honest unavailable values."""

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = function()
    cpu_seconds = time.process_time() - cpu_start
    elapsed = time.perf_counter() - wall_start
    report = ResourceReport(
        experiment_id=experiment_id,
        seed=seed,
        git_revision=git_revision(repo),
        configuration_hash=configuration_hash(config),
        persistent_bytes=persistent_bytes,
        temporary_bytes=temporary_bytes,
        peak_ram_bytes=_peak_ram_bytes(),
        peak_vram_bytes=None,
        elapsed_seconds=elapsed,
        cpu_seconds=cpu_seconds,
        model_parameters=model_parameters,
        observations_consumed=observations,
        score=float(score_of(result)),
        compute_proxy=compute_proxy,
    )
    return result, report
