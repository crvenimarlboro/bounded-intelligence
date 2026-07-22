"""Read-only environment inspection with explicit unavailable values."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from bilab.config import LabConfig
from bilab.resources import directory_bytes, git_revision


def _read_first(path: Path, prefix: str) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[-1].strip()
    except OSError:
        return None
    return None


def _memory() -> tuple[int | None, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable"}:
                values[name] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _git_dirty(repo: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout)


def _binary_version(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "present": False, "version": None, "note": "not configured"}
    if not path.is_file():
        return {"path": str(path), "present": False, "version": None, "note": "not found"}
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        first_line = output.splitlines()[0] if output else ""
        unsupported = result.returncode != 0 or "usage:" in output.lower()
        version = first_line if first_line and not unsupported else None
        note = (
            None
            if version
            else f"version unavailable: --version returned exit code {result.returncode}"
        )
    except (OSError, subprocess.SubprocessError) as error:
        version = None
        note = f"present but version unavailable: {error}"
    return {"path": str(path), "present": True, "version": version, "note": note}


def _file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "present": False, "size_bytes": None, "note": "not configured"}
    try:
        present = path.is_file()
        size = path.stat().st_size if present else None
    except OSError as error:
        return {"path": str(path), "present": False, "size_bytes": None, "note": str(error)}
    return {
        "path": str(path),
        "present": present,
        "size_bytes": size,
        "note": None if present else "not found",
    }


def _artifacts(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "present": False, "files": [], "note": "not configured"}
    if not path.is_dir():
        return {"path": str(path), "present": False, "files": [], "note": "not found"}
    try:
        files = sorted(
            candidate.name
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in {".csv", ".jsonl", ".txt"}
        )
    except OSError as error:
        return {"path": str(path), "present": True, "files": [], "note": str(error)}
    return {"path": str(path), "present": True, "files": files, "note": None}


def collect_environment(config: LabConfig) -> dict[str, Any]:
    total_ram, available_ram = _memory()
    proc_version = None
    with suppress(OSError):
        proc_version = Path("/proc/version").read_text(encoding="utf-8").strip()
    is_wsl = proc_version is not None and "microsoft" in proc_version.lower()
    return {
        "schema_version": "1.0",
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "wsl": is_wsl,
            "wsl_interop": os.environ.get("WSL_INTEROP") is not None,
        },
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "cpu": {
            "model": _read_first(Path("/proc/cpuinfo"), "model name"),
            "logical_processors": os.cpu_count(),
        },
        "memory": {"total_bytes": total_ram, "available_bytes": available_ram},
        "git": {
            "revision": git_revision(config.repository),
            "dirty": _git_dirty(config.repository),
        },
        "llama_bench": _binary_version(config.llama_bench),
        "model": _file(config.model),
        "benchmarks": {
            "external": _artifacts(config.external_benchmarks),
            "repository": _artifacts(config.repository_benchmarks),
        },
        "paths": {
            "repository": str(config.repository),
            "config_source": str(config.source) if config.source else None,
            "llama_bench": str(config.llama_bench) if config.llama_bench else None,
            "model": str(config.model) if config.model else None,
            "external_benchmarks": (
                str(config.external_benchmarks) if config.external_benchmarks else None
            ),
            "repository_benchmarks": str(config.repository_benchmarks),
        },
        "repository_disk_usage": {
            "bytes": directory_bytes(config.repository, {".git", ".venv", "__pycache__"}),
            "scope": "working tree excluding .git, .venv, and __pycache__",
        },
        "unavailable": {
            "peak_vram_bytes": "not sampled by the read-only bootstrap doctor",
            "physical_cpu_cores": "not measured portably without an added dependency",
        },
    }


def render_environment(report: dict[str, Any]) -> str:
    def show(value: object) -> str:
        return "unavailable" if value is None else str(value)

    lines = [
        "Bounded Intelligence Laboratory — environment doctor",
        f"OS: {show(report['os']['platform'])}",
        f"WSL: {report['os']['wsl']} (interop={report['os']['wsl_interop']})",
        f"Python: {report['python']['version']} ({report['python']['executable']})",
        f"CPU: {show(report['cpu']['model'])}; logical={show(report['cpu']['logical_processors'])}",
        (
            f"RAM: total={show(report['memory']['total_bytes'])} bytes; "
            f"available={show(report['memory']['available_bytes'])} bytes"
        ),
        f"Git: revision={show(report['git']['revision'])}; dirty={show(report['git']['dirty'])}",
        (
            f"llama-bench: present={report['llama_bench']['present']}; "
            f"version={show(report['llama_bench']['version'])}; "
            f"path={show(report['llama_bench']['path'])}"
        ),
        (
            f"Model: present={report['model']['present']}; "
            f"size={show(report['model']['size_bytes'])} bytes; "
            f"path={show(report['model']['path'])}"
        ),
        (
            f"Benchmark artifacts: external={len(report['benchmarks']['external']['files'])}; "
            f"repository={len(report['benchmarks']['repository']['files'])}"
        ),
        (
            f"Repository disk usage: {report['repository_disk_usage']['bytes']} bytes "
            f"({report['repository_disk_usage']['scope']})"
        ),
        "Unavailable metrics:",
    ]
    lines.extend(f"- {key}: {reason}" for key, reason in report["unavailable"].items())
    note = report["llama_bench"].get("note")
    if note:
        lines.append(f"llama-bench note: {note}")
    return "\n".join(lines)
