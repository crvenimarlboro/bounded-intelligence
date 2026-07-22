import subprocess
from pathlib import Path

from bilab.config import LabConfig
from bilab.doctor import _binary_version, collect_environment, render_environment


def test_doctor_reports_missing_optional_assets_honestly(tmp_path: Path) -> None:
    config = LabConfig(
        repository=tmp_path,
        llama_bench=tmp_path / "missing.exe",
        model=None,
        external_benchmarks=None,
        repository_benchmarks=tmp_path / "benchmarks",
        source=None,
    )
    report = collect_environment(config)
    assert report["llama_bench"]["present"] is False
    assert report["model"]["size_bytes"] is None
    assert "unavailable" in render_environment(report)
    assert report["unavailable"]["peak_vram_bytes"]


def test_binary_usage_is_not_misreported_as_a_version(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "llama-bench.exe"
    binary.touch()
    completed = subprocess.CompletedProcess(
        [str(binary), "--version"], 1, stdout="usage: llama-bench.exe [options]\n", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    report = _binary_version(binary)
    assert report["version"] is None
    assert "unavailable" in report["note"]
