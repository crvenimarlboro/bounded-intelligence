from pathlib import Path

from bilab.resources import configuration_hash, deep_size, measure_call


def test_configuration_hash_is_order_independent() -> None:
    assert configuration_hash({"b": 2, "a": 1}) == configuration_hash({"a": 1, "b": 2})


def test_measure_call_reports_required_instrumentation(tmp_path: Path) -> None:
    result, report = measure_call(
        lambda: 4,
        experiment_id="test.measurement",
        seed=11,
        config={"mode": "test"},
        repo=tmp_path,
        observations=3,
        score_of=lambda value: value / 4,
        temporary_bytes=deep_size([1, 2, 3]),
        compute_proxy="one constant",
    )
    assert result == 4
    assert report.score == 1.0
    assert report.observations_consumed == 3
    assert report.elapsed_seconds >= 0
    assert report.cpu_seconds is not None
    assert report.peak_vram_bytes is None
    assert len(report.configuration_hash) == 64
