from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_large_runner():
    runner_path = Path("experiments/cognitive_core_v3/run_large_d1.py")

    spec = importlib.util.spec_from_file_location(
        "run_large_d1",
        runner_path,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_development_config_is_materially_larger_and_keeps_final_closed() -> None:
    path = Path("experiments/cognitive_core_v3/configs/development_large_d1.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    training = document["training"]

    assert document["seeds"] == [
        3501,
        3502,
        3503,
        3504,
        3505,
        3506,
    ]
    assert training["optimizer_steps"] == 1600
    assert training["observations_per_candidate_seed"] == 1_228_800
    assert training["validation_groups"] == 128
    assert document["evaluation_groups"] == 256
    assert document["long_sequence_lengths"][-1] == 1_000_000
    assert document["evaluation_seed_start"] < 700_000
    assert set(document["candidates"]) == {
        "v3c_large_robust_b3201",
        "v3c_large_fragile_b3202",
        "v3c_large_random",
    }


def test_large_development_staged_donors_are_fixed_across_target_seeds() -> None:
    path = Path("experiments/cognitive_core_v3/configs/development_large_d1.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    seeds = {str(seed) for seed in document["seeds"]}

    robust = document["candidates"]["v3c_large_robust_b3201"]
    robust_sources = robust["staged_sources"]

    assert set(robust_sources["raw_relation"]["source_seed_map"]) == seeds
    assert set(robust_sources["preservation"]["source_seed_map"]) == seeds
    assert set(robust_sources["raw_relation"]["source_seed_map"].values()) == {3202}
    assert set(robust_sources["preservation"]["source_seed_map"].values()) == {3201}

    fragile = document["candidates"]["v3c_large_fragile_b3202"]
    fragile_sources = fragile["staged_sources"]

    assert set(fragile_sources["raw_relation"]["source_seed_map"].values()) == {3201}
    assert set(fragile_sources["preservation"]["source_seed_map"].values()) == {3202}


def test_large_runner_bundles_all_candidates_per_seed() -> None:
    module = _load_large_runner()

    config_path = Path("experiments/cognitive_core_v3/configs/development_large_d1.json")
    base = json.loads(config_path.read_text(encoding="utf-8"))

    bundled = module._job_config(base, 3501)

    assert bundled["seeds"] == [3501]
    assert set(bundled["candidates"]) == set(base["candidates"])
    assert len(bundled["candidates"]) >= 2
    assert bundled["experiment_id"].endswith("-seed-3501")


def test_large_completed_job_requires_every_candidate(tmp_path: Path) -> None:
    module = _load_large_runner()

    expected = {
        "v3c_large_robust_b3201",
        "v3c_large_fragile_b3202",
        "v3c_large_random",
    }

    result_path = tmp_path / "pilot_results.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "rows": [{"candidate": candidate} for candidate in sorted(expected)],
            }
        ),
        encoding="utf-8",
    )

    completed = module._completed_job(tmp_path, expected)

    assert completed is not None
    assert len(completed["rows"]) == 3


def test_large_completed_job_rejects_single_candidate_result(
    tmp_path: Path,
) -> None:
    module = _load_large_runner()

    expected = {
        "v3c_large_robust_b3201",
        "v3c_large_fragile_b3202",
        "v3c_large_random",
    }

    result_path = tmp_path / "pilot_results.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "rows": [{"candidate": ("v3c_large_robust_b3201")}],
            }
        ),
        encoding="utf-8",
    )

    assert module._completed_job(tmp_path, expected) is None
