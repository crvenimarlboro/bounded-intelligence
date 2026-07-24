from __future__ import annotations

import json
from pathlib import Path


def test_large_development_config_is_materially_larger_and_keeps_final_closed() -> None:
    path = Path("experiments/cognitive_core_v3/configs/development_large_d1.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    training = document["training"]

    assert document["seeds"] == [3501, 3502, 3503, 3504, 3505, 3506]
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
