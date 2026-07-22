from dataclasses import replace
from pathlib import Path

import pytest

from bilab.benchmarks import (
    BenchmarkError,
    compare_records,
    normalize_sources,
    parse_file,
    render_summary,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_jsonl_distinguishes_prompt_and_generation() -> None:
    records = parse_file(FIXTURES / "llama_bench.jsonl")
    assert [record.phase for record in records] == ["prompt_processing", "token_generation"]
    assert records[0].input_tokens == 16
    assert records[1].output_tokens == 8
    assert records[0].sample_tokens_per_second == (16000.0,)


def test_csv_is_normalized_and_summarized() -> None:
    document = normalize_sources([FIXTURES / "llama_bench.csv"])
    assert document["records"][0]["gpu_layers"] == 4
    assert document["summaries"][0]["mean_tokens_per_second"] == 32000.0
    assert document["comparisons"] == []
    assert "prompt_processing" in render_summary(document)


def test_missing_required_fields_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"build_commit":"only-one-field"}\n', encoding="utf-8")
    with pytest.raises(BenchmarkError, match="missing fields"):
        parse_file(bad)


def test_comparison_rejects_changed_controlled_condition() -> None:
    left = parse_file(FIXTURES / "llama_bench.csv")[0]
    allowed_intervention = replace(left, gpu_layers=6, device="Vulkan0")
    incompatible = replace(allowed_intervention, threads=3)
    assert compare_records(left, allowed_intervention)["status"] == "comparable"
    comparison = compare_records(left, incompatible)
    assert comparison["status"] == "rejected_incompatible"
    assert comparison["differences"]["threads"] == (2, 3)


def test_compatible_layer_intervention_produces_derived_ratio() -> None:
    document = normalize_sources([FIXTURES / "llama_bench.jsonl", FIXTURES / "llama_bench.csv"])
    prompt_summaries = [
        summary for summary in document["summaries"] if summary["phase"] == "prompt_processing"
    ]
    assert len(prompt_summaries) == 2
    assert document["comparisons"][0]["tokens_per_second_ratio"] == 2.0
    assert not document["warnings"]


def test_incompatible_cohorts_warn_and_do_not_compare(tmp_path: Path) -> None:
    original = (FIXTURES / "llama_bench.csv").read_text()
    incompatible = tmp_path / "different-threads.csv"
    incompatible.write_text(original.replace(",2,0x0,", ",3,0x0,"), encoding="utf-8")
    document = normalize_sources([FIXTURES / "llama_bench.csv", incompatible])
    assert document["comparisons"] == []
    assert "incompatible condition cohorts" in document["warnings"][0]


def test_invalid_optional_input_is_reported_as_warning(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text("GPU_Layers,Prompt_TPS\n6,10\n", encoding="utf-8")
    document = normalize_sources([FIXTURES / "llama_bench.csv", summary])
    assert "skipped non-native or invalid input" in document["warnings"][0]
