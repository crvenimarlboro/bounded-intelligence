import json
from pathlib import Path

from bilab.resources import directory_bytes
from bilab.training.v1_final import (
    compact_confirmatory_summary,
    compare_v1_result_files,
    evaluate_v1_checkpoint_file,
    regenerate_v1_report,
    run_confirmatory,
)


def test_tiny_confirmatory_run_writes_reproducible_artifacts(tmp_path: Path) -> None:
    config = {
        "protocol_version": "test",
        "experiment_id": "test-v1-confirmatory",
        "seeds": [19],
        "persistent_state": {"float32_values": 2, "bytes": 8},
        "training": {
            "world_seed_start": 10000,
            "validation_seed_start": 11000,
            "optimizer_steps": 2,
            "batch_groups": 1,
            "episode_steps": 12,
            "validation_groups": 1,
            "validation_interval": 1,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "torch_threads": 1,
            "checkpoint_selection": "final_step",
        },
        "variants": {
            "core": {"family": "factorized", "hidden_dim": 16},
            "no_memory": {"family": "no_memory", "hidden_dim": 16},
            "episodic": {
                "family": "episodic",
                "hidden_dim": 16,
                "budget_bytes": 8,
            },
        },
        "evaluation": {
            "groups": 2,
            "reversal_seed_start": 12000,
            "delay_seed_start": 13000,
            "composition_seed_start": 14000,
            "context_seed_start": 15000,
            "random_seed_start": 16000,
            "rule_probe_seed_start": 18000,
            "surface_probe_seed_start": 19000,
            "delay_steps": 2,
            "delay_query_steps": 3,
            "composition_steps": 6,
            "context_steps": 6,
        },
        "success_thresholds": {
            "minimum_delay_accuracy": 0.0,
            "minimum_composition_accuracy": 0.0,
            "minimum_recovery_accuracy": 0.0,
            "minimum_retention_accuracy": 0.0,
            "minimum_surface_relabelled_accuracy": 0.0,
            "maximum_change_step_accuracy": 1.0,
            "maximum_random_advantage": 1.0,
            "minimum_reset_drop": -1.0,
            "minimum_donor_consistency": 0.0,
            "minimum_rule_probe_accuracy": 0.0,
            "minimum_episodic_advantage": -1.0,
            "minimum_no_memory_advantage": -1.0,
            "parameter_match_tolerance_fraction": 1.0,
        },
    }
    manifest = {
        "schema_version": "1.0",
        "experiment_id": "test-v1-confirmatory",
        "hypothesis": "A tiny test run writes and reloads every artifact.",
        "baseline": "No memory and episodic memory.",
        "intervention": "Factorized state.",
        "controlled_variables": {"config": "temporary"},
        "independent_variables": ["variant"],
        "dependent_metrics": ["accuracy"],
        "resource_budgets": {"wall_seconds": 60},
        "expected_result": "The artifact pipeline completes.",
        "falsification_condition": "Any checkpoint fails to reload.",
        "repetition_count": 1,
        "seeds": [19],
        "artifact_locations": {"output": "temporary"},
        "stopping_rule": "Run every declared variant once.",
        "status": "planned",
        "final_conclusion": None,
        "evidence_classification": "project_hypothesis",
    }
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    results_directory = tmp_path / "results"
    result = run_confirmatory(
        Path(__file__).parents[1], config_path, manifest_path, results_directory
    )
    assert result["aggregate"]["core"]["checkpoint_reproduction_max_error"] == 0
    assert result["resources"]["output_bytes"] == directory_bytes(results_directory)
    assert (results_directory / "results.json").is_file()
    assert (results_directory / "raw_metrics.jsonl").is_file()
    assert (results_directory / "learning_curves.csv").is_file()
    assert (results_directory / "adaptation_curves.csv").is_file()

    checkpoint = results_directory / "checkpoints/core/seed-19.pt"
    replay = evaluate_v1_checkpoint_file(checkpoint, config_path, tmp_path / "replay.json")
    assert replay["weights_unchanged"] is True
    assert replay["evaluation"]["delay"] == result["per_seed"][0]["evaluation"]["delay"]

    summary = compact_confirmatory_summary(result)
    assert summary["run_count"] == 3
    assert summary["final_seed_count"] == 1
    regenerated = regenerate_v1_report(
        results_directory / "results.json",
        tmp_path / "report.md",
        summary_output=tmp_path / "summary.json",
    )
    assert regenerated["assessment"] == result["assessment"]
    assert "Conclusion:" in (tmp_path / "report.md").read_text(encoding="utf-8")
    comparison = compare_v1_result_files(
        results_directory / "results.json",
        results_directory / "results.json",
        tmp_path / "comparison.json",
    )
    assert comparison["stable_rows_equal"] is True
    assert comparison["all_checkpoint_model_digests_equal"] is True
    reproduced_report = regenerate_v1_report(
        results_directory / "results.json",
        tmp_path / "reproduced-report.md",
        reproduction_comparison=tmp_path / "comparison.json",
    )
    assert reproduced_report["committed_source_reproduction"]["checkpoint_model_digest_count"] == 3
    assert "Committed-source full reproduction" in (tmp_path / "reproduced-report.md").read_text(
        encoding="utf-8"
    )

    changed = json.loads((results_directory / "results.json").read_text(encoding="utf-8"))
    changed["per_seed"][0]["evaluation"]["delay"]["post_evidence_accuracy"] += 0.125
    changed_path = results_directory / "changed-results.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    mismatch = compare_v1_result_files(
        results_directory / "results.json",
        changed_path,
        tmp_path / "mismatch.json",
    )
    assert mismatch["stable_rows_equal"] is False
    assert mismatch["stable_row_maximum_numeric_error"] == 0.125
