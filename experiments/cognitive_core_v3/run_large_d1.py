from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bilab.v3.runner import run_v3_pilot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the resumable V3C large development replication."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/cognitive_core_v3/configs/development_large_d1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cognitive_core_v3/large-development-d1"),
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _job_directory(output: Path, seed: int) -> Path:
    """Return one resumable job directory containing all candidates for a seed."""
    return output / "jobs" / f"seed-{seed}"


def _completed_job(
    path: Path,
    expected_candidates: set[str],
) -> dict[str, Any] | None:
    result_path = path / "pilot_results.json"
    if not result_path.exists():
        return None

    try:
        document = _read_json(result_path)
    except (OSError, json.JSONDecodeError):
        return None

    rows = document.get("rows", [])
    observed_candidates = {str(row.get("candidate")) for row in rows if isinstance(row, dict)}

    if document.get("status") != "completed":
        return None
    if len(rows) != len(expected_candidates):
        return None
    if observed_candidates != expected_candidates:
        return None

    return document


def _preserve_incomplete(path: Path) -> None:
    if not path.exists():
        return

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}-incomplete-{stamp}")
    shutil.move(str(path), str(backup))
    print(f"preserved incomplete job as {backup}", flush=True)


def _job_config(base: dict[str, Any], seed: int) -> dict[str, Any]:
    """Create a one-seed configuration while retaining all serious candidates."""
    document = copy.deepcopy(base)
    document["experiment_id"] = f"{base['experiment_id']}-seed-{seed}"
    document["seeds"] = [seed]
    return document


def _aggregate(base: dict[str, Any], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []

    candidate_names = list(base["candidates"])
    expected_candidates = set(candidate_names)
    seeds = [int(seed) for seed in base["seeds"]]

    for seed in seeds:
        job_path = _job_directory(output, seed)
        result = _completed_job(job_path, expected_candidates)
        completed = result is not None

        jobs.append(
            {
                "seed": seed,
                "completed": completed,
                "expected_candidates": candidate_names,
                "result": str(job_path / "pilot_results.json"),
            }
        )

        if completed:
            rows.extend(result["rows"])

    summary: dict[str, Any] = {}

    for candidate_name in candidate_names:
        selected = [row for row in rows if row.get("candidate") == candidate_name]
        behavioral = sum(row.get("passes_behavioral_stage_gates") is True for row in selected)
        exact = sum(row.get("passes_exact_preservation") is True for row in selected)
        combined = sum(row.get("passes_stage_gates") is True for row in selected)

        summary[candidate_name] = {
            "completed_seeds": len(selected),
            "total_seeds": len(seeds),
            "behavioral_passes": behavioral,
            "exact_preservation_passes": exact,
            "combined_passes": combined,
            "primary_six_of_six": (len(selected) == len(seeds) and combined == len(seeds)),
            "near_replication_five_of_six": (
                len(selected) == len(seeds) and combined == len(seeds) - 1
            ),
        }

    expected_rows = len(candidate_names) * len(seeds)

    return {
        "schema_version": "1.0",
        "experiment_id": base["experiment_id"],
        "status": ("completed" if len(rows) == expected_rows else "in_progress"),
        "completed_seed_bundles": sum(job["completed"] is True for job in jobs),
        "total_seed_bundles": len(seeds),
        "completed_jobs": len(rows),
        "total_jobs": expected_rows,
        "candidate_selection_rule": base["candidate_selection_rule"],
        "jobs": jobs,
        "rows": rows,
        "summary": summary,
    }


def main() -> int:
    arguments = _parser().parse_args()
    repo = Path.cwd()
    base = _read_json(arguments.config)

    arguments.output.mkdir(parents=True, exist_ok=True)
    aggregate_path = arguments.output / "scale_results.json"

    candidate_names = list(base["candidates"])
    expected_candidates = set(candidate_names)
    seeds = [int(seed) for seed in base["seeds"]]
    total_bundles = len(seeds)

    for ordinal, seed in enumerate(seeds, start=1):
        job_path = _job_directory(arguments.output, seed)
        completed = _completed_job(job_path, expected_candidates)

        if completed is not None:
            print(
                f"[{ordinal}/{total_bundles}] "
                f"skipping completed seed={seed} "
                f"with {len(candidate_names)} candidates",
                flush=True,
            )
            _write_json(
                aggregate_path,
                _aggregate(base, arguments.output),
            )
            continue

        _preserve_incomplete(job_path)
        job_path.mkdir(parents=True, exist_ok=True)

        config_path = job_path / "job_config.json"
        _write_json(config_path, _job_config(base, seed))

        print(
            f"[{ordinal}/{total_bundles}] running seed={seed} "
            f"with {len(candidate_names)} serious candidates",
            flush=True,
        )

        run_v3_pilot(
            repo,
            config_path,
            job_path,
            stage="v3c",
        )

        result = _completed_job(job_path, expected_candidates)
        if result is None:
            raise RuntimeError(
                f"seed bundle did not produce {len(candidate_names)} completed rows: seed={seed}"
            )

        for row in result["rows"]:
            print(
                f"[{ordinal}/{total_bundles}] "
                f"completed {row.get('candidate')} seed={seed} "
                f"behavioral={row.get('passes_behavioral_stage_gates')} "
                f"exact={row.get('passes_exact_preservation')} "
                f"all={row.get('passes_stage_gates')}",
                flush=True,
            )

        _write_json(
            aggregate_path,
            _aggregate(base, arguments.output),
        )

    final = _aggregate(base, arguments.output)
    _write_json(aggregate_path, final)

    print(
        f"status={final['status']} "
        f"seed_bundles="
        f"{final['completed_seed_bundles']}/"
        f"{final['total_seed_bundles']} "
        f"candidate_seed_rows="
        f"{final['completed_jobs']}/{final['total_jobs']}",
        flush=True,
    )

    for candidate_name, summary in final["summary"].items():
        total = summary["total_seeds"]
        print(
            f"{candidate_name}: "
            f"behavioral={summary['behavioral_passes']}/{total} "
            f"exact={summary['exact_preservation_passes']}/{total} "
            f"combined={summary['combined_passes']}/{total} "
            f"primary={summary['primary_six_of_six']}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
