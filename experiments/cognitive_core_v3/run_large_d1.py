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


def _job_directory(output: Path, candidate: str, seed: int) -> Path:
    return output / "jobs" / candidate / f"seed-{seed}"


def _completed_job(path: Path) -> dict[str, Any] | None:
    result_path = path / "pilot_results.json"
    if not result_path.exists():
        return None
    try:
        document = _read_json(result_path)
    except (OSError, json.JSONDecodeError):
        return None
    if document.get("status") != "completed" or len(document.get("rows", [])) != 1:
        return None
    return document


def _preserve_incomplete(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}-incomplete-{stamp}")
    shutil.move(str(path), str(backup))
    print(f"preserved incomplete job as {backup}", flush=True)


def _job_config(base: dict[str, Any], candidate_name: str, seed: int) -> dict[str, Any]:
    document = copy.deepcopy(base)
    document["experiment_id"] = f"{base['experiment_id']}-{candidate_name}-seed-{seed}"
    document["seeds"] = [seed]
    document["candidates"] = {candidate_name: copy.deepcopy(base["candidates"][candidate_name])}
    return document


def _aggregate(base: dict[str, Any], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    candidate_names = list(base["candidates"])
    seeds = [int(seed) for seed in base["seeds"]]

    for candidate_name in candidate_names:
        for seed in seeds:
            job_path = _job_directory(output, candidate_name, seed)
            result = _completed_job(job_path)
            completed = result is not None
            jobs.append(
                {
                    "candidate": candidate_name,
                    "seed": seed,
                    "completed": completed,
                    "result": str(job_path / "pilot_results.json"),
                }
            )
            if completed:
                rows.append(result["rows"][0])

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
            "primary_six_of_six": len(selected) == len(seeds) and combined == len(seeds),
            "near_replication_five_of_six": len(selected) == len(seeds)
            and combined == len(seeds) - 1,
        }

    return {
        "schema_version": "1.0",
        "experiment_id": base["experiment_id"],
        "status": "completed" if len(rows) == len(candidate_names) * len(seeds) else "in_progress",
        "completed_jobs": len(rows),
        "total_jobs": len(candidate_names) * len(seeds),
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
    seeds = [int(seed) for seed in base["seeds"]]
    total = len(candidate_names) * len(seeds)
    ordinal = 0

    for candidate_name in candidate_names:
        for seed in seeds:
            ordinal += 1
            job_path = _job_directory(arguments.output, candidate_name, seed)
            if _completed_job(job_path) is not None:
                print(
                    f"[{ordinal}/{total}] skipping completed {candidate_name} seed={seed}",
                    flush=True,
                )
                _write_json(aggregate_path, _aggregate(base, arguments.output))
                continue

            _preserve_incomplete(job_path)
            job_path.mkdir(parents=True, exist_ok=True)
            config_path = job_path / "job_config.json"
            _write_json(config_path, _job_config(base, candidate_name, seed))

            print(
                f"[{ordinal}/{total}] running {candidate_name} seed={seed}",
                flush=True,
            )
            run_v3_pilot(repo, config_path, job_path, stage="v3c")
            result = _completed_job(job_path)
            if result is None:
                raise RuntimeError(
                    f"job did not produce one completed row: {candidate_name} seed={seed}"
                )

            row = result["rows"][0]
            print(
                f"[{ordinal}/{total}] completed {candidate_name} seed={seed} "
                f"behavioral={row.get('passes_behavioral_stage_gates')} "
                f"exact={row.get('passes_exact_preservation')} "
                f"all={row.get('passes_stage_gates')}",
                flush=True,
            )
            _write_json(aggregate_path, _aggregate(base, arguments.output))

    final = _aggregate(base, arguments.output)
    _write_json(aggregate_path, final)
    print(
        f"status={final['status']} jobs={final['completed_jobs']}/{final['total_jobs']}",
        flush=True,
    )
    for candidate_name, summary in final["summary"].items():
        print(
            f"{candidate_name}: behavioral={summary['behavioral_passes']}/6 "
            f"exact={summary['exact_preservation_passes']}/6 "
            f"combined={summary['combined_passes']}/6 "
            f"primary={summary['primary_six_of_six']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
