from __future__ import annotations

import argparse
from pathlib import Path

from bilab.v3.runner import run_v3_pilot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the preregistered V3C cross-pair audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/cognitive_core_v3/configs/pilot_xpair.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cognitive_core_v3/xpair-v1.0"),
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    repo = Path.cwd()
    result_path = arguments.output / "pilot_results.json"
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite an existing cross-pair result: {result_path}")

    result = run_v3_pilot(
        repo,
        arguments.config,
        arguments.output,
        stage="v3c",
    )

    print(f"status={result['status']} rows={len(result['rows'])}")
    for row in result["rows"]:
        metadata = row["training"]["initialization_metadata"]
        raw_seed = metadata["raw_relation_source"]["source_seed"]
        preservation_seed = metadata["preservation_source"]["source_seed"]
        initial = row["initial_evaluation"]
        print(
            f"{row['candidate']}: "
            f"A={raw_seed} B={preservation_seed} "
            f"initial_behavioral={initial['passes_behavioral_stage_gates']} "
            f"initial_exact={initial['passes_exact_preservation']} "
            f"post_behavioral={row['passes_behavioral_stage_gates']} "
            f"post_exact={row['passes_exact_preservation']} "
            f"post_all={row['passes_stage_gates']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
