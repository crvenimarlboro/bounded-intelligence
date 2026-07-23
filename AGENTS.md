# Repository agent guide

Read `docs/PROJECT_CHARTER.md`, `docs/RESEARCH_PRINCIPLES.md`, and the relevant ADRs before changing
methodology. Inspect before editing; preserve user changes and raw evidence.

## Non-negotiable operations

- Work inside this repository. Never modify `/mnt/e/AI`, download models/large datasets, call paid
  services, expose secrets, push, rewrite history, use `sudo`, or install global software.
- Use Python 3.12 and repo-local `uv` dependencies. Prefer the standard library. Keep decisions
  reversible and record major architectural or methodological choices in `docs/adr/`.
- Keep experimental cognitive mechanisms in `src/bilab/models` separate from training/evaluation
  instrumentation. Keep runtime state, temporary state, reproducibility artifacts, and telemetry
  explicitly separate.
- Every experiment declares hypothesis, baseline, controls, budgets, seeds, falsifier, stopping rule,
  artifacts, and evidence label. Report negative results and incompatible conditions.
- Do not fabricate metrics. Use `null`/"unavailable" and explain why. Do not generalize hardware
  measurements beyond their recorded model, build, workload, and environment.
- Update docs when behavior changes. If a mistake repeats, add concise prevention guidance here or
  to the deeper governing document.

## Map

`src/bilab/` implementation; `tests/` fixtures/tests; `experiments/` manifests;
`benchmarks/hardware/raw/` immutable evidence; `benchmarks/hardware/normalized/` derived ignored
outputs; `results/` reproducible ignored outputs; `telemetry/` optional ignored research logs;
`schemas/` physical contracts; `docs/` governance and ADRs; `config/` example plus ignored override.

For code discovery, prefer codebase-memory graph tools in this order when available: `search_graph`,
`trace_path`, `get_code_snippet`, `query_graph`, then `get_architecture`. Fall back to `rg` for literals,
non-code files, configuration, or insufficient graph results.

## Exact commands

```bash
uv sync --frozen
uv build
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run bilab doctor
uv run bilab manifest validate experiments/smoke/manifest.json
uv run bilab bench ingest benchmarks/hardware/raw --output benchmarks/hardware/normalized/current.json --summary-output benchmarks/hardware/normalized/current.txt
uv run bilab smoke --output results/smoke
uv run bilab core validate-worlds --config experiments/cognitive_core_v0/configs/final.json
uv run bilab core smoke --config experiments/cognitive_core_v0/configs/pilot.json --output results/cognitive_core_v0/smoke
uv run bilab core run --config experiments/cognitive_core_v0/configs/final.json --output results/cognitive_core_v0/final-v1.2
uv run bilab core evaluate --checkpoint results/cognitive_core_v0/final-v1.2/checkpoints/cognitive_core-seed101.pt --output results/cognitive_core_v0/reproduction/evaluate-seed101.json
uv run bilab core ablate --checkpoint results/cognitive_core_v0/final-v1.2/checkpoints/cognitive_core-seed101.pt --output results/cognitive_core_v0/reproduction/ablate-seed101.json
uv run bilab core report --results results/cognitive_core_v0/final-v1.2/results.json --output results/cognitive_core_v0/reproduction/report.md
uv run bilab manifest validate experiments/cognitive_core_v1/manifest.json
uv run bilab v1 validate
uv run bilab v1 overfit
uv run bilab v1 pilot
uv run bilab v1 pilot-level5
uv run bilab v1 pilot-compression
uv run bilab v1 pilot-level7
uv run bilab v1 pilot-level8
uv run bilab v1 pilot-level9
nice -n 5 uv run bilab v1 final --config experiments/cognitive_core_v1/configs/final.json --manifest experiments/cognitive_core_v1/manifest.json --output results/cognitive_core_v1/final-v1.0
uv run bilab v1 evaluate --checkpoint results/cognitive_core_v1/final-v1.0/checkpoints/core/seed-1701.pt --output results/cognitive_core_v1/reproduction/evaluate-seed1701.json
uv run bilab v1 probe --checkpoint results/cognitive_core_v1/final-v1.0/checkpoints/core/seed-1701.pt --output results/cognitive_core_v1/reproduction/probe-seed1701.json
uv run bilab v1 intervene --checkpoint results/cognitive_core_v1/final-v1.0/checkpoints/core/seed-1701.pt --output results/cognitive_core_v1/reproduction/intervene-seed1701.json
uv run bilab v1 ablate --checkpoint results/cognitive_core_v1/final-v1.0/checkpoints/core/seed-1701.pt --output results/cognitive_core_v1/reproduction/ablate-seed1701.json
uv run bilab v1 compare --left results/cognitive_core_v1/final-v1.0/results.json --right results/cognitive_core_v1/committed-reproduction-v1.0/results.json --output experiments/cognitive_core_v1/reproduction_summary.json
uv run bilab v1 report --results results/cognitive_core_v1/final-v1.0/results.json --reproduction-comparison experiments/cognitive_core_v1/reproduction_summary.json --output experiments/cognitive_core_v1/REPORT.md --summary-output experiments/cognitive_core_v1/final_summary.json
```

Before completion: run sync, tests, lint, doctor, relevant real ingestion/experiments, and review the
entire diff. Confirm output retention, resource accounting, docs/commands, no large additions, and
no external asset changes. Fix discovered defects. Commit only coherent verified work when identity
is configured; never publish it. See `docs/EVALUATION_PROTOCOL.md` and `docs/PLANS.md`.
