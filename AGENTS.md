# Repository agent guide

Read `docs/PROJECT_CHARTER.md`, `docs/RESEARCH_PRINCIPLES.md`, and the relevant ADRs before changing
methodology. Inspect before editing; preserve user changes and raw evidence.

## Non-negotiable operations

- Work inside this repository. Never modify `/mnt/e/AI`, download models/large datasets, call paid
  services, expose secrets, push, rewrite history, use `sudo`, or install global software.
- Use Python 3.12 and repo-local `uv` dependencies. Prefer the standard library. Keep decisions
  reversible and record major architectural or methodological choices in `docs/adr/`.
- Treat `src/bilab` as research instrumentation, not cognition. Keep runtime cognitive state,
  temporary state, reproducibility artifacts, and telemetry explicitly separate.
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
```

Before completion: run sync, tests, lint, doctor, relevant real ingestion/experiments, and review the
entire diff. Confirm output retention, resource accounting, docs/commands, no large additions, and
no external asset changes. Fix discovered defects. Commit only coherent verified work when identity
is configured; never publish it. See `docs/EVALUATION_PROTOCOL.md` and `docs/PLANS.md`.
