# Bounded Intelligence Laboratory

This repository is a small, reproducible laboratory for testing whether machine capability can
improve through better internal organization while persistent cognitive state remains bounded. It
is not an AGI implementation, an LLM wrapper, or a claim of superiority over frontier models.

The research target is higher capability per persistent byte, parameter, joule, observation, and
unit of compute on consumer hardware. A valid result must identify which resource paid for a gain,
survive equal-resource comparison, and keep research telemetry separate from runtime cognition.
See [the project charter](docs/PROJECT_CHARTER.md) and
[research principles](docs/RESEARCH_PRINCIPLES.md).

## Quick start

Requires Python 3.12 and `uv`.

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run bilab doctor
uv run bilab manifest validate experiments/smoke/manifest.json
uv run bilab bench ingest benchmarks/hardware/raw \
  --output benchmarks/hardware/normalized/current.json \
  --summary-output benchmarks/hardware/normalized/current.txt
uv run bilab smoke --output results/smoke
```

Build the package with `uv build`. Machine-specific paths come from
`config/lab.local.json` (ignored), environment variables, or the checked-in
`config/lab.example.json`; the doctor never loads the model or mutates external assets.

## Repository map

- `src/bilab/`: doctor, benchmark ingestion, experiment contracts, resource accounting, smoke CLI.
- `tests/`: deterministic unit and end-to-end tests using tiny fixtures.
- `docs/`: constitution, evaluation protocol, roadmap, inventory, and ADRs.
- `schemas/`: physical JSON schema for the versioned experiment contract.
- `experiments/`: reviewed, tracked experiment manifests and code/data specific to experiments.
- `benchmarks/hardware/raw/`: tracked, immutable copies of small source benchmark reports.
- `benchmarks/hardware/normalized/`: ignored derived results; regenerate from raw sources.
- `results/`: ignored reproducible experiment outputs.
- `telemetry/`: ignored optional research telemetry, never runtime cognitive state.
- `data/temporary/`: ignored disposable task data; `models/` is ignored local model storage.

Artifact roles and retention rules are defined in
[the resource constitution](docs/RESOURCE_CONSTITUTION.md). The first smoke experiment only proves
the pipeline can run; it deliberately makes no intelligence claim.

## Cognitive Core v0

The first trainable architecture experiment is implemented under
`experiments/cognitive_core_v0/`. It compares a 4 KiB gated latent workspace against a stateless
neural predictor and an exact-byte episodic ring buffer in procedural Rule Worlds. The corrected
three-seed result is **unsupported**: all systems remained near four-class chance and the full core
did not beat no memory. See [the report](experiments/cognitive_core_v0/REPORT.md).

```bash
uv run bilab core validate-worlds --config experiments/cognitive_core_v0/configs/final.json
uv run bilab core smoke --config experiments/cognitive_core_v0/configs/pilot.json \
  --output results/cognitive_core_v0/smoke
uv run bilab core run --config experiments/cognitive_core_v0/configs/final.json \
  --output results/cognitive_core_v0/final-v1.2
uv run bilab core evaluate \
  --checkpoint results/cognitive_core_v0/final-v1.2/checkpoints/cognitive_core-seed101.pt \
  --output results/cognitive_core_v0/reproduction/evaluate-seed101.json
uv run bilab core ablate \
  --checkpoint results/cognitive_core_v0/final-v1.2/checkpoints/cognitive_core-seed101.pt \
  --output results/cognitive_core_v0/reproduction/ablate-seed101.json
uv run bilab core report --results results/cognitive_core_v0/final-v1.2/results.json \
  --output results/cognitive_core_v0/reproduction/report.md
```

Generated checkpoints and metrics are reproducible but ignored under `results/`; the preregistration,
amendments, configuration, code, tests, and concise report are tracked.

## Cognitive Core v1

V1 replaces the overloaded v0 task with an oracle-backed adaptation ladder. Its selected model has
38,952 parameters and an eight-byte workspace (two float32 values). It learns from public
input/outcome feedback with full within-episode BPTT, then adapts online with frozen weights. The
three-seed confirmatory result is **SUPPORTED AT LEVEL H within the narrow Boolean ladder**: delayed,
composed, relabelled, retention, and one-feedback-step reversal-recovery accuracy were 1.000;
no-memory was 0.500 and the equal-byte episodic control averaged 0.591 on delayed queries. Random
control remained at 0.493. See the [v1 report](experiments/cognitive_core_v1/REPORT.md) for causal
interventions and limitations.

The strongest caveat is architectural scaffolding: the writer receives a hand-computed public
input/outcome relation and fixed context-to-slot routing. V1 establishes causal bounded state use,
not autonomous discovery of the sufficient statistic.

```bash
uv run bilab manifest validate experiments/cognitive_core_v1/manifest.json
uv run bilab v1 validate
uv run bilab v1 overfit
uv run bilab v1 pilot
uv run bilab v1 pilot-level5
uv run bilab v1 pilot-compression
uv run bilab v1 pilot-level7
uv run bilab v1 pilot-level8
uv run bilab v1 pilot-level9
nice -n 5 uv run bilab v1 final \
  --config experiments/cognitive_core_v1/configs/final.json \
  --manifest experiments/cognitive_core_v1/manifest.json \
  --output results/cognitive_core_v1/final-v1.0

uv run bilab v1 evaluate \
  --checkpoint results/cognitive_core_v1/final-v1.0/checkpoints/core/seed-1701.pt \
  --output results/cognitive_core_v1/reproduction/evaluate-seed1701.json
uv run bilab v1 probe \
  --checkpoint results/cognitive_core_v1/final-v1.0/checkpoints/core/seed-1701.pt \
  --output results/cognitive_core_v1/reproduction/probe-seed1701.json
uv run bilab v1 intervene \
  --checkpoint results/cognitive_core_v1/final-v1.0/checkpoints/core/seed-1701.pt \
  --output results/cognitive_core_v1/reproduction/intervene-seed1701.json
uv run bilab v1 ablate \
  --checkpoint results/cognitive_core_v1/final-v1.0/checkpoints/core/seed-1701.pt \
  --output results/cognitive_core_v1/reproduction/ablate-seed1701.json
uv run bilab v1 report \
  --results results/cognitive_core_v1/final-v1.0/results.json \
  --reproduction-comparison experiments/cognitive_core_v1/reproduction_summary.json \
  --output experiments/cognitive_core_v1/REPORT.md \
  --summary-output experiments/cognitive_core_v1/final_summary.json
```

The original final runner undercounted normalized output bytes, and its first temporal-credit probe
did not apply intervening updates. The reporting-only corrections are reproducible and idempotent:

```bash
uv run bilab v1 report \
  --results results/cognitive_core_v1/final-v1.0/results.json \
  --checkpoint-root results/cognitive_core_v1/final-v1.0/checkpoints \
  --refresh-resource-accounting --refresh-temporal-credit \
  --reproduction-comparison experiments/cognitive_core_v1/reproduction_summary.json \
  --output experiments/cognitive_core_v1/REPORT.md \
  --summary-output experiments/cognitive_core_v1/final_summary.json
```

`v1 final` trains all seven declared variants across all three seeds and writes 21 compact
checkpoints, raw/normalized metrics, and CSV learning/adaptation curves. The pilot commands train the
candidate families and advance one curriculum factor at a time. All generated files under `results/`
are ignored intentionally; the compact [final summary](experiments/cognitive_core_v1/final_summary.json),
protocol, amendments, and report are tracked.
