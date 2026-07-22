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
