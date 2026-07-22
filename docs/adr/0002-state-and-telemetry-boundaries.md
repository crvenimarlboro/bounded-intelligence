# ADR 0002: Separate cognitive state from research artifacts

- Status: accepted
- Date: 2026-07-22
- Evidence: project principle implemented as repository policy

## Decision

Classify persistent artifacts by runtime dependency, not directory name. `benchmarks/hardware/raw`,
manifests, schemas, fixtures, code, and accepted documentation are reproducibility evidence.
`benchmarks/hardware/normalized`, `results`, and `telemetry` are regenerable/optional research outputs
and ignored. `data/temporary`, caches, and local `models` are ignored. An experiment records persistent
cognitive state separately and cannot access excluded telemetry during final evaluation.

## Consequences and replacement

Research disk usage may exceed claimed cognitive state, but must be disclosed separately. Any runtime
dependency moves its bytes into cognitive accounting even if stored externally. If future artifact
storage changes, preserve these semantic categories, add clean-room tests that run without telemetry,
and document migration/rollback in a superseding ADR.
