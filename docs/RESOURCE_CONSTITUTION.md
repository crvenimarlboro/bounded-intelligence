# Resource constitution

## Accounted boundaries

Each experiment separates and reports:

- **Persistent cognitive state:** bytes required by the tested runtime after learning. If runtime
  behavior depends on an artifact, it belongs here regardless of its directory or marketing label.
- **Temporary task state:** peak disposable bytes used during a run.
- **Research telemetry:** diagnostics used only by researchers; excluded from cognitive claims and
  forbidden as a hidden runtime dependency.
- **Reproducibility artifacts:** manifests, code, compact results, and immutable source evidence.
- **Optional archives:** large, nonessential material, never required by the final benchmark.

Also record peak RAM/VRAM, elapsed and CPU time, parameters, input observations, attempts, model/tool
access, compute proxy, score, seed, revision, and configuration hash. `null` means unavailable; zero
means measured/defined zero. Energy must remain unavailable until directly measured or a clearly
labeled proxy is justified.

## Metrics

- Byte capability density: `score / persistent_cognitive_bytes` (undefined at zero bytes; report the
  pair instead of infinity).
- Parameter capability density: `score / model_parameters` (same zero rule).
- Experience efficiency: `held_out_capability_gain / new_observations_consumed`.
- Compute efficiency: `held_out_capability_gain / declared_compute_proxy` when units are comparable.
- Retention: old-task score after learning divided by old-task score before learning.
- Plateau test: over a preregistered task sequence, held-out capability trend is positive while
  persistent cognitive bytes stay below a fixed budget and raw experience is not retained elsewhere.

Ratios never replace the underlying numerators, denominators, confidence intervals, or task coverage.
Resource growth is allowed only when predeclared and justified; a gain purchased merely by storage is
not architectural progress.

## Storage policy

Raw hardware reports are immutable tracked evidence. Normalized reports and smoke outputs are compact
regenerable ignored artifacts. Manifests, schemas, fixtures, and accepted summaries are tracked.
Temporary data, telemetry, caches, and local models are ignored. A future format must document its
semantic loss, provenance, costs, update rules, usefulness test, and migration/rollback path.
