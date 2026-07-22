# Research principles

1. **Bound every resource.** Record persistent/temporary bytes, RAM/VRAM where measurable, wall and
   CPU time, parameters, observations, compute proxy, attempts, and capability. State unavailable
   values; never infer measurements from hardware specifications.
2. **Turn experience into structure.** Prefer generators to stored outputs, procedures to transcripts,
   causal models to correlations, and compact sufficient state to full history. Test reconstruction,
   transfer, and safe forgetting.
3. **Keep every layer replaceable.** Text, tokens, models, graphs, programs, JSON, objectives, and
   interfaces are provisional. Components need semantic contracts, versions, ablations, independent
   tests, rollback, and a documented replacement path. Abstraction must serve an experiment.
4. **Use equal-resource controls.** Match model, parameters, data, context, compute, storage, attempts,
   tools, seeds, and evaluation wherever possible. Name the independent variable and residual
   confounders.
5. **Prefer falsification.** Use deterministic seeds, repeated runs, held-out transfer, adversarial
   cases, uncertainty, failure analysis, and negative-result retention. Never select favorable runs.
6. **Constrain autonomy.** Architectural proposals proceed through sandboxed candidate, evaluation,
   regression tests, accept/reject decision, and recoverable rollback.
7. **Separate telemetry from cognition.** Research logs may be large, but final runtime evaluation
   must execute without them and their bytes cannot be excluded from cognition if the runtime reads
   them.
8. **Use minimum complexity.** Build the smallest discriminating experiment. No database, service,
   dashboard, framework, or special encoding earns a place without measured need.
9. **Account for representation.** Document preserved meaning, discarded information, acceptable
   loss, byte/read/write costs, update semantics, provenance, replacement, and utility test.

Important claims use one label: **established**, **supported but uncertain**, **project hypothesis**,
**implementation assumption**, **speculative possibility**, or **disproven by current evidence**.
Labels describe current evidence, not desirability.
