# Failure modes

| Failure | Detection | Required response |
|---|---|---|
| Storage masquerades as learning | Persistent dependency/byte audit | Price all reachable state; reject unbounded gain |
| Telemetry leaks into cognition | Run benchmark with telemetry absent | Reclassify bytes or remove dependency |
| Incompatible comparison | Canonical condition diff | Reject comparison; rerun matched conditions |
| Cherry-picked seeds/runs | Manifest vs emitted run IDs | Report all runs; invalidate undeclared selection |
| Benchmark or task leakage | Held-out provenance/adversarial tests | Replace contaminated evaluation |
| Capability metric gaming | Per-task results and counterexamples | Add orthogonal metrics; narrow or reject claim |
| Catastrophic interference | Old-task pre/post scores | Reject change unless within declared tolerance |
| More compute/attempts buys gain | Resource and attempt accounting | Equalize or price the difference |
| False precision in hardware data | Environment metadata and repeats | Report uncertainty; avoid tiny-change claims |
| Unmeasured value reported as zero | Schema/semantic review | Use `null` and explain unavailability |
| Format becomes architecture | Semantic contract and replacement ADR | Add migration/ablation or remove abstraction |
| Irreproducible self-change | Clean checkout reconstruction | Reject candidate and rollback |
| Complexity without discriminating power | Ablation/dependency review | Remove unused layer or justify experimentally |

Safety issues, destructive external changes, inaccessible credentials/resources, or irreversible
high-cost forks are genuine stop conditions. Ordinary ambiguity is resolved by reversible assumptions
recorded in the plan or ADR.
