# Experiment contract

The authoritative semantic validator is `bilab.manifest`; the physical, machine-readable contract is
`schemas/experiment-manifest.schema.json`, version `1.0`. Required fields cover ID, hypothesis,
baseline/intervention, variables/metrics, budgets, expected result/falsifier, repetitions/seeds,
artifacts, stopping rule, lifecycle status, conclusion, and evidence classification. Terminal status
requires a conclusion; planned/running status forbids one.

JSON was chosen because Python reads it without a runtime dependency, values hash deterministically,
and the schema is portable. It preserves declared experimental intent and rejects unknown fields; it
does not encode arbitrary procedures or prove that claims are true. Files are small, linearly
read/written, reviewed in Git, and updated by replacement with provenance in history.

Replacement path: introduce a new schema version and adapter, validate old fixtures and manifests,
produce semantic equivalence tests, migrate copies, and retain rollback until all experiment commands
consume the new version. Encoding is provisional and is not a cognitive representation.
