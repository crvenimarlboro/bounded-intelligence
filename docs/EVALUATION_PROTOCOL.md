# Evaluation protocol

1. Preregister a versioned manifest: unique ID, falsifiable hypothesis, baseline, intervention,
   controls, independent variable, metrics, resource budgets, expected result, falsifier, repetitions,
   seeds, artifacts, stopping rule, status, conclusion, and evidence label.
2. Freeze evaluation code and held-out cases before observing intervention results. Record revision and
   stable configuration hash. Preserve raw inputs unchanged.
3. Match model, parameter count, observations, context, compute/attempt budget, storage, tools, and
   evaluation. If a mismatch is intrinsic to the intervention, price it and report it rather than
   pretending it is controlled.
4. Run all declared seeds and failures. Report distributions or confidence intervals when repetitions
   support them, plus adversarial cases, transfer, retention, and ablations appropriate to the claim.
5. Reject the comparison if conditions differ outside declared independent variables, source
   provenance is missing, budgets are exceeded, results cannot be reconstructed, or hidden telemetry
   is required. Reject the claimed improvement for leakage, cherry-picking, more attempts/data/storage,
   changed scoring, or a regression that violates predeclared acceptance criteria.
6. Classify evidence, record negative results, and state limitations. Do not generalize hardware
   results beyond exact build/model/quantization/workload/driver/environment evidence.

Implementation changes remain reversible. Candidate self-modification happens in isolation, then
faces the same evaluation and regression suite. Accept only on preregistered evidence; otherwise
reject and retain a compact failure record. Git revision plus declarative configuration provide the
current rollback path; destructive history rewriting is prohibited.
