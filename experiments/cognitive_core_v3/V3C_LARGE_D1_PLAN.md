# Cognitive Core V3C large development replication

Status: preregistered large development test. Confirmatory seeds 3701--3705 and worlds beginning
at 700000 remain unopened.

## Motivation

The cross-pair audit used one target-training seed, 800 optimizer updates, 307,200 observations,
64 evaluation groups, and a maximum preservation span of 100,000 distractors. It established one
exactly preserving off-diagonal composition, but one run cannot establish seed robustness.

This test scales the evidence in four directions at once:

- six independent target-training seeds;
- three preregistered conditions, for eighteen trained models;
- 1,600 optimizer updates and 1,228,800 observations per model;
- 256 evaluation groups and a 1,000,000-distractor exact-preservation stress test.

## Conditions

1. `v3c_large_robust_b3201`: V3A seed 3202 plus V3B seed 3201. This is the off-diagonal pair
   that passed behavioral and exact gates in the cross-pair audit.
2. `v3c_large_fragile_b3202`: V3A seed 3201 plus V3B seed 3202. This pair retained behavior but
   failed exact preservation in the cross-pair audit.
3. `v3c_large_random`: the same soft-routing architecture from random initialization. This tests
   whether a larger fixed development budget is sufficient for autonomous joint discovery.

## Fixed resources

- Target seeds: 3501--3506.
- Training worlds begin at 610000.
- Validation worlds begin at 640000.
- Evaluation worlds begin at 650000; existing diagnostic offsets remain below 700000.
- 1,600 AdamW updates.
- 12 batch groups and 16 steps per episode.
- 1,228,800 training observations per model.
- 128 validation groups and 256 evaluation groups.
- Preservation spans: 10, 100, 1,000, 10,000, 100,000, and 1,000,000 distractors.
- 48,100 trainable parameters, two float32 state values, and exactly eight persistent bytes.
- Six CPU threads and final-step checkpoint selection.

## Gates

Every row is evaluated with the existing V3C behavioral gates and strengthened exact-preservation
gate. Exact preservation requires zero distractor writes, zero changed-state events, zero drift,
zero code transitions, and bit-identical predictions at every measured span.

Primary success for a condition requires all six seeds to pass both behavioral and exact gates.
Results of five out of six are reported as near-replication but do not satisfy the primary claim.

## Interpretation

- Robust donor passes 6/6: strong development evidence for a seed-robust modular V3C composition.
- Robust donor passes 5/6: promising but not yet stable enough for confirmatory testing.
- B3202 remains behaviorally strong but exact-fragile: preservation robustness is donor-specific.
- Random passes 6/6: larger-budget autonomous joint V3C becomes development-supported.
- Random remains unstable: the remaining problem is optimization/coordination, not test size.

The runner is resumable at the candidate-seed job boundary. An interrupted overnight run can be
restarted without repeating completed jobs.
