# ADR 0004: Minimal factorized predictive state

- Status: accepted for Cognitive Core v1; intentionally provisional
- Date: 2026-07-23
- Evidence: development ladder, reserved pilots, and completed three-seed confirmation

## Decision

Replace v0's 4,096-byte unstructured workspace with two float32 state values (eight bytes) for the
two-rule Boolean ladder. Use a learned gated convex writer and learned reader around a public signed
input/outcome relation. Fixed public context masks route primitive operations to state slots and
prevent composition or marked distractor events from writing. Use one shared thought cycle, outcome
feedback only, and full within-episode BPTT. Freeze all weights during held-out adaptation.

Compare against an empty-state neural control and an exact eight-byte packed episodic ring, both
parameter-matched within 2%. Treat PyTorch tensors, float32, slot meanings, Boolean relations, and the
physical JSON/checkpoint formats as replaceable experimental choices—not final cognitive ontology.

## Evidence and consequences

Across three untouched final seeds, the 38,952-parameter core reached 1.000 delayed, compositional,
relabelled-delay, post-change recovery, and retention accuracy. The no-memory control stayed at 0.500;
the equal-byte episodic control averaged 0.591 on delayed queries. Reset/frozen/random state removed
the effect, donor-state swaps transferred it, rule decoding reached 0.958, and random outcomes stayed
at chance. K=3 and explicit prediction-error variants earned no advantage and are rejected for v1.

This establishes bounded causal state use only in a narrow scaffolded family. The public relation and
fixed routing mask substantially simplify sufficient-statistic discovery. They must not be described
as autonomous abstraction.

## Replacement and rollback

V2 should remove the relation feature and context mask one at a time while preserving the environment,
budgets, seeds discipline, controls, and causal diagnostics. Accept a generic writer only if it
recovers the same result without hidden-rule inputs. Rollback is the versioned v1 configuration,
checkpoints, concise evidence, and report; do not rewrite v0 or v1 evidence when a successor fails.
