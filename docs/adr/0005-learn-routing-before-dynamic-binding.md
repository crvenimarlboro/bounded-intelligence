# ADR 0005: Retain learned routing evidence and defer dynamic binding

Status: accepted, 2026-07-23.

Evidence label: supported but uncertain.

## Context

Cognitive Core v1 solved the Boolean ladder with an eight-byte state, but code supplied both the
input/outcome relation and operation-to-slot/no-write masks. V2 separately removed these scaffolds.
The raw writers learned decodable relations but did not reproducibly achieve one-feedback recovery
and stable routing. A relation-supplied soft router passed both reserved pilot seeds and all five
confirmatory seeds.

## Decision

Retain the V2B soft router as evidence that routing and distractor suppression can move from
hand-written code into learned computation. Do not treat it as the preferred final cognitive
representation and do not advance to per-world dynamic binding or trained quantization until a
raw-field-only V2C mechanism passes.

Use two float32 state values, `K=1`, full 12-step BPTT, outcome loss, final-step checkpoints, and the
mixed curriculum as the reproducible comparison point. Continue to label the signed relation as an
engineered scaffold.

## Consequences

- V2 supports learned organization of fixed state, not autonomous sufficient-statistic discovery.
- Equal-byte episodic memory remains a strong delay baseline and must stay in future comparisons.
- Future non-write objectives must include much longer distractor sequences because V2B content
  drifted despite fixed bytes.
- Generic candidates may use learned multiplicative layers, but no external XOR/equality feature or
  hidden relation label may enter final V2C training/evaluation.

## Rejected alternatives

- More optimizer steps: doubling observations did not stabilize V2A or V2C.
- Selecting a favorable generic seed: rejected by the frozen all-seeds pilot rule.
- Advancing directly to operation relabelling: rejected because joint discovery did not pass.
- Calling decodable writer features success: rejected because causal behavior remained unstable.

## Replacement and rollback

The router is replaceable through the existing model-family contract and removable by routing
ablation. A future ADR may supersede this decision after a raw-field-only mechanism passes at least
the same five-seed causal protocol. Rollback is Git revert plus regeneration of ignored results; v0,
v1, and V2 pilot/final evidence remains unchanged.
