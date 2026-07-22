# ADR 0003: Fixed-byte online adaptation comparison

- Status: accepted for Cognitive Core v0; mechanism unsupported by current evidence
- Date: 2026-07-22
- Evidence: completed three-seed controlled experiment

## Decision

Use procedural Rule Worlds and three parameter-matched CPU neural systems to isolate persistent-state representation. Cognitive Core v0 owns exactly 1,024 float32 workspace values (4,096 bytes), reads them during prediction, applies one shared gated residual thought block three times, and updates them from the observation, hidden state, outcome, and detached prediction error. The episodic control owns an exact 4,096-byte ring buffer; the no-memory control has no cross-step state. Model weights are frozen during held-out online evaluation.

PyTorch CPU is the only substantial runtime dependency. JSON configuration, ordinary state dictionaries, and flat result files remain replaceable encodings; none is treated as a final cognitive representation.

## Consequences and replacement

This makes state bytes, parameters, observations, checkpoints, compute, leakage boundaries, and ablations auditable. It also exposes that v0 failed to learn the intended cross-world procedure: disabling state did not materially hurt performance. Retain the negative result and tests. A successor should first prove within-episode rule inference in a smaller curriculum with an analytic oracle and state-decoding probes, then reintroduce composition, delay, reversal, and relabeling one factor at a time. Replacement is a new versioned model/config/manifest; do not rewrite v0 or its amendments.
