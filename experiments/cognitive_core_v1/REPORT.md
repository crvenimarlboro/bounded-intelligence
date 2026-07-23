# Cognitive Core v1 report

**Conclusion: SUPPORTED AT LEVEL H.**

The confirmatory result supports bounded outcome-only adaptation through curriculum Level H (unmarked rule revision) in the deliberately narrow Boolean ladder. It does not establish general intelligence or an architecture-independent learning principle.

## Evidence classification and completed ladder

The final classification is **supported at Level H within this test family**. Environment validity and the observed measurements are established for the checked seeds; transfer beyond this synthetic family remains a project hypothesis.

| Level | Requirement | Result |
|---|---|---:|
| A | Exhaustive binary oracle, 496 cases | 0.500 before / 1.000 after evidence |
| B | One sequence, one world, several worlds, explicit state | all 1.000 |
| C/D | Supervised-state ladder and outcome-only adaptation | both 1.000 in pilot |
| E | Global binary surface relabelling | 1.000 final delay accuracy |
| F | XOR composition of two inferred rules | 1.000 |
| G | Eight marked irrelevant steps | 1.000, zero state drift |
| H | Unmarked context-0 rule reversal | 1.000 after one feedback step |

The binary oracle needs one bit; the two-rule oracle needs two bits. Current input alone is paired with opposite labels under different public histories, so no-memory Bayes accuracy is 0.500 on the balanced evaluation.

## Architecture and controls

The selected core encodes current input, public operation context, and a phase marker with a 64-wide MLP. Its two-float workspace is read by a learned projection. A single shared gated residual thought block transforms the active hidden state, and a learned output head predicts the binary outcome. After feedback, the writer forms a public signed input/outcome relation, produces a learned candidate and gate, and performs a convex update. Fixed public context masks route the two primitive operations to the two state components; composition and marked distractor contexts do not write. Weights are frozen and only this workspace changes online.

The no-memory control has an empty state. The equal-byte control uses one header byte and seven deterministic packed public-event bytes in a ring; its learned reader receives no hidden rule. Parameter differences from the core are 1.49% and 0.74%, respectively.

## Development, supervision, and candidate decisions

One-sequence and one-world overfit both reached 1.000 in 150 steps; the explicit-state solver reached 1.000 in 250 steps; several-world post-evidence accuracy reached 1.000. Full, weak, annealed, and zero auxiliary rule supervision all reached 1.000 in the minimal pilot. Final training therefore used outcome loss only.

The generic GRU candidate was rejected as unstable: seed 402 reached 0.805 post-evidence accuracy, 0.863 probe accuracy, and 0.500 donor consistency. The predictive-state GRU and factorized candidates both passed two pilot seeds at 1.000. The factorized model was selected by the frozen tie-breaker because it used fewer parameters (38,904 versus 41,804) and less wall time. A one-float state was rejected for unstable decoding; two floats were the smallest passing float32 state.

## Frozen protocol

Final model seeds were 1701, 1702, and 1703. Every variant used 800 Adam optimizer steps, 307,200 observations per seed, batch groups of eight, and the final optimizer step as its checkpoint. Training used full 12-step BPTT with no detach inside an episode and detach only at world boundaries. Evaluation performed no gradient descent. Training, validation, probe, and evaluation generation ranges were disjoint before final seeds were opened.
The frozen manifest SHA-256 is `e68c2d970185ace9e0b64876ac66c096c9abb39c4dbc71689a6638424cb103c5`; the final configuration file SHA-256 is `f44fd4a711c6c041af13cc47e3d35706a1755e7020ee6d53000b53a63c77ec56`.

## Confirmatory comparison

| Variant | Parameters | State bytes | Delay | Composition | Recovery | Retention | Random | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core | 38,952 | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.493 | 17.26 |
| no_memory | 39,532 | 0 | 0.500 | 0.500 | 0.500 | 0.500 | 0.498 | 13.60 |
| episodic | 39,242 | 8 | 0.591 | 0.655 | 0.839 | 0.994 | 0.503 | 22.59 |
| error_detached | 38,952 | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.487 | 17.47 |
| error_differentiable | 38,952 | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.487 | 18.23 |
| error_surprise | 38,952 | 8 | 0.986 | 1.000 | 1.000 | 1.000 | 0.491 | 17.94 |
| recurrence_k3 | 38,952 | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.487 | 24.70 |

All values are means over 3 untouched final seeds. Each variant consumed 307,200 training observations per seed and used the final optimizer step; no final-seed checkpoint selection occurred.

Standard deviations for core delay, composition, recovery, and retention were all 0.000. Context accuracy was 0.986 ± 0.014; random control was 0.493 ± 0.004.

### Core per final seed

| Seed | Delay | Context | Composition | Change step | Recovery | Retention | Random |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1701 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.495 |
| 1702 | 1.000 | 0.993 | 1.000 | 0.000 | 1.000 | 1.000 | 0.497 |
| 1703 | 1.000 | 0.967 | 1.000 | 0.031 | 1.000 | 1.000 | 0.487 |

### Learning and online adaptation curves

| Optimizer step | 1 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 | 800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean validation accuracy | 0.500 | 0.580 | 0.833 | 0.910 | 0.914 | 0.914 | 0.914 | 0.914 | 0.913 | 0.911 | 0.914 | 0.914 | 0.914 | 0.914 | 0.914 | 0.914 | 0.915 |

In delayed evaluation, task accuracy was 0.500 before either rule was known (positions 0 and 1), then 1.000 at the first query after both evidence events and the configured non-events. It remained 1.000 through position 15. At an unmarked rule change, accuracy was 0.010 on the change event—as expected when the new rule is unknowable—then 1.000 from the first post-feedback opportunity.

## Mechanistic evidence

- The core used two float32 values (8 bytes) and 38,952 trainable parameters.
- Reset and frozen-state accuracy were both 0.500; random state was 0.503.
- Donor-state consistency was 1.000; held-out rule decoding was 0.958.
- The surface-label probe remained at 0.516, consistent with the state carrying rules rather than the relabelling bit.
- Delayed state drift was 0.000; the minimum measured early-write gradient norm was 0.356.
- K=1 scored 1.000; forcing K=3 at evaluation scored 0.951. Extra recurrence was not useful.
- Mean write-gate activation was 0.086; 0.812 of gates were below 0.05 and none exceeded 0.95. The high low-gate fraction is partly imposed by context masking.
- Removing state component 0 or 1 reduced mean accuracy to 0.681 and 0.639.
- Detached error, differentiable error, and surprise inputs did not improve the outcome-only core. Explicit prediction error is therefore rejected for this level.

## Compression

| Bits/value | Canonical bytes (ceiling) | Accuracy |
|---:|---:|---:|
| 1 | 1 | 0.750 |
| 2 | 1 | 0.923 |
| 4 | 1 | 1.000 |
| 8 | 2 | 1.000 |
| 16 | 4 | 1.000 |

Four-bit quantization per value preserved full accuracy (one canonical byte total); two-bit quantization retained 0.923 mean accuracy, while one-bit values fell to 0.750. This is a post-training intervention, not a separately trained quantized core.

## Capability-density accounting

The declared capability measure is delayed accuracy gain over pre-evidence accuracy: 0.500. This is 0.062500 gain per persistent float32 byte and 0.007812 per state bit. Delay accuracy per trainable parameter is 0.00002567; adaptation gain per training observation is 0.0000016276. A delayed episode uses 16 prediction and 16 update calls, giving 0.015625 gain per call. These are task-specific density measures, not measures of general intelligence.

The post-hoc four-bit-per-value state has an eight-bit canonical total and retained full accuracy, corresponding to 0.062500 gain per bit. It is not credited as a trained one-byte runtime until quantized-state training and serialization are implemented.

## Resources and reproducibility

- Full 21-run confirmatory wall time: 409.26 seconds.
- Peak measured resident memory: 330.8 MiB; VRAM was not used.
- Checkpoints: 3,510,183 bytes total; normalized output: 6,384,137 bytes.
- Maximum checkpoint replay error: 0.0.
- CPU time and temporary allocation bytes were not measured reliably; peak RSS and wall time are reported instead. VRAM was unused.

## Scope and strongest counterevidence

The selected writer is deliberately scaffolded: it receives a hand-computed public input/outcome relation, uses fixed context-to-slot masking, and treats marked distractors as non-writes. Thus the experiment proves that a learned bounded reader/writer can retain, compose, transfer, and revise a sufficient statistic; it does not prove autonomous discovery of that statistic. The environment is Boolean, the surface relabelling is a global bit flip, only three model seeds were run, and quantization was post hoc. These are the strongest limits on generalization.

## Bugs, amendments, and comparison with v0

Two pilot defects were found before preregistration. A deterministic reversal position allowed anticipation; amendment 0001 randomized valid change opportunities and added no-change training worlds. The first Level-9 pilot also overlapped generated training and evaluation seeds; amendment 0002 preserved that invalid run, added a range-audit regression test, and reran the pilot on disjoint ranges. Amendment 0003 corrects only post-run output-byte accounting. Amendment 0004 strengthens the temporal-credit probe to cross eight real update operations and preserves the superseded diagnostic. No confirmatory metric or threshold changed.

V0 used 1,203,000 parameters and 4,096 workspace bytes yet stayed near four-class chance; workspace, error, and K=3 ablations were inert. V1 uses 38,952 parameters and eight bytes, passes basic learnability first, carries delayed temporal gradients, decodes the rule, and changes behavior causally under state swaps. What remains unsolved is whether a generic writer can discover the sufficient statistic without the engineered relation and routing mask, and whether the effect survives richer relabellings and tasks.

## v2 recommendation

Remove the engineered relation and context mask one at a time. Require a generic writer to infer the same two-bit sufficient statistic from raw public fields, retain donor-state causality, and match the 8-byte control under the same observations. If that passes, make operation identities relabel per world before increasing task complexity.

Detailed per-seed metrics, learning curves, interventions, and checkpoint metadata are in `results/cognitive_core_v1/final-v1.0/`.
The checkpoints record training parent revision `b482a8930941d6d1713c9dd175e37f99d2c5fc67`; the completed implementation commit and clean-worktree status are reported in the final repository handoff because a file cannot contain its own commit hash.
