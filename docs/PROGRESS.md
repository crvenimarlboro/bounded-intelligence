# Progress log

## 2026-07-22 — bootstrap

- **Established:** repository began as an uncommitted `uv init` skeleton with empty source/test/docs
  directories; Python 3.12.3, uv 0.11.31, Git, Clang, CMake, Ninja, and jq are available in WSL2.
- **Established:** read-only inspection found the Qwen3.5 0.8B model, Windows `llama-bench.exe`, and ten
  small benchmark reports under `/mnt/e/AI`; native reports contain build 10088 / commit 67b9b0e7f.
- **Implementation assumption:** versioned JSON/JSONL plus flat files are sufficient for the bootstrap;
  no database or runtime framework is justified.
- Added governance, ADRs, package/CLI, doctor, native benchmark normalization, resource primitives,
  manifest validation, fixtures/tests, and the disposable smoke experiment.
- **Established:** `uv sync --frozen` and `uv build` succeed; 20 tests pass; Ruff lint and format checks
  report no issues.
- **Established:** the doctor detects WSL2, 12 logical CPUs, available RAM, the 579,615,840-byte model,
  ten external/repository benchmark artifacts, and reports VRAM/physical-core/binary-version gaps as
  unavailable rather than fabricating them.
- **Established:** real ingestion normalizes 34 records into prompt/generation phases, emits 24
  compatibility-gated comparisons, and warns on two non-native summary schemas.
- **Established:** the three-seed CPU smoke pipeline emits complete accounting, succeeds under its
  declared budgets, demonstrates a failed threshold and rejected incompatible comparison, and makes
  no intelligence claim.
- **Established:** all ten repository source reports and current `/mnt/e/AI/benchmarks` files match the
  checked-in SHA-256 manifest. Generated outputs remain outside cognitive-state accounting and follow
  the storage policy.

## 2026-07-22 — Cognitive Core v0

- **Established:** Rule Worlds v0, three real CPU neural systems, bounded state, training/evaluation,
  checkpoints, ablations, and substantive leakage/reproduction tests are implemented. Parameter
  counts are 1,203,056 core, 1,203,065 episodic, and 1,203,197 no-memory.
- **Established:** the core and episodic runtime states are exactly 4,096 bytes per world; tests hold
  this constant through 10, 100, 1,000, and 10,000 observations. The no-memory system is stateless.
- **Established:** corrected protocol 1.2 trained seeds 101, 202, and 303 for 32,768 observations per
  model/seed and evaluated all required ablations. It completed in 227.32 seconds at 475,090,944-byte
  peak RAM with 43,444,830 checkpoint bytes; every checkpoint reproduced exactly.
- **Disproven by current evidence:** Cognitive Core v0 is unsupported in Rule Worlds v0. Structured
  accuracy was 0.248 versus 0.251 no-memory and 0.247 equal-byte episodic; its step-0-to-32 adaptation
  changed by -0.049, retention was 0.243, and no ablation produced the required degradation.
- **Established:** a repeated-world pilot motivated a resource-neutral diversity amendment. Review
  then found a query-shuffle defect; it was recorded, fixed, tested, and every final condition rerun.
  See `experiments/cognitive_core_v0/REPORT.md` and ADR 0003.

## 2026-07-23 — Cognitive Core v1

- **Established:** an exhaustive 496-case binary oracle is 0.500 before evidence and 1.000 after one
  public outcome; paired identical current observations require opposite answers under different
  histories. One-sequence, one-world, several-world, and explicit-state neural checks reached 1.000.
- **Established:** generic GRU, predictive-state GRU, and factorized-state cores were trained. The
  generic GRU was unstable on one pilot seed; predictive and factorized models passed, and the
  factorized model won the preregistered lower-parameter/lower-compute tie-breaker.
- **Established:** full, weak, annealed, and absent rule-state auxiliary loss all solved the minimal
  task. Final training therefore used outcome loss only, full 12-step BPTT, and no online gradients.
- **Established:** the selected final core has 38,952 parameters and exactly eight persistent bytes.
  Across final seeds 1701/1702/1703 it scored 1.000 on delayed, compositional, relabelled-delay,
  post-change recovery, and unrelated-rule retention evaluations. No-memory was 0.500; the exact
  eight-byte episodic ring scored 0.591 ± 0.031 on delayed queries; random control was 0.493 ± 0.004.
- **Established:** reset/frozen/random state returned accuracy to approximately 0.500; donor-state
  swaps transferred behavior at 1.000; rule probes decoded 0.958; both state-component ablations hurt;
  delayed gradients reached the writer; checkpoints reproduced with zero numerical error.
- **Supported but uncertain:** four-bit post-hoc quantization per state value preserved 1.000 accuracy
  with a one-byte canonical ceiling; two bits/value retained 0.923. Quantized training was not tested.
- **Disproven by current evidence:** K=3 did not improve K=1, and detached/differentiable prediction
  error or surprise inputs did not improve outcome-only feedback. These mechanisms are not retained
  as v1 advantages.
- **Supported at Level H within this environment, not generally:** v1 establishes causal bounded
  adaptation in the Boolean ladder. Its hand-computed input/outcome relation and fixed slot routing
  are substantial scaffolds; autonomous sufficient-statistic discovery remains unproven.
- Two invalid pilot shortcuts were found and preserved before preregistration: deterministic reversal
  timing and overlapping generation ranges. Both were corrected, regression-tested, and rerun. A
  post-run amendment corrects normalized-output byte accounting; another strengthens the
  temporal-credit probe to cross eight actual updates while preserving the original diagnostic.
- **Established:** the complete 21-condition protocol was rerun from committed implementation
  `a588386d6708366c5a606b787c9d93e373e2d1dc`. All deterministic result fields matched exactly and
  all 21 model-state digests were identical; only runtime/resource and Git provenance fields were
  excluded from exact comparison.

## 2026-07-23 — Cognitive Core v2

- **Established:** frozen v1 checkpoint interventions reduced delay from 1.000 to 0.533 without the
  supplied relation, to 0.594 under uniform routing, and to 0.676 when distractor writes were
  permitted. The relation and fixed routing/no-write scaffolds were causally important.
- **Established:** the raw writer interface excludes external XOR/equality/relation, correct slot,
  write target, hidden rule, future event, seed, and history. One-sequence, one-world, several-world,
  explicit-relation, and supervised-relation diagnostics all overfit at 1.000.
- **Disproven by current evidence:** V2A and both serious V2C families were not reproducible across
  reserved pilot seeds. Doubling training observations and adding learned multiplicative features
  did not stabilize joint relation/routing discovery.
- **Supported at V2-B:** the relation-supplied learned router passed all five final seeds with 0.998
  delay, 1.000 composition, 0.997 recovery, 1.000 retention, 1.000 donor consistency, and 0.480
  random-control accuracy. Reset, freeze, and writer-disabled conditions were 0.500; uniform routing
  was 0.675. All checkpoints and diagnostics reproduced exactly.
- **Established:** the final core has 43,284 parameters and an exactly eight-byte state. Parameter-
  matched controls were within 1.21%. Equal-byte episodic memory matched delay at 0.999 but recovered
  at 0.691; no memory remained 0.500.
- **Supported but uncertain:** learned routes separated the two primitive operations by 0.839 and
  learned distractor write strength of only 0.038--0.058 without a mask. Both state components were
  causal and corresponding slot/parameter permutations preserved behavior.
- **Established counterevidence:** content drifted under extremely long distractor streams; mean
  retained-rule accuracy fell from 0.975 at 10 updates to 0.725 at 100,000 while bytes stayed fixed.
- One final-run diagnostic defect required amendment 0001: behavior-preserving float32
  reparameterization changed loss by `8.38e-9`. The invalid partial run was preserved, a tolerance
  regression added, and all final conditions rerun unchanged.
