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
