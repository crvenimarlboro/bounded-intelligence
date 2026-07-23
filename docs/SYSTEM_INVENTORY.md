# System inventory

Observed 2026-07-22; exact runtime output comes from `uv run bilab doctor`.

## Host and development

- Windows 10 host; WSL2 Ubuntu development environment.
- AMD Ryzen 5 3600: 6 physical / 12 logical processors (host-provided; doctor reports logical count).
- 16 GiB system RAM; Radeon RX 570 with nominal 8 GiB VRAM. VRAM availability is not sampled by the
  bootstrap doctor and must not be inferred during experiments.
- Python 3.12.3 via uv; Git 2.43, Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1, jq 1.7.
- Repo-local CPU training runtime: PyTorch 2.13.0+cpu and NumPy 2.4.4. No ROCm/GPU training path is
  assumed or used by Cognitive Core v0 or v1.
- Cognitive Core v1 confirmatory execution used six PyTorch threads, peaked at 346,906,624 resident
  bytes, and completed 21 training/evaluation runs in 409.26 seconds. Its 21 checkpoints total
  3,510,183 bytes. These are measurements of this exact run, not host guarantees.
- Cognitive Core v2 confirmatory execution used six PyTorch threads, peaked at 343,293,952 resident
  bytes, and completed 20 training runs plus five independent primary reproductions in 883.01
  seconds. Its 20 checkpoints total 3,696,710 bytes. The longer time includes five 100,000-step
  stability traces; it is not directly comparable with v1 wall time.

## Read-only external inference assets

- Windows root `E:\AI`, visible in WSL as `/mnt/e/AI`.
- `llama.cpp-vulkan/llama-bench.exe`: reported build 10088, commit 67b9b0e7f in source artifacts.
- `models/Qwen3.5-0.8B-Q4_K_M.gguf`: Q4_K_M, source record size 568,653,056 bytes and 772,845,888
  parameters. The doctor stats the file without loading it.
- `benchmarks/`: native CPU, Vulkan, layer-sweep, and final 6/7-layer JSONL/CSV reports. Repository
  copies preserve these small files; `/mnt/e/AI` remains read-only.

Known results are scoped to that model, quantization, llama.cpp build, workload, driver, and desktop
state. Browser/desktop activity was present during much of collection. Six GPU layers showed the best
measured interactive balance in the final 512-input/128-output comparison; this is supported but
uncertain as a general machine setting and must not be generalized to other conditions.
