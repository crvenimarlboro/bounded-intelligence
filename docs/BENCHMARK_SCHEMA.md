# Hardware benchmark canonical schema

`bilab bench ingest` reads native llama-bench JSONL/CSV (including UTF-8 BOM), validates required
fields, and emits schema version `1.0`. Each record preserves source filename/row; build commit/number;
model filename/type/bytes/parameters; backend/device/CPU/GPU; GPU layers, threads, batch/microbatch,
KV types and flash-attention mode; prompt-vs-generation phase and token counts; timestamp; latency,
throughput, standard deviations, and available samples. Record IDs are deterministic truncated hashes.

Meaning preserved: performance and compatibility metadata needed for the present hardware baseline.
Discarded: unknown llama-bench fields not in the contract and any unrecorded host activity. Source
files remain authoritative and immutable, making loss acceptable and normalization replaceable.

Comparisons treat GPU-layer count (and its coupled device selection) as the intervention. All other
compatibility conditions must match. A condition ID hashes those controls; ratios are emitted only
within one condition ID against its lowest-layer result. The tool separates incompatible cohorts and
warns rather than deriving a combined result. Flat JSON costs roughly source-size scale, is
read/written linearly, and is regenerated rather than incrementally mutated. Replace it with a
versioned adapter/migration only when query scale proves a need; no database is justified now.
