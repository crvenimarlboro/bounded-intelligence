# Cognitive Core v3C joint pilot plan

Status: implementation prepared; joint reserved-pilot evidence not yet observed.

## Entry evidence

V3C is opened only because the isolated reserved pilots reported:

- V3A `raw_bilinear_overwrite` passed both isolated pilot seeds with the signed relation absent and the learned raw relation pathway causally necessary.
- V3B `relation_hard_skip_softtrain` passed both preservation-revision pilot seeds with exact zero distractor writes, zero drift through 100,000 updates, and successful one-feedback revision.

The runner validates both result files and refuses to start V3C if either candidate lacks passing rows for every recorded seed.

## Joint hypothesis

A raw bilinear evidence encoder, learned router, and learned exact evaluation skip can jointly support delayed prediction, composition, one-feedback revision, unrelated-rule retention, and at least 0.90 retained accuracy after 100,000 distractors using exactly two float32 runtime values.

## Candidates

Two runtime mechanisms are tested under both random and staged initialization:

1. `raw_soft_router_hard_skip_softtrain`
   - raw public fields only;
   - learned bilinear relation encoder;
   - learned soft routing during training and evaluation;
   - soft write strength during training;
   - deterministic exact binary skip/write during evaluation.

2. `raw_hard_router_softtrain`
   - same raw relation and write mechanism;
   - soft routing during training;
   - deterministic one-hot routing during evaluation.

Random initialization is required for a full V3-C claim. Staged initialization is a compatibility diagnostic and can support at most V3-C-STAGED.

## Staged transfer policy

Staged initialization is explicit and source-audited:

- V3A contributes only `writer_encoder.*`.
- V3B contributes every target-compatible parameter except `writer_encoder.*`, including the learned router, write controller, value candidate, reader, thought block, and output path.
- exact parameter names and tensor shapes must match;
- source checkpoint family, seed, digest, path, and transferred parameter provenance are recorded;
- no supplied-relation encoder parameter is copied into V3C.

## Pilot separation and resources

- Joint pilot model seeds: 3301 and 3302.
- Joint training worlds begin at 680000.
- Validation worlds begin at 688000.
- Evaluation worlds begin at 690000.
- Final confirmatory worlds beginning at 700000 remain unopened.
- 800 AdamW updates, 307,200 observations, full 12-step BPTT, K=1.
- Exactly eight persistent runtime bytes.
- Six CPU threads and existing dependencies only.

Staged seed pairing is 3301 <- 3201 and 3302 <- 3202. The target joint training and evaluation worlds remain fresh.

## Frozen pilot gates

Every seed must satisfy the existing common gates plus:

- learned raw relation pathway ablation drop >= 0.20;
- retained accuracy after 100,000 distractors >= 0.90;
- one-feedback recovery after the long span >= 0.90;
- unrelated-rule retention after the long span >= 0.90.

A high relation probe without behavioral deployment fails. Exact preservation without revision fails. Staged success does not count as random-initialized autonomous joint discovery.

## Decision

- Random and staged pass: advance the random candidate to five-seed preregistration.
- Staged passes, random fails: classify pilot result as staged compatibility only and diagnose joint optimization.
- Both fail: preserve V3-AB as the highest isolated result and do not open final seeds.
