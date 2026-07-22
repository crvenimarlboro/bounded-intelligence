# Cognitive Core v0 execution plan

## Identity and evidence state

- Experiment: `cognitive-core-v0-rule-worlds-v1`; preregistered 2026-07-22 at revision `1f3decd`.
- Evidence before execution: project hypothesis; no model result has been observed.
- Reversible assumptions: 4 KiB float32 workspace, fixed three-cycle thought recurrence, procedural
  four-class Rule Worlds, CPU PyTorch, final-epoch checkpoint selection. JSON configuration and model
  classes are replaceable; no external data or model is used.

## Claim

A learned, gated 4 KiB latent workspace updated with outcome and prediction error will improve
held-out structured-rule adaptation over no persistent memory and a deterministic equal-byte episodic
ring buffer. It should not create the same advantage on random-control worlds. Full thresholds and
falsifiers are frozen in `manifest.json` and `configs/final.json` before pilot execution.

## Controls and resources

- Same observations, targets, train/test worlds, optimizer steps, seeds, encoder vocabulary, outcome
  classes, and evaluation checkpoints. All neural variants target parameter counts within 2%.
- Independent variables: persistent representation/update pathway and full-core ablation mode.
- Persistent state: exactly 4096 bytes per evaluated world. Weights/checkpoints and telemetry are
  reported separately. CPU only; less than 2 GiB peak RAM, 250 MiB checkpoints, and 3600 seconds hard
  task ceiling.
- Final model seeds: 101, 202, 303. Generator seed ranges are disjoint and declared in config.
- Amendment 0001, frozen after the repeated-world pilot and before any final-seed run, preserves the
  32,768-observation and 64-step budget but uses 512 distinct training worlds for one epoch. It does
  not change models, evaluation, seeds, or success thresholds.
- Amendment 0002 fixes a query-block shuffling defect discovered during review of the first final
  execution. All final conditions are rerun from new checkpoints; the first execution is retained
  only as superseded diagnostic evidence.

## Execution

1. Implement leakage-audited Rule Worlds, three neural systems, differentiable gated updates,
   bounded episodic storage, training/evaluation/checkpoint/report commands, and substantive tests.
2. Run the tiny smoke config, then the one-seed pilot. If runtime or learnability requires protocol
   changes, record a numbered amendment before final execution; never overwrite this preregistration.
3. Run all final seeds and required ablations, reload checkpoints independently, generate normalized
   metrics/curves/tables, review failures and resource costs, then accept or reject the hypothesis.

Rollback is ordinary Git revert plus deletion of ignored generated results. Raw telemetry is never an
input to evaluation. The model receives only the current public observation and bounded state.
