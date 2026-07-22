# Amendment 0001: Training-world diversity

Date: 2026-07-22
Status: frozen before final execution

## Original protocol

Protocol 1.0 specified 128 training worlds of length 64 repeated for four epochs. This was 32,768 observations and 64 optimizer steps per model and seed. Training world seeds were 10000 through 10127. All success thresholds, final model seeds, evaluation worlds, model configurations, and resource budgets were frozen in `manifest.json` and `configs/final.json`.

## Pilot evidence

The one-seed pilot used 64 worlds of length 48 for three epochs (9,216 observations per model; configuration hash `4c5d46c4f68fdad83a04d40a5da1f81aff2ca4e7fd73e3152b15cb1bbaec764c`). Training accuracy rose to 0.315 for no memory, 0.378 for episodic memory, and 0.354 for the core, while structured held-out accuracy remained 0.245, 0.225, and 0.258 respectively. The full core's largest required-ablation drop was only 0.015. This pattern is consistent with repeated-world memorization and does not establish a generally useful update rule.

## Change

Protocol 1.1 replaces four passes over 128 worlds with one pass over 512 disjoint worlds, using seeds 10000 through 10511. Episode length, total observations (32,768), batch size, optimizer steps (64), optimizer settings, parameter counts, persistent bytes, model seeds, validation and evaluation worlds, checkpoints, and all success/failure thresholds are unchanged.

The diversity pilot uses 256 worlds of length 64 for one epoch and seed 17. Its purpose is only to verify that the amended final run remains computationally feasible and that the task produces finite learning signals. It is not final evidence.

## Rationale and scope

This reversible change gives the same resource budget to learning a cross-world procedure instead of repeatedly fitting a small set of world identities. It does not make a favorable result easier by increasing observations or optimizer steps. No result from final seeds was inspected before this amendment was frozen.
