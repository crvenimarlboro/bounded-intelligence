# Roadmap

## 0. Laboratory bootstrap — current

Establish contracts, accounting, environment inspection, benchmark ingestion, tests, and a disposable
smoke pipeline. Exit only when commands reproduce locally and governance matches behavior.

## 1. First falsifiable memory experiment

Create a tiny synthetic sequential environment and compare a fixed-byte learned procedure against
equal-byte episodic storage and a no-memory baseline. Measure held-out transfer, retention,
observations, compute, and persistent bytes across enough seeds to estimate uncertainty.

## 2. Compression and transfer

Test whether learned generators/procedures reconstruct useful behavior, transfer to changed tasks,
and permit raw episode deletion. Add adversarial and ablation tests; reject mere memorization.

## 3. Continual learning under a storage plateau

Predeclare a task sequence and fixed persistent-byte budget. Test replacement, consolidation, and
active forgetting against equal-resource baselines while tracking old-task retention.

## 4. Adaptive computation and modular replacement

Only after earlier evidence, test conditional compute and sandboxed component proposals with explicit
accept/reject/rollback. Scale models or introduce specialized representations only when a smaller
experiment demonstrates the need.

Every phase may disprove the intended direction. Advancement requires evidence, not schedule.
