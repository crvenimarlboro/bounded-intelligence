# ADR 0001: Minimal flat-file Python laboratory

- Status: accepted
- Date: 2026-07-22
- Evidence: implementation assumption

## Context and decision

The bootstrap needs reproducible contracts, parsers, accounting, and tests—not scalable services.
Use Python 3.12 in a `src/bilab` package managed by uv; standard-library runtime code; pytest/Ruff as
development dependencies; versioned JSON/JSONL/CSV flat files; and no database, web framework, model
SDK, or static type checker yet. Components expose small semantic functions behind one CLI.

This minimizes installation, makes costs visible, and allows every component to be independently
tested/removed. JSON preserves current declarations and metrics but discards no source evidence because
raw benchmark inputs remain immutable.

## Consequences and replacement

Linear scans and manual semantic validation are acceptable at current scale. JSON is not endorsed as
a final intelligence representation. Replace a component only after a benchmark demonstrates need:
add a versioned adapter/schema, equivalence fixtures, migration, ablation, and rollback. A database or
type checker requires an ADR with measured benefit. Rollback is removal of the package/entry point and
restoration through Git; source evidence remains independent.
