# Cognitive Core V3C cross-pair compatibility audit

Status: preregistered development audit. Confirmatory seeds remain unopened.

## Trigger

The reserved joint pilot established behavioral success only for the soft-routed staged candidate.
Random initialization failed, hard evaluation routing was seed-unstable, and one behaviorally passing
soft model changed state on every distractor. The original V3C gate therefore overstated exact
preservation and matching source seeds left a latent-compatibility confound.

## Hypothesis

A successful V3A raw relation encoder can be composed with a successful V3B preservation subsystem
without requiring matching source seeds. Joint training should retain behavioral competence and
recover exact no-write preservation under the fixed V3 budget.

## Pairing matrix

The audit composes all four source pairs:

- V3A seed 3201 + V3B seed 3201;
- V3A seed 3201 + V3B seed 3202;
- V3A seed 3202 + V3B seed 3201;
- V3A seed 3202 + V3B seed 3202.

All pairs use the same target seed, training worlds, validation worlds, evaluation worlds, optimizer,
and budget so source pairing is the only intended difference.

## Procedure

For each pair:

1. Compose the V3A `writer_encoder.*` tensors with the V3B compatible non-writer tensors.
2. Run the complete diagnostic battery before joint optimization.
3. Train the soft-routing V3C model for 800 AdamW updates and 307,200 observations.
4. Run the same diagnostics after training.
5. Record behavioral and exact-preservation gates separately.

Hard routing is not trained as a second candidate because the previous soft and hard candidates had
identical trained model digests. Hard-routing evidence from the joint pilot is retained as a separate evaluation result; it is not
retrained in this source-compatibility audit.

## Exact preservation gate

Behavioral retention is insufficient for an exact-preservation claim. Every stress measurement must
have:

- zero cumulative changed-state events;
- zero cumulative nonzero writes;
- zero drift norm;
- predictions bit-identical to the pre-distractor reference.

The aggregate stress result must also report constant state size, zero changed-state events, zero
code transitions, and zero nonzero distractor writes.

## Resources and separation

- Target model seed: 3401.
- Training worlds begin at 580000 and remain below the reserved-pilot range.
- Validation worlds begin at 587000.
- Audit evaluation worlds begin at 588000; diagnostic offsets remain below 600000.
- Worlds beginning at 700000 and confirmatory seeds 3701--3705 remain unopened.
- Exactly eight persistent bytes, 48,100 trainable parameters, K=1, six CPU threads.

## Falsifiers and decision

- Both off-diagonal pairs pass behavioral and exact gates: staged subsystems are modularly compatible.
- Only diagonal pairs pass: compatibility depends on aligned source conventions.
- Behavioral gates pass but exact gates fail: robust staged behavior exists, exact preservation does not.
- No pair passes behavioral gates: the earlier staged result is not stable under controlled recomposition.

No final-seed run follows automatically from this audit.
