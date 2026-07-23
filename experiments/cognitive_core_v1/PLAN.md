# Cognitive Core v1 execution plan

Status: development and pilot. Cognitive Core v0 is immutable evidence.

## Question and starting point

Test whether a frozen-weight neural system can learn an online update algorithm that infers a
balanced hidden COPY/FLIP rule from public input/outcome feedback and stores the useful statistic
in a fixed state. The first environment deliberately contains one bit of hidden structure:
`outcome = input XOR rule`. Paired worlds expose identical current observations with opposite
targets, so current input cannot solve the task. An exact oracle is the analytic upper bound.

V0 used full episode BPTT, but its overloaded curriculum, unconstrained 4 KiB state, and lack of
state-content interventions did not localize the near-chance failure. V1 begins with K=1, four
float32 state values (16 bytes), and a direct delayed-gradient audit.

## Development seed policy

- Development world seeds: 10,000--19,999; initialization seeds: 101--109.
- Pilot world seeds: 30,000--39,999; initialization seeds: 401--403. Each executable
  configuration must prove its actual consumed train, validation, probe, and evaluation ranges
  are disjoint; a nominal range declaration is not sufficient.
- Confirmatory world seeds: 70,000--80,511 in explicitly disjoint ranges; initialization seeds:
  1701--1703.
- Ranges are disjoint. Confirmatory seeds remain unopened until a versioned manifest is frozen.

## Ladder and gates

1. Exhaustively verify the paired environment and oracle; oracle post-evidence accuracy >= 0.99
   and current-observation Bayes accuracy <= 0.55.
2. Require > 0.99 one-sequence and one-world neural overfit and > 0.99 accuracy from an explicit
   correct rule state.
3. Require supervised state decoding >= 0.95, post-evidence task accuracy >= 0.90, and >= 0.20
   loss under wrong/reset state.
4. Compare full, weak, annealed, and zero rule-state auxiliary supervision.
5. Outcome-only adaptation must reach >= 0.85 after evidence, gain >= 0.25 from pre-evidence,
   lose >= 0.20 under reset/frozen/wrong state, decode rule >= 0.90, and improve for every final
   seed. This is the first autonomous bounded-adaptation claim.
6. Advance one factor at a time only after the preceding gate passes: multiple rules, surface
   relabelling, composition, delay, then reversal. The first failed level is the capability
   frontier.

## Candidates and controls

- Exact analytic oracle and a current-observation-only neural control.
- Exact-byte ring-buffer episodic control using only public events.
- General GRU state core: learned recurrent writer and learned reader.
- Predictive-state core: recurrent state plus a future-outcome sufficient-statistic objective.
- Factorized evidence core: controlled evidence/confidence state with a learned public-feedback
  update. Its structural bias is reported explicitly.

Candidate selection uses pilot seeds only: pass lower-level gates, maximize held-out adaptation,
require causal state sensitivity, then prefer fewer state bytes, parameters, and compute. At most
three revisions per family are allowed, and each revision must answer a measured failure.

## Temporal credit and resources

Training uses full BPTT within each short world and detaches only at world boundaries. A loss whose
only query occurs several steps after evidence must produce non-zero gradient in the first writer.
Held-out evaluation uses `no_grad`, frozen weights, and only the fixed state across steps.

Use at most six PyTorch threads, under 4 GiB RAM when practical, under 500 MiB generated artifacts,
and under 250 MiB retained checkpoints. Record optimizer steps, training observations, wall time,
peak RAM, parameters, state bytes, forward/update counts, and checkpoint bytes.

## Confirmatory integrity

The selected level, candidate, budgets, seeds, checkpoint rule, thresholds, controls, ablations,
and stopping rule will be frozen in `manifest.json` before confirmatory seeds are evaluated. Bugs
after freezing require a preserved invalid run, regression test, versioned amendment, and complete
rerun of affected conditions. Final results are never used for candidate selection.
