# Amendment 0001: slot-permutation floating-point tolerance

Status: accepted before the replacement final run.

## Original protocol and defect

Commit `8f22f619313904310f6e97845ba2db359684f466` required a corresponding
slot/parameter permutation to preserve behavior. The implementation incorrectly required bit-exact
equality of cross-entropy loss after reordering two algebraically equivalent float32 dot products.
Final seed 2701 produced identical predictions, adaptation curve, and accuracies but losses
`0.1911131584765826` and `0.19111316685848578` (difference `8.38e-9`). The runner therefore marked
the intervention false even though its behavioral invariant passed.

The initial run was interrupted and preserved at
`results/cognitive_core_v2/superseded-final-slot-permutation-v1.0`.

## Correction

The permutation still requires exact equality of every discrete prediction-derived metric and the
complete adaptation curve. Mean loss may differ by at most `1e-7`, well below the smallest
behavioral resolution and typical float32 accumulation error. A regression test now checks both
conditions. No model, seed, threshold, data, optimizer, checkpoint rule, or other intervention
changed.

All final variants and seeds are rerun from scratch. The interrupted run is superseded and cannot
support a claim.
