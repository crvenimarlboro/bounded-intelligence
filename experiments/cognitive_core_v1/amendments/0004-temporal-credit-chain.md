# Post-run correction 0004: temporal-credit update chain

Status: diagnostic-only correction after confirmatory execution.

The original temporal-credit probe wrote evidence, issued several unused prediction calls, and then
backpropagated a query loss through the original state. It proved that a later reader loss reached the
writer, but it did not thread state through intervening update operations as claimed. Training itself
used full 12-step BPTT and all behavioral evaluations were correct, so this defect does not affect
model weights, checkpoints, capability metrics, seed selection, or thresholds.

The corrected probe applies eight actual state updates between evidence and the only supervised query,
retains every intermediate state gradient, and requires non-zero gradients across the complete chain.
For the final four-context core, these are publicly marked distractor updates whose fixed mask leaves
the state value unchanged but whose autograd identity path is explicit. The original values remain in
normalized results as `temporal_credit_pre_amendment_0004`; raw metrics are unchanged. Corrected
diagnostics are rerun from all three frozen checkpoints with the documented report refresh command.
