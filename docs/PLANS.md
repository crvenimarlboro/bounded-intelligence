# Experiment and implementation plan template

## Identity and evidence state

- Plan/experiment ID, owner, date, revision:
- Current evidence label:
- Reversible assumptions and alternatives considered:

## Claim

- Falsifiable hypothesis:
- Baseline and intervention:
- Expected result and explicit falsification condition:
- Why this is the smallest discriminating experiment:

## Controls and resources

- Independent/dependent/controlled variables:
- Persistent/temporary bytes, RAM/VRAM, time/CPU, parameters, observations, compute/energy proxy,
  attempts, context, tools, and model budgets:
- Seeds, repetitions, uncertainty method, held-out/adversarial cases:
- Telemetry/cognition boundary and artifact retention:

## Execution

- Ordered reversible changes:
- Exact commands and stopping rule:
- Acceptance, rejection, regression, and rollback criteria:
- Representation semantics/loss/cost/provenance/update/replacement:

## Result

- All runs and failures:
- Resource-paid gain and incompatible conditions:
- Conclusion, limitations, evidence label:
- Retain/revise/reject decision and next falsifiable test:

For executable experiments, encode the contract in the versioned JSON manifest and validate it before
running. The prose plan explains judgment; it does not replace machine validation.
