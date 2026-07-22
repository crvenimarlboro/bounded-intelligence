# Amendment 0002: Query-schedule fix

Date: 2026-07-22
Status: frozen before corrected final execution

## Defect and affected result

Review after the protocol 1.1 final execution found that `_query_kind` recreated its ten-element list on every call. It shuffled only on steps divisible by ten, then discarded the shuffled list. Consequently, most positions followed a fixed query-family order instead of retaining the shuffled order for the whole block. Hidden rules, outcomes, future observations, and targets were not exposed, and the same schedule was used by every model. Nevertheless, this contradicted the intended Rule Worlds definition, so the 238.59-second protocol 1.1 execution is superseded as confirmatory evidence. It had concluded unsupported, with core structured accuracy 0.246 versus 0.253 no-memory and 0.250 episodic.

## Correction

Protocol 1.2 keeps one shuffled list for each complete ten-step block and includes a test that every block has the declared five direct, two composed, one counterfactual, and two delayed queries, with varying order. The first query cannot be delayed because no public predecessor exists.

No model, optimizer, world seed, observation count, memory budget, parameter count, evaluation size, success threshold, or stopping rule changes. All three models, all three seeds, all required ablations, checkpoint saves, and independent reload evaluations are rerun to `results/cognitive_core_v0/final-v1.2`.
