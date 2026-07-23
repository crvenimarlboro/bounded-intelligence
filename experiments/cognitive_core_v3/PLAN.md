# Cognitive Core v3 execution plan

Status: development and reserved pilot. Cognitive Core v0, v1, and v2 are immutable evidence.

## Claim and isolation ladder

V3 tests two failures separately before any joint claim:

1. **V3A:** derive and behaviorally deploy the COPY/FLIP sufficient statistic from raw public
   input/outcome fields while using an explicitly scaffolded, fixed research route.
2. **V3B:** preserve supplied-relation state through 100,000 irrelevant public events using a
   learned exact skip or learned attractor, without a semantic no-write mask.
3. **V3C:** attempted only if both isolated pilot gates pass; remove relation, fixed routing, and
   semantic write masks together.

The strict raw interface is versioned in `input_contract.json`. V3A/V3C receive no XOR, equality,
signed relation, hidden rule, correct slot, write label, future event, seed, transcript, probe
output, or telemetry.

## Prior failure reproduction

Committed V2 pilots were rerun unchanged under `results/cognitive_core_v3/v2-reproduction/`.
All raw writers again decoded the relation, but behavioral deployment remained unstable:

- raw fixed-route recovery was 0.844--0.847 at equal budget and 0.821--0.856 at double budget;
- raw soft-router recovery was 0.864--0.869 and route separation collapsed to 0.140 for one seed;
- raw GRU retention fell to 0.836 for one equal-budget seed;
- learned bilinear candidates still failed at least one reserved seed;
- relation-supplied V2B alone passed, reproducing 0.997--1.000 recovery.

Implementation assumption: partial convex writes allow old state to dominate contradictory raw
evidence. V3 tests exact overwrite/discrete-code mechanisms before changing model or data scale.

## Candidates

V3A:

- `raw_bilinear_overwrite`: generic learned input/outcome projections and multiplicative
  interaction; exact overwrite through the fixed research route.
- `raw_discrete_overwrite`: generic raw encoder selects a learned scalar code through a
  straight-through categorical decision; exact overwrite through the fixed research route.

V3B:

- `relation_hard_skip`: supplied relation, learned addressing, and a learned straight-through
  write/skip decision whose evaluation skip is bit-exact identity.
- `relation_attractor`: supplied relation, learned addressing, and projection onto learned
  state prototypes after each update.

If eligible, V3C integrates the retained raw encoder and preservation mechanism using a learned hard
address and learned hard write decision. Random initialization is required for full V3-C support;
compatible isolated weights are also tested as staged initialization.

## Resources, seeds, and selection

- Six CPU threads; soft 4 GiB/hard 8 GiB RAM; no GPU.
- Two float32 state values: exactly eight runtime bytes; `K=1`.
- Primary equal budget: 800 AdamW updates and 307,200 observations per candidate/seed.
- Development model seeds 3101--3199 and generation seeds 500000--559999.
- Reserved pilot model seeds 3201/3202 and generation seeds 600000--679999.
- Confirmatory model seeds 3701--3705 and generation seeds beginning at 700000; these remain
  unopened until the manifest, configuration, and hashes are committed.
- Candidate selection order: contract and bytes; learnability; temporal gradient; relation
  deployment/recovery; causal state dependence; long preservation/revision; donor consistency;
  seed stability; then bytes, parameters, observations, and time.

At most two evidence-driven revisions per family are allowed. More data is only a separately priced
secondary point. V3C is blocked unless one V3A and one V3B candidate pass every reserved-pilot gate.

## Gates and stopping

V3A requires every selected seed to reach at least 0.95 delay, composition, recovery, retention,
surface transfer, donor consistency, and relation probe; state interventions must cost 0.30 and
random outcomes remain within 0.05 of chance.

V3B requires the same ordinary behavior and recovery plus retained accuracy of
0.99/0.99/0.98/0.97/0.95 after 10/100/1,000/10,000/100,000 distractors, exact fixed bytes,
finite detached evaluation state, and one-feedback revision after long preservation.

V3C uses the frozen gates in its confirmatory manifest, including at least 0.90 retention after
100,000 distractors. Stop opening branches after the declared candidates/revisions fail; preserve a
negative frontier instead of random search. Rollback is Git revert plus regeneration of ignored V3
results. Prior evidence is never rewritten.
