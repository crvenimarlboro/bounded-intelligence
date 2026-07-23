# Cognitive Core v2 final report

Conclusion: **SUPPORTED AT V2-B**.

This is a narrow result. V2 learned routing, write strength, and distractor suppression when the
COPY/FLIP relation was supplied. It did **not** reproducibly discover both that relation and its
memory organization from raw fields. V2C is unsupported; V2D dynamic binding and V2E trained
quantization were not opened because their prerequisite failed.

## Claim boundary

**Established:** across five preregistered seeds, the selected 43,284-parameter model used exactly
two float32 state values (eight bytes), adapted with frozen weights, passed every V2B threshold, and
reproduced exactly from all five checkpoints.

**Supported but uncertain:** a generic neural controller can replace v1's fixed operation-to-slot
mask and fixed distractor no-write mask on this Boolean ladder.

**Disproven by current evidence:** the tested raw MLP, dense GRU, and learned-bilinear candidates did
not jointly discover the sufficient relation and stable routing across both reserved pilot seeds.

**Not tested:** per-world operation-ID binding and trained quantized generic state.

## What v1's hand-written code supplied

The frozen v1 checkpoint audit found these task-specific pathways:

- `FactorizedStateCore.update` computed a signed input/outcome product equivalent to the hidden
  COPY/FLIP relation.
- the first two context one-hot values directly selected the two state components;
- composition and distractor contexts received an all-zero state-write mask;
- state interpretation therefore depended on fixed slot meanings;
- composition output was learned rather than hard-coded, and hidden rules/seeds remained telemetry.

Post-hoc interventions over all three immutable v1 checkpoints measured:

| V1 intervention | Delay | Composition | Recovery | Retention | Donor | Rule probe |
|---|---:|---:|---:|---:|---:|---:|
| Native | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 |
| Relation zero/raw-only | 0.533 | 0.501 | 0.499 | 0.509 | 0.500 | 0.302 |
| Random balanced relation | 0.561 | 0.503 | 0.493 | 0.735 | 0.500 | 0.473 |
| Incorrect relation | 0.333 | 1.000 | 0.452 | 0.000 | 1.000 | 0.958 |
| Routing swapped | 0.667 | 1.000 | 0.568 | 0.485 | 1.000 | 0.813 |
| Routing shuffled per world | 0.833 | 1.000 | 0.783 | 0.767 | 1.000 | 0.775 |
| Uniform routing | 0.594 | 0.788 | 0.696 | 0.651 | 0.813 | 0.745 |
| Correct relation, random routing | 0.612 | 0.949 | 0.701 | 0.700 | 1.000 | 0.633 |
| Distractor writes enabled | 0.676 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 |

Enabling distractor writes produced 0.492 mean distractor drift. This audit establishes that both
the relation and routing/non-write scaffolds carried causal capability in v1.

## Writer interface and architecture

The strict V2C raw interface contains only current input, a generic four-way public operation
encoding, phase, observed outcome after prediction, and previous state. It prohibits externally
computed equality/XOR/relation, hidden rule, correct slot, write target, future event, or seed.

The final V2B model intentionally adds one derived field: the v1 signed public relation. Everything
else is learned:

1. the raw eight fields plus relation pass through a `9 → 64 → 32` value encoder;
2. raw fields plus the previous two-value state pass through a `10 → 32 → 2` softmax router;
3. a separate `10 → 32 → 1` controller learns write strength;
4. a scalar learned candidate is expanded across the two state coordinates;
5. `gate = route × write_strength` performs a bounded convex state update;
6. the current-observation encoder, state reader, one shared gated thought application (`K=1`), and
   output head generate the prediction.

No full history, cache, gradient update, or growing state is available during held-out evaluation.
The relation scaffold means this architecture is not V2C.

## Development ladder and candidate decisions

Fixed-data learnability passed at 1.000 for one sequence, one world, several worlds, an explicit
relation diagnostic, and a relation-supervised raw diagnostic. Full BPTT spans all 12 episode steps.
Delayed query gradients crossed eight earlier state updates.

Outcome-only pilot evidence is retained in `candidate_ledger.md`. The decisive results were:

- V2A raw fixed routing decoded the relation at 1.000 but recovery stayed near 0.845.
- V2B relation-supplied learned routing passed both pilot seeds.
- V2C raw soft routing had one route-separation collapse to 0.140.
- V2C dense GRU had unstable retention and weak state organization.
- doubling training to 614,400 observations did not stabilize V2A/V2C;
- learned multiplicative features helped one seed but the other V2C seed still fell to 0.666
  retention and 0.065 route separation.

The pilot rule therefore selected V2B. No final seed influenced that choice.

## Confirmatory protocol

Final source revision was `a1ec8895e72aa7a0261d966da8956beb4f5ecdb1`. Model seeds were
2701--2705. Every variant received 800 AdamW updates, 307,200 training observations per seed, the
same mixed context/composition/delay/reversal curriculum, the same final worlds, full 12-step BPTT,
and final-step checkpoint selection. Evaluation froze every learned weight.

Parameters were matched to the 43,284-parameter primary:

| Variant | Parameters | Difference | Runtime state |
|---|---:|---:|---:|
| V2B learned router | 43,284 | reference | 8 B |
| V1 scaffolded reference | 43,808 | +1.21% | 8 B |
| No memory | 43,588 | +0.70% | 0 B |
| Equal-byte episodic | 43,322 | +0.09% | 8 B |

## Final behavior

Means and population standard deviations span all five final seeds.

| Variant | Delay | Composition | Recovery | Retention | Surface relabel | Random |
|---|---:|---:|---:|---:|---:|---:|
| V2B learned router | 0.9977 ± 0.0017 | 1.0000 ± 0.0000 | 0.9973 ± 0.0053 | 1.0000 ± 0.0000 | 0.9964 ± 0.0033 | 0.4799 ± 0.0075 |
| V1 scaffolded reference | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4766 ± 0.0043 |
| No memory | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4956 ± 0.0156 |
| Episodic, 8 B | 0.9991 ± 0.0018 | 0.9904 ± 0.0051 | 0.6913 ± 0.0300 | 0.9734 ± 0.0190 | 0.9992 ± 0.0016 | 0.5105 ± 0.0042 |

The episodic control essentially matched delayed prediction. V2B's meaningful advantage over it was
rapid rule revision, not ordinary delay. The project-specific delay gain per byte was 0.06221 for
V2B and 0.06239 for episodic; recovery gain per byte was 0.06217 versus 0.02391. These ratios are
task metrics, not general intelligence measures.

The episodic policy is deterministic and intentionally strong: one header byte plus seven packed
public event bytes form a ring; each record stores input, observed outcome, phase, validity, and
public context. It skips publicly marked context-3 distractors and evicts by ring overwrite. That
hard policy is research code rather than learned cognition, but it uses no hidden rule, future
outcome, seed, or more than eight canonical bytes.

Per primary seed:

| Seed | Delay | Recovery | Surface | Random | Route separation | Rule probe | Passed |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2701 | 0.9961 | 1.0000 | 0.9909 | 0.4766 | 0.9470 | 0.7917 | yes |
| 2702 | 1.0000 | 0.9867 | 1.0000 | 0.4805 | 0.9316 | 0.7943 | yes |
| 2703 | 0.9967 | 1.0000 | 0.9967 | 0.4941 | 0.6417 | 0.8073 | yes |
| 2704 | 0.9961 | 1.0000 | 0.9948 | 0.4727 | 0.8936 | 0.9583 | yes |
| 2705 | 0.9993 | 1.0000 | 0.9993 | 0.4759 | 0.7832 | 0.9583 | yes |

The delayed curve was 0.500 before either public feedback event, 0.9883 at the first held-out query
after evidence and distractors, 0.9984 at the next query, and at least 0.9992 thereafter. Full curves
are in `results/cognitive_core_v2/final-v1.0/adaptation_curves.csv`.

## Mechanistic evidence

All means below use the same final worlds.

| Intervention | Informed accuracy | Drop from 0.9977 |
|---|---:|---:|
| Reset state every step | 0.5000 | 0.4977 |
| Freeze initial state | 0.5000 | 0.4977 |
| Random initial state | 0.4984 | 0.4992 |
| Shuffle state across worlds | 0.4882 | 0.5095 |
| Add state noise | 0.9202 | 0.0775 |
| Uniform routing | 0.6749 | 0.3228 |
| Random routing | 0.7458 | 0.2518 |
| Swap routing destinations only | 0.6757 | 0.3220 |
| Disable writer | 0.5000 | 0.4977 |
| Remove state component 0 | 0.7137 | 0.2840 |
| Remove state component 1 | 0.6815 | 0.3161 |

Donor-state consistency was 1.000 for every seed. Simultaneously permuting state coordinates and all
corresponding learned routing/read coordinates preserved every discrete behavior for every seed.
Incorrectly changing routes did not. This supports a causal learned addressing interpretation rather
than a passive state correlate.

Primitive route separation was 0.839 ± 0.114 (minimum 0.642). Distractor write strength ranged
0.038--0.058, learned without v1's mask. It was not exactly zero: this explains the long-horizon
drift below. Overall gates averaged 0.156--0.184; 57.0--72.7% were below 0.05 and none exceeded 0.95.

The hidden relation was linearly decoded at 1.000; the two-rule state probe averaged 0.862 ± 0.079.
The irrelevant surface probe was 0.531, near binary chance. Probe decodability is diagnostic and did
not train the final model.

Every delayed-gradient audit reached the writer encoder, candidate, write controller, router,
reader, observation encoder, and output head. Minimum retained-state gradient norms ranged
0.780--23.682 across seeds; no path was zero or non-finite.

## Fixed bytes and long-horizon failure

State shape remained `[batch, 2]`, canonical state remained eight bytes per world, values remained
finite, and no autograd history was retained through 100,000 updates for every seed.

| Distractor updates | Mean retained rule accuracy |
|---:|---:|
| 10 | 0.975 |
| 100 | 0.825 |
| 1,000 | 0.825 |
| 10,000 | 0.725 |
| 100,000 | 0.725 |

Mean 100,000-step drift was 0.963 in state-vector norm. This is the strongest direct limitation of
the learned no-write controller: storage remains bounded, but content is not indefinitely stable.

## Resources and reproducibility

- Complete confirmatory wall time: 883.01 seconds.
- Sum of model-training time: 439.29 seconds.
- Peak process RAM: 343,293,952 bytes (cumulative process peak; per-variant peaks are not separable).
- Total training observations: 6,144,000 across 20 variant/seed runs; 1,536,000 for the primary.
- Checkpoints: 20 model-only files, 3,696,710 bytes total; no optimizer state.
- Complete final output: 5,377,688 bytes.
- Primary training: 131.27 seconds total.
- All five primary model digests and all recomputed diagnostics matched checkpoint evaluation
  exactly.

The result artifacts are ignored reproducible outputs under
`results/cognitive_core_v2/final-v1.0/`; the protocol, amendments, summary, and report are tracked.

## Defect and amendment

The first final attempt exposed an overly strict slot-permutation assertion: identical predictions
and adaptation curves differed in cross-entropy by `8.38e-9` because float32 dot-product order
changed. The interrupted run is preserved under
`results/cognitive_core_v2/superseded-final-slot-permutation-v1.0/`. Amendment 0001 requires exact
discrete behavior and loss agreement within `1e-7`; a regression test was added, the fix committed,
and all final conditions rerun. No seed, model, threshold, or training resource changed.

## Comparison with v1 and remaining engineering

V2 removed intelligence from two hand-written v1 mechanisms:

- operation identity no longer directly indexes a state component;
- distractors no longer receive a manually defined zero-write mask.

The learned router causally organized two state values and learned strong distractor suppression.
However, the writer still receives the signed public relation, public operation IDs remain globally
fixed, phase semantics are explicit, the two-value state size is chosen by the researcher, and the
environment is a narrow balanced Boolean curriculum. V2 did not autonomously discover the sufficient
statistic, dynamic binding, or a discrete runtime code.

Strongest supporting evidence is the five-seed combination of near-perfect recovery, large
reset/freeze/writer-disabled drops, routing-specific degradation, donor-state causality, slot
reparameterization invariance, fixed bytes, and exact checkpoint reproduction.

Strongest counterevidence is that every fully generic V2C candidate failed at least one pilot seed;
episodic memory matched ordinary delayed prediction; long distractor sequences degraded learned
state; and the final relation was still engineered.

## V3 recommendation

Test one focused hypothesis: a small associative or bilinear writer with an explicit learned
evidence-preservation objective can discover the raw input/outcome relation and achieve one-feedback
revision without any relation, slot, or write labels. Keep eight bytes, `K=1`, 43k parameters,
307,200 observations, five final seeds, and the same controls. Pilot selection must require both
raw-field compliance and long-distractor stability; only after reproducible V2C should operation IDs
be permuted per world. Failure would localize the remaining frontier to autonomous sufficient-
statistic discovery rather than routing.

Tracked source commits used before this report:

- `8f22f619313904310f6e97845ba2db359684f466`: implementation and preregistration;
- `a1ec8895e72aa7a0261d966da8956beb4f5ecdb1`: slot-permutation amendment and final-run source.

The final evidence/report commit is recorded in repository history; a commit cannot contain its own
hash without changing that hash.
