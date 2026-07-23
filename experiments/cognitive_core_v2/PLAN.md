# Cognitive Core v2 execution plan

Status: development and reserved pilot. Cognitive Core v0 and v1 are immutable evidence.

## Claim and staged removal

V2 tests whether an eight-byte frozen-weight learner can discover from raw public feedback both
the COPY/FLIP sufficient statistic and its state organization. V2A removes the engineered relation
while retaining fixed routing. V2B retains the relation while learning routing and write suppression.
V2C removes both. V2D is attempted only after V2C passes; trained quantization follows only after a
generic float candidate passes.

The raw writer contract is frozen in `writer_input_contract.json`. A final V2C writer receives only
current public input, generic one-hot encoding of the public operation symbol, public phase, observed
outcome, and previous fixed state. It receives no relation, correct slot, write mask, hidden rule,
seed, future event, or history.

## V1 scaffold-dependence audit

The post-hoc checkpoint audit uses all three v1 core seeds without changing their tensors. Native
delay/composition/recovery/retention were 1.000. Removing the relation reduced these to
0.533/0.501/0.499/0.509 and donor consistency to 0.500. Random relation produced
0.561/0.503/0.493/0.735. Uniform routing reduced delay to 0.594 and donor consistency to 0.813.
Enabling distractor writes reduced delayed accuracy to 0.676 with 0.492 mean per-step drift.
Thus both relation and non-write/routing scaffolds carry substantial causal capability.

## Development and pilot protocol

- Model seeds: development 2101--2105; reserved pilot 2201--2202.
- Procedural worlds: development 100000--199999; pilot training 200000--206399,
  validation 208000+, evaluation 250000+.
- Final seeds/ranges remain unopened until a hashed confirmatory manifest is committed.
- Initial budget: 800 Adam steps, eight groups, 12 steps, 307,200 observations, six CPU threads,
  K=1, two float32 values (eight bytes), full 12-step BPTT.
- Mixed training cycles through equal-length context, composition, delay, and reversal worlds. This
  is necessary because a generic writer must observe distractors to learn non-write behavior; the
  observation count does not increase.
- Candidate order: protocol compliance, relation discovery, held-out adaptation, state-ablation
  drop, donor causality, routing causality, then lower bytes/parameters/observations/runtime.

## Candidates and stopping

1. Raw fixed-route writer (V2A): MLP relation discovery with the v1 mask retained.
2. Relation-supplied soft-slot router (V2B): learned route and write strength, no mask.
3. Raw soft-slot router (V2C): learned relation, route, and write strength.
4. Raw dense GRU writer (second serious V2C family): no slot decomposition.

Diagnostic auxiliary relation/routing losses are development-only. Primary pilots and final evidence
use outcome loss only. At most one evidence-driven revision per measured failure is attempted before
selecting the highest passing stage. Rollback is deletion of ignored V2 outputs plus Git revert;
prior evidence is never rewritten.

## Pilot decision 1: explicit learning-efficiency curve

At the equal v1 budget, V2B passed both reserved seeds. V2A learned the relation perfectly but
recovery was 0.844--0.847. The raw router reached 0.971--1.000 delay but only 0.864--0.869 recovery,
and one seed's routing separation collapsed to 0.140. The raw GRU recovered at 0.980--0.985 but
retention varied from 0.836 to 0.963. These failures are preserved in
`results/cognitive_core_v2/pilot-v1.0`.

The first evidence-driven revision is a 1,600-step learning-efficiency point for V2A and both V2C
families. Thresholds, model widths, state bytes, optimizer, batch size, and pilot seeds remain
unchanged; only observations and steps double. This tests slow joint optimization before introducing
staged or auxiliary training. A passing extended run must be reported as spending twice v1's
training observations, never as an equal-budget result.

## Pilot decision 2: learned multiplicative raw features

The 1,600-step V2A/raw-router results did not materially improve the bottleneck, so extra experience
was rejected as the preferred remedy. The final permitted within-family revision adds generic
learned input and outcome projections plus a learned elementwise interaction before the writer
MLP. The external contract remains the same eight raw fields; no XOR, equality, relation label,
slot target, or write target is computed outside the network. Both fixed-route V2A and learned-route
V2C variants return to the equal 800-step budget. This revision tests whether the ordinary MLP's
feature interaction was the source of slow or unstable sufficient-statistic discovery.

## Confirmatory selection

Neither doubled training nor the learned bilinear revision made V2A or V2C pass both pilot seeds.
V2B was the only protocol-compliant candidate passing every reserved pilot gate. The confirmatory
claim is therefore intentionally limited to learned routing with the relation scaffold retained.
Five fresh final model seeds (2701--2705) and generation ranges beginning at 400000 are frozen in
`manifest.json` and `configs/final.json`. Final controls are parameter matched within two percent:
v1 scaffolded reference (43,808 parameters), no memory (43,588), and eight-byte episodic (43,322)
against the 43,284-parameter V2B core.
