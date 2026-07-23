# Cognitive Core v2 candidate ledger

All pilots used reserved seeds 2201 and 2202. Auxiliary relation/routing labels were used only in
fixed-data diagnostic overfits; every procedural result below used outcome loss only. Scores are
held-out minima across the two seeds unless a range is shown.

| Candidate | Params | State | Budget | Delay | Recovery | Retention | Route separation | Rule probe | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V2A raw MLP, fixed route | 41,400 | 8 B | 800 steps | 1.000 | 0.844 | 1.000 | fixed | 0.882 | Reject: one-feedback overwrite failed |
| V2B relation + learned router | 43,284 | 8 B | 800 steps | 1.000 | 0.997 | 1.000 | 0.594 | 0.803 | Retain: only candidate passing both seeds |
| V2C raw MLP router | 43,220 | 8 B | 800 steps | 0.971 | 0.864 | 0.985 | 0.140 | 0.656 | Reject: unstable routing and recovery |
| V2C raw dense GRU | 41,474 | 8 B | 800 steps | 0.980 | 0.980 | 0.836 | 0.257 | 0.623 | Reject: unstable retention/state organization |
| V2A raw MLP, fixed route | 41,400 | 8 B | 1,600 steps | 1.000 | 0.821 | 1.000 | fixed | 0.889 | Reject: twice the experience did not fix it |
| V2C raw MLP router | 43,220 | 8 B | 1,600 steps | 0.960 | 0.937 | 0.837 | 0.121 | 0.603 | Reject: twice the experience remained unstable |
| V2C raw dense GRU | 41,474 | 8 B | 1,600 steps | 0.917 | 0.945 | 0.913 | 0.007 | 0.582 | Reject: extra experience remained seed-unstable |
| V2A learned bilinear, fixed route | 46,004 | 8 B | 800 steps | 1.000 | 0.871--1.000 | 0.987--1.000 | fixed | 0.894--0.958 | Reject: one seed still failed recovery |
| V2C learned bilinear router | 46,004 | 8 B | 800 steps | 0.947--0.995 | 0.839--0.971 | 0.666--1.000 | 0.065--0.624 | 0.671--0.842 | Reject: one seed collapsed |

The bilinear families have more parameters than the first pilots because their learned input,
outcome, and interaction projections are explicit. They were not selected, so no parameter-matched
confirmatory claim is made for them.

## What the ledger establishes

- A raw writer can internally expose a linearly decodable relation, but probe decodability alone
  did not yield fast, stable rule revision.
- The learned router reliably learned near-zero distractor write strength when the relation was
  supplied; v1's hard no-write mask is therefore removable.
- Joint relation/routing optimization remained seed-sensitive under both equal and doubled
  experience budgets and under two state mechanisms plus one feature-interaction revision.
- The final protocol tests V2B only. It cannot establish V2C, V2D, or trained quantized V2E.
