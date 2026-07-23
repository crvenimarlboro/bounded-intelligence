# Cognitive Core v1 candidate ledger

All entries are development or reserved-pilot evidence; final seeds were not used for selection.

| Candidate | Hypothesis | Parameters | State | Pilot result | Decision |
|---|---|---:|---:|---|---|
| Generic GRU | An unconstrained gated writer can infer one hidden bit | 41,512 | 16 B | Seed 401 passed; seed 402 scored 0.805 post-evidence, 0.863 probe, 0.500 donor consistency | Reject: unstable causal state |
| Predictive-state GRU | A future-outcome auxiliary objective improves state organization | 41,804 | 16 B | Both pilot seeds reached 1.000 with 1.000 probes/swaps | Retain as valid alternative; not selected |
| Factorized state | A public relation plus learned slot writer solves credit assignment cheaply | 38,904 | 16 B | Both pilot seeds reached 1.000 with 1.000 probes/swaps | Select by lower parameters and wall time |
| One-float factorized | One continuous scalar may encode both rule bits | 38,576 | 4 B | Accuracy sometimes passed, but probe decoding was 0.636--0.817 and unstable | Reject |
| Two-float factorized | One learned scalar per primitive rule is the smallest stable state | 38,750 pilot / 38,952 final | 8 B | All compression and ladder pilot seeds passed | Select for confirmation |

Supervision pilots compared full rule-state loss, weak loss, annealing to zero, and outcome-only
training. All solved the minimal task, so final confirmation used outcome loss only. The predictive
auxiliary and supervised labels are not credited for the final result.

The candidate family limit was respected: GRU, predictive-state GRU, and factorized state. Revisions
responded only to measured instability, excessive state size, reversal anticipation, and seed overlap.
