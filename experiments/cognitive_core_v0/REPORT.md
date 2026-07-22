# Cognitive Core v0 final report

## Conclusion

**Disproven by current evidence:** the preregistered Cognitive Core v0 hypothesis is unsupported in Rule Worlds v0. The learned workspace did not improve structured held-out prediction over no memory or equal-byte episodic memory, did not improve with informative experience, and was not materially weakened by disabling its proposed mechanisms. This is a result about this architecture, curriculum, and resource setting—not a claim about bounded latent workspaces in general.

## Executed design

Rule Worlds generates disjoint structured, random-control, surface-relabelled, and marked rule-change episodes. A model receives only seven integer-valued fields from the current public observation. Hidden values, operator biases, generation seeds, future observations, targets, and researcher annotations are excluded from model inputs. Direct, composed, counterfactual, delayed, and post-change probes are generated procedurally.

Cognitive Core v0 uses a 160-wide observation encoder and a fixed 1,024-float32 workspace. It projects the observation and workspace read, applies one shared gated residual thought block three times, and predicts one of four outcomes. After the outcome arrives, it encodes the detached error `one_hot(outcome) - softmax(logits)` plus the outcome, observation, and final hidden state. Separate sigmoid-gate and tanh-candidate projections perform `state + gate * (candidate - state)`. Held-out weights are frozen; only this workspace changes.

The no-memory baseline applies matched active MLP capacity and the same three-cycle shared thought block to the current observation, returning no state. The episodic baseline stores 511 eight-byte records (seven public fields plus outcome) and two int32 ring-buffer counters: exactly 4,096 bytes. It retrieves the 16 most recent records through a learned query and evicts by deterministic overwrite. All three systems use the same worlds, outcomes, optimizer settings, observation count, seeds, and evaluation procedure.

| system | trainable parameters | persistent experience bytes | mean training seconds |
|---|---:|---:|---:|
| no memory | 1,203,197 | 0 | 15.21 ± 0.30 |
| equal-byte episodic | 1,203,065 | 4,096 | 24.11 ± 0.15 |
| Cognitive Core v0 | 1,203,056 | 4,096 | 16.32 ± 0.08 |

Maximum parameter mismatch was 0.0117%, below the frozen 2% tolerance. State-size tests cover sequence lengths 10, 100, 1,000, and 10,000. Weights/checkpoints are excluded from experience-memory bytes and reported separately.

## Protocol and runs

The final amended protocol used seeds 101, 202, and 303. Each neural checkpoint trained once over 512 distinct 64-step worlds: 32,768 observations and 64 optimizer steps. Held-out evaluation used 48 worlds in each category, with 80 steps for rule-change worlds. The final epoch was the only checkpoint selection rule. All four ablations were evaluated for every core seed, and all nine primary checkpoints were independently reloaded and reevaluated.

The repeated-world pilot showed rising training accuracy with chance validation, so amendment 0001 exchanged four repetitions of 128 worlds for one pass over 512 worlds without changing observations or optimizer steps. Review after the first final execution found that shuffled query blocks were accidentally discarded after their first element. Amendment 0002 preserves that superseded negative run, fixes the query schedule, adds a regression test, and reruns every final condition. Only corrected protocol 1.2 below is confirmatory.

## Corrected final results

Values are mean ± sample standard deviation across three seeds. Four-class chance is 0.25.

| condition | structured | surface relabel | random | rule change | retention | eval seconds |
|---|---:|---:|---:|---:|---:|---:|
| no memory | 0.251 ± 0.002 | 0.248 ± 0.009 | 0.252 ± 0.004 | 0.249 ± 0.002 | 0.246 ± 0.006 | 1.61 ± 0.03 |
| equal-byte episodic | 0.247 ± 0.008 | 0.247 ± 0.000 | 0.254 ± 0.010 | 0.239 ± 0.005 | 0.234 ± 0.007 | 2.85 ± 0.04 |
| Cognitive Core full | 0.248 ± 0.002 | 0.254 ± 0.005 | 0.253 ± 0.008 | 0.245 ± 0.007 | 0.243 ± 0.012 | 1.90 ± 0.04 |

Core structured advantage was -0.0029 over no memory and +0.0016 over episodic, versus frozen requirements of +0.08 and +0.03. Surface advantage over no memory was +0.0054, far below +0.06, and absolute surface accuracy was 0.254 rather than at least 0.45. Random-control advantage was +0.0011, correctly below the +0.04 specificity ceiling, but this is uninformative because the structured advantage was absent. None of the three seeds beat no memory on structured accuracy.

The full core's structured adaptation points were 0.278 at observation 0, 0.243 at 1, 0.285 at 2, 0.229 at 4, 0.229 at 8, 0.285 at 16, 0.229 at 32, and 0.174 at 63. The preregistered 0-to-32 change was **-0.049**, not the required +0.10. Hidden-rule accuracy was 0.249 ± 0.009; composed/counterfactual accuracy was 0.254 ± 0.012.

## Ablations

All ablations use the same trained full-core checkpoint for a seed and change only the evaluated mechanism.

| core condition | structured accuracy | drop from full | interpretation |
|---|---:|---:|---|
| full | 0.2483 | — | proposed mechanism |
| workspace disabled/reset each step | 0.2471 | +0.0012 | negligible |
| workspace frozen | 0.2497 | -0.0014 | slightly better without updates |
| no explicit prediction error | 0.2495 | -0.0012 | slightly better without error signal |
| recurrence K=1 | 0.2484 | -0.0001 | extra cycles had no useful effect |

No ablation approached the required 0.04 degradation. The strongest mechanistic evidence is therefore against the proposed explanation: updates, explicit error, and extra recurrence were not carrying measurable held-out capability.

## Resources and artifacts

- Corrected final wall time: 227.32 seconds; peak RAM: 475,090,944 bytes (about 453 MiB).
- Training observations: 294,912 across nine primary checkpoints.
- Evaluation observations: 391,680 including all ablations and independent reload checks.
- Checkpoints: 43,444,830 bytes total; each is about 4.83 MB and contains model config/state, experiment ID, source Git revision, seed, parameters, 4 KiB budget, training step, validation score, and configuration hash.
- Corrected checkpoints: `results/cognitive_core_v0/final-v1.2/checkpoints/`.
- Machine-readable results, raw JSONL, CSV learning measurements, and SVG: `results/cognitive_core_v0/final-v1.2/`.
- Maximum checkpoint reproduction error: 0.0. Evaluation verified frozen weights and observed exactly one state size per condition.
- Capability-density measurement for the full core is 0.2483 / 4,096 = 0.0000606 structured-accuracy units per persistent byte. It has no evidential value as an advantage because capability did not exceed the controls.

Generated results are ignored research telemetry and are never read by a model. The tracked manifest, amendments, config, source, tests, and this concise report reproduce their semantics. No Qwen model, external API, large dataset, GPU trainer, database, RAG system, or growing transcript participates.

## Defects and protocol record

Implementation smoke testing fixed relative output-path provenance, one-seed standard-deviation handling, retention aggregation, and accounting of reload-evaluation observations before confirmatory execution. Amendment 0001 records the resource-neutral training-diversity change after the pilot. Amendment 0002 records the query-schedule defect discovered after the first final run and the complete corrected rerun. Thresholds, final seeds, parameters, observations, memory, and checkpoint selection were never changed after results to improve the conclusion.

## Interpretation and v1 recommendation

The most plausible current reading is that v0 learned neither a reliable within-world solver nor a workspace update algorithm. Final training and validation accuracy remained close to chance, so the experiment cannot distinguish a fundamentally poor workspace idea from an architecture/curriculum that never learned to use any history. The strongest evidence against the negative conclusion is the tiny +0.005 surface advantage over no memory and +0.002 structured advantage over episodic, but both are smaller than seed variation and neither appears as increasing adaptation or ablation sensitivity.

Cognitive Core v1 should not merely scale width, data, or workspace. First create a two-stage falsifiable curriculum: (1) a binary or two-operator world where an analytic bounded-state oracle proves attainable accuracy and a trained core must overfit a single world then generalize across many worlds; (2) add one factor at a time—surface relabeling, composition, delay, then reversal. Add linear probes that decode the exact latent rule from workspace state, compare explicit supervised state targets against pure outcome loss, and record gate saturation/gradient norms. Advance to the full Rule Worlds battery only after K=1 and frozen-state controls separate cleanly on held-out adaptation.

## Reproduction commands

```bash
uv sync --frozen
uv run bilab manifest validate experiments/cognitive_core_v0/manifest.json
uv run bilab core validate-worlds --config experiments/cognitive_core_v0/configs/final.json
uv run bilab core smoke --config experiments/cognitive_core_v0/configs/pilot.json --output results/cognitive_core_v0/smoke
uv run bilab core run --config experiments/cognitive_core_v0/configs/final.json --output results/cognitive_core_v0/final-v1.2
uv run bilab core evaluate --checkpoint results/cognitive_core_v0/final-v1.2/checkpoints/cognitive_core-seed101.pt --output results/cognitive_core_v0/reproduction/evaluate-seed101.json
uv run bilab core ablate --checkpoint results/cognitive_core_v0/final-v1.2/checkpoints/cognitive_core-seed101.pt --output results/cognitive_core_v0/reproduction/ablate-seed101.json
uv run bilab core report --results results/cognitive_core_v0/final-v1.2/results.json --output results/cognitive_core_v0/reproduction/report.md
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
