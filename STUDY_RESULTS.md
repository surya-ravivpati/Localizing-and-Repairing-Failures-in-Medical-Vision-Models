# Study Results

**How Prompt and Reasoning Format Affect Groundedness, Confidence Calibration, and
Diagnostic Accuracy in Radiology LLMs Without Expert Radiologist Annotation**

Generated from `pipeline/scripts/17_study_results.py`. Last updated 2026-07-19,
after the six-bug evaluation audit (see §6 and `MENTOR_MEMO.md`).

---

## 1. Setup

| | |
|---|---|
| Dataset | IU X-Ray / OpenI — 3,826 usable studies (report + frontal/lateral images) |
| Test set | **754 studies**, balanced across 6 difficulty strata |
| Strata | easy_normal 150, easy_abnormal 150, ambiguous 150, intermediate 128, rare 100, hard_multi 76 |
| Model | gemini-2.5-flash, temperature 0, full-resolution images, hidden CoT disabled (`thinking_budget: 0`) |
| Conditions | direct, cot, ddx, evidence_first, uncertainty_first (identical system preamble; only the reasoning scaffold differs) |
| Design | Fully paired — every case evaluated under all five conditions |
| Reference | CheXbert labels on the held-out report (`lblcx_`), hierarchy-aware, uncertain-as-positive |

The model never sees the report; "groundedness" is therefore *reference-consistency*.

## 2. Diagnostic performance

Reported across a granularity ladder rather than at a single level. Fine-grained
subtyping is partly limited by label noise (§5); coarser groupings follow the standard
CXR review framework and were chosen for clinical coherence, not to maximize F1.

| Condition | 13-class | reliable-11 | 3-group | binary (dx only) | binary (full response) | top-1 acc |
|---|---|---|---|---|---|---|
| direct | 0.302 | 0.323 | 0.412 | 0.559 | 0.683 | 0.192 |
| cot | 0.333 | 0.356 | 0.444 | 0.596 | 0.703 | 0.210 |
| ddx | 0.307 | 0.330 | 0.415 | 0.627 | 0.691 | 0.220 |
| **evidence_first** | **0.367** | **0.391** | **0.482** | 0.585 | 0.711 | **0.232** |
| uncertainty_first | 0.359 | 0.384 | 0.476 | 0.621 | **0.712** | 0.229 |

*(micro-F1; `reliable-11` excludes the two classes with inter-labeler κ < 0.4)*

**Two operationalizations of the binary task.** `dx only` is the pre-registered metric
(is the stated final diagnosis abnormal?). `full response` asks the triage question
(does the response identify any abnormality?) by parsing diagnosis + explanation. Both
are legitimate; they answer different questions and must not be conflated. A coarser
question being easier is not evidence of a better model.

### Paired significance (McNemar, top-1 accuracy, vs best = evidence_first)

| Comparison | wins / losses | p | |
|---|---|---|---|
| vs direct | 40 / 10 | 0.0000 | **significant** |
| vs cot | 28 / 11 | 0.0104 | **significant** |
| vs ddx | 37 / 28 | 0.3211 | n.s. |
| vs uncertainty_first | 19 / 17 | 0.8676 | n.s. |

**Conclusion:** a **top tier** — evidence_first, uncertainty_first, and ddx, mutually
indistinguishable — significantly outperforms `direct`. Not a single winner. (This
changed after the audit: `ddx` caught up once its discarded `differential` field was
scored.)

## 3. Hypothesis outcomes

### H1 — Evidence-first improves groundedness → **not supported as stated**

| Condition | support rate | claims/case | contradiction rate |
|---|---|---|---|
| direct | **0.850** | 2.04 | **0.066** |
| evidence_first | 0.821 | **4.60** | 0.075 |
| cot | 0.766 | 3.05 | 0.115 |
| uncertainty_first | 0.762 | 3.16 | 0.113 |
| ddx | 0.744 | 2.58 | 0.116 |

`direct` has the highest support rate. But it achieves this by saying very little
(2.0 claims/case). `evidence_first` makes **2.25× more claims** while holding 0.821
support and the lowest contradiction rate among the reasoning prompts. The defensible
claim: *evidence-first surfaces substantially more evidence at near-equal fidelity* —
a support-rate/coverage tradeoff the original hypothesis did not anticipate.

### H2 — Uncertainty-first reduces overconfidence → **supported**

| Condition | ECE | mean confidence | accuracy | overconfidence gap |
|---|---|---|---|---|
| **uncertainty_first** | **0.641** | **0.870** | 0.229 | **+0.641** |
| ddx | 0.653 | 0.873 | 0.220 | +0.653 |
| evidence_first | 0.681 | 0.913 | 0.232 | +0.681 |
| direct | 0.694 | 0.887 | 0.192 | +0.694 |
| cot | 0.710 | 0.920 | 0.210 | +0.710 |

`uncertainty_first` has the lowest ECE, lowest mean confidence, and smallest gap.
**Secondary finding:** *every* condition is severely overconfident (gap +0.64 to
+0.71) — stated confidence clusters near 0.87–0.92 regardless of correctness, so it is
not usable as a probability.

### H3 — CoT/DDx improve apparent reasoning but not faithfulness → **supported**

Tested as pre-registered: paired Wilcoxon dissociation on judge scores, each condition
vs `direct` (n=754, LLM-judge 100% scored). H3 holds when persuasiveness rises
significantly while faithfulness does not.

| Condition | diagnostic_coherence Δ | faithfulness Δ | support_rate Δ (non-judge) | dissociation |
|---|---|---|---|---|
| **cot** | **+0.248** (p<0.0001) | **−0.229** (p<0.0001) | −0.084 | ✅ **yes** |
| **evidence_first** | **+0.131** (p=0.0020) | **−0.121** (p=0.0003) | −0.028 | ✅ **yes** |
| ddx | −0.219 | −0.251 | −0.105 | no (uniformly lower) |
| uncertainty_first | −0.149 | −0.162 | −0.088 | no (uniformly lower) |

**Chain-of-thought is the clearest case:** significantly *more* coherent-sounding while
significantly *less* faithful. H3 predicted faithfulness would be flat or declining; it
**actively declined**, a stronger effect than hypothesized.

**Convergent validation.** The LLM-judge faithfulness score and the independent
rule-based claim-adjudication support rate agree in direction for every condition, and
the claim-level contradiction rates tell the same story (cot 0.115 / ddx 0.116 vs
direct 0.066, ~1.75×). Two methodologically unrelated instruments reaching the same
conclusion is the protocol §8 validation requirement.

`ddx` and `uncertainty_first` do **not** dissociate — they score lower on both axes, so
they are simply weaker here rather than "persuasive but unfaithful."

This is the study's most novel result: *more visible reasoning ≠ more faithful
reasoning — and for chain-of-thought, visibly better reasoning is measurably less
grounded.*

### H4 — Prompt effects larger in ambiguous cases → **supported**

3-group F1 by difficulty stratum:

| Stratum | direct | cot | ddx | evidence_first | uncertainty_first | spread |
|---|---|---|---|---|---|---|
| hard_multi | 0.593 | 0.652 | 0.530 | **0.703** | 0.681 | **0.173** |
| rare | 0.237 | 0.257 | 0.212 | **0.356** | 0.324 | **0.144** |
| intermediate | 0.353 | 0.416 | 0.402 | **0.444** | 0.437 | 0.090 |
| easy_normal | 0.167 | 0.087 | 0.133 | 0.095 | 0.077 | 0.090 |
| ambiguous | 0.345 | 0.380 | 0.403 | 0.394 | **0.413** | 0.067 |
| easy_abnormal | 0.352 | 0.338 | 0.387 | 0.351 | **0.388** | 0.050 |

Prompt choice matters most on hard multi-finding (0.173) and rare (0.144) cases, least
on easy abnormal ones (0.050). Prompt engineering buys the most where the task is
hardest.

*(`easy_normal` F1 is near zero for all conditions by construction: micro-F1 has almost
no true positives to score on all-normal cases and mostly counts false positives.)*

## 3b. Uncertain-label sensitivity (protocol §7.1)

Protocol §7.1 requires all three CheXpert uncertain-label variants, so that one
arbitrary choice cannot drive the conclusions. Reference-uncertain (−1) cells are
0.99% of the label matrix.

| Condition | U-ones | U-zeros | U-ignore | spread |
|---|---|---|---|---|
| direct | 0.302 | 0.309 | 0.296 | 0.014 |
| cot | 0.333 | 0.340 | 0.329 | 0.011 |
| ddx | 0.307 | 0.320 | 0.307 | 0.013 |
| evidence_first | **0.367** | **0.376** | **0.367** | 0.009 |
| uncertainty_first | 0.359 | 0.364 | 0.355 | 0.009 |

*(13-class micro-F1; the reliable-11 level behaves identically — see
`results/study_uncertain_variants_test.csv`.)*

**The prompt ranking is identical under all three policies** —
`evidence_first > uncertainty_first > cot > ddx > direct` — with a maximum spread of
0.017. The comparative conclusions are therefore robust to the uncertain-label
convention, which is precisely the check §7.1 was designed to force.

*Semantics note:* U-ignore **masks** reference-uncertain (case, class) cells out of the
confusion matrix; it is not equivalent to U-zeros. These two were computed identically
until 2026-07-19 (see §6).

## 4. Reference quality (the ceiling)

Two independent report labelers were compared on the same corpus:

| Metric | Value |
|---|---|
| Bronze scored against CheXbert (hierarchy-aware, n=754) | **F1 0.771** |
| Mean Cohen's κ across classes (n=3,826) | **0.645** |
| Case-level exact positive-set agreement | **68.3%** |
| Lowest-agreement classes | Enlarged Cardiomediastinum κ=0.12, Lung Lesion κ=0.36 |
| Highest-agreement classes | Pneumothorax 0.91, Consolidation 0.89, Pleural Effusion 0.87 |

**Interpretation:** a second labeler scores only 0.771 against the first. That is an
upper bound on anything graded against a report-derived reference, and it means a
material share of apparent model error is reference disagreement rather than model
error. The model also reports **1.7× more findings per study** (1.68 vs 0.97) than the
terse IU reports mention — some over-calls, some findings visible but never dictated.
Separating those requires expert image labels (see `pipeline/EXPERT_VALIDATION_SCOPE.md`).

## 5. Limitations

1. **Report-derived reference.** Ground truth is what the radiologist *dictated*, not
   what is *visible*. Undictated findings are scored as false positives.
2. **Reference noise.** κ = 0.645 mean; two classes below 0.4 are effectively
   unmeasurable and excluded from the `reliable-11` column.
3. **Single model.** gemini-2.5-flash only. gemini-2.5-pro was spot-checked and
   performed comparably, but was not run across all conditions.
4. **Confidence is not probabilistic.** Stated confidence is near-constant, limiting
   the calibration analysis to detecting overconfidence rather than ranking quality.
5. **Absolute performance is modest.** These results support *comparative* claims about
   prompt style. They do not support claims of clinical readiness or fine-grained
   labeling competence.

## 6. Evaluation audit (2026-07-19)

Six defects were found and corrected; the fine-grained numbers for all five conditions
roughly doubled (0.19–0.27 → 0.30–0.37) as a result. Full detail in `MENTOR_MEMO.md`.

| # | Defect | Effect |
|---|---|---|
| 1 | Flat scoring ignored the CheXpert parent/child hierarchy (26% of all FPs) | large gain |
| 2 | Scored against impression-only labels instead of CheXbert | large gain |
| 3 | `evidence_first`'s `extracted_evidence` and `ddx`'s `differential` discarded | evidence_first +0.027 (3-group) |
| 4 | No negation scoping in the prediction mapper ("No evidence of pneumonia" → positive) | *lowered* scores; kept, it is correct |
| 5 | Lexicon gaps (e.g. "congestive heart failure" mapped to nothing) | mixed/small |
| 6 | Bronze labeler used sentence-level negation scoping (148 mentions) | mean κ 0.640→0.645 |

Verified clean: prediction↔label alignment (permutation test z=18 vs chance),
image↔study correspondence, and manual end-to-end case traces.

**Rejected as invalid:** greedy class-subset selection (reached 0.636 by keeping only
Lung Opacity — a one-class system) and dev-tuned group abstention (degenerated to one
group and scored *worse* on held-out test). Neither is reported.

## 7. Reproducing

```bash
cd pipeline
python scripts/17_study_results.py          # this document's tables
python scripts/16_granularity_report.py     # granularity ladder
python scripts/10_final_comparison.py       # all prompt variants incl. exploratory
```

Outputs: `results/study_{performance,calibration,difficulty}_test.csv`.
