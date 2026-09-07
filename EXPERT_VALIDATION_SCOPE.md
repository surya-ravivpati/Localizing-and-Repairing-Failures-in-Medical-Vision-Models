# Expert-Labeled Validation Set — Scope

**Purpose.** Every F1 we report (best: targeted prompt, hierarchy-aware, **0.462**)
is scored against *report-derived* labels (CheXbert on the radiology report). Two
problems cap that number no matter how good the model is:

1. **Report incompleteness.** The model reads the *image* and reports
   comprehensively (1.68 findings/study); the terse IU reports mention only what
   the radiologist chose to dictate (0.97 findings/study). Findings that are
   *visible but undictated* score as false positives.
2. **Reference noise.** Two independent labelers (bronze rule-based, CheXbert)
   agree on only 68.5% of studies (κ=0.64); their mutual F1 is 0.76 — an upper
   bound on anything scored against them.

An expert reading the **image** (not the report) removes both. This set answers
two questions the report-derived reference cannot:

- **G1 — True ceiling:** what is the model's F1 against a human image-reader?
- **G2 — Is the over-calling real?** Of the model's 819 disputed over-calls
  (model-present / reference-absent), how many are genuine false positives vs
  visible-but-unreported findings?

---

## Design

**Size: 200 studies**, two strata (see `scripts/13_build_expert_set.py`,
`results/expert_manifest.csv`):

| Stratum | n | Purpose | Sampling |
|---|---|---|---|
| **Random** | 120 | Unbiased F1 (G1) | Proportional across the 6 difficulty strata, seeded |
| **Disputed** | 80 | Resolve over-calling (G2) | Enriched for studies with model-present/ref-absent calls, spread across the 4 highest-FP classes (Atelectasis, Support Devices, Lung Opacity, Cardiomegaly) |

This yields **312 disputed over-call cells** to adjudicate directly, plus a clean
random subset for an unbiased estimate. Order is shuffled and reference/model
labels are hidden from the labeler (columns prefixed `_` are for analysis only).

**Why 200?** Powered to (a) estimate F1 to ±0.05 at 95% CI on the random stratum,
and (b) put ≥40 adjudicated cases behind each of the 4 problem classes. It is also
a realistic single-session workload: ~3–5 hours for one reader at ~1–1.5 min/study.

## Labeling protocol

- **Unit:** each study (frontal ± lateral). Read the **image only**; the clinical
  indication is provided, the report is **not**.
- **Task:** for each of the 13 CheXpert findings, mark **present / absent /
  uncertain**. "No Finding" is implied when all 13 are absent.
- **Definitions:** use the standard CheXpert definitions; apply the same per-class
  criteria the model was given (see `_OVERCALL_GUIDANCE` in `src/prompts.py`) so
  human and model are held to one standard — e.g. Cardiomegaly = CTR > 0.5;
  Atelectasis = linear/band opacity with volume loss; Support Devices = a visibly
  present tube/line/device.
- **Hierarchy:** mark the specific finding; parent propagation (Lung Opacity from
  its children) is applied in scoring, not by the labeler.

**Raters.** Minimum one board-certified radiologist or senior radiology resident.
**Recommended:** two independent readers on ≥50 overlapping studies to report
inter-rater κ (establishes the human ceiling); one reader adjudicates the rest.

## Analysis plan (once labels return)

1. **G1 — Ceiling:** recompute targeted-prompt P/R/F1 vs expert labels on the
   random stratum, hierarchy-aware. Compare to the 0.462 report-derived number.
2. **G2 — Over-call resolution:** for the 312 disputed cells, tabulate
   expert-confirmed-present (model was right; report was incomplete) vs
   expert-absent (genuine model FP). The confirmed-present fraction is the
   estimated report-incompleteness rate — it directly re-prices precision.
3. **Reference re-pricing:** re-estimate the CheXbert reference's own precision/
   recall vs expert, to state how much of the 0.46→0.76 gap is reference noise.
4. **Inter-rater κ** on the overlap subset → the human ceiling for context.

## Deliverables in this repo

- `scripts/13_build_expert_set.py` — regenerates the sample (seeded, reproducible).
- `results/expert_manifest.csv` — the 200-study blinded manifest with blank
  `label_*` columns for the expert to fill.
- `scripts/14_stage_expert_images.py` — copies the 200 images into a self-contained
  `results/expert_label/` folder with a browser labeling tool (next step).
- `scripts/15_score_expert.py` — scores model vs returned expert labels (G1 + G2).

## Cost / timeline

- **Labeling:** ~3–5 hours, one reader (a second reader on 50 overlaps adds ~1 hr).
- **No API cost.** All compute is local scoring.
- **Turnaround:** dominated by radiologist availability, not engineering.
