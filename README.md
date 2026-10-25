# RadLLM Prompt-Eval Pipeline

Reference implementation of the evaluation framework in
[`../STUDY_PROTOCOL.md`](../STUDY_PROTOCOL.md): *How prompt/reasoning format
affects groundedness, calibration, and diagnostic accuracy in radiology LLMs
without expert radiologist annotation.*

It runs **end-to-end today** on the local IU X-Ray (OpenI) archive with a
deterministic **mock** VLM/judge (no API key, no GPU), and swaps to real
Claude vision + judge backends by editing one config field.

## What it does

```
reports.csv ─▶ rule-based CheXpert-style labeler ─▶ 14-class reference labels
projections.csv                                        + uncertainty (-1)
images/            └▶ difficulty stratifier (hedging + labeler-vs-MeSH disagreement)

               ┌─ 5 prompt conditions (direct / cot / ddx / evidence-first / uncertainty-first)
frozen VLM ────┤   over a locked, balanced test set → schema-validated JSON
               └─ evaluate: accuracy · groundedness · hallucination · calibration · LLM-judge
                  analyze:  McNemar · Wilcoxon · bootstrap ΔECE · GLMM(prompt×difficulty) · H3 dissociation
```

## Quickstart (offline, mock backend)

```bash
cd pipeline
pip install -r requirements.txt
make dataset    # build labels + locked balanced test/dev manifests
make eda        # confirm ambiguity signals exist in the reports
make infer      # 754 studies × 5 prompts  (mock VLM)
make eval       # all 5 evaluators + 2 mock judges
make analyze    # paired stats + H1–H4 hypothesis tests
make figures    # F2–F5 PNGs in results/figures/
make test       # 10 unit tests on the deterministic core
```

## Switching to real models

Edit `configs/config.yaml`:
```yaml
inference:
  backend: anthropic
  model_id: claude-opus-4-8      # a vision-capable Claude model
judge:
  backend: anthropic
  models: [claude-opus-4-8]      # use a DIFFERENT family than the subject VLM
```
then `export ANTHROPIC_API_KEY=...` and rerun `make infer eval analyze`.

> **Compliance (protocol §0.2):** only send **IU X-Ray** (this archive, fully
> de-identified) to commercial APIs. **Never** route MIMIC-CXR / CheXpert
> credentialed data to a public API endpoint — use local open models for those.

## Layout

| Path | Purpose |
|---|---|
| `configs/config.yaml` | one config = one experiment (seed, backend, sampling, stats) |
| `src/data.py` | load + join reports/projections/images at study level |
| `src/labeling.py` | rule-based 14-class CheXpert-style labeler (negation + uncertainty) |
| `src/difficulty.py` | difficulty strata via hedging + inter-source disagreement |
| `src/normalize.py` | free-text diagnosis → CheXpert label space |
| `src/schema.py` | output JSON schema + validation/repair |
| `src/prompts.py` | the 5 prompt templates (shared preamble) |
| `src/inference.py` | Mock + Anthropic VLM backends |
| `src/eval_*.py` | accuracy, groundedness, hallucination, calibration |
| `src/judge.py` | multi-judge harness (bias controls) |
| `src/stats.py` | McNemar, Wilcoxon, bootstrap, GLMM, dissociation |
| `scripts/0*.py` | ordered pipeline stages (see Makefile) |
| `tests/` | deterministic-core unit tests |

## Reproducibility

- Single `seed` in config drives sampling + bootstrap.
- `make dataset` writes `results/test_manifest.csv` (uid + stratum + report hash)
  — **freeze this before reporting**; it is the locked test set.
- Every prediction row carries `prompt_version`, `model_id`, `schema_valid`.

## Honest limitations (read before reporting anything)

1. **The mock backend is a test harness, not a model.** It peeks at labels with
   injected noise/hallucination to exercise the metrics. Numbers from it are
   meaningless as findings — they only prove the pipeline computes correctly.
2. **The rule-based labeler is a "bronze" reference.** Validate against / replace
   with **CheXbert** on the dev set before publication (protocol §3.2, §23).
3. **Lexicon claim extraction** stands in for **RadGraph** — swap it in for
   stronger groundedness (`eval_groundedness.extract_claims`).
4. All results are **conditional on the report as an imperfect reference**, not
   clinical ground truth (protocol §0.1, §3.3).
```
