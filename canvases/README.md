# Canvases

Self-contained HTML snapshots of the write-ups for this study, exported from
Claude artifacts on 2026-10-25. Open any file directly in a browser.

| File | What it is | Status |
|---|---|---|
| `01_consolidated_paper.html` | **What moves a radiology VLM — prompt, model, and image.** The single consolidated working paper: original prompt/model study, two-stage perception–reasoning split, Experiments 1–5 (visual input), Exp 6 (probes, quantisation dissociation, inside-the-LLM probes) and the three-arm alignment intervention (§9.9). | **Current — use this one** |
| `02_where_the_f1_goes.html` | Diagnostic note: why 13-class F1 is low (per-class operating points, not discrimination or reference noise) and the trained-readout ladder. | Folded into 01 (§9) |
| `03_perception_reasoning_split.html` | Two-stage perception/reasoning experiment, first version (Gemini only). | Superseded by 01 (§5) |
| `04_complete_results_appendix.html` | Full metric tables for the prompt × model study. | Superseded by 01 |
| `05_prompt_appendix.html` | Every prompt verbatim. | Folded into 01 (appendix A) |
| `06_working_paper_v2.html` | Earlier working paper (prompt format & model choice). | Superseded by 01 |
| `07_working_paper_v1.html` | First working paper draft. | Superseded by 06, then 01 |

## Known issue — read before citing §9.9 of the consolidated paper

The connector and connector+LoRA arms reported in §9.9 were trained with
`mlx_vlm`'s stock SFT loss, which for Gemma-3 builds an all-ones attention mask:
every training position can see the token it is predicting. This was verified
directly (editing one late token moved earlier-position logits by up to 37 on the
training path, exactly 0 on the causal path) and is the root cause of the earlier
fine-tuning failures. The §9.9 arm results — including "degradation is monotone
with intervention depth" — should be treated as unverified until re-run with the
causal loss in `src/readout_ft.py`.

Results after the fix (per-finding readout fine-tune, scripts 56–59) are not yet
in any canvas: with a causal loss and small learning rate, LoRA preserves the base
model's ranking (no collapse) but did not significantly beat it on a held-out dev
confirmation set (pooled AUROC 0.737 → 0.749, 95% CI [−0.024, +0.047]).
