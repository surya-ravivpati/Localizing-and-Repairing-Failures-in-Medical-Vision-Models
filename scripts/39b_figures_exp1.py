"""Experiment 1 figures (Step 9). Four only, as pre-registered."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from _bootstrap import load_cfg

out = load_cfg()["paths"]["out_dir"]
figdir = os.path.join(out, "figures"); os.makedirs(figdir, exist_ok=True)
r = pd.read_csv(os.path.join(out, "exp1_results_exp1f_gem.csv")).set_index("arm")
order = ["baseline_f", "highres_f", "multicrop_f", "oracle_f"]
labels = ["A\nbaseline", "B\nhigher-res", "C\nmulti-crop", "D\noracle"]
r = r.loc[order]
C = ["#5b6570", "#7c5cff", "#0f766e", "#b45309"]

def bar(ax, vals, title, ylab, fmt="{:.3f}", base_line=True):
    b = ax.bar(labels, vals, color=C, width=.62)
    if base_line:
        ax.axhline(vals[0], color="#8b95a1", ls="--", lw=1, zorder=0)
    for x, v in zip(b, vals):
        ax.text(x.get_x()+x.get_width()/2, v, fmt.format(v), ha="center",
                va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11, weight="bold"); ax.set_ylabel(ylab, fontsize=9)
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=9)

# F1 — Stage-A finding F1 across the four visual conditions
fig, ax = plt.subplots(figsize=(6.4,4))
bar(ax, r.stageA_f1_13.tolist(), "Stage-A perception: 13-class F1", "micro-F1")
ax.set_ylim(0, .5); ax.text(.5,.02,"all differences vs baseline n.s. (paired bootstrap)",
    transform=ax.transAxes, ha="center", fontsize=8.5, color="#5b6570")
fig.tight_layout(); fig.savefig(f"{figdir}/exp1_F1_stageA_f1.png", dpi=130); plt.close(fig)

# F2 — perception-originated error %
fig, ax = plt.subplots(figsize=(6.4,4))
bar(ax, (r.pct_errors_perception_caused*100).tolist(),
    "Share of Stage-B errors already wrong in perception", "% of triage errors", "{:.1f}%")
ax.set_ylim(0,100); fig.tight_layout()
fig.savefig(f"{figdir}/exp1_F2_perception_share.png", dpi=130); plt.close(fig)

# F3 — hallucination / false-positive rate  (the one thing that DOES move)
fig, ax = plt.subplots(figsize=(6.4,4))
bar(ax, r.stageA_fp_per_study.tolist(),
    "Hallucinated findings per study (false positives)", "FP / study", "{:.2f}")
ax.set_ylim(0, 2.4)
ax.text(.5,.02,"B and C significantly exceed baseline (Wilcoxon, BH-FDR)",
    transform=ax.transAxes, ha="center", fontsize=8.5, color="#b45309")
fig.tight_layout(); fig.savefig(f"{figdir}/exp1_F3_false_positives.png", dpi=130); plt.close(fig)

# F4 — Stage-B diagnostic performance
fig, ax = plt.subplots(figsize=(6.4,4))
bar(ax, r.stageB_bin_acc.tolist(), "Stage-B diagnosis: binary-triage accuracy", "accuracy")
ax.set_ylim(0,.85); fig.tight_layout()
fig.savefig(f"{figdir}/exp1_F4_stageB_accuracy.png", dpi=130); plt.close(fig)
print("wrote 4 figures to", figdir)
for f in sorted(os.listdir(figdir)):
    if f.startswith("exp1_"): print("  ", f)
