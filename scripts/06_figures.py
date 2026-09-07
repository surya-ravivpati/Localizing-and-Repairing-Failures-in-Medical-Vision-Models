"""Publication figures (protocol §17). Writes PNGs to results/figures/.

F2 reliability diagrams, F3 error-profile bars, F4 prompt x difficulty accuracy,
F5 faithfulness-vs-persuasiveness scatter.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_calibration import reliability_bins


def main(split="test"):
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)
    preds = pd.read_csv(os.path.join(out, f"predictions_scored_{split}.csv"))
    hall = pd.read_csv(os.path.join(out, f"hallucination_{split}.csv"))
    judge = pd.read_csv(os.path.join(out, f"judge_scores_{split}.csv"))
    ground = pd.read_csv(os.path.join(out, f"groundedness_{split}.csv"))
    conds = sorted(preds.prompt_condition.unique())

    # F2 reliability diagrams
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 3.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), conds):
        rb = reliability_bins(preds, c, 10)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(rb.mean_conf, rb.accuracy, "o-", color="#2b6cb0")
        ax.set_title(c, fontsize=9); ax.set_xlabel("confidence"); ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("empirical accuracy")
    fig.suptitle("F2 · Reliability diagrams by prompt")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "F2_reliability.png"), dpi=130)
    plt.close(fig)

    # F3 error-profile stacked bars
    prof = hall.groupby("prompt_condition")[["n_contradicted", "n_fabricated",
                                             "n_omitted"]].mean()
    ax = prof.plot(kind="bar", stacked=True, figsize=(7, 4),
                   color=["#c53030", "#dd6b20", "#805ad5"])
    ax.set_ylabel("mean count / case"); ax.set_title("F3 · Error profile by prompt")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "F3_error_profile.png"), dpi=130); plt.close()

    # F4 accuracy by prompt x difficulty
    piv = preds.pivot_table(index="difficulty", columns="prompt_condition",
                            values="correct", aggfunc="mean")
    ax = piv.plot(kind="bar", figsize=(9, 4))
    ax.set_ylabel("accuracy"); ax.set_title("F4 · Accuracy by prompt × difficulty (H4)")
    ax.legend(fontsize=7, ncol=5); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "F4_prompt_difficulty.png"), dpi=130); plt.close()

    # F5 faithfulness vs persuasiveness (judge coherence)
    jm = judge.groupby("prompt_condition")[["score_faithfulness",
                                            "score_diagnostic_coherence"]].mean()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(jm.score_diagnostic_coherence, jm.score_faithfulness, s=80,
               color="#2b6cb0")
    for c, r in jm.iterrows():
        ax.annotate(c, (r.score_diagnostic_coherence, r.score_faithfulness),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("judge coherence (persuasiveness)")
    ax.set_ylabel("judge faithfulness")
    ax.set_title("F5 · Persuasiveness vs faithfulness (H3)")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "F5_dissociation.png"), dpi=130)
    plt.close()

    print(f"Wrote 4 figures to {figdir}")


if __name__ == "__main__":
    main()
