"""Statistical analysis + hypothesis tests (protocol §10).

Paired McNemar (accuracy, H1/H4), Wilcoxon (support-rate H1, ECE-per-case H2),
bootstrap ΔECE, GLMM interaction (H4), dissociation test (H3).
"""
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.stats import (mcnemar_all_pairs, wilcoxon_all_pairs, bootstrap_diff,
                       glmm_interaction, dissociation_test)
from src.eval_calibration import calibration_metrics


def per_case_ece(preds, group_cols=("prompt_condition",)):
    """Approximate per-case calibration contribution for Wilcoxon: |conf-correct|."""
    d = preds.copy()
    d["cal_err"] = (d["confidence"] / 100.0 - d["correct"].astype(int)).abs()
    return d


def main(split="test"):
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    preds = pd.read_csv(os.path.join(out, f"predictions_scored_{split}.csv"))
    ground = pd.read_csv(os.path.join(out, f"groundedness_{split}.csv"))
    judge = pd.read_csv(os.path.join(out, f"judge_scores_{split}.csv"))

    print("===== H1/H4: Accuracy — McNemar (paired, FDR) =====")
    mc = mcnemar_all_pairs(preds, "correct")
    print(mc.round(4).to_string(index=False))

    print("\n===== H1: Groundedness support-rate — Wilcoxon =====")
    wg = wilcoxon_all_pairs(ground, "support_rate")
    print(wg.round(4).to_string(index=False))

    print("\n===== H2: Per-case calibration error — Wilcoxon =====")
    dce = per_case_ece(preds)
    wce = wilcoxon_all_pairs(dce, "cal_err")
    print(wce.round(4).to_string(index=False))

    print("\n===== H2: Bootstrap ΔECE vs 'direct' =====")
    base = preds[preds.prompt_condition == "direct"]
    base_ece = calibration_metrics(base.confidence/100, base.correct.astype(int))["ece"]
    for cond in preds.prompt_condition.unique():
        if cond == "direct":
            continue
        g = preds[preds.prompt_condition == cond]
        # paired by uid on |conf-correct| as an ECE-surrogate bootstrap
        m = base.merge(g, on="uid", suffixes=("_d", "_c"))
        xb = (m.confidence_c/100 - m.correct_c.astype(int)).abs().values
        yb = (m.confidence_d/100 - m.correct_d.astype(int)).abs().values
        bs = bootstrap_diff(xb, yb, n_boot=cfg["stats"]["n_bootstrap"], seed=cfg["seed"])
        print(f"  {cond:18s} Δ|cal_err| vs direct: {bs['point']:+.3f} "
              f"[{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}]")

    print("\n===== H3: Persuasiveness vs faithfulness dissociation =====")
    diss = dissociation_test(judge, ground)
    print(diss.round(3).to_string(index=False))

    print("\n===== H4: GLMM  correct ~ prompt*difficulty + (1|uid) =====")
    try:
        summ = glmm_interaction(preds, "correct")
        # print only the coefficient table tail to keep it readable
        print("\n".join(summ.splitlines()[:40]))
    except Exception as e:
        print(f"GLMM failed: {e}")

    print(f"\n(Full tables persisted under {out})")
    mc.to_csv(os.path.join(out, f"stats_mcnemar_accuracy_{split}.csv"), index=False)
    wg.to_csv(os.path.join(out, f"stats_wilcoxon_groundedness_{split}.csv"), index=False)
    diss.to_csv(os.path.join(out, f"stats_dissociation_{split}.csv"), index=False)


if __name__ == "__main__":
    main()
