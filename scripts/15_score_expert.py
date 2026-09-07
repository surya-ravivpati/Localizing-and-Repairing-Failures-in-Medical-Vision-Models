"""Score the model against returned expert labels (answers G1 + G2).

Input: results/expert_label/labels.csv (uid + label_<class> in present/uncertain/
absent), from the labeling tool. Joins to the model's targeted predictions and the
CheXbert reference via the manifest strata.

G1 — true ceiling: targeted-prompt P/R/F1 vs EXPERT on the random stratum,
     hierarchy-aware, next to the report-derived (CheXbert) F1 on the same studies.
G2 — over-call resolution: for each disputed cell (model-present / CheXbert-absent),
     did the expert confirm it present? confirmed = report was incomplete, not a
     model error. Reports the report-incompleteness rate that re-prices precision.
"""
import argparse
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import (_pred_set, _truth_positive_set, propagate_hierarchy,
                               RELIABLE_CLASSES)
from src.labeling import CHEXPERT_CLASSES
from sklearn.metrics import precision_recall_fscore_support

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]


def expert_positive(row, u_policy="ignore"):
    pos = set()
    for c in ALL:
        v = str(row.get(f"label_{c}", "")).lower()
        if v == "present" or (v == "uncertain" and u_policy == "ones"):
            pos.add(c)
    return pos


def prf(truth, pred, uids, classes):
    yt = np.array([[1 if c in propagate_hierarchy(truth[u]) else 0 for c in classes]
                   for u in uids])
    yp = np.array([[1 if c in propagate_hierarchy(pred[u]) else 0 for c in classes]
                   for u in uids])
    return precision_recall_fscore_support(yt, yp, average="micro", zero_division=0)[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="expert_label/labels.csv")
    args = ap.parse_args()
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    lab = pd.read_csv(os.path.join(out, args.labels)).set_index("uid")
    man = pd.read_csv(os.path.join(out, "expert_manifest.csv")).set_index("uid")
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    tg = pd.read_csv(os.path.join(out, "predictions_test_targeted.csv")).set_index("uid")

    uids = [u for u in lab.index if u in tg.index and u in man.index]
    expert = {u: expert_positive(lab.loc[u]) for u in uids}
    model = {u: _pred_set(tg.loc[u]) for u in uids}
    chex = {u: _truth_positive_set(s.loc[u], "ignore", "lblcx_") for u in uids}
    rand = [u for u in uids if man.loc[u, "_stratum"] == "random"]

    print(f"scored {len(uids)} expert-labeled studies ({len(rand)} random stratum)\n")
    print("=== G1: TRUE CEILING (random stratum, hierarchy-aware) ===")
    for name, cls in [("full-13", ALL), ("reliable", RELIABLE_CLASSES)]:
        pe, re, fe = prf(expert, model, rand, cls)
        pc, rc, fc = prf(chex, model, rand, cls)
        print(f"  {name:<9} vs EXPERT: F1={fe:.3f} (P{pe:.2f}/R{re:.2f})   "
              f"vs CheXbert-report: F1={fc:.3f} (P{pc:.2f}/R{rc:.2f})   "
              f"Δ={fe-fc:+.3f}")

    print("\n=== G2: OVER-CALL RESOLUTION (all disputed model+/CheXbert- cells) ===")
    conf = wrong = 0
    by_class = {}
    for u in uids:
        disputed = propagate_hierarchy(model[u]) - propagate_hierarchy(chex[u])
        for c in disputed:
            ok = c in expert[u]
            conf += ok
            wrong += (not ok)
            d = by_class.setdefault(c, [0, 0])
            d[0 if ok else 1] += 1
    tot = conf + wrong
    if tot:
        print(f"  {tot} disputed over-calls | expert-CONFIRMED present: {conf} "
              f"({100*conf/tot:.0f}%) -> report was incomplete, NOT a model error")
        print(f"                          | expert-ABSENT (true FP): {wrong} "
              f"({100*wrong/tot:.0f}%)")
        print("  by class (confirmed / true-FP):")
        for c, (ok, bad) in sorted(by_class.items(), key=lambda kv: -sum(kv[1])):
            print(f"    {c:<22} {ok:>3} / {bad:<3}  ({100*ok/(ok+bad):.0f}% confirmed)")
        print(f"\n  => report-incompleteness rate {100*conf/tot:.0f}%: this fraction of "
              f"'false positives' are real findings. Re-pricing precision upward by "
              f"this factor is the honest ceiling estimate.")


if __name__ == "__main__":
    main()
