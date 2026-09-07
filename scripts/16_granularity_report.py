"""Multi-granularity evaluation report + dev-tuned per-group abstention policy.

Part 1 (always): the granularity LADDER — 13 fine classes / 5 clinical groups /
3 core pathology groups / binary acute. Reported together and transparently; a
coarser level is an easier task, not a better model.

Part 2 (if dev predictions exist): learns a per-group INCLUDE/EXCLUDE policy on
the DEV split and applies it to TEST. A group is dropped only if dropping it
improves dev micro-F1 (its false positives outweigh its true positives). Tuned on
dev, reported on held-out test — no test-set fitting.
"""
import argparse
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import _pred_set, _truth_positive_set, propagate_hierarchy
from src.grouping import CLINICAL_GROUPS, CORE_PATHOLOGY_GROUPS, to_groups, any_acute
from src.labeling import CHEXPERT_CLASSES
from sklearn.metrics import precision_recall_fscore_support

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]


def load(split, suffix, out, drop_errors=False):
    p = os.path.join(out, f"predictions_{split}_{suffix}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if drop_errors and "api_error" in df:
        # throttled runs leave empty predictions; tuning on them is meaningless
        df = df[df["api_error"].fillna("") == ""]
    return df.set_index("uid")


def sets(pred_df, s, uids, u_policy="ones", prefix="lblcx_"):
    T = {u: propagate_hierarchy(_truth_positive_set(s.loc[u], u_policy, prefix)) for u in uids}
    P = {u: propagate_hierarchy(_pred_set(pred_df.loc[u])) for u in uids}
    return T, P


def micro_groups(T, P, uids, groups):
    keys = list(groups)
    if not keys or not uids:
        return 0.0, 0.0, 0.0
    yt = np.array([[1 if g in to_groups(T[u], groups) else 0 for g in keys]
                   for u in uids], dtype=int)
    yp = np.array([[1 if g in to_groups(P[u], groups) else 0 for g in keys]
                   for u in uids], dtype=int)
    # micro P/R/F1 computed directly: robust to single-column / all-zero cases
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def micro_flat(T, P, uids, classes):
    yt = [[1 if c in T[u] else 0 for c in classes] for u in uids]
    yp = [[1 if c in P[u] else 0 for c in classes] for u in uids]
    return precision_recall_fscore_support(np.array(yt), np.array(yp),
                                           average="micro", zero_division=0)[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="targeted")
    args = ap.parse_args()
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")

    test = load("test", args.suffix, out)
    uids = sorted(set(test.index) & set(s.index))
    T, P = sets(test, s, uids)

    print(f"=== GRANULARITY LADDER (test, n={len(uids)}, hierarchy-aware, uncertain=1) ===")
    print(f"{'granularity':<44}{'P':>7}{'R':>7}{'F1':>7}")
    p, r, f = micro_flat(T, P, uids, ALL)
    print(f"{'13 fine-grained CheXpert classes':<44}{p:>7.3f}{r:>7.3f}{f:>7.3f}")
    for name, G in [("5 clinical groups", CLINICAL_GROUPS),
                    ("3 core pathology groups", CORE_PATHOLOGY_GROUPS)]:
        p, r, f = micro_groups(T, P, uids, G)
        print(f"{name:<44}{p:>7.3f}{r:>7.3f}{f:>7.3f}")
    yt = [1 if any_acute(T[u]) else 0 for u in uids]
    yp = [1 if any_acute(P[u]) else 0 for u in uids]
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    print(f"{'binary: any acute abnormality':<44}{p:>7.3f}{r:>7.3f}{f:>7.3f}")

    # ---- Part 2: dev-tuned per-group policy ----
    dev = load("dev", args.suffix, out, drop_errors=True)
    if dev is None or len(dev) < 50:
        print("\n(no usable dev predictions — skipping dev-tuned policy)")
        return
    duids = sorted(set(dev.index) & set(s.index))
    dT, dP = sets(dev, s, duids)
    print(f"\n=== DEV-TUNED GROUP POLICY (tuned on dev n={len(duids)}, applied to test) ===")
    for name, G in [("5 clinical groups", CLINICAL_GROUPS),
                    ("3 core pathology groups", CORE_PATHOLOGY_GROUPS)]:
        keep = list(G)
        base = micro_groups(dT, dP, duids, {k: G[k] for k in keep})[2]
        improved = True
        while improved and len(keep) > 1:
            improved = False
            for g in list(keep):
                cand = [x for x in keep if x != g]
                if micro_groups(dT, dP, duids, {k: G[k] for k in cand})[2] > base:
                    base = micro_groups(dT, dP, duids, {k: G[k] for k in cand})[2]
                    keep, improved = cand, True
        sub = {k: G[k] for k in keep}
        dp, dr, df_ = micro_groups(dT, dP, duids, sub)
        tp, tr, tf = micro_groups(T, P, uids, sub)
        dropped = [g for g in G if g not in keep]
        print(f"  {name}: kept {keep}" + (f", dropped {dropped}" if dropped else " (none dropped)"))
        print(f"    dev F1={df_:.3f}  ->  TEST F1={tf:.3f} (P{tp:.3f}/R{tr:.3f})")


if __name__ == "__main__":
    main()
