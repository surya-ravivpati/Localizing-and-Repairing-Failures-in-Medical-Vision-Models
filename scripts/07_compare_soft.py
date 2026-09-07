"""Compare structured_soft (100-study validation run) against structured_balanced
on the SAME 100 uids, vs the CheXbert reference (lblcx_).

Audit follow-up (2026-07-09): balanced routes visible-but-subtle findings to
'uncertain' (scored as a miss). soft tells the model to commit when evidence is
visible. This script reports paired micro P/R/F1 plus the uncertain->present
shift on the top-missed classes.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import _pred_set, _truth_positive_set, multilabel_prf_by_difficulty
from src.labeling import CHEXPERT_CLASSES
from sklearn.metrics import precision_recall_fscore_support


def micro_prf(pred_df, truth_by_uid, prefix="lblcx_"):
    classes = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
    yt, yp = [], []
    for _, r in pred_df.iterrows():
        t = _truth_positive_set(truth_by_uid.loc[r["uid"]], "ignore", prefix)
        p = _pred_set(r)
        yt.append([1 if c in t else 0 for c in classes])
        yp.append([1 if c in p else 0 for c in classes])
    pr, rc, f1, _ = precision_recall_fscore_support(
        np.array(yt), np.array(yp), average="micro", zero_division=0)
    return pr, rc, f1


def presence_counts(pred_df):
    """How often each class is marked present / uncertain across the run."""
    pres, unc = {}, {}
    for _, r in pred_df.iterrows():
        try:
            f = json.loads(r["response"]).get("findings", {}) or {}
        except Exception:
            continue
        for cls, v in f.items():
            if not isinstance(v, dict):
                continue
            p = str(v.get("presence", "")).lower()
            if p == "present":
                pres[cls] = pres.get(cls, 0) + 1
            elif p == "uncertain":
                unc[cls] = unc.get(cls, 0) + 1
    return pres, unc


def bootstrap_f1_ci(pred_df, truth_by_uid, prefix="lblcx_", n=2000, seed=0):
    """Bootstrap 95% CI for micro-F1 by resampling cases."""
    classes = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
    yt, yp = [], []
    for _, r in pred_df.iterrows():
        t = _truth_positive_set(truth_by_uid.loc[r["uid"]], "ignore", prefix)
        p = _pred_set(r)
        yt.append([1 if c in t else 0 for c in classes])
        yp.append([1 if c in p else 0 for c in classes])
    yt, yp = np.array(yt), np.array(yp)
    rng = np.random.default_rng(seed)
    f1s = []
    m = len(yt)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        _, _, f1, _ = precision_recall_fscore_support(
            yt[idx], yp[idx], average="micro", zero_division=0)
        f1s.append(f1)
    return np.percentile(f1s, 2.5), np.percentile(f1s, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft", default="predictions_test_soft100.csv",
                    help="soft predictions file (basename in results/)")
    ap.add_argument("--ci", action="store_true", help="bootstrap 95% CI on F1")
    args = ap.parse_args()

    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")

    soft = pd.read_csv(os.path.join(out, args.soft))
    bal = pd.read_csv(os.path.join(out, "predictions_test_balanced.csv"))
    uids = set(soft["uid"])
    bal = bal[bal["uid"].isin(uids)].reset_index(drop=True)
    print(f"paired comparison on {len(uids)} uids "
          f"(soft n={len(soft)}, balanced n={len(bal)})\n")

    print(f"{'condition':<22} {'P':>7} {'R':>7} {'F1':>7}   {'F1 95% CI':>16}")
    for name, df in [("structured_balanced", bal), ("structured_soft", soft)]:
        pr, rc, f1 = micro_prf(df, s)
        ci = ""
        if args.ci:
            lo, hi = bootstrap_f1_ci(df, s)
            ci = f"[{lo:.3f}, {hi:.3f}]"
        print(f"{name:<22} {pr:>7.3f} {rc:>7.3f} {f1:>7.3f}   {ci:>16}")

    print("\npresence/uncertain counts on the top-missed classes:")
    pb, ub = presence_counts(bal)
    ps, us = presence_counts(soft)
    print(f"{'class':<26} {'bal pres':>9} {'bal unc':>8} {'soft pres':>10} {'soft unc':>9}")
    for c in ["Cardiomegaly", "Lung Opacity", "Atelectasis", "Support Devices"]:
        print(f"{c:<26} {pb.get(c, 0):>9} {ub.get(c, 0):>8} "
              f"{ps.get(c, 0):>10} {us.get(c, 0):>9}")

    print("\nF1 by difficulty (soft):")
    soft2 = soft.copy()
    print(multilabel_prf_by_difficulty(soft2, s.reset_index(), prefix="lblcx_")
          .to_string(index=False))


if __name__ == "__main__":
    main()
