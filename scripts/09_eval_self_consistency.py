"""Evaluate self-consistency predictions by sweeping the vote threshold vote_k.

Key question: is the ensemble vote fraction a DISCRIMINATIVE signal (unlike the
model's flat self-reported confidence)? If so, raising vote_k should trade recall
for precision along a real curve — and some vote_k should beat the single-sample
soft F1 of 0.338. Compares against the same 150 uids from the single-shot soft run.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import _pred_set, _truth_positive_set
from src.normalize import normalize_dx
from src.labeling import CHEXPERT_CLASSES
from sklearn.metrics import precision_recall_fscore_support

CLASSES = [c for c in CHEXPERT_CLASSES if c != "No Finding"]


def pred_set_votek(resp, k, drop=frozenset()):
    f = resp.get("findings", {}) or {}
    out = set()
    for cls, d in f.items():
        if isinstance(d, dict) and d.get("votes", 0) >= k:
            hit = normalize_dx(cls)
            out |= hit if hit else {cls}
    return {c for c in out if c != "No Finding" and c not in drop}


def micro(y_true, y_pred):
    return precision_recall_fscore_support(
        np.array(y_true), np.array(y_pred), average="micro", zero_division=0)[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", default="predictions_test_sc5.csv")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    sc = pd.read_csv(os.path.join(out, args.sc)).set_index("uid")
    soft = pd.read_csv(os.path.join(out, "predictions_test_soft.csv")).set_index("uid")
    uids = sorted(set(sc.index) & set(s.index))

    truth = {u: _truth_positive_set(s.loc[u], "ignore", "lblcx_") for u in uids}
    yt = [[1 if c in truth[u] else 0 for c in CLASSES] for u in uids]

    # single-sample soft baseline on the SAME uids
    yp_soft = [[1 if c in _pred_set(soft.loc[u]) else 0 for c in CLASSES] for u in uids]
    p, r, f = micro(yt, yp_soft)
    print(f"baseline single-shot soft (same {len(uids)} uids): "
          f"P={p:.3f} R={r:.3f} F1={f:.3f}\n")

    print(f"=== VOTE-THRESHOLD SWEEP (N={args.n} samples) ===")
    print(f"{'vote_k':>7}{'P':>7}{'R':>7}{'F1':>7}   meaning")
    resp = {u: json.loads(sc.loc[u]["response"]) for u in uids}
    best = None
    for k in range(1, args.n + 1):
        yp = [[1 if c in pred_set_votek(resp[u], k) else 0 for c in CLASSES] for u in uids]
        p, r, f = micro(yt, yp)
        tag = {1: "any sample", (args.n + 1) // 2 + (args.n % 2 == 0): "majority",
               args.n: "unanimous"}.get(k, "")
        print(f"{k:>7}{p:>7.3f}{r:>7.3f}{f:>7.3f}   >={k}/{args.n} samples  {tag}")
        if best is None or f > best[1]:
            best = (k, f, p, r)

    # also drop the 3 junk classes at the best k
    k = best[0]
    JUNK = frozenset({"Enlarged Cardiomediastinum", "Lung Lesion", "Pleural Other"})
    keep = [c for c in CLASSES if c not in JUNK]
    yt2 = [[1 if c in truth[u] else 0 for c in keep] for u in uids]
    yp2 = [[1 if c in pred_set_votek(resp[u], k, JUNK) else 0 for c in keep] for u in uids]
    p, r, f = micro(yt2, yp2)
    print(f"\nbest vote_k={best[0]} (F1={best[1]:.3f}) + drop 3 junk classes: "
          f"P={p:.3f} R={r:.3f} F1={f:.3f}")


if __name__ == "__main__":
    main()
