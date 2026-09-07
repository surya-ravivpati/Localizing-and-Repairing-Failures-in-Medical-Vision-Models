"""Master comparison of every prompt/decoding variant on the full 754 test set,
scored vs the CheXbert reference (lblcx_), on both the full-13 classes and the
reliable classes (κ≥0.4). One authoritative table + bootstrap CIs.

Auto-discovers whichever predictions_test_*.csv exist; missing ones are skipped.
"""
import json
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import (_pred_set, _truth_positive_set, RELIABLE_CLASSES,
                               propagate_hierarchy)
from src.labeling import CHEXPERT_CLASSES
from sklearn.metrics import precision_recall_fscore_support

ALL_CLASSES = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
HIER = os.environ.get("HIER", "1") == "1"   # hierarchy-aware scoring (default on)
U_POLICY = os.environ.get("U_POLICY", "ignore")

# label -> predictions file (single-shot variants). Ensemble handled separately.
VARIANTS = [
    ("structured_balanced", "predictions_test_balanced.csv"),
    ("structured_soft", "predictions_test_soft.csv"),
    ("structured_targeted", "predictions_test_targeted.csv"),
]


def score(pred_by_uid, truth, uids, classes, n_boot=2000):
    if HIER:
        truth = {u: propagate_hierarchy(truth[u]) for u in uids}
        pred_by_uid = {u: propagate_hierarchy(pred_by_uid[u]) for u in uids}
    yt = np.array([[1 if c in truth[u] else 0 for c in classes] for u in uids])
    yp = np.array([[1 if c in pred_by_uid[u] else 0 for c in classes] for u in uids])
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average="micro", zero_division=0)
    rng = np.random.default_rng(0)
    m = len(uids)
    fs = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        fs.append(precision_recall_fscore_support(
            yt[idx], yp[idx], average="micro", zero_division=0)[2])
    return p, r, f, np.percentile(fs, 2.5), np.percentile(fs, 97.5)


def ensemble_pred(sc_df, k=5):
    from src.normalize import normalize_dx
    out = {}
    for u, row in sc_df.iterrows():
        f = json.loads(row["response"]).get("findings", {}) or {}
        s = set()
        for c, d in f.items():
            if isinstance(d, dict) and d.get("votes", 0) >= k:
                hit = normalize_dx(c)
                s |= hit if hit else {c}
        out[u] = {c for c in s if c != "No Finding"}
    return out


def main():
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")

    # collect available variants
    preds = {}
    for label, fname in VARIANTS:
        p = os.path.join(out, fname)
        if os.path.exists(p):
            df = pd.read_csv(p).set_index("uid")
            preds[label] = {u: _pred_set(df.loc[u]) for u in df.index}
    for label, fname in [("soft-ensemble unanimous", "predictions_test_sc5.csv"),
                         ("targeted-ensemble unanimous", "predictions_test_sc5_targeted.csv")]:
        sc_path = os.path.join(out, fname)
        if os.path.exists(sc_path):
            sc = pd.read_csv(sc_path).set_index("uid")
            preds[label] = ensemble_pred(sc, k=5)

    # common uid set across all variants
    uids = sorted(set.intersection(*[set(d.keys()) for d in preds.values()]))
    truth = {u: _truth_positive_set(s.loc[u], U_POLICY, "lblcx_") for u in uids}
    print(f"scored on {len(uids)} common uids, CheXbert (lblcx_) reference | "
          f"hierarchy={HIER} u_policy={U_POLICY}\n")

    for title, classes in [("FULL 13 CLASSES", ALL_CLASSES),
                           (f"RELIABLE CLASSES (κ≥0.4, n={len(RELIABLE_CLASSES)})", RELIABLE_CLASSES)]:
        print(f"=== {title} ===")
        print(f"{'variant':<26}{'P':>7}{'R':>7}{'F1':>7}   {'F1 95% CI':>16}")
        for label, pb in preds.items():
            p, r, f, lo, hi = score(pb, truth, uids, classes)
            print(f"{label:<26}{p:>7.3f}{r:>7.3f}{f:>7.3f}   [{lo:.3f}, {hi:.3f}]")
        print()


if __name__ == "__main__":
    main()
