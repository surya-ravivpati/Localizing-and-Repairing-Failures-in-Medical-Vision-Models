"""Why is 13-class F1 low? Decompose the loss instead of guessing at it.

Every prompt/decoding lever for raising this number has already been tried and is
exhausted (ensembling twice, chain-of-verification, few-shot twice, targeted
per-class prompts) and class-dropping was rejected as metric gaming. What was
never available before is a measurement of how much of the low score is a
DISCRIMINATION failure and how much is everything else. The Exp 6 probe supplies
it: the same studies, the same reference, but a readout whose threshold we control.

Four scorings of the same predictions separate four different causes:

  1. matched commitment   probe forced to the VLM's positive count  (the fair
                          head-to-head of §9 - but NOT the best the
                          representation can do)
  2. dev-tuned threshold  per-class cutoff chosen on the 552-study dev split,
                          applied once to test. Legitimate: no test peeking,
                          no class dropped.
  3. test-oracle          per-class cutoff chosen ON test. NOT a claimable
                          number - an upper bound on what thresholding alone
                          could ever buy.
  4. maj3 reference       same predictions, scored against >=2-of-3 agreement
                          among the rule-based, CheXbert and VisualCheXbert
                          labelers instead of CheXbert alone. Isolates how much
                          of the loss is reference noise rather than model error.

Usage
  python3 scripts/48_diagnose_f1.py --features 8bit --site proj
"""
import argparse
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.eval_accuracy import (_pred_set, _truth_positive_set, propagate_hierarchy,
                               RELIABLE_CLASSES)
from src.labeling import CHEXPERT_CLASSES

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
C_GRID = [0.001, 0.01, 0.1, 1.0]


def load_features(feat_dir, uids, site):
    have = [u for u in uids if os.path.exists(os.path.join(feat_dir, f"{u}.npz"))]
    X = np.stack([np.load(os.path.join(feat_dir, f"{u}.npz"))[site] for u in have])
    return X, have


def truth_matrix(studies, uids, classes, prefix):
    by_uid = studies.set_index("uid")
    rows = []
    for u in uids:
        t = propagate_hierarchy(_truth_positive_set(by_uid.loc[u], "ones", prefix))
        rows.append([1 if c in t else 0 for c in classes])
    return np.array(rows, dtype=int)


def maj3_matrix(studies, uids, classes):
    """Class is positive if >=2 of the 3 independent labelers call it positive.
    Denoises labeler idiosyncrasy; does NOT escape the report ceiling, since two
    of the three read reports rather than images."""
    by_uid = studies.set_index("uid")
    rows = []
    for u in uids:
        row = by_uid.loc[u]
        votes = []
        for pfx in ("lbl_", "lblcx_", "lblvcx_"):
            votes.append(propagate_hierarchy(_truth_positive_set(row, "ones", pfx)))
        rows.append([1 if sum(c in v for v in votes) >= 2 else 0 for c in classes])
    return np.array(rows, dtype=int)


def propagate_matrix(M, classes):
    out = []
    for row in M:
        s = propagate_hierarchy({c for c, v in zip(classes, row) if v})
        out.append([1 if c in s else 0 for c in classes])
    return np.array(out, dtype=int)


def micro(yt, yp):
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_threshold(scores, truth):
    """Cutoff maximising this class's F1 on the split it is given."""
    order = np.argsort(-scores)
    ts, best, bt = truth[order], -1.0, scores.max() + 1.0
    tp = 0; npos = int(truth.sum())
    for i, t in enumerate(ts):
        tp += int(t)
        p = tp / (i + 1); r = tp / npos if npos else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best:
            best, bt = f, scores[order][i]
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="8bit")
    ap.add_argument("--site", default="proj")
    ap.add_argument("--stage-a", default="predictions_test_twostage_mg_stageA.csv")
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    feat_dir = os.path.join(out, "exp6_features", args.features)
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    test_uids = list(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    dev_uids = list(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    train_uids = [u for u in studies.uid if u not in set(test_uids) | set(dev_uids)]
    stage_a = pd.read_csv(os.path.join(out, args.stage_a)).set_index("uid")

    Xtr, utr = load_features(feat_dir, train_uids, args.site)
    Xde, ude = load_features(feat_dir, dev_uids, args.site)
    Xte, ute = load_features(feat_dir, test_uids, args.site)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xde, Xte = sc.transform(Xtr), sc.transform(Xde), sc.transform(Xte)

    ytr = truth_matrix(studies, utr, ALL, "lblcx_")
    yde = truth_matrix(studies, ude, ALL, "lblcx_")
    yte = truth_matrix(studies, ute, ALL, "lblcx_")
    yte_maj = maj3_matrix(studies, ute, ALL)

    # the VLM's own calls on the same studies
    vlm_raw = np.array([[1 if c in _pred_set(stage_a.loc[u]) else 0 for c in ALL]
                        if u in stage_a.index else [0] * len(ALL) for u in ute])
    vlm = propagate_matrix(vlm_raw, ALL)

    print(f"[diag] site={args.site} features={args.features} "
          f"train {len(utr)} dev {len(ude)} test {len(ute)}")

    dev_sc = np.zeros((len(ude), len(ALL)))
    te_sc = np.zeros((len(ute), len(ALL)))
    aurocs = {}
    for ci, cls in enumerate(ALL):
        if ytr[:, ci].sum() < 2:
            te_sc[:, ci] = -1e9; dev_sc[:, ci] = -1e9; continue
        best, bacc = None, -1
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                     random_state=args.seed).fit(Xtr, ytr[:, ci])
            if len(np.unique(yde[:, ci])) < 2:
                best = clf; break
            a = roc_auc_score(yde[:, ci], clf.decision_function(Xde))
            if a > bacc:
                best, bacc = clf, a
        dev_sc[:, ci] = best.decision_function(Xde)
        te_sc[:, ci] = best.decision_function(Xte)
        if 0 < yte[:, ci].sum() < len(ute):
            aurocs[cls] = roc_auc_score(yte[:, ci], te_sc[:, ci])

    def decide(mode):
        yp = np.zeros_like(yte)
        for ci in range(len(ALL)):
            if mode == "matched":
                k = int(vlm_raw[:, ci].sum())
                if k:
                    yp[np.argsort(-te_sc[:, ci])[:k], ci] = 1
            elif mode == "dev":
                if len(np.unique(yde[:, ci])) > 1:
                    yp[:, ci] = (te_sc[:, ci] >= best_threshold(dev_sc[:, ci], yde[:, ci])).astype(int)
            elif mode == "oracle":
                if yte[:, ci].sum():
                    yp[:, ci] = (te_sc[:, ci] >= best_threshold(te_sc[:, ci], yte[:, ci])).astype(int)
        return propagate_matrix(yp, ALL)

    def report(name, yp, truth, classes=ALL):
        idx = [ALL.index(c) for c in classes]
        p, r, f = micro(truth[:, idx], yp[:, idx])
        print(f"  {name:<34} P {p:.3f}  R {r:.3f}  F1 {f:.4f}   ({int(yp[:, idx].sum())} calls)")
        return f

    print(f"\n[macro AUROC at {args.site}] {np.mean(list(aurocs.values())):.3f}  "
          f"-- discrimination is NOT the whole story if F1 is far below this")

    print("\n=== 13-class, CheXbert reference (the paper's headline convention) ===")
    report("VLM Stage A", vlm, yte)
    m = decide("matched"); report("probe, matched commitment", m, yte)
    d = decide("dev");     report("probe, dev-tuned threshold", d, yte)
    o = decide("oracle");  report("probe, test-oracle (upper bd)", o, yte)

    print("\n=== same predictions, majority-of-3-labeler reference ===")
    report("VLM Stage A", vlm, yte_maj)
    report("probe, dev-tuned threshold", d, yte_maj)

    print("\n=== reliable-11 (drops the 2 classes labelers disagree on) ===")
    report("VLM Stage A", vlm, yte, RELIABLE_CLASSES)
    report("probe, dev-tuned threshold", d, yte, RELIABLE_CLASSES)

    print("\n=== per-class F1 at the dev-tuned threshold (CheXbert ref) ===")
    rows = []
    for ci, c in enumerate(ALL):
        p, r, f = micro(yte[:, [ci]], d[:, [ci]])
        pv, rv, fv = micro(yte[:, [ci]], vlm[:, [ci]])
        rows.append(dict(cls=c, n_pos=int(yte[:, ci].sum()),
                         auroc=aurocs.get(c, np.nan), probe_f1=f, vlm_f1=fv,
                         probe_calls=int(d[:, ci].sum()), vlm_calls=int(vlm[:, ci].sum())))
    df = pd.DataFrame(rows).sort_values("n_pos", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(os.path.join(out, "exp6_f1_diagnosis.csv"), index=False)
    print(f"\n[diag] wrote {os.path.join(out, 'exp6_f1_diagnosis.csv')}")


if __name__ == "__main__":
    main()
