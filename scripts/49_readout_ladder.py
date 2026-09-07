"""Push the trained-readout score as far as it honestly goes, on the full ladder.

48_diagnose_f1.py established that the low headline F1 is a per-class OPERATING
POINT failure rather than a discrimination failure: macro AUROC 0.77 while the
model scores 0.31, and per-class call counts that are wrong in both directions
(578 Support Devices calls against 61 true; 5 Atelectasis against 144). This
script asks how far a readout of the SAME frozen features goes once the two
things 48 identified as broken are fixed properly.

Two knobs, both selected on the dev split and applied once to test:

  FEATURES   proj alone (the vision-language interface) versus a concatenation of
             mid-stack, late and interface representations. 48 showed information
             is fully formed by layer 12 and merely persists, so the later sites
             may carry partly redundant but not identical error.

  THRESHOLD  F1-maximising on dev (what 48 used) versus PREVALENCE-MATCHED --
             emit the number of positives dev prevalence predicts. 48's known
             weakness was rare classes: F1-maximising picked a cutoff giving
             Pleural Other 134 calls for 11 true positives, because with a
             handful of dev positives the F1 curve is nearly flat and its argmax
             is noise. Prevalence matching is stable exactly where that fails.

Everything is scored on the paper's full granularity ladder (13 classes,
reliable-11, 3 clinical groups, binary any-acute) under the paper's conventions
(CheXbert reference, hierarchy-aware, uncertain = positive), so the numbers sit
directly beside every other table.

HONESTY CONSTRAINT: this readout is SUPERVISED on the 2 520-study training pool.
Every prompt result in this project is zero-shot. These numbers belong in a
"trained readout" row, never in the prompting tables. No class is ever dropped.

Usage
  python3 scripts/49_readout_ladder.py --features 8bit
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
from src.grouping import CORE_PATHOLOGY_GROUPS, to_groups, any_acute
from src.labeling import CHEXPERT_CLASSES

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
C_GRID = [0.001, 0.01, 0.1, 1.0]
FEATURE_SETS = {"proj": ["proj"], "concat": ["layer_12", "layer_24", "proj"]}


def load_features(feat_dir, uids, sites):
    have = [u for u in uids if os.path.exists(os.path.join(feat_dir, f"{u}.npz"))]
    mats = []
    for u in have:
        z = np.load(os.path.join(feat_dir, f"{u}.npz"))
        mats.append(np.concatenate([z[s] for s in sites]))
    return np.stack(mats), have


def truth_matrix(studies, uids, classes, prefix="lblcx_"):
    by_uid = studies.set_index("uid")
    return np.array([[1 if c in propagate_hierarchy(
        _truth_positive_set(by_uid.loc[u], "ones", prefix)) else 0 for c in classes]
        for u in uids], dtype=int)


def propagate_matrix(M, classes):
    return np.array([[1 if c in propagate_hierarchy(
        {k for k, v in zip(classes, row) if v}) else 0 for c in classes]
        for row in M], dtype=int)


def micro(yt, yp):
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def f1max_threshold(scores, truth):
    order = np.argsort(-scores)
    ts = truth[order]; npos = int(truth.sum())
    if not npos:
        return scores.max() + 1.0
    tp = 0; best = -1.0; bt = scores.max() + 1.0
    for i, t in enumerate(ts):
        tp += int(t)
        p = tp / (i + 1); r = tp / npos
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best:
            best, bt = f, scores[order][i]
    return bt


def prevalence_threshold(dev_scores, dev_truth, test_scores):
    """Emit the count dev prevalence predicts. Stable for rare classes, where the
    F1 curve is flat and its argmax is noise."""
    rate = dev_truth.mean()
    k = int(round(rate * len(test_scores)))
    if k <= 0:
        return test_scores.max() + 1.0
    k = min(k, len(test_scores))
    return np.sort(test_scores)[::-1][k - 1]


def ladder(pred_sets, truth_sets):
    """13-class / reliable-11 / 3-group micro-F1 and binary accuracy + F1."""
    def mat(sets, classes):
        return np.array([[1 if c in s else 0 for c in classes] for s in sets], dtype=int)
    out = {}
    for name, cls in (("f1_13", ALL), ("f1_rel11", RELIABLE_CLASSES)):
        out[name] = micro(mat(truth_sets, cls), mat(pred_sets, cls))[2]
    g = sorted(CORE_PATHOLOGY_GROUPS)
    gp = [to_groups(s, CORE_PATHOLOGY_GROUPS) for s in pred_sets]
    gt = [to_groups(s, CORE_PATHOLOGY_GROUPS) for s in truth_sets]
    out["f1_3grp"] = micro(mat(gt, g), mat(gp, g))[2]
    bp = np.array([any_acute(s) for s in pred_sets], dtype=int)
    bt = np.array([any_acute(s) for s in truth_sets], dtype=int)
    out["bin_acc"] = float((bp == bt).mean())
    out["bin_f1"] = micro(bt.reshape(-1, 1), bp.reshape(-1, 1))[2]
    return out


def to_sets(M, classes):
    return [{c for c, v in zip(classes, row) if v} for row in M]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="8bit")
    ap.add_argument("--stage-a", default="predictions_test_twostage_mg_stageA.csv")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    feat_dir = os.path.join(out, "exp6_features", args.features)
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    test_uids = list(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    dev_uids = list(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    train_uids = [u for u in studies.uid if u not in set(test_uids) | set(dev_uids)]
    stage_a = pd.read_csv(os.path.join(out, args.stage_a)).set_index("uid")

    rows = []
    _, ute0 = load_features(feat_dir, test_uids, ["proj"])
    yte = truth_matrix(studies, ute0, ALL)
    truth_sets = to_sets(yte, ALL)

    vlm_raw = np.array([[1 if c in _pred_set(stage_a.loc[u]) else 0 for c in ALL]
                        if u in stage_a.index else [0] * len(ALL) for u in ute0])
    vlm_sets = to_sets(propagate_matrix(vlm_raw, ALL), ALL)
    base = ladder(vlm_sets, truth_sets)
    rows.append(dict(readout="MedGemma Stage A (zero-shot)", features="-", threshold="-",
                     calls=int(vlm_raw.sum()), **base))
    print(f"[base] VLM  13cls {base['f1_13']:.4f}  rel11 {base['f1_rel11']:.4f}  "
          f"3grp {base['f1_3grp']:.4f}  bin {base['bin_acc']:.4f}", flush=True)

    for fname, sites in FEATURE_SETS.items():
        Xtr, utr = load_features(feat_dir, train_uids, sites)
        Xde, ude = load_features(feat_dir, dev_uids, sites)
        Xte, ute = load_features(feat_dir, test_uids, sites)
        assert ute == ute0, "test order drifted between feature sets"
        sc = StandardScaler().fit(Xtr)
        Xtr, Xde, Xte = sc.transform(Xtr), sc.transform(Xde), sc.transform(Xte)
        ytr = truth_matrix(studies, utr, ALL)
        yde = truth_matrix(studies, ude, ALL)

        dev_s = np.zeros((len(ude), len(ALL))); te_s = np.zeros((len(ute), len(ALL)))
        aur = []
        for ci in range(len(ALL)):
            if ytr[:, ci].sum() < 2:
                dev_s[:, ci] = -1e9; te_s[:, ci] = -1e9; continue
            best, ba = None, -1
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                         random_state=args.seed).fit(Xtr, ytr[:, ci])
                if len(np.unique(yde[:, ci])) < 2:
                    best = clf; break
                a = roc_auc_score(yde[:, ci], clf.decision_function(Xde))
                if a > ba:
                    best, ba = clf, a
            dev_s[:, ci] = best.decision_function(Xde)
            te_s[:, ci] = best.decision_function(Xte)
            if 0 < yte[:, ci].sum() < len(ute):
                aur.append(roc_auc_score(yte[:, ci], te_s[:, ci]))

        for tname in ("f1max", "prevalence"):
            yp = np.zeros_like(yte)
            for ci in range(len(ALL)):
                if te_s[0, ci] == -1e9:
                    continue
                if tname == "f1max":
                    if len(np.unique(yde[:, ci])) > 1:
                        yp[:, ci] = (te_s[:, ci] >= f1max_threshold(dev_s[:, ci], yde[:, ci])).astype(int)
                else:
                    yp[:, ci] = (te_s[:, ci] >= prevalence_threshold(
                        dev_s[:, ci], yde[:, ci], te_s[:, ci])).astype(int)
            sets = to_sets(propagate_matrix(yp, ALL), ALL)
            res = ladder(sets, truth_sets)
            rows.append(dict(readout="linear probe (supervised)", features=fname,
                             threshold=tname, calls=int(yp.sum()),
                             macro_auroc=float(np.mean(aur)), **res))
            print(f"[{fname:>6}/{tname:<10}] 13cls {res['f1_13']:.4f}  "
                  f"rel11 {res['f1_rel11']:.4f}  3grp {res['f1_3grp']:.4f}  "
                  f"bin {res['bin_acc']:.4f}  ({int(yp.sum())} calls)", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(out, "exp6_readout_ladder.csv")
    df.to_csv(path, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[ladder] wrote {path}")
    print("[ladder] REMINDER: the probe rows are SUPERVISED on 2520 studies; the "
          "VLM row is zero-shot. Never merge these into the prompting tables.")


if __name__ == "__main__":
    main()
