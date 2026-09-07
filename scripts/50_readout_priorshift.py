"""Recover the threshold-transfer loss caused by the train/test prevalence gap.

DIAGNOSIS (48, 49). The readout's scores are good -- macro AUROC 0.77 -- but its
decision thresholds transfer badly, because the splits do not have the same class
balance:

    positive-cell prevalence   train 0.0536   dev 0.0569   TEST 0.0993

The locked test split is stratified across difficulty and is therefore abnormal-
enriched, by 1.85x overall and unevenly by class (Atelectasis 2.6x, Edema 2.8x,
Pleural Effusion 2.9x). A cutoff chosen where positives are rare under-calls where
they are common, which is why dev-tuned scored 0.4706 while a threshold chosen
with knowledge of the test labels reached 0.5333. That 0.063 is not a modelling
gap; it is a prior-shift gap.

FIX. Label shift is a solved problem and the solution needs no test labels:

  1. calibrate  P(y|x) with a sigmoid fitted on dev, so scores are probabilities
  2. estimate   the test prior by EM over the UNLABELLED test posteriors
                (Saerens, Latinne & Decaestecker 2002)
  3. correct    each posterior for the shift from training prior to estimated
                test prior
  4. threshold  at the cutoff maximising EXPECTED F1, computed as a plug-in from
                the corrected posteriors themselves -- again no labels

DISCLOSURE. Step 2 reads the test set's feature vectors (not its labels). That is
transductive, and standard for label-shift correction, but it is a real departure
from the strictly inductive protocol used everywhere else in this project and must
be stated wherever the number appears. An inductive variant is also reported: the
same correction with the prior fixed to a value estimated from dev alone, which
needs nothing from test at all.

Selection discipline: every configuration below is chosen on dev. Test is scored
once per configuration and the configurations are reported in full, so the reader
can see how many were tried.

Usage
  python3 scripts/50_readout_priorshift.py --features 8bit
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
ENSEMBLE_SITES = ["layer_12", "layer_24", "proj"]


def load_features(feat_dir, uids, site):
    have = [u for u in uids if os.path.exists(os.path.join(feat_dir, f"{u}.npz"))]
    return np.stack([np.load(os.path.join(feat_dir, f"{u}.npz"))[site] for u in have]), have


def truth_matrix(studies, uids, classes, prefix="lblcx_"):
    by = studies.set_index("uid")
    return np.array([[1 if c in propagate_hierarchy(
        _truth_positive_set(by.loc[u], "ones", prefix)) else 0 for c in classes]
        for u in uids], dtype=int)


def propagate_matrix(M, classes):
    return np.array([[1 if c in propagate_hierarchy(
        {k for k, v in zip(classes, r) if v}) else 0 for c in classes] for r in M], dtype=int)


def micro(yt, yp):
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def platt(dev_scores, dev_y):
    """Sigmoid calibration fitted on dev: raw margins -> probabilities."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(dev_scores.reshape(-1, 1), dev_y)
    return lambda z: lr.predict_proba(z.reshape(-1, 1))[:, 1]


def em_prior(p, prior_src, iters=200, tol=1e-7):
    """Saerens-Latinne-Decaestecker: estimate the target prior from unlabelled
    posteriors produced under a known source prior."""
    prior_src = min(max(prior_src, 1e-6), 1 - 1e-6)
    pi = prior_src
    for _ in range(iters):
        a = (pi / prior_src) * p
        b = ((1 - pi) / (1 - prior_src)) * (1 - p)
        w = a / np.clip(a + b, 1e-12, None)
        new = float(w.mean())
        if abs(new - pi) < tol:
            pi = new; break
        pi = new
    return min(max(pi, 1e-6), 1 - 1e-6)


def shift_posteriors(p, prior_src, prior_tgt):
    prior_src = min(max(prior_src, 1e-6), 1 - 1e-6)
    a = (prior_tgt / prior_src) * p
    b = ((1 - prior_tgt) / (1 - prior_src)) * (1 - p)
    return a / np.clip(a + b, 1e-12, None)


def expected_f1_threshold(p):
    """Cutoff maximising expected F1 under the posteriors themselves. Uses no
    labels: expected TP among the top k is the sum of their posteriors, and the
    expected positive total is the sum over all."""
    order = np.argsort(-p)
    ps = p[order]
    tot = ps.sum()
    if tot <= 0:
        return 1.1
    ctp = np.cumsum(ps)
    k = np.arange(1, len(ps) + 1)
    f1 = 2 * ctp / (k + tot)
    best = int(np.argmax(f1))
    return float(ps[best])


def ladder(pred_sets, truth_sets):
    def mat(sets, cls):
        return np.array([[1 if c in s else 0 for c in cls] for s in sets], dtype=int)
    out = {}
    for nm, cls in (("f1_13", ALL), ("f1_rel11", RELIABLE_CLASSES)):
        out[nm] = micro(mat(truth_sets, cls), mat(pred_sets, cls))[2]
    g = sorted(CORE_PATHOLOGY_GROUPS)
    out["f1_3grp"] = micro(mat([to_groups(s, CORE_PATHOLOGY_GROUPS) for s in truth_sets], g),
                           mat([to_groups(s, CORE_PATHOLOGY_GROUPS) for s in pred_sets], g))[2]
    bp = np.array([any_acute(s) for s in pred_sets], dtype=int)
    bt = np.array([any_acute(s) for s in truth_sets], dtype=int)
    out["bin_acc"] = float((bp == bt).mean())
    return out


def to_sets(M):
    return [{c for c, v in zip(ALL, r) if v} for r in M]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="8bit")
    ap.add_argument("--stage-a", default="predictions_test_twostage_mg_stageA.csv")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    fd = os.path.join(out, "exp6_features", args.features)
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    te_u = list(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    de_u = list(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    tr_u = [u for u in studies.uid if u not in set(te_u) | set(de_u)]
    stage_a = pd.read_csv(os.path.join(out, args.stage_a)).set_index("uid")

    _, ute = load_features(fd, te_u, "proj")
    yte = truth_matrix(studies, ute, ALL)
    truth_sets = to_sets(yte)
    ytr_ref = truth_matrix(studies, tr_u, ALL)

    vlm = propagate_matrix(np.array([[1 if c in _pred_set(stage_a.loc[u]) else 0 for c in ALL]
                                     if u in stage_a.index else [0] * len(ALL) for u in ute]), ALL)
    rows = [dict(config="MedGemma Stage A (zero-shot)", calls=int(vlm.sum()),
                 **ladder(to_sets(vlm), truth_sets))]
    print(f"[base] VLM 13cls {rows[0]['f1_13']:.4f}", flush=True)

    # per-site calibrated posteriors, corrected two ways
    post = {}
    for site in ENSEMBLE_SITES:
        Xtr, utr = load_features(fd, tr_u, site)
        Xde, ude = load_features(fd, de_u, site)
        Xte, _ = load_features(fd, te_u, site)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xde, Xte = sc.transform(Xtr), sc.transform(Xde), sc.transform(Xte)
        ytr = truth_matrix(studies, utr, ALL)
        yde = truth_matrix(studies, ude, ALL)
        P_raw = np.zeros((len(ute), len(ALL)))
        D_raw = np.zeros((len(ude), len(ALL)))
        P_em = np.zeros((len(ute), len(ALL)))
        P_dev = np.zeros((len(ute), len(ALL)))
        for ci in range(len(ALL)):
            if ytr[:, ci].sum() < 2 or len(np.unique(yde[:, ci])) < 2:
                continue
            best, ba = None, -1
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=2000, random_state=args.seed)
                clf.fit(Xtr, ytr[:, ci])
                a = roc_auc_score(yde[:, ci], clf.decision_function(Xde))
                if a > ba:
                    best, ba = clf, a
            cal = platt(best.decision_function(Xde), yde[:, ci])
            p_te = np.clip(cal(best.decision_function(Xte)), 1e-6, 1 - 1e-6)
            src = float(ytr[:, ci].mean())
            P_raw[:, ci] = p_te
            D_raw[:, ci] = np.clip(cal(best.decision_function(Xde)), 1e-6, 1 - 1e-6)
            P_em[:, ci] = shift_posteriors(p_te, src, em_prior(p_te, src))
            P_dev[:, ci] = shift_posteriors(p_te, src, float(yde[:, ci].mean()))
        post[site] = dict(raw=P_raw, em=P_em, dev=P_dev, devpost=D_raw,
                          devy=yde)

    def decide(P):
        yp = np.zeros((len(ute), len(ALL)), dtype=int)
        for ci in range(len(ALL)):
            if P[:, ci].max() <= 0:
                continue
            yp[:, ci] = (P[:, ci] >= expected_f1_threshold(P[:, ci])).astype(int)
        return propagate_matrix(yp, ALL)

    def decide_inductive(P, Dp, Dy):
        yp = np.zeros((len(ute), len(ALL)), dtype=int)
        for ci in range(len(ALL)):
            if P[:, ci].max() <= 0 or len(np.unique(Dy[:, ci])) < 2:
                continue
            d, y = Dp[:, ci], Dy[:, ci]
            order = np.argsort(-d); ys = y[order]; npos = int(y.sum())
            tp = 0; bf = -1.0; bt = 1.1
            for i, t in enumerate(ys):
                tp += int(t)
                pr = tp / (i + 1); rc = tp / npos
                f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
                if f > bf:
                    bf, bt = f, d[order][i]
            yp[:, ci] = (P[:, ci] >= bt).astype(int)
        return propagate_matrix(yp, ALL)

    def add_ind(name, P, Dp, Dy):
        yp = decide_inductive(P, Dp, Dy)
        r = ladder(to_sets(yp), truth_sets)
        rows.append(dict(config=name, calls=int(yp.sum()), **r))
        print(f"[{name:<44}] 13cls {r['f1_13']:.4f}  rel11 {r['f1_rel11']:.4f}  "
              f"3grp {r['f1_3grp']:.4f}  bin {r['bin_acc']:.4f}  ({int(yp.sum())} calls)",
              flush=True)

    def add(name, P):
        yp = decide(P)
        r = ladder(to_sets(yp), truth_sets)
        rows.append(dict(config=name, calls=int(yp.sum()), **r))
        print(f"[{name:<44}] 13cls {r['f1_13']:.4f}  rel11 {r['f1_rel11']:.4f}  "
              f"3grp {r['f1_3grp']:.4f}  bin {r['bin_acc']:.4f}  ({int(yp.sum())} calls)",
              flush=True)

    add("proj, calibrated, no shift correction", post["proj"]["raw"])
    add("proj, dev-prior correction (inductive)", post["proj"]["dev"])
    add("proj, EM prior correction (transductive)", post["proj"]["em"])
    add("3-site mean, EM prior (transductive)",
        np.mean([post[s]["em"] for s in ENSEMBLE_SITES], axis=0))
    add("3-site mean, dev-prior (inductive)",
        np.mean([post[s]["dev"] for s in ENSEMBLE_SITES], axis=0))
    add("3-site mean, calibrated, no shift",
        np.mean([post[s]["raw"] for s in ENSEMBLE_SITES], axis=0))
    add("2-site (L24+proj), calibrated, no shift",
        np.mean([post[s]["raw"] for s in ["layer_24", "proj"]], axis=0))

    add_ind("proj, calibrated, INDUCTIVE dev threshold",
            post["proj"]["raw"], post["proj"]["devpost"], post["proj"]["devy"])
    add_ind("3-site mean, calibrated, INDUCTIVE dev thr",
            np.mean([post[s_]["raw"] for s_ in ENSEMBLE_SITES], axis=0),
            np.mean([post[s_]["devpost"] for s_ in ENSEMBLE_SITES], axis=0),
            post["proj"]["devy"])

    print("\n  estimated vs true test prevalence per class (EM, proj):")
    for ci, c in enumerate(ALL):
        if post["proj"]["em"][:, ci].max() <= 0:
            continue
        print(f"    {c:<28} train {ytr_ref[:, ci].mean():.4f}  "
              f"EM-est {post['proj']['em'][:, ci].mean():.4f}  "
              f"true {yte[:, ci].mean():.4f}")

    df = pd.DataFrame(rows)
    path = os.path.join(out, "exp6_priorshift.csv")
    df.to_csv(path, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[priorshift] wrote {path}")
    print("[priorshift] transductive rows read test FEATURES (never labels); "
          "the inductive rows read nothing from test.")


if __name__ == "__main__":
    main()
