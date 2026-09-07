"""Close the two statistical gaps in the Exp 6 probe result.

The probe section currently reports a point estimate where every other comparison
in this project carries a paired interval, and a shuffle control drawn once per
finding where a distribution is wanted. Both are fixed here, on features already
on disk, without refitting anything the reader has not already seen.

  1. PAIRED BOOTSTRAP on the interface gap. Resample STUDIES with replacement
     (never cells -- the 13 findings within a study are not independent), recompute
     micro-F1 for probe and model on the same resample, and take the difference.
     This is the same procedure used for every visual-condition and dose-response
     comparison in the parent study, so the interval is directly comparable.

  2. SHUFFLE CONTROL AS A DISTRIBUTION. The published control permuted each
     finding's training labels once, which is why its per-class values scatter
     from 0.311 to 0.544 -- that spread is the sampling noise of a single draw,
     not evidence about the probe. Repeating the permutation gives a null band,
     and the honest question is whether the observed AUROC sits outside it.

Both are reported at the vision-language interface (`proj`), the site the section's
claims are made at.

Usage
  python3 scripts/51_probe_inference.py --features 8bit --boot 5000 --shuffles 25
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
MIN_POS = 10


def load_features(fd, uids, site):
    have = [u for u in uids if os.path.exists(os.path.join(fd, f"{u}.npz"))]
    return np.stack([np.load(os.path.join(fd, f"{u}.npz"))[site] for u in have]), have


def truth_matrix(studies, uids, classes, prefix="lblcx_"):
    by = studies.set_index("uid")
    return np.array([[1 if c in propagate_hierarchy(
        _truth_positive_set(by.loc[u], "ones", prefix)) else 0 for c in classes]
        for u in uids], dtype=int)


def propagate_matrix(M, classes):
    return np.array([[1 if c in propagate_hierarchy(
        {k for k, v in zip(classes, r) if v}) else 0 for c in classes] for r in M], dtype=int)


def micro_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def micro_f1(yt, yp):
    return micro_from_counts(int((yt & yp).sum()), int(((1 - yt) & yp).sum()),
                             int((yt & (1 - yp)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="8bit")
    ap.add_argument("--site", default="proj")
    ap.add_argument("--stage-a", default="predictions_test_twostage_mg_stageA.csv")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--shuffles", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    fd = os.path.join(out, "exp6_features", args.features)
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    te_u = list(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    de_u = list(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    tr_u = [u for u in studies.uid if u not in set(te_u) | set(de_u)]
    stage_a = pd.read_csv(os.path.join(out, args.stage_a)).set_index("uid")
    rng = np.random.default_rng(args.seed)

    Xtr, utr = load_features(fd, tr_u, args.site)
    Xde, ude = load_features(fd, de_u, args.site)
    Xte, ute = load_features(fd, te_u, args.site)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xde, Xte = sc.transform(Xtr), sc.transform(Xde), sc.transform(Xte)
    ytr = truth_matrix(studies, utr, ALL)
    yde = truth_matrix(studies, ude, ALL)
    yte = truth_matrix(studies, ute, ALL)
    print(f"[51] site={args.site} train {len(utr)} dev {len(ude)} test {len(ute)}")

    vlm_raw = np.array([[1 if c in _pred_set(stage_a.loc[u]) else 0 for c in ALL]
                        if u in stage_a.index else [0] * len(ALL) for u in ute])
    vlm = propagate_matrix(vlm_raw, ALL)

    # --- fit the probe once, exactly as the published result does -------------
    te_scores = np.zeros((len(ute), len(ALL)))
    obs_auc = {}
    for ci, cls in enumerate(ALL):
        if ytr[:, ci].sum() < 2 or len(np.unique(yde[:, ci])) < 2:
            te_scores[:, ci] = -1e9
            continue
        best, ba = None, -1
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                     random_state=args.seed).fit(Xtr, ytr[:, ci])
            a = roc_auc_score(yde[:, ci], clf.decision_function(Xde))
            if a > ba:
                best, ba = clf, a
        te_scores[:, ci] = best.decision_function(Xte)
        if MIN_POS <= yte[:, ci].sum() < len(ute):
            obs_auc[cls] = roc_auc_score(yte[:, ci], te_scores[:, ci])

    # matched commitment, same rule as the published table
    probe_raw = np.zeros_like(vlm_raw)
    for ci in range(len(ALL)):
        k = int(vlm_raw[:, ci].sum())
        if k:
            probe_raw[np.argsort(-te_scores[:, ci])[:k], ci] = 1
    probe = propagate_matrix(probe_raw, ALL)

    def gap(idx, classes):
        cols = [ALL.index(c) for c in classes]
        t, p, v = yte[np.ix_(idx, cols)], probe[np.ix_(idx, cols)], vlm[np.ix_(idx, cols)]
        return micro_f1(t, p) - micro_f1(t, v), micro_f1(t, p), micro_f1(t, v)

    print(f"\n[1] PAIRED BOOTSTRAP over studies ({args.boot} resamples)")
    for name, classes in (("13-class", ALL), ("reliable-11", RELIABLE_CLASSES)):
        obs, pf, vf = gap(np.arange(len(ute)), classes)
        boots = np.empty(args.boot)
        n = len(ute)
        for b in range(args.boot):
            idx = rng.integers(0, n, n)
            boots[b] = gap(idx, classes)[0]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "n.s. (interval spans 0)"
        print(f"  {name:<12} probe {pf:.4f}  model {vf:.4f}  "
              f"delta {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")

    print(f"\n[2] SHUFFLE CONTROL as a distribution ({args.shuffles} permutations)")
    null = {c: [] for c in obs_auc}
    for s in range(args.shuffles):
        for cls in obs_auc:
            ci = ALL.index(cls)
            yp = rng.permutation(ytr[:, ci])
            clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced",
                                     random_state=args.seed + s).fit(Xtr, yp)
            null[cls].append(roc_auc_score(yte[:, ci], clf.decision_function(Xte)))
        print(f"    permutation {s+1}/{args.shuffles}", end="\r", flush=True)
    print(" " * 40, end="\r")

    rows = []
    for cls, obs in sorted(obs_auc.items(), key=lambda kv: -kv[1]):
        arr = np.array(null[cls])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        # one-sided permutation p, with the +1 correction for a finite draw count
        p = (int((arr >= obs).sum()) + 1) / (len(arr) + 1)
        rows.append(dict(cls=cls, n_test_pos=int(yte[:, ALL.index(cls)].sum()),
                         auroc=obs, null_mean=float(arr.mean()),
                         null_lo=lo, null_hi=hi, perm_p=p))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  null band mean {df.null_mean.mean():.3f} "
          f"[{df.null_lo.mean():.3f}, {df.null_hi.mean():.3f}] — "
          f"{int((df.perm_p < 0.05).sum())}/{len(df)} findings above it at p<0.05")

    path = os.path.join(out, "exp6_probe_inference.csv")
    df.to_csv(path, index=False)
    print(f"\n[51] wrote {path}")


if __name__ == "__main__":
    main()
