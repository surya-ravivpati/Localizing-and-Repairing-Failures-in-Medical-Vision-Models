r"""Experiment 6, stage 2: linear probes on MedGemma's visual representations.

Question: when MedGemma's language head fails to report a finding, was the
information ever there? Probes are deliberately linear (logistic regression) -
we are not building a better detector, we are asking whether the information is
already LINEARLY RECOVERABLE at each point along the vision->language pathway.

    pixels -> [encoder layer 0..27] -> [projector output] -> language model
              \______________________________________/       \___________/
                        probed here                          scored already
                                                          (Stage A, twostage_mg)

HEADLINE METRIC IS MATCHED-COMMITMENT F1, NOT AUROC.
-----------------------------------------------------
This paper's central finding is that every lever tried so far (prompt format,
model, resolution, crops, reasoning depth) moves the model's COMMITMENT - how
much it calls - rather than its decision quality. A probe's AUROC is threshold-
free while the VLM has already committed, so "probe AUROC .82 beats VLM F1 .40"
would be exactly the confound this project spent five experiments documenting.

So each class's probe is thresholded to predict the SAME NUMBER of positives the
VLM predicted for that class on the same studies. Both sides then get identical
CheXpert hierarchy propagation and are scored against the same reference
(lblcx_, u=ones) with the same micro-F1. Equal commitment budget, so any gap is
readout quality. AUROC is reported alongside as the ceiling-if-perfectly-calibrated.

CONTROLS (a probe result is uninterpretable without these):
  pixels     32x32 grayscale  - is the task simply linearly easy from raw pixels?
  randinit   untrained SigLIP - do TRAINED features beat random ones of same shape?
  shuffle    permuted labels  - is above-chance AUROC real?

Split discipline: probes fit on train (2520), C selected on dev (552), test (754)
scored once. The test set has been locked since the main study.

Usage
  python3 scripts/47_probe_layers.py --features 8bit
  python3 scripts/47_probe_layers.py --features 8bit --compare 4bit randinit
"""
import argparse
import json
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
# every class can be below MIN_POS on a small pilot -> all-NaN nanmean slices
warnings.filterwarnings("ignore", message="Mean of empty slice")

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
C_GRID = [0.001, 0.01, 0.1, 1.0]
MIN_POS = 10          # fewest test positives for a per-class AUROC to be reported


def load_features(feat_dir: str, uids) -> tuple[dict[str, np.ndarray], list[int]]:
    """-> {site: [N, D]} plus the uid order. Only studies present on disk."""
    have = [u for u in uids if os.path.exists(os.path.join(feat_dir, f"{u}.npz"))]
    if not have:
        raise SystemExit(f"no features in {feat_dir}")
    sites, stacks = None, {}
    for u in have:
        z = np.load(os.path.join(feat_dir, f"{u}.npz"))
        if sites is None:
            sites = list(z.files)
            stacks = {s: [] for s in sites}
        for s in sites:
            stacks[s].append(z[s])
    return {s: np.stack(v) for s, v in stacks.items()}, have


def label_matrix(studies: pd.DataFrame, uids, classes) -> np.ndarray:
    """[N, C] binary truth, u=ones + CheXpert hierarchy - the paper's convention."""
    by_uid = studies.set_index("uid")
    rows = []
    for u in uids:
        t = propagate_hierarchy(_truth_positive_set(by_uid.loc[u], "ones", "lblcx_"))
        rows.append([1 if c in t else 0 for c in classes])
    return np.array(rows, dtype=int)


def vlm_prediction_matrix(stage_a: pd.DataFrame, uids, classes):
    """[N, C] the VLM's own committed calls, RAW (pre-hierarchy) so that matching
    the probe's commitment budget compares like with like; propagation is applied
    afterwards to both sides identically, exactly as multilabel_prf does."""
    by_uid = stage_a.set_index("uid")
    raw, prop = [], []
    for u in uids:
        if u not in by_uid.index:
            raw.append(None); prop.append(None); continue
        p = _pred_set(by_uid.loc[u])
        raw.append([1 if c in p else 0 for c in classes])
        pp = propagate_hierarchy(p)
        prop.append([1 if c in pp else 0 for c in classes])
    return raw, prop


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int((y_true & y_pred).sum())
    fp = int(((1 - y_true) & y_pred).sum())
    fn = int((y_true & (1 - y_pred)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def propagate_matrix(M: np.ndarray, classes) -> np.ndarray:
    out = []
    for row in M:
        s = propagate_hierarchy({c for c, v in zip(classes, row) if v})
        out.append([1 if c in s else 0 for c in classes])
    return np.array(out, dtype=int)


def fit_probe(Xtr, ytr, Xdev, ydev, seed):
    """One logistic regression per class; C chosen on dev by AUROC."""
    best, best_auc = None, -1.0
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                 random_state=seed)
        clf.fit(Xtr, ytr)
        if len(np.unique(ydev)) < 2:
            return clf, C
        auc = roc_auc_score(ydev, clf.decision_function(Xdev))
        if auc > best_auc:
            best, best_auc, best_C = clf, auc, C
    return best, best_C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="8bit", help="feature dir under exp6_features/")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="additional feature dirs to probe (e.g. 4bit randinit)")
    ap.add_argument("--stage-a", default="predictions_test_twostage_mg_stageA.csv",
                    help="the VLM's own perception pass = the comparator")
    ap.add_argument("--classes", default="all", choices=["all", "reliable"])
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out-suffix", default="exp6")
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    classes = ALL if args.classes == "all" else RELIABLE_CLASSES
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    test_uids = list(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    dev_uids = list(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    train_uids = [u for u in studies.uid if u not in set(test_uids) | set(dev_uids)]
    stage_a = pd.read_csv(os.path.join(out, args.stage_a))
    rng = np.random.default_rng(args.seed)

    rows, per_class_rows = [], []
    for tag in [args.features] + list(args.compare):
        feat_dir = os.path.join(out, "exp6_features", tag)
        if not os.path.isdir(feat_dir):
            print(f"[exp6] skip {tag}: {feat_dir} missing"); continue
        Ftr, utr = load_features(feat_dir, train_uids)
        Fdev, ude = load_features(feat_dir, dev_uids)
        Fte, ute = load_features(feat_dir, test_uids)
        ytr = label_matrix(studies, utr, classes)
        ydev = label_matrix(studies, ude, classes)
        yte = label_matrix(studies, ute, classes)
        raw_vlm, prop_vlm = vlm_prediction_matrix(stage_a, ute, classes)
        scored = [i for i, r in enumerate(raw_vlm) if r is not None]
        print(f"\n[exp6] {tag}: train {len(utr)} dev {len(ude)} test {len(ute)} "
              f"({len(scored)} with a VLM prediction)")

        # the VLM's own numbers on exactly these studies = the thing to beat
        yv_raw = np.array([raw_vlm[i] for i in scored])
        yv = propagate_matrix(yv_raw, classes)
        yt_s = yte[scored]
        vp, vr, vf1 = micro_f1(yt_s, yv)
        print(f"  VLM Stage A: P {vp:.3f} R {vr:.3f} micro-F1 {vf1:.4f} "
              f"({yv_raw.sum()} calls)")

        sites = [s for s in Ftr if s.startswith("layer_")]
        sites = sorted(sites, key=lambda s: int(s.split("_")[1])) + \
            [s for s in ("proj", "pixels") if s in Ftr]

        for site in sites:
            scaler = StandardScaler().fit(Ftr[site])
            Xtr, Xdev, Xte = (scaler.transform(Ftr[site]),
                              scaler.transform(Fdev[site]),
                              scaler.transform(Fte[site]))
            scores = np.zeros((len(ute), len(classes)))
            aucs, shuf_aucs = [], []
            for ci, cls in enumerate(classes):
                if ytr[:, ci].sum() < 2:
                    scores[:, ci] = -1e9; aucs.append(np.nan); shuf_aucs.append(np.nan)
                    continue
                clf, C = fit_probe(Xtr, ytr[:, ci], Xdev, ydev[:, ci], args.seed)
                scores[:, ci] = clf.decision_function(Xte)
                npos = int(yte[:, ci].sum())
                auc = (roc_auc_score(yte[:, ci], scores[:, ci])
                       if npos >= MIN_POS and npos < len(ute) else np.nan)
                aucs.append(auc)
                # label-shuffle control: same pipeline, permuted training labels
                sh = fit_probe(Xtr, rng.permutation(ytr[:, ci]), Xdev,
                               ydev[:, ci], args.seed)[0]
                shuf_aucs.append(roc_auc_score(yte[:, ci], sh.decision_function(Xte))
                                 if npos >= MIN_POS and npos < len(ute) else np.nan)
                per_class_rows.append(dict(features=tag, site=site, cls=cls, C=C,
                                           n_test_pos=npos, auroc=auc,
                                           auroc_shuffled=shuf_aucs[-1]))

            # MATCHED COMMITMENT: give the probe exactly the VLM's budget per class
            yp_raw = np.zeros_like(yv_raw)
            for ci in range(len(classes)):
                k = int(yv_raw[:, ci].sum())
                if k == 0:
                    continue
                sc = scores[scored, ci]
                yp_raw[np.argsort(-sc)[:k], ci] = 1
            yp = propagate_matrix(yp_raw, classes)
            pp, pr, pf1 = micro_f1(yt_s, yp)

            # does the probe recover what the VLM MISSED? (the non-confounded form
            # of the TP-vs-FN comparison: severity confounds a raw separability test)
            miss = (yt_s == 1) & (yv == 0)
            recovered = int((miss & (yp == 1)).sum())
            rows.append(dict(
                features=tag, site=site,
                macro_auroc=float(np.nanmean(aucs)),
                macro_auroc_shuffled=float(np.nanmean(shuf_aucs)),
                probe_P=pp, probe_R=pr, probe_F1=pf1,
                vlm_P=vp, vlm_R=vr, vlm_F1=vf1, delta_F1=pf1 - vf1,
                n_vlm_misses=int(miss.sum()), n_recovered=recovered,
                frac_misses_recovered=recovered / miss.sum() if miss.sum() else np.nan))
            print(f"  {site:>10}  AUROC {np.nanmean(aucs):.3f} "
                  f"(shuf {np.nanmean(shuf_aucs):.3f})  "
                  f"matched-F1 {pf1:.4f} (vs VLM {vf1:.4f}, "
                  f"{pf1 - vf1:+.4f})  recovers {recovered}/{int(miss.sum())} misses",
                  flush=True)

    if not rows:
        raise SystemExit("[exp6] no features probed")
    df = pd.DataFrame(rows)
    pc = pd.DataFrame(per_class_rows)
    f1 = os.path.join(out, f"{args.out_suffix}_probe_results.csv")
    f2 = os.path.join(out, f"{args.out_suffix}_probe_per_class.csv")
    df.to_csv(f1, index=False); pc.to_csv(f2, index=False)
    print(f"\n[exp6] wrote {f1}\n[exp6] wrote {f2}")

    main_tag = args.features
    m = df[df.features == main_tag]
    if len(m):
        best = m.loc[m.probe_F1.idxmax()]
        print(f"\n[exp6] best site for {main_tag}: {best.site} "
              f"matched-F1 {best.probe_F1:.4f} vs VLM {best.vlm_F1:.4f}")
        proj = m[m.site == "proj"]
        # A verdict is only meaningful with a real test set and scorable classes.
        # Small pilots would otherwise print "OUTCOME B" purely from noise.
        n_scored = int(pc[pc.features == main_tag].auroc.notna().sum())
        powered = len(ute) >= 200 and n_scored > 0
        if len(proj) and powered:
            p = proj.iloc[0]
            verdict = ("OUTCOME A - information reaches the language model and is "
                       "NOT used => readout/alignment bottleneck"
                       if p.probe_F1 > p.vlm_F1 else
                       "OUTCOME B - information is not linearly present at the "
                       "language interface => representation bottleneck")
            print(f"[exp6] at the vision->language interface (proj): "
                  f"probe {p.probe_F1:.4f} vs VLM {p.vlm_F1:.4f}\n[exp6] {verdict}")
        elif len(proj):
            print(f"[exp6] UNDERPOWERED (test n={len(ute)}, scorable classes "
                  f"{n_scored}) - verdict withheld. This is a code check, not a "
                  f"result; do not read the numbers above.")
        if powered:
            print("[exp6] NOTE: read this against the pixels/randinit controls "
                  "before concluding anything.")


if __name__ == "__main__":
    main()
