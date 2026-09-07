"""Experiment 6c, stage 2: which of the three failure modes is it?

Section 9 places the diagnostic information at the vision-language interface and
shows the model does not express all of it. Three mechanisms remain:

  A  present, but the language model does not attend to the right visual tokens
  B  attended to, but the visual->clinical mapping is weak
  C  internally recoverable at the decision point, but generation fails to express it

`52_extract_llm_states.py` recorded the model's own hidden states at two places per
block. Probing them separates the modes on evidence rather than argument:

  img_L{k}     what the language model has made of the 256 image tokens by block k
  last_L{k}    the final position at block k -- the vector generation runs from
  last_final   the same after the output norm: the model's last word before decoding

DECISION RULE, fixed before looking:
  * If `last_*` recovers a comparable share of the model's misses to the interface
    probe, the concept reaches the decision point and is not expressed -> MODE C.
    Attention/connector tuning would then be aimed upstream of the real failure.
  * If `last_*` recovers substantially fewer, the information is present at the
    interface but not carried to the decision point -> MODE A or B, and the
    connector/attention interventions are the indicated fix.

The headline is not AUROC. It is the 129 findings that the interface probe recovered
from the model's own 621 misses (section 9.6): how many of those exact cases survive
to each depth. That set is the one an alignment intervention would have to convert.

Usage
  python3 scripts/53_probe_llm_states.py --features llm8bit --interface 8bit
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
warnings.filterwarnings("ignore", message="Mean of empty slice")

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
C_GRID = [0.001, 0.01, 0.1, 1.0]
MIN_POS = 10


def load_site(feat_dir, uids, site):
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


def fit_site(Xtr, ytr, Xde, yde, Xte, seed):
    """Per-class logistic regression, C chosen on dev; returns test decision scores."""
    scores = np.zeros((Xte.shape[0], len(ALL)))
    for ci in range(len(ALL)):
        if ytr[:, ci].sum() < 2 or len(np.unique(yde[:, ci])) < 2:
            scores[:, ci] = -1e9
            continue
        best, ba = None, -1
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                     random_state=seed).fit(Xtr, ytr[:, ci])
            a = roc_auc_score(yde[:, ci], clf.decision_function(Xde))
            if a > ba:
                best, ba = clf, a
        scores[:, ci] = best.decision_function(Xte)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="llm8bit", help="LLM-state feature dir")
    ap.add_argument("--interface", default="8bit", help="vision-side dir, for `proj`")
    ap.add_argument("--stage-a", default="predictions_test_twostage_mg_stageA.csv")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out-suffix", default="exp6c")
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    fd = os.path.join(out, "exp6_features", args.features)
    ifd = os.path.join(out, "exp6_features", args.interface)
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    te_u = list(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    de_u = list(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    tr_u = [u for u in studies.uid if u not in set(te_u) | set(de_u)]
    stage_a = pd.read_csv(os.path.join(out, args.stage_a)).set_index("uid")

    probe_sites = sorted(
        [k for k in np.load(os.path.join(fd, f"{te_u[0]}.npz")).files
         if k != "n_image_tokens"],
        key=lambda s: (not s.startswith("img"), s))
    _, ute = load_site(fd, te_u, "last_final")
    yte = truth_matrix(studies, ute, ALL)
    print(f"[6c] {len(ute)} test studies | sites: {probe_sites}")

    vlm_raw = np.array([[1 if c in _pred_set(stage_a.loc[u]) else 0 for c in ALL]
                        if u in stage_a.index else [0] * len(ALL) for u in ute])
    vlm = propagate_matrix(vlm_raw, ALL)
    vp, vr, vf = micro(yte, vlm)
    miss = (yte == 1) & (vlm == 0)
    print(f"[6c] model Stage A: P {vp:.3f} R {vr:.3f} F1 {vf:.4f} | "
          f"{int(miss.sum())} missed finding-instances")

    def matched(scores):
        yp = np.zeros_like(vlm_raw)
        for ci in range(len(ALL)):
            k = int(vlm_raw[:, ci].sum())
            if k:
                yp[np.argsort(-scores[:, ci])[:k], ci] = 1
        return propagate_matrix(yp, ALL)

    # ---- the interface probe, refit here so the recovered set is exactly comparable
    Xtr_i, utr_i = load_site(ifd, tr_u, "proj")
    Xde_i, ude_i = load_site(ifd, de_u, "proj")
    Xte_i, ute_i = load_site(ifd, te_u, "proj")
    assert ute_i == ute, "interface and LLM feature dirs disagree on test order"
    sc = StandardScaler().fit(Xtr_i)
    iface = matched(fit_site(sc.transform(Xtr_i), truth_matrix(studies, utr_i, ALL),
                             sc.transform(Xde_i), truth_matrix(studies, ude_i, ALL),
                             sc.transform(Xte_i), args.seed))
    iface_rec = miss & (iface == 1)
    N_IFACE = int(iface_rec.sum())
    print(f"[6c] interface probe recovers {N_IFACE} of {int(miss.sum())} misses "
          f"— this is the set to track\n")

    rows = []
    for site in probe_sites:
        Xtr, utr = load_site(fd, tr_u, site)
        Xde, ude = load_site(fd, de_u, site)
        Xte, _ = load_site(fd, te_u, site)
        s2 = StandardScaler().fit(Xtr)
        scores = fit_site(s2.transform(Xtr), truth_matrix(studies, utr, ALL),
                          s2.transform(Xde), truth_matrix(studies, ude, ALL),
                          s2.transform(Xte), args.seed)
        aucs = [roc_auc_score(yte[:, ci], scores[:, ci])
                for ci in range(len(ALL))
                if MIN_POS <= yte[:, ci].sum() < len(ute) and scores[0, ci] != -1e9]
        yp = matched(scores)
        p, r, f = micro(yte, yp)
        rec = int((miss & (yp == 1)).sum())
        kept = int((iface_rec & (yp == 1)).sum())
        rows.append(dict(site=site, macro_auroc=float(np.mean(aucs)),
                         matched_F1=f, prec=p, rec_=r,
                         misses_recovered=rec,
                         of_interface_129=kept,
                         frac_of_interface=kept / N_IFACE if N_IFACE else np.nan))
        print(f"  {site:>11}  AUROC {np.mean(aucs):.3f}  matched-F1 {f:.4f}  "
              f"recovers {rec:>3}/{int(miss.sum())}  "
              f"keeps {kept:>3}/{N_IFACE} of the interface set "
              f"({100*kept/max(N_IFACE,1):.0f}%)", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(out, f"{args.out_suffix}_llm_probe.csv")
    df.to_csv(path, index=False)
    print(f"\n[6c] wrote {path}")

    last = df[df.site.str.startswith("last")]
    if len(last):
        best = last.loc[last.frac_of_interface.idxmax()]
        print(f"\n[6c] best decision-point site: {best.site} keeps "
              f"{int(best.of_interface_129)}/{N_IFACE} "
              f"({100*best.frac_of_interface:.0f}%) of the interface-recovered set")
        print("[6c] READ THIS AGAINST THE PRE-SET RULE: a high share => the concept "
              "reaches the decision point and is not expressed (mode C); a low share "
              "=> it is not carried there (mode A/B).")


if __name__ == "__main__":
    main()
