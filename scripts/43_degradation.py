"""Experiment 3 — controlled visual degradation, with an information floor.

Experiment 2 found the model insensitive to resolution: 256px reads as well as
1024px. That sharpens the question from "how much detail helps?" to "is the image
being used at all?" A degradation ladder cannot answer that without a floor, so
`blank` (a uniform grey field) is included: whatever it scores is what the model
obtains from priors alone — indication text, base rates, prompt — with no image
evidence. Every rung is 1024px, one image part; only content is degraded.

The decisive quantity is not any single rung but the SPREAD between baseline and
blank. If it is small, image evidence contributes little regardless of quality.
"""
import argparse, os
import numpy as np, pandas as pd
from _bootstrap import load_cfg
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from src.eval_accuracy import _pred_set, _truth_positive_set, propagate_hierarchy
from src.grouping import any_acute
from src.labeling import CHEXPERT_CLASSES

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
RUNGS = [("baseline",   "exp1f_gem_baseline_f"),
         ("blur σ=4",   "exp3_blur4_f"),
         ("blur σ=12",  "exp3_blur12_f"),
         ("scramble",   "exp3_scramble_f"),
         ("blank",      "exp3_blank_f")]


def micro(T, P, us):
    tp = fp = fn = 0
    for u in us:
        for c in ALL:
            t, p = c in T[u], c in P[u]
            tp += t and p; fp += (not t) and p; fn += t and (not p)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids-file", default="results/exp2_subset_uids.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = load_cfg()["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    keep = set(pd.read_csv(args.uids_file).uid)

    d = {}
    for lab, stem in RUNGS:
        f = os.path.join(out, f"predictions_test_{stem}_stageA.csv")
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
        d[lab] = pd.read_csv(f).set_index("uid")
    uids = sorted(set.intersection(*[set(x.index) for x in d.values()]) & keep & set(s.index))
    T = {u: propagate_hierarchy(_truth_positive_set(s.loc[u], "ones", "lblcx_")) for u in uids}
    yt = np.array([1 if any_acute(T[u]) else 0 for u in uids])

    print(f"n={len(uids)} · identical studies at every rung · abnormal base rate {yt.mean():.3f}\n")
    print(f"{'rung':<11}{'13cls F1':>10}{'prec':>8}{'rec':>8}{'binAcc':>9}"
          f"{'find/std':>10}{'FP/std':>9}{'%abn':>7}")
    P, C = {}, {}
    for lab, _ in RUNGS:
        Pm = {u: propagate_hierarchy(_pred_set(d[lab].loc[u])) for u in uids}
        pr, rc, f1, fp = micro(T, Pm, uids)
        yp = np.array([1 if any_acute(Pm[u]) else 0 for u in uids])
        P[lab] = Pm; C[lab] = (yp == yt)
        print(f"{lab:<11}{f1:>10.4f}{pr:>8.4f}{rc:>8.4f}{(yp==yt).mean():>9.4f}"
              f"{np.mean([len(Pm[u]) for u in uids]):>10.3f}{fp/len(uids):>9.3f}{yp.mean():>7.1%}")

    rng = np.random.default_rng(20260705)
    boots = [rng.integers(0, len(uids), len(uids)) for _ in range(args.n_boot)]
    base = P["baseline"]; f1b = micro(T, base, uids)[2]
    print(f"\nΔ13-class F1 vs baseline (paired bootstrap) and paired McNemar on triage:")
    pv, rows = [], []
    for lab, _ in RUNGS[1:]:
        dd = micro(T, P[lab], uids)[2] - f1b
        ds = [micro(T, P[lab], [uids[i] for i in b])[2] - micro(T, base, [uids[i] for i in b])[2]
              for b in boots]
        lo, hi = np.percentile(ds, [2.5, 97.5])
        a, b_ = C["baseline"], C[lab]
        n01 = int((a & ~b_).sum()); n10 = int((~a & b_).sum())
        p = mcnemar([[0, n01], [n10, 0]], exact=False, correction=True).pvalue
        pv.append(p); rows.append((lab, dd, lo, hi, p))
    fdr = multipletests(pv, method="fdr_bh")[1]
    for (lab, dd, lo, hi, p), q in zip(rows, fdr):
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {lab:<11} ΔF1={dd:+.4f} CI [{lo:+.4f},{hi:+.4f}] {sig:<5}"
              f" | McNemar p={p:.4f} → FDR {q:.4f} {'SIG' if q<0.05 else 'n.s.'}")

    span = f1b - micro(T, P["blank"], uids)[2]
    print(f"\nINFORMATION SPAN (baseline − blank): ΔF1 = {span:+.4f}, "
          f"binary acc {C['baseline'].mean():.4f} → {C['blank'].mean():.4f} "
          f"({C['baseline'].mean()-C['blank'].mean():+.4f})")
    print("  = how much the image is worth, over and above priors alone.")


if __name__ == "__main__":
    main()
