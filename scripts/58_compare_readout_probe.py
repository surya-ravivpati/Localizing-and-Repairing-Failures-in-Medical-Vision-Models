"""Paired base-vs-adapter comparison on the SAME probe studies (scripts 57 outputs).

On 80 studies a point AUROC difference means little, so resample STUDIES with
replacement (never cells: the 13 findings of one study are not independent) and
report the pooled-cell AUROC difference with a 95% interval, plus per-class AUROC
side by side and the call-rate shift.

  python3 scripts/58_compare_readout_probe.py --a base --b readout_probe
"""
import argparse
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from sklearn.metrics import roc_auc_score

from src.readout_ft import FINDABLE, truth_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="base")
    ap.add_argument("--b", default="readout_probe")
    ap.add_argument("--split", default="probe")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    out = load_cfg("configs/config_medgemma_estr.yaml")["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    A = pd.read_csv(os.path.join(out, f"readout_slots_{args.a}_{args.split}.csv")).set_index("uid")
    B = pd.read_csv(os.path.join(out, f"readout_slots_{args.b}_{args.split}.csv")).set_index("uid")
    uids = [u for u in A.index if u in B.index]
    cols = [f"p_{c}" for c in FINDABLE]
    PA, PB = A.loc[uids, cols].to_numpy(), B.loc[uids, cols].to_numpy()
    T = np.array([[1 if c in truth_set(studies.loc[u]) else 0 for c in FINDABLE] for u in uids])

    print(f"n={len(uids)} paired studies, {T.sum()} positive cells\n")
    print(f"{'finding':<28}{'pos':>4} {'AUROC ' + args.a:>14} {'AUROC ' + args.b:>22} "
          f"{'yes% ' + args.a:>10} {'yes% ' + args.b:>10}")
    for j, c in enumerate(FINDABLE):
        ok = 0 < T[:, j].sum() < len(T)
        aa = roc_auc_score(T[:, j], PA[:, j]) if ok else np.nan
        ab = roc_auc_score(T[:, j], PB[:, j]) if ok else np.nan
        print(f"{c:<28}{T[:, j].sum():>4} {aa:>14.3f} {ab:>22.3f} "
              f"{(PA[:, j] >= .5).mean():>10.0%} {(PB[:, j] >= .5).mean():>10.0%}")

    pooled = lambda P, idx: roc_auc_score(T[idx].ravel(), P[idx].ravel())
    full = np.arange(len(uids))
    d0 = pooled(PB, full) - pooled(PA, full)
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(args.boot):
        idx = rng.integers(0, len(uids), len(uids))
        if T[idx].sum() == 0:
            continue
        ds.append(pooled(PB, idx) - pooled(PA, idx))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"\npooled-cell AUROC  {args.a} {pooled(PA, full):.3f}  {args.b} {pooled(PB, full):.3f}"
          f"  delta {d0:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"({'significant' if lo > 0 or hi < 0 else 'n.s.'})")
    # MACRO AUROC = mean of within-class AUROCs. This is the metric that matters
    # for a per-class-threshold readout: pooled-cell AUROC also rewards rescaling
    # one class's scores relative to another's, which per-class thresholds give
    # for free. (Learned the hard way: the gentle sweep's best pooled checkpoint
    # was WORSE than base within most classes.)
    def macro(P, idx, min_pos=5):
        v = [roc_auc_score(T[idx, j], P[idx, j]) for j in range(T.shape[1])
             if T[idx, j].sum() >= min_pos and T[idx, j].sum() < len(idx)]
        return np.mean(v) if v else np.nan
    m0 = macro(PB, full) - macro(PA, full)
    ms = []
    rng = np.random.default_rng(1)
    for _ in range(args.boot):
        idx = rng.integers(0, len(uids), len(uids))
        a, b = macro(PA, idx), macro(PB, idx)
        if not (np.isnan(a) or np.isnan(b)):
            ms.append(b - a)
    mlo, mhi = np.percentile(ms, [2.5, 97.5])
    print(f"macro AUROC        {args.a} {macro(PA, full):.3f}  {args.b} {macro(PB, full):.3f}"
          f"  delta {m0:+.3f}  95% CI [{mlo:+.3f}, {mhi:+.3f}]  "
          f"({'significant' if mlo > 0 or mhi < 0 else 'n.s.'})")
    print(f"greedy yes-rate    {args.a} {(PA >= .5).mean():.1%}  {args.b} {(PB >= .5).mean():.1%}"
          f"  truth {T.mean():.1%}")
    print(f"distinct patterns  {args.a} {len({tuple(r) for r in PA >= .5})}  "
          f"{args.b} {len({tuple(r) for r in PB >= .5})}  of {len(uids)}")


if __name__ == "__main__":
    main()
