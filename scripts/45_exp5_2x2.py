"""Experiment 5 — visual information x reasoning effort, a 2x2.

The paper's thesis is that error is perception-bound and the reasoning stage is a
faithful pass-through. That predicts something specific and falsifiable: varying
reasoning effort should do little at EITHER visual level, and crucially should not
help MORE when vision is degraded (no interaction). If instead elaborated reasoning
rescues degraded perception, reasoning is compensatory and the thesis is too strong.

Visual factor  : Stage-A findings from the intact 1024px read vs the blur12 read.
Reasoning factor: Stage B commits directly, vs reasons explicitly before committing.
Stage A is byte-identical within each visual level, so the factors are cleanly crossed.
"""
import argparse, json, os
import numpy as np, pandas as pd
from _bootstrap import load_cfg
from statsmodels.stats.contingency_tables import mcnemar
from src.eval_accuracy import _truth_positive_set, propagate_hierarchy
from src.grouping import any_acute
from src.normalize import normalize_dx

CELLS = {("intact","direct"):   "exp1f_gem_baseline_f",
         ("intact","elaborated"):"exp5_exp1f_gem_baseline_f_cot",
         ("degraded","direct"):  "exp3_blur12_f",
         ("degraded","elaborated"):"exp5_exp3_blur12_f_cot"}


def dx_set(row):
    try: o = json.loads(row["response"])
    except Exception: o = {}
    return propagate_hierarchy({x for x in normalize_dx(str(o.get("primary_diagnosis","") or ""))
                                if x != "No Finding"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids-file", default="results/exp2_subset_uids.csv")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    out = load_cfg()["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out,"studies_labeled.csv")).set_index("uid")
    keep = set(pd.read_csv(args.uids_file).uid)

    d = {}
    for k, stem in CELLS.items():
        f = os.path.join(out, f"predictions_test_{stem}_stageB.csv")
        if not os.path.exists(f): raise SystemExit(f"missing {f}")
        d[k] = pd.read_csv(f).set_index("uid")
    uids = sorted(set.intersection(*[set(x.index) for x in d.values()]) & keep & set(s.index))
    T = {u: propagate_hierarchy(_truth_positive_set(s.loc[u],"ones","lblcx_")) for u in uids}
    yt = np.array([1 if any_acute(T[u]) else 0 for u in uids])

    C = {}
    print(f"n={len(uids)} paired studies · abnormal base rate {yt.mean():.3f}\n")
    print(f"{'visual':<10}{'reasoning':<13}{'bin acc':>9}{'flags abn':>11}")
    for k in CELLS:
        yp = np.array([1 if any_acute(dx_set(d[k].loc[u])) else 0 for u in uids])
        C[k] = (yp == yt)
        print(f"{k[0]:<10}{k[1]:<13}{C[k].mean():>9.4f}{yp.mean():>11.1%}")

    rng = np.random.default_rng(20260705)
    bs = [rng.integers(0,len(uids),len(uids)) for _ in range(args.n_boot)]
    def ci(v):
        a = np.array([v[b].mean() for b in bs]); return np.percentile(a,[2.5,97.5])

    vis = ((C[("intact","direct")].astype(int)+C[("intact","elaborated")].astype(int))
           -(C[("degraded","direct")].astype(int)+C[("degraded","elaborated")].astype(int)))/2
    rea = ((C[("intact","elaborated")].astype(int)+C[("degraded","elaborated")].astype(int))
           -(C[("intact","direct")].astype(int)+C[("degraded","direct")].astype(int)))/2
    inter = ((C[("degraded","elaborated")].astype(int)-C[("degraded","direct")].astype(int))
             -(C[("intact","elaborated")].astype(int)-C[("intact","direct")].astype(int)))

    print("\n2x2 effects on binary-triage accuracy (paired bootstrap):")
    for name, v in (("main effect · VISUAL (intact − degraded)", vis),
                    ("main effect · REASONING (elaborated − direct)", rea),
                    ("INTERACTION (reasoning gain when degraded − when intact)", inter)):
        lo,hi = ci(v)
        print(f"  {name:<56} {v.mean():+.4f}  CI [{lo:+.4f},{hi:+.4f}]  "
              f"{'SIGNIFICANT' if (lo>0 or hi<0) else 'n.s.'}")

    print("\nSimple effects of reasoning, within each visual level (paired McNemar):")
    for lvl in ("intact","degraded"):
        a,b = C[(lvl,"direct")], C[(lvl,"elaborated")]
        n01=int((a&~b).sum()); n10=int((~a&b).sum())
        p=mcnemar([[0,n01],[n10,0]],exact=False,correction=True).pvalue
        print(f"  {lvl:<9} {a.mean():.4f} → {b.mean():.4f} ({b.mean()-a.mean():+.4f})  "
              f"n01={n01} n10={n10}  p={p:.4f}  {'SIG' if p<0.05 else 'n.s.'}")


if __name__ == "__main__":
    main()
