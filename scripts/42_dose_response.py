"""Experiment 2 — resolution dose-response on a fixed stratified subset.

Experiment 1 arm B tested a single upward step (1024 -> 2048) and found nothing.
That is ambiguous: the model may SATURATE at or below 1024 (so extra pixels are
wasted but resolution still matters below some threshold), or it may be largely
INSENSITIVE to resolution at any level. Only going downward separates them.

MEASURED before running: Gemini's image payload is QUANTIZED, not continuous --
256px -> 610 tokens, 512/768/1024/1536 -> 1642 (all identical), 2048 -> 2932. So
there are three effective doses, not six. The sweep measures 256 and 512 (new) and
reuses the existing 1024 / 2048 runs restricted to the same studies; 512 is
included precisely to confirm the plateau at the OUTCOME level, not merely in the
token count.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from _bootstrap import load_cfg
from src.eval_accuracy import _pred_set, _truth_positive_set, propagate_hierarchy
from src.grouping import any_acute
from src.labeling import CHEXPERT_CLASSES

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
# (label, px, predictions-file stem)
RUNGS = [("256px",  256,  "exp2_res256_f"),
         ("512px",  512,  "exp2_res512_f"),
         ("1024px", 1024, "exp1f_gem_baseline_f"),
         ("2048px", 2048, "exp1f_gem_highres_f")]


def micro(T, P, uids):
    tp = fp = fn = 0
    for u in uids:
        for c in ALL:
            t, p = c in T[u], c in P[u]
            tp += t and p; fp += (not t) and p; fn += t and (not p)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--uids-file", default="results/exp2_subset_uids.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = load_cfg()["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    keep = set(pd.read_csv(args.uids_file).uid)

    dfs = {}
    for lab, px, stem in RUNGS:
        f = os.path.join(out, f"predictions_{args.split}_{stem}_stageA.csv")
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
        dfs[lab] = pd.read_csv(f).set_index("uid")
    uids = sorted(set.intersection(*[set(d.index) for d in dfs.values()]) & keep & set(s.index))
    T = {u: propagate_hierarchy(_truth_positive_set(s.loc[u], "ones", "lblcx_")) for u in uids}
    yt = np.array([1 if any_acute(T[u]) else 0 for u in uids])

    print(f"n={len(uids)} (fixed stratified subset, identical studies at every rung)\n")
    print(f"{'rung':<9}{'tokens':>8}{'13cls F1':>10}{'prec':>8}{'rec':>8}"
          f"{'binAcc':>9}{'find/std':>10}{'FP/std':>9}")
    res = {}
    for lab, px, _ in RUNGS:
        d = dfs[lab]
        P = {u: propagate_hierarchy(_pred_set(d.loc[u])) for u in uids}
        pr, rc, f1, fp = micro(T, P, uids)
        yp = np.array([1 if any_acute(P[u]) else 0 for u in uids])
        tok = pd.to_numeric(d.loc[uids, "prompt_tokens"], errors="coerce").median()
        res[lab] = (P, f1, (yp == yt).mean())
        print(f"{lab:<9}{tok:>8.0f}{f1:>10.4f}{pr:>8.4f}{rc:>8.4f}"
              f"{(yp==yt).mean():>9.4f}{np.mean([len(P[u]) for u in uids]):>10.3f}"
              f"{fp/len(uids):>9.3f}")

    # every rung vs the 1024 reference, paired bootstrap on micro-F1
    print(f"\nΔ13-class F1 vs the 1024px reference (paired bootstrap, {args.n_boot}x):")
    rng = np.random.default_rng(20260705)
    boots_idx = [rng.integers(0, len(uids), len(uids)) for _ in range(args.n_boot)]
    ref = res["1024px"][0]
    for lab, px, _ in RUNGS:
        if lab == "1024px":
            continue
        P = res[lab][0]
        d = micro(T, P, uids)[2] - micro(T, ref, uids)[2]
        ds = []
        for b in boots_idx:
            su = [uids[i] for i in b]
            ds.append(micro(T, P, su)[2] - micro(T, ref, su)[2])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {lab:<8} Δ={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")


if __name__ == "__main__":
    main()
