"""Condition D (oracle) — difference-in-differences analysis.

Attaching the oracle crop costs payload: a 2-part request drops from ~1643 to ~880
prompt tokens, because Gemini only grants the single-image MEDIA_RESOLUTION_HIGH
budget to a lone image. So a straight D-vs-baseline comparison confounds "was told
where to look" with "was given less to look at".

The fix is built into the arm. Every study receives exactly one crop; for ~30% the
zone is the CORRECT one (parsed from the report's MeSH anatomy) and for the rest it
is an ARBITRARY zone drawn from the matched marginal. Both subgroups take the
identical payload hit, so it cancels in the difference:

    DiD = E[D - A | correct zone] - E[D - A | arbitrary zone]

A positive DiD means being handed the right region genuinely helps. Note the two
subgroups differ in case mix (localizable studies are abnormal by construction),
which is exactly why the DiD is used rather than a raw subgroup comparison -- each
subgroup is its own paired control against baseline.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from _bootstrap import load_cfg
from src import imaging
from src.eval_accuracy import _pred_set, _truth_positive_set, propagate_hierarchy
from src.grouping import any_acute
from src.labeling import CHEXPERT_CLASSES

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--prefix", default="exp1f_gem")
    ap.add_argument("--base", default="baseline_f")
    ap.add_argument("--oracle", default="oracle_f")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    out = load_cfg()["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")

    def load(arm):
        return pd.read_csv(os.path.join(
            out, f"predictions_{args.split}_{args.prefix}_{arm}_stageA.csv")).set_index("uid")
    A, D = load(args.base), load(args.oracle)

    man = pd.read_csv(imaging.manifest_path(out, args.oracle))
    src = (man[man.kind == "crop"].set_index("uid")["zone_source"]).to_dict()
    uids = sorted(set(A.index) & set(D.index) & set(s.index) & set(src))

    T = {u: propagate_hierarchy(_truth_positive_set(s.loc[u], "ones", "lblcx_")) for u in uids}
    PA = {u: propagate_hierarchy(_pred_set(A.loc[u])) for u in uids}
    PD = {u: propagate_hierarchy(_pred_set(D.loc[u])) for u in uids}
    yt = np.array([1 if any_acute(T[u]) else 0 for u in uids])
    ca = np.array([1 if (1 if any_acute(PA[u]) else 0) == yt[i] else 0 for i, u in enumerate(uids)])
    cd = np.array([1 if (1 if any_acute(PD[u]) else 0) == yt[i] else 0 for i, u in enumerate(uids)])
    # per-study finding overlap (Jaccard) as a finer-grained outcome than triage
    def jac(P, u):
        t, p = T[u], P[u]
        return 1.0 if not t and not p else len(t & p) / max(1, len(t | p))
    ja = np.array([jac(PA, u) for u in uids]); jd = np.array([jac(PD, u) for u in uids])
    loc = np.array([src[u] == "localizable" for u in uids])

    print(f"n={len(uids)}   correct-zone (localizable) {loc.sum()}   "
          f"arbitrary-zone (assigned) {(~loc).sum()}")
    tokA = pd.to_numeric(A.loc[uids, "prompt_tokens"], errors="coerce")
    tokD = pd.to_numeric(D.loc[uids, "prompt_tokens"], errors="coerce")
    print(f"payload: baseline {tokA.median():.0f} -> oracle {tokD.median():.0f} tokens "
          f"(identical hit in both subgroups: "
          f"{tokD[loc].median():.0f} vs {tokD[~loc].median():.0f})\n")

    rng = np.random.default_rng(20260705)
    for name, va, vd in (("binary-triage correctness", ca, cd),
                         ("per-study finding Jaccard", ja, jd)):
        d = vd - va
        dl, da_ = d[loc].mean(), d[~loc].mean()
        did = dl - da_
        li, ai = np.where(loc)[0], np.where(~loc)[0]
        boots = [ (d[rng.choice(li, li.size)].mean() - d[rng.choice(ai, ai.size)].mean())
                  for _ in range(args.n_boot) ]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * min((np.array(boots) <= 0).mean(), (np.array(boots) >= 0).mean())
        print(f"{name}")
        print(f"  baseline: correct-zone {va[loc].mean():.4f} | arbitrary {va[~loc].mean():.4f}")
        print(f"  oracle  : correct-zone {vd[loc].mean():.4f} | arbitrary {vd[~loc].mean():.4f}")
        print(f"  Δ(D-A)  : correct-zone {dl:+.4f} | arbitrary {da_:+.4f}")
        print(f"  DiD = {did:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p={p:.3f}   "
              f"=> {'SIGNIFICANT' if (lo > 0 or hi < 0) else 'not significant'}\n")


if __name__ == "__main__":
    main()
