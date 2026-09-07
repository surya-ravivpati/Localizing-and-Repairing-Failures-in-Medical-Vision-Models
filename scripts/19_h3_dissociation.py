"""H3 as pre-registered: persuasiveness/coherence UP while faithfulness FLAT.

Protocol §2 specifies "Dissociation test: significant on judge score, null on
faithfulness". The previous implementation (src.stats.dissociation_test) compared
group MEANS with no significance test, so it could not evaluate that claim — and it
was run on a judge output that was 98.7% missing (now quarantined in results/invalid/).

This version does the paired test the protocol asks for: for each prompt vs `direct`,
paired Wilcoxon on (a) the persuasiveness axis and (b) the faithfulness axis, on the
same cases. H3 is supported for a prompt when (a) is significantly positive and (b) is
NOT significantly positive.
"""
import argparse
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from scipy.stats import wilcoxon

BASE = "direct"
PERSUASIVE = ["score_diagnostic_coherence", "score_logical_consistency"]
FAITHFUL = ["score_faithfulness", "score_evidence_identification"]


def paired(judge, dim, cond, base=BASE):
    """Paired Wilcoxon of `cond` vs `base` on `dim`, over shared uids."""
    a = judge[judge.prompt_condition == cond].set_index("uid")[dim]
    b = judge[judge.prompt_condition == base].set_index("uid")[dim]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].astype(float), b.loc[common].astype(float)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    if len(a) < 10 or (a - b).abs().sum() == 0:
        return np.nan, np.nan, len(a)
    stat, p = wilcoxon(a, b)
    return float(a.mean() - b.mean()), float(p), int(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    judge = pd.read_csv(os.path.join(out, f"judge_scores_{args.split}.csv"))
    ground = pd.read_csv(os.path.join(out, f"groundedness_{args.split}.csv"))

    scored = judge["score_faithfulness"].notna().mean()
    print(f"judge rows={len(judge)}  scored={100*scored:.1f}%")
    if scored < 0.9:
        raise SystemExit("judge output is incomplete (<90% scored) — do not interpret; "
                         "re-run scripts/18_run_judge.py")

    conds = [c for c in judge.prompt_condition.unique() if c != BASE]
    print(f"\n=== H3 DISSOCIATION vs '{BASE}' (paired Wilcoxon) ===")
    print("H3 holds when PERSUASIVENESS rises significantly while FAITHFULNESS does not.\n")
    rows = []
    for c in sorted(conds):
        print(f"--- {c} ---")
        pers_sig = faith_sig = False
        rec = {"prompt_condition": c}
        for label, dims, flag in [("persuasive", PERSUASIVE, "p"), ("faithful", FAITHFUL, "f")]:
            for d in dims:
                delta, p, n = paired(judge, d, c)
                sig = (not np.isnan(p)) and p < args.alpha and delta > 0
                if flag == "p":
                    pers_sig |= sig
                else:
                    faith_sig |= sig
                rec[f"d_{d}"] = delta
                rec[f"p_{d}"] = p
                mark = "SIG+" if sig else ("n.s." if not np.isnan(p) else "n/a")
                print(f"  [{label:<10}] {d:<32} Δ={delta:+.3f}  p={p:.4f}  {mark}  n={n}")
        # groundedness support rate (independent of the judge)
        sr = ground.groupby("prompt_condition")["support_rate"].mean()
        rec["d_support_rate"] = sr.get(c, np.nan) - sr.get(BASE, np.nan)
        print(f"  [reference ] support_rate (non-judge)     Δ={rec['d_support_rate']:+.3f}")
        rec["dissociation"] = bool(pers_sig and not faith_sig)
        print(f"  => H3 dissociation: {'YES' if rec['dissociation'] else 'no'}"
              f"  (persuasive_up={pers_sig}, faithful_up={faith_sig})\n")
        rows.append(rec)

    df = pd.DataFrame(rows)
    path = os.path.join(out, f"stats_dissociation_{args.split}.csv")
    df.round(4).to_csv(path, index=False)
    hits = df[df.dissociation].prompt_condition.tolist()
    print(f"H3 supported for: {hits or 'none'}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
