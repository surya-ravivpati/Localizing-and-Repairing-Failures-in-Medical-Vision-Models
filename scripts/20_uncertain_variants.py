"""Uncertain-label sensitivity: U-ones / U-zeros / U-ignore (protocol §7.1).

§7.1 requires all three CheXpert uncertain-label variants be reported so that a
single arbitrary choice cannot drive the conclusions. Headline numbers use U-ones;
this script supplies the other two and checks whether the RANKING of prompts is
stable across them (which is what actually matters for a comparative study).

Semantics (these are genuinely different — see _truth_uncertain_set):
  U-ones   : reference uncertain (-1) counted POSITIVE
  U-zeros  : reference uncertain counted NEGATIVE
  U-ignore : reference-uncertain (case, class) cells MASKED OUT of the matrix
"""
import argparse
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import (_pred_set, _truth_positive_set, _truth_uncertain_set,
                               propagate_hierarchy, RELIABLE_CLASSES)
from src.grouping import CORE_PATHOLOGY_GROUPS, to_groups
from src.labeling import CHEXPERT_CLASSES

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
CONDS = ["direct", "cot", "ddx", "evidence_first", "uncertainty_first"]


def micro_masked(pairs):
    """micro P/R/F1 from (truth_vec, pred_vec, mask_vec) triples; mask 0 = skip."""
    tp = fp = fn = 0
    for t, p, m in pairs:
        for ti, pi, mi in zip(t, p, m):
            if not mi:
                continue
            tp += ti and pi
            fp += (not ti) and pi
            fn += ti and (not pi)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def evaluate(pred_df, s, uids, classes, policy, prefix="lblcx_", hierarchy=True):
    triples = []
    g = pred_df
    for u in uids:
        row = s.loc[u]
        truth = _truth_positive_set(row, "ones" if policy == "ones" else "zeros", prefix)
        unc = _truth_uncertain_set(row, prefix)
        pred = _pred_set(g.loc[u])
        if hierarchy:
            truth, pred = propagate_hierarchy(truth), propagate_hierarchy(pred)
            unc = propagate_hierarchy(unc)
        t = [c in truth for c in classes]
        p = [c in pred for c in classes]
        # U-ignore masks out cells the reference called uncertain
        m = [not (policy == "ignore" and c in unc) for c in classes]
        triples.append((t, p, m))
    return micro_masked(triples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    pr = pd.read_csv(os.path.join(out, f"predictions_{args.split}.csv"))

    # how many cells are actually uncertain (i.e. how much this can matter)
    n_unc = sum(len(_truth_uncertain_set(s.loc[u])) for u in s.index)
    print(f"reference uncertain (-1) cells across corpus: {n_unc} "
          f"({100*n_unc/(len(s)*len(ALL)):.2f}% of all cells)\n")

    rows = []
    for level, classes in [("13-class", ALL), ("reliable", RELIABLE_CLASSES)]:
        print(f"=== {level} (micro-F1, CheXbert ref, hierarchy-aware) ===")
        print(f"{'condition':<20}{'U-ones':>9}{'U-zeros':>9}{'U-ignore':>10}{'spread':>9}")
        for c in CONDS:
            g = pr[pr.prompt_condition == c].set_index("uid")
            uids = [u for u in g.index if u in s.index]
            f = {}
            for pol in ("ones", "zeros", "ignore"):
                f[pol] = evaluate(g, s, uids, classes, pol)[2]
            spread = max(f.values()) - min(f.values())
            rows.append({"level": level, "prompt_condition": c,
                         "f1_u_ones": f["ones"], "f1_u_zeros": f["zeros"],
                         "f1_u_ignore": f["ignore"], "spread": spread})
            print(f"{c:<20}{f['ones']:>9.3f}{f['zeros']:>9.3f}{f['ignore']:>10.3f}{spread:>9.3f}")
        sub = [r for r in rows if r["level"] == level]
        rank = {pol: [r["prompt_condition"] for r in
                      sorted(sub, key=lambda r: -r[f"f1_u_{pol}"])]
                for pol in ("ones", "zeros", "ignore")}
        same = rank["ones"] == rank["zeros"] == rank["ignore"]
        print(f"  ranking identical across all three policies: {'YES' if same else 'NO'}")
        for pol in ("ones", "zeros", "ignore"):
            print(f"    U-{pol:<7}: {' > '.join(rank[pol])}")
        print()

    df = pd.DataFrame(rows).round(4)
    path = os.path.join(out, f"study_uncertain_variants_{args.split}.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
