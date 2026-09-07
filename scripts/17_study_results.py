"""THE STUDY RESULTS — the 5 proposal conditions, corrected methodology, H1-H4.

This is the paper's results section. It evaluates ONLY the five pre-registered
prompt conditions (direct / cot / ddx / evidence_first / uncertainty_first) — the
structured/soft/targeted prompts explored later are a separate methods note, not
part of the pre-registered comparison.

Corrections applied vs the original run (all found in the 2026-07 audit):
  * reference: CheXbert (lblcx_) instead of impression-only (lblimp_)
  * hierarchy-aware scoring (CheXpert parent propagation on both sides)
  * uncertain reference labels counted positive
  * results reported across the granularity ladder, not a single level
"""
import argparse
import os

import json

import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from statsmodels.stats.contingency_tables import mcnemar

from src.eval_accuracy import (_pred_set, _truth_positive_set, propagate_hierarchy,
                               RELIABLE_CLASSES)
from src.grouping import CORE_PATHOLOGY_GROUPS, CLINICAL_GROUPS, to_groups, any_acute
from src.normalize import normalize_dx
from src.labeling import CHEXPERT_CLASSES
from sklearn.metrics import precision_recall_fscore_support

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
CONDS = ["direct", "cot", "ddx", "evidence_first", "uncertainty_first"]


def micro(yt, yp):
    yt, yp = np.array(yt, dtype=int), np.array(yp, dtype=int)
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ece(conf, acc, bins=15):
    conf, acc = np.asarray(conf, float), np.asarray(acc, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    pr = pd.read_csv(os.path.join(out, f"predictions_{args.split}.csv"))

    sets = {}
    for c in CONDS:
        g = pr[pr.prompt_condition == c].set_index("uid")
        uids = [u for u in g.index if u in s.index]
        sets[c] = (g, uids,
                   {u: propagate_hierarchy(_truth_positive_set(s.loc[u], "ones", "lblcx_")) for u in uids},
                   {u: propagate_hierarchy(_pred_set(g.loc[u])) for u in uids})
    common = sorted(set.intersection(*[set(v[1]) for v in sets.values()]))

    # ---- 1. accuracy / F1 across the granularity ladder
    print(f"=== 1. DIAGNOSTIC PERFORMANCE (n={len(common)}, CheXbert ref, hierarchy-aware) ===")
    print(f"{'condition':<20}{'13-cls':>8}{'reliable':>10}{'3-group':>9}{'binary':>8}{'bin-full':>10}{'top1':>7}")
    rows = []
    for c in CONDS:
        g, _, T, P = sets[c]
        f13 = micro([[1 if x in T[u] else 0 for x in ALL] for u in common],
                    [[1 if x in P[u] else 0 for x in ALL] for u in common])[2]
        frel = micro([[1 if x in RELIABLE_CLASSES and x in T[u] else 0 for x in RELIABLE_CLASSES] for u in common],
                     [[1 if x in P[u] else 0 for x in RELIABLE_CLASSES] for u in common])[2]
        k = list(CORE_PATHOLOGY_GROUPS)
        f3 = micro([[1 if x in to_groups(T[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in common],
                   [[1 if x in to_groups(P[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in common])[2]
        yt = [1 if any_acute(T[u]) else 0 for u in common]
        yp = [1 if any_acute(P[u]) else 0 for u in common]
        fb = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)[2]
        # SECONDARY operationalization: does the FULL response (diagnosis +
        # explanation) identify an abnormality? Answers the triage question rather
        # than the pre-registered "is the stated final diagnosis abnormal?".
        # Only valid because normalize_dx now does negation scoping — otherwise
        # every "no pleural effusion" in an explanation would count as positive.
        ypf = []
        for u in common:
            try:
                o = json.loads(g.loc[u, "response"])
            except Exception:
                o = {}
            pos = normalize_dx(str(o.get("primary_diagnosis", "") or "")) | \
                normalize_dx(str(o.get("explanation", "") or ""))
            ypf.append(1 if any_acute(propagate_hierarchy(
                {x for x in pos if x != "No Finding"})) else 0)
        fbf = precision_recall_fscore_support(yt, ypf, average="binary", zero_division=0)[2]
        acc = np.mean([(len(P[u]) == 0) if not T[u] else bool(P[u] & T[u]) for u in common])
        rows.append({"prompt_condition": c, "f1_13class": f13, "f1_reliable": frel,
                     "f1_3group": f3, "f1_binary": fb, "f1_binary_fullresp": fbf, "top1_accuracy": acc})
        print(f"{c:<20}{f13:>8.3f}{frel:>10.3f}{f3:>9.3f}{fb:>8.3f}{fbf:>10.3f}{acc:>7.3f}")
    perf = pd.DataFrame(rows)
    best = perf.loc[perf.f1_3group.idxmax(), "prompt_condition"]
    print(f"  -> best overall: {best}")

    # ---- 2. H2 calibration
    print("\n=== 2. H2 — CALIBRATION (corrected correctness; lower ECE = better) ===")
    cal = []
    for c in CONDS:
        g, _, T, P = sets[c]
        conf = np.array([float(g.loc[u, "confidence"]) / 100 for u in common])
        acc = np.array([(len(P[u]) == 0) if not T[u] else bool(P[u] & T[u]) for u in common], float)
        e = ece(conf, acc)
        cal.append({"prompt_condition": c, "ece": e, "mean_conf": conf.mean(),
                    "accuracy": acc.mean(), "overconfidence_gap": conf.mean() - acc.mean()})
        print(f"  {c:<20}ECE={e:.3f}  mean_conf={conf.mean():.3f}  overconf_gap={conf.mean()-acc.mean():+.3f}")
    cal = pd.DataFrame(cal)
    print(f"  -> H2 {'SUPPORTED' if cal.loc[cal.ece.idxmin(),'prompt_condition']=='uncertainty_first' else 'NOT supported'}: "
          f"lowest ECE = {cal.loc[cal.ece.idxmin(),'prompt_condition']}")

    # ---- 3. H1/H3 groundedness (from the groundedness pipeline)
    gpath = os.path.join(out, f"summary_groundedness_{args.split}.csv")
    if os.path.exists(gpath):
        gsum = pd.read_csv(gpath).set_index("prompt_condition")
        print("\n=== 3. H1/H3 — GROUNDEDNESS (report-consistency) ===")
        print(gsum[["support_rate", "mean_claims", "contradiction_rate"]].round(3).to_string())
        top = gsum.support_rate.idxmax()
        print(f"  -> H1 {'SUPPORTED' if top=='evidence_first' else 'NOT supported'}: "
              f"highest support_rate = {top} (evidence_first makes "
              f"{gsum.loc['evidence_first','mean_claims']:.1f} claims/case vs "
              f"{gsum.loc['direct','mean_claims']:.1f} for direct)")
        print(f"  -> H3: cot/ddx contradiction {gsum.loc['cot','contradiction_rate']:.3f}/"
              f"{gsum.loc['ddx','contradiction_rate']:.3f} vs direct "
              f"{gsum.loc['direct','contradiction_rate']:.3f}")

    # ---- 4. H4 difficulty interaction
    print("\n=== 4. H4 — PROMPT EFFECT BY DIFFICULTY (3-group F1) ===")
    k = list(CORE_PATHOLOGY_GROUPS)
    strat = {}
    for c in CONDS:
        g, _, T, P = sets[c]
        for diff in s.loc[common, "difficulty"].unique():
            uu = [u for u in common if s.loc[u, "difficulty"] == diff]
            strat.setdefault(diff, {})[c] = micro(
                [[1 if x in to_groups(T[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in uu],
                [[1 if x in to_groups(P[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in uu])[2]
    sdf = pd.DataFrame(strat).T
    sdf["spread"] = sdf.max(axis=1) - sdf.min(axis=1)
    print(sdf.round(3).to_string())
    print(f"  -> H4: largest prompt effect in "
          f"{sdf.spread.idxmax()} ({sdf.spread.max():.3f}), smallest in "
          f"{sdf.spread.idxmin()} ({sdf.spread.min():.3f})")

    # ---- 5. paired significance vs best
    print(f"\n=== 5. PAIRED McNEMAR — {best} vs others (top-1 accuracy) ===")
    ab = {c: np.array([(len(sets[c][3][u]) == 0) if not sets[c][2][u]
                       else bool(sets[c][3][u] & sets[c][2][u]) for u in common], int)
          for c in CONDS}
    for c in CONDS:
        if c == best:
            continue
        n01 = int(((ab[best] == 1) & (ab[c] == 0)).sum())
        n10 = int(((ab[best] == 0) & (ab[c] == 1)).sum())
        p = mcnemar([[0, n01], [n10, 0]], exact=False, correction=True).pvalue
        print(f"  vs {c:<20} wins={n01:<4} losses={n10:<4} p={p:.4f} "
              f"{'SIG' if p < 0.05 else 'n.s.'}")

    perf.round(4).to_csv(os.path.join(out, f"study_performance_{args.split}.csv"), index=False)
    cal.round(4).to_csv(os.path.join(out, f"study_calibration_{args.split}.csv"), index=False)
    sdf.round(4).to_csv(os.path.join(out, f"study_difficulty_{args.split}.csv"))
    print(f"\nWrote study_{{performance,calibration,difficulty}}_{args.split}.csv")


if __name__ == "__main__":
    main()
