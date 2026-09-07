"""Score the two-stage perception/reasoning experiment (run by 35_run_twostage.py)
and attribute error to PERCEPTION vs REASONING.

Answers the mentor's question directly: does the structured prompt improve visual
PERCEPTION, or does it just make the model REPORT MORE findings?

Comparators (all Gemini, CheXbert ref, hierarchy-aware, uncertain=ones — the
paper's headline convention, identical to 17_study_results.py):

  evidence_first        single pass, unstructured   (predictions_test.csv)
  evidence_structured   single pass, structured     (predictions_test_estr.csv)
  perception_only       Stage A: image -> findings   (twostage_stageA)  == PERCEPTION
  two_stage_diagnose    Stage B: findings -> dx      (twostage_stageB)  == REASONING

Stage A is scored on the full granularity ladder (it is a findings list => a
perception measurement). Stage B is a single diagnosis => scored on the triage
decision (binary / 3-group) plus its FIDELITY to the Stage-A findings it was given.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from sklearn.metrics import precision_recall_fscore_support

from src.eval_accuracy import (_pred_set, _truth_positive_set, propagate_hierarchy,
                               RELIABLE_CLASSES)
from src.grouping import CORE_PATHOLOGY_GROUPS, to_groups, any_acute
from src.labeling import CHEXPERT_CLASSES
from src.normalize import normalize_dx

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]


def micro(yt, yp):
    yt, yp = np.array(yt, dtype=int), np.array(yp, dtype=int)
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def dx_set(row):
    """Positive CheXpert set from a diagnosis row's FULL response (dx+explanation),
    negation-scoped — the same 'bin-full' operationalization as the main study."""
    try:
        o = json.loads(row["response"])
    except Exception:
        o = {}
    pos = normalize_dx(str(o.get("primary_diagnosis", "") or "")) | \
        normalize_dx(str(o.get("explanation", "") or ""))
    return {x for x in pos if x != "No Finding"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--suffix", default="twostage")
    ap.add_argument("--ef-file", default="predictions_test.csv",
                    help="single-pass unstructured comparator file")
    ap.add_argument("--ef-cond", default="evidence_first")
    ap.add_argument("--es-file", default="predictions_test_estr.csv",
                    help="single-pass structured comparator file (skipped if absent)")
    ap.add_argument("--es-cond", default="evidence_structured")
    args = ap.parse_args()
    out = load_cfg()["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")

    def load(path, cond=None):
        df = pd.read_csv(os.path.join(out, path))
        if cond:
            df = df[df.prompt_condition == cond]
        return df.set_index("uid")

    sa = load(f"predictions_{args.split}_{args.suffix}_stageA.csv")
    sb = load(f"predictions_{args.split}_{args.suffix}_stageB.csv")
    # single-pass comparators are model-specific and optional (MedGemma has no
    # single-pass structured run on the test split): --ef-file / --es-file, each
    # skipped if the file is absent so the two-stage core still scores.
    comps = []  # (name, indexed_df, pred_fn)
    for fname, cond, label in [(args.ef_file, args.ef_cond, "evidence_first"),
                               (args.es_file, args.es_cond, "evidence_structured")]:
        if fname and os.path.exists(os.path.join(out, fname)):
            comps.append((label, load(fname, cond), _pred_set))

    idx_sets = [set(d.index) for _, d, _ in comps] + [set(sa.index), set(sb.index), set(s.index)]
    uids = sorted(set.intersection(*idx_sets))
    T = {u: propagate_hierarchy(_truth_positive_set(s.loc[u], "ones", "lblcx_")) for u in uids}
    base = sum(any_acute(T[u]) for u in uids) / len(uids)
    print(f"n={len(uids)}   true-abnormal base rate = {base:.3f}   (CheXbert ref, hierarchy-aware)\n")

    # predicted positive set per representation (comparators first, then the two stages)
    P = {label: {u: propagate_hierarchy(fn(d.loc[u])) for u in uids} for label, d, fn in comps}
    P["perception_only(A)"] = {u: propagate_hierarchy(_pred_set(sa.loc[u])) for u in uids}
    P["two_stage_dx(B)"] = {u: propagate_hierarchy(dx_set(sb.loc[u])) for u in uids}

    # ---- ladder + operating-point decomposition ---------------------------------
    print("=== LADDER + BINARY DECOMPOSITION (micro-F1; binary P/R/acc) ===")
    hdr = f"{'representation':<22}{'13cls':>7}{'reliab':>8}{'3grp':>7}{'binF1':>7}" \
          f"{'find/std':>9}{'%abn':>6}{'binP':>7}{'binR':>7}{'binAcc':>8}"
    print(hdr)
    for name, Pm in P.items():
        f13 = micro([[1 if x in T[u] else 0 for x in ALL] for u in uids],
                    [[1 if x in Pm[u] else 0 for x in ALL] for u in uids])[2]
        frel = micro([[1 if x in T[u] else 0 for x in RELIABLE_CLASSES] for u in uids],
                     [[1 if x in Pm[u] else 0 for x in RELIABLE_CLASSES] for u in uids])[2]
        k = list(CORE_PATHOLOGY_GROUPS)
        f3 = micro([[1 if x in to_groups(T[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in uids],
                   [[1 if x in to_groups(Pm[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in uids])[2]
        yt = [1 if any_acute(T[u]) else 0 for u in uids]
        yp = [1 if any_acute(Pm[u]) else 0 for u in uids]
        bp, br, bf, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        acc = np.mean([(yt[i] == yp[i]) for i in range(len(uids))])
        fps = np.mean([len(Pm[u]) for u in uids])
        pabn = np.mean(yp)
        print(f"{name:<22}{f13:>7.3f}{frel:>8.3f}{f3:>7.3f}{bf:>7.3f}"
              f"{fps:>9.2f}{pabn:>6.0%}{bp:>7.3f}{br:>7.3f}{acc:>8.3f}")

    # ---- perception vs reasoning attribution ------------------------------------
    print("\n=== PERCEPTION vs REASONING ATTRIBUTION (binary triage) ===")
    # Stage B, dx-ONLY (primary_diagnosis, no explanation mining) — the pre-registered
    # operationalization; a stricter read of the reasoning step's committed decision.
    def dx_only(row):
        try:
            o = json.loads(row["response"])
        except Exception:
            o = {}
        return propagate_hierarchy({x for x in normalize_dx(str(o.get("primary_diagnosis", "") or ""))
                                    if x != "No Finding"})
    yBdo = np.array([1 if any_acute(dx_only(sb.loc[u])) else 0 for u in uids])
    pabn_do = yBdo.mean(); accBdo = (yBdo == np.array([1 if any_acute(T[u]) else 0 for u in uids])).mean()
    print(f"[Stage-B dx-ONLY: %abnormal={pabn_do:.0%}, binary acc={accBdo:.3f}]")

    yt = np.array([1 if any_acute(T[u]) else 0 for u in uids])
    yA = np.array([1 if any_acute(P["perception_only(A)"][u]) else 0 for u in uids])
    # use the COMMITTED primary diagnosis (dx-only) as the reasoning decision — the
    # explanation merely echoes the findings it was handed, so bin-full over-counts.
    yB = yBdo
    accA = (yA == yt).mean()
    accB = (yB == yt).mean()
    fidelity = (yA == yB).mean()   # does Stage-B dx agree with the Stage-A findings it was given?
    print(f"Stage-A findings imply-abnormal accuracy (PERCEPTION ceiling): {accA:.3f}")
    print(f"Stage-B diagnosis accuracy (perception + reasoning)          : {accB:.3f}")
    print(f"Stage-B/Stage-A agreement (REASONING FIDELITY)              : {fidelity:.3f}")
    # where does Stage B disagree with its own findings, and is it right to?
    disagree = yA != yB
    if disagree.sum():
        b_right = (yB[disagree] == yt[disagree]).mean()
        print(f"  on the {disagree.sum()} cases B overrides A's abnormal/normal call, "
              f"B is correct {b_right:.0%} of the time")
    # of Stage-B errors, how many were already wrong in perception?
    b_err = yB != yt
    perc_err = (yA[b_err] != yt[b_err]).mean() if b_err.sum() else 0.0
    print(f"  of Stage-B's {int(b_err.sum())} triage errors, {perc_err:.0%} were "
          f"ALREADY wrong in Stage-A perception (=> perception-caused, not reasoning)")


if __name__ == "__main__":
    main()
