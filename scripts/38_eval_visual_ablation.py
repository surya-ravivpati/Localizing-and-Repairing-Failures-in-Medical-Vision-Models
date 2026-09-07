"""Score Experiment 1 (visual-information ablation) arms and persist the results.

Scoring conventions are IDENTICAL to 36_eval_twostage.py / 17_study_results.py --
CheXbert reference (lblcx_), uncertain=positive, CheXpert hierarchy propagated on
both sides -- so Experiment 1 numbers sit on the same scale as everything else in
the paper. Unlike 36 (which only prints), this writes CSV.

  python3 scripts/38_eval_visual_ablation.py --arms baseline --suffix-prefix exp1_pilot --pilot-checks
  python3 scripts/38_eval_visual_ablation.py --arms baseline,highres --suffix-prefix exp1_gem
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from sklearn.metrics import precision_recall_fscore_support as prf

from src import imaging
from src.eval_accuracy import (_pred_set, _truth_positive_set, propagate_hierarchy,
                               RELIABLE_CLASSES)
from src.grouping import CORE_PATHOLOGY_GROUPS, to_groups, any_acute
from src.labeling import CHEXPERT_CLASSES
from src.normalize import normalize_dx

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]


def micro(yt, yp):
    yt, yp = np.array(yt, int), np.array(yp, int)
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, fp, fn


def dx_only(row):
    """Stage B's COMMITTED diagnosis. Mining the explanation instead double-counts
    the findings it was handed (incl. negated ones) -- see 36_eval_twostage.py."""
    try:
        o = json.loads(row["response"])
    except Exception:
        o = {}
    return propagate_hierarchy({x for x in normalize_dx(str(o.get("primary_diagnosis", "") or ""))
                                if x != "No Finding"})


def load_arm(out, split, prefix, arm):
    a = os.path.join(out, f"predictions_{split}_{prefix}_{arm}_stageA.csv")
    b = os.path.join(out, f"predictions_{split}_{prefix}_{arm}_stageB.csv")
    for p in (a, b):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run 35_run_twostage.py for arm '{arm}' first")
    return pd.read_csv(a).set_index("uid"), pd.read_csv(b).set_index("uid")


def score_arm(arm, sa, sb, s, uids):
    T = {u: propagate_hierarchy(_truth_positive_set(s.loc[u], "ones", "lblcx_")) for u in uids}
    P = {u: propagate_hierarchy(_pred_set(sa.loc[u])) for u in uids}

    p13, r13, f13, tp, fp, fn = micro([[1 if x in T[u] else 0 for x in ALL] for u in uids],
                                      [[1 if x in P[u] else 0 for x in ALL] for u in uids])
    frel = micro([[1 if x in T[u] else 0 for x in RELIABLE_CLASSES] for u in uids],
                 [[1 if x in P[u] else 0 for x in RELIABLE_CLASSES] for u in uids])[2]
    k = list(CORE_PATHOLOGY_GROUPS)
    f3 = micro([[1 if x in to_groups(T[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in uids],
               [[1 if x in to_groups(P[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in uids])[2]

    yt = np.array([1 if any_acute(T[u]) else 0 for u in uids])
    yA = np.array([1 if any_acute(P[u]) else 0 for u in uids])
    bp, br, bf, _ = prf(yt, yA, average="binary", zero_division=0)

    yB = np.array([1 if any_acute(dx_only(sb.loc[u])) else 0 for u in uids])
    b_err = yB != yt
    perc_err = float((yA[b_err] != yt[b_err]).mean()) if b_err.sum() else 0.0

    n_parts = pd.to_numeric(sa.loc[uids, "n_image_parts"], errors="coerce") \
        if "n_image_parts" in sa.columns else pd.Series(dtype=float)
    ptok = pd.to_numeric(sa.loc[uids, "prompt_tokens"], errors="coerce") \
        if "prompt_tokens" in sa.columns else pd.Series(dtype=float)

    return {
        "arm": arm, "n": len(uids),
        # --- Stage A: perception ---
        "stageA_f1_13": round(f13, 4), "stageA_p_13": round(p13, 4), "stageA_r_13": round(r13, 4),
        "stageA_f1_reliable": round(frel, 4), "stageA_f1_3group": round(f3, 4),
        "stageA_tp": tp, "stageA_fp_hallucinated": fp, "stageA_fn_missed": fn,
        "stageA_fp_per_study": round(fp / len(uids), 3),
        "stageA_fn_per_study": round(fn / len(uids), 3),
        "stageA_bin_p": round(bp, 4), "stageA_bin_r": round(br, 4),
        "stageA_bin_f1": round(bf, 4), "stageA_bin_acc": round(float((yA == yt).mean()), 4),
        "findings_per_study": round(float(np.mean([len(P[u]) for u in uids])), 3),
        "pct_flagged_abnormal": round(float(yA.mean()), 4),
        # --- Stage B: reasoning ---
        "stageB_bin_acc": round(float((yB == yt).mean()), 4),
        "stageB_pct_flagged_abnormal": round(float(yB.mean()), 4),
        "reasoning_fidelity": round(float((yA == yB).mean()), 4),
        "pct_errors_perception_caused": round(perc_err, 4),
        # --- provenance: what was actually SENT ---
        "mean_image_parts": round(float(n_parts.mean()), 3) if len(n_parts) else None,
        "mean_prompt_tokens": round(float(ptok.mean()), 1) if len(ptok) else None,
        "median_prompt_tokens": float(ptok.median()) if len(ptok) else None,
        "stageA_schema_valid": round(float(sa.loc[uids, "schema_valid"].mean()), 4),
        "stageB_schema_valid": round(float(sb.loc[uids, "schema_valid"].mean()), 4),
    }, yA, yB, yt


def pilot_checks(out, arm, sa, sb, uids):
    """Step-6 gate. Returns list of failure strings (empty == pass)."""
    bad = []
    # 1. image parts sent == parts rendered for that uid
    try:
        man = imaging.load_manifest(out, arm)
        exp = {u: len(man[u]["frontal"]) + len(man[u]["lateral"]) for u in uids if u in man}
        got = pd.to_numeric(sa.loc[uids, "n_image_parts"], errors="coerce")
        mism = [u for u in uids if u in exp and got.get(u) != exp[u]]
        if mism:
            bad.append(f"[parts] {len(mism)} studies sent != rendered part count "
                       f"(e.g. {mism[:3]}: sent {got.get(mism[0])} vs rendered {exp[mism[0]]})")
    except SystemExit as e:
        bad.append(f"[parts] {e}")
    # 2. token provenance actually recorded
    tok = pd.to_numeric(sa.loc[uids, "prompt_tokens"], errors="coerce")
    if tok.isna().all() or (tok.fillna(0) <= 0).all():
        bad.append("[tokens] prompt_tokens missing/zero -- cannot verify visual payload")
    # 3. Stage A must NOT diagnose
    dxa = sa.loc[uids, "primary_diagnosis"].astype(str).str.strip().str.upper()
    frac = float((dxa == "DEFERRED").mean())
    if frac < 0.90:
        bad.append(f"[stageA] only {frac:.0%} of Stage-A rows say DEFERRED "
                   f"(perception stage may be leaking a diagnosis)")
    # 4. Stage B must have received the frozen findings, and no image
    if "stage_a_findings" not in sb.columns:
        bad.append("[stageB] stage_a_findings column missing")
    else:
        empty = sb.loc[uids, "stage_a_findings"].astype(str).isin(["", "{}", "nan"]).sum()
        if empty > 0.10 * len(uids):
            bad.append(f"[stageB] {empty}/{len(uids)} rows got empty frozen findings")
    if "n_image_parts" in sb.columns and pd.to_numeric(
            sb.loc[uids, "n_image_parts"], errors="coerce").fillna(0).sum() > 0:
        bad.append("[stageB] received image parts -- must be text-only")
    # 5. schema health
    for tag, d in (("A", sa), ("B", sb)):
        v = float(d.loc[uids, "schema_valid"].mean())
        if v < 0.90:
            bad.append(f"[schema] Stage {tag} only {v:.0%} schema-valid")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--suffix-prefix", default="exp1_gem")
    ap.add_argument("--arms", default="baseline", help="comma-separated")
    ap.add_argument("--pilot-checks", action="store_true",
                    help="run the Step-6 gate; exit non-zero on failure")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    loaded = {a: load_arm(out, args.split, args.suffix_prefix, a) for a in arms}
    common = sorted(set.intersection(*[set(sa.index) & set(sb.index)
                                       for sa, sb in loaded.values()]) & set(s.index))
    print(f"Experiment 1 | arms={arms} | paired n={len(common)} "
          f"(CheXbert ref, hierarchy-aware, uncertain=positive)\n")

    rows, ys = [], {}
    for a in arms:
        sa, sb = loaded[a]
        rec, yA, yB, yt = score_arm(a, sa, sb, s, common)
        rows.append(rec); ys[a] = (yA, yB, yt)

    res = pd.DataFrame(rows)
    cols = ["arm", "n", "stageA_f1_13", "stageA_p_13", "stageA_r_13", "stageA_f1_reliable",
            "stageA_f1_3group", "stageA_bin_acc", "findings_per_study",
            "pct_flagged_abnormal", "stageA_fp_per_study", "stageA_fn_per_study",
            "stageB_bin_acc", "reasoning_fidelity", "pct_errors_perception_caused",
            "mean_image_parts", "mean_prompt_tokens"]
    print(res[cols].to_string(index=False))

    dest = args.out_csv or os.path.join(out, f"exp1_results_{args.suffix_prefix}.csv")
    res.to_csv(dest, index=False)
    print(f"\nWrote {dest}")

    if args.pilot_checks:
        print("\n=== STEP-6 GATE ===")
        failures = []
        for a in arms:
            sa, sb = loaded[a]
            bad = pilot_checks(out, a, sa, sb, common)
            print(f"  {a}: " + ("PASS" if not bad else "FAIL"))
            for b in bad:
                print(f"      {b}")
            failures += bad
        # cross-arm: the visual manipulation must actually change the payload
        if len(arms) > 1:
            # PER-STUDY, not mean: a mean can rise on a handful of outliers while the
            # manipulation reaches almost nothing. That is exactly what happened on the
            # first highres pilot -- 22/24 studies were byte-different on disk but
            # token-IDENTICAL at the API, and the mean still rose. Require the payload
            # to actually increase for a large majority of studies.
            sa0 = loaded[arms[0]][0]
            t0 = pd.to_numeric(sa0.loc[common, "prompt_tokens"], errors="coerce")
            for a in arms[1:]:
                ta = pd.to_numeric(loaded[a][0].loc[common, "prompt_tokens"],
                                   errors="coerce")
                frac = float((ta > t0).mean())
                print(f"      [payload] arm '{a}': {frac:.0%} of studies received MORE "
                      f"prompt tokens than '{arms[0]}' "
                      f"(median {t0.median():.0f} -> {ta.median():.0f})")
                if frac < 0.90:
                    msg = (f"[payload] arm '{a}' delivered more tokens for only "
                           f"{frac:.0%} of studies (<90%): the visual manipulation did "
                           f"NOT reach the model for the rest")
                    print(f"      {msg}"); failures.append(msg)
        if failures:
            print(f"\nGATE FAILED ({len(failures)} issue(s)). Fix before the full run.")
            sys.exit(1)
        print("\nGATE PASSED.")


if __name__ == "__main__":
    main()
