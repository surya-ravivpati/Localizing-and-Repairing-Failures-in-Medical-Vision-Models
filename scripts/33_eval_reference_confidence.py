"""Existing primary metric (micro-F1) stratified by REFERENCE CONFIDENCE (EDIT 4).

Answers: does VLM performance increase as reference-label reliability increases?
Scores each prompt on all cells, then on cells where the 3 labelers agree at HIGH
(3/3), MEDIUM (2/3), and LOW (disagree) — reusing the EXISTING evaluation
(multilabel_prf / multilabel_prf_masked), not a new metric. Uses structured scoring
(normalize.structured_pred_set) for the structured evidence-first condition so its
finding table is read directly.
"""
import argparse
import os

import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import multilabel_prf, multilabel_prf_masked, RELIABLE_CLASSES
from src.reference_confidence import cell_mask_factory, available_labelers
from src.normalize import structured_pred_set


def _pred_fn_for(preds):
    # for the structured condition, score the machine-readable findings table directly
    import json
    def fn(r):
        try:
            return structured_pred_set(json.loads(r["response"]))
        except Exception:
            return set()
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="predictions_test.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--structured", action="store_true",
                    help="score via the structured findings table (structured cond.)")
    args = ap.parse_args()
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    preds = pd.read_csv(os.path.join(out, args.preds))

    labs = available_labelers(s)
    print(f"labelers: {labs} ({len(labs)}-way agreement)\n")
    pred_fn = _pred_fn_for(preds) if args.structured else None

    all_prf = multilabel_prf(preds, s, u_policy="ones", prefix="lblcx_",
                             classes=RELIABLE_CLASSES, hierarchy=True)
    f_all = all_prf[all_prf.average == "micro"].set_index("prompt_condition")["f1"]

    per_level = {}
    for level in ["HIGH", "MEDIUM", "LOW"]:
        mask = cell_mask_factory(s, level, exact=True)
        prf = multilabel_prf_masked(preds, s, mask, u_policy="ones", prefix="lblcx_",
                                    classes=RELIABLE_CLASSES, hierarchy=True,
                                    pred_fn=pred_fn)
        per_level[level] = prf.set_index("prompt_condition")

    print(f"{'prompt':<22}{'All':>8}{'HIGH':>8}{'MEDIUM':>8}{'LOW':>8}")
    rows = []
    for cond in f_all.index:
        h = per_level['HIGH']['f1'].get(cond, float('nan'))
        m = per_level['MEDIUM']['f1'].get(cond, float('nan'))
        lo = per_level['LOW']['f1'].get(cond, float('nan'))
        print(f"{cond:<22}{f_all[cond]:>8.3f}{h:>8.3f}{m:>8.3f}{lo:>8.3f}")
        rows.append({"prompt_condition": cond, "f1_all": f_all[cond],
                     "f1_high": h, "f1_medium": m, "f1_low": lo})
    pd.DataFrame(rows).round(4).to_csv(
        os.path.join(out, f"reference_confidence_f1_{args.split}.csv"), index=False)
    print(f"\n(HIGH=3/3 agree, MEDIUM=2/3, LOW=disagree)  "
          f"Wrote reference_confidence_f1_{args.split}.csv")


if __name__ == "__main__":
    main()
