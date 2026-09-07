"""Run the full evaluation pipeline over predictions (protocol §7-8).

Writes per-response and per-prompt summary tables + judge scores.
"""
import argparse
import os

import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import (score_accuracy, accuracy_by_prompt,
                               accuracy_by_prompt_difficulty, multilabel_prf,
                               accuracy_f1_summary, RELIABLE_CLASSES)
from src.eval_groundedness import score_groundedness, groundedness_by_prompt
from src.eval_hallucination import score_hallucination, hallucination_by_prompt
from src.eval_calibration import (calibration_by_prompt,
                                  calibration_by_prompt_difficulty)
from src.judge import run_judging


def main(split="test", config=None):
    cfg = load_cfg(config)
    print(f"judge backend: {cfg['judge']['backend']} | models: {cfg['judge']['models']}")
    out = cfg["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    preds = pd.read_csv(os.path.join(out, f"predictions_{split}.csv"))

    print("1/5 Accuracy + F1 ...")
    # PRIMARY reference: impression-only ("acute") labels -> real misses, not
    # chronic/incidental label noise (protocol §3.3). 'correct' column is set here.
    acc, preds = accuracy_f1_summary(preds, studies, prefix="lblimp_")
    acc_diff = accuracy_by_prompt_difficulty(preds)
    prf = multilabel_prf(preds, studies, prefix="lblimp_")
    # SECONDARY (comparison): comprehensive findings+impression labels.
    acc_findings, _ = accuracy_f1_summary(preds, studies, prefix="lbl_")
    # RELIABLE-CLASSES view (drop the 3 low-agreement classes where the CheXbert
    # reference itself is noisy, κ<0.4): separates model error from reference error.
    prf_reliable = multilabel_prf(preds, studies, prefix="lblcx_",
                                  classes=RELIABLE_CLASSES)

    print("2/5 Groundedness ...")
    ground = score_groundedness(preds, studies)
    ground_sum = groundedness_by_prompt(ground)

    print("3/5 Hallucination ...")
    hall = score_hallucination(preds, studies)
    hall_sum = hallucination_by_prompt(hall)

    print("4/5 Calibration ...")
    calib = calibration_by_prompt(preds, cfg["stats"]["ece_bins"])
    calib_diff = calibration_by_prompt_difficulty(preds)

    print("5/5 LLM judge ...")
    jckpt = os.path.join(out, f"judge_scores_{split}.partial.csv")
    judge = run_judging(preds, studies, ground, cfg, checkpoint=jckpt)

    # persist everything
    preds.to_csv(os.path.join(out, f"predictions_scored_{split}.csv"), index=False)
    ground.to_csv(os.path.join(out, f"groundedness_{split}.csv"), index=False)
    hall.to_csv(os.path.join(out, f"hallucination_{split}.csv"), index=False)
    judge.to_csv(os.path.join(out, f"judge_scores_{split}.csv"), index=False)
    for name, t in [("accuracy", acc), ("accuracy_findings_ref", acc_findings),
                    ("accuracy_by_difficulty", acc_diff),
                    ("multilabel_prf", prf),
                    ("multilabel_prf_reliable", prf_reliable),
                    ("groundedness", ground_sum),
                    ("hallucination", hall_sum), ("calibration", calib),
                    ("calibration_by_difficulty", calib_diff)]:
        t.to_csv(os.path.join(out, f"summary_{name}_{split}.csv"), index=False)

    print("\n===== SUMMARY (per prompt) =====")
    print("\nAccuracy + F1  [PRIMARY: impression/acute reference]:\n",
          acc.to_string(index=False))
    print("\nAccuracy + F1  [comparison: findings+impression reference]:\n",
          acc_findings[["prompt_condition", "accuracy", "f1_micro"]].to_string(index=False))
    print("\nMulti-label P/R/F1  [reliable classes (κ≥0.4), CheXbert ref, micro]:\n",
          prf_reliable[prf_reliable.average == "micro"]
          [["prompt_condition", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print("\nGroundedness:\n", ground_sum.to_string(index=False))
    print("\nHallucination:\n", hall_sum.to_string(index=False))
    print("\nCalibration:\n",
          calib[["prompt_condition", "ece", "mce", "brier",
                 "overconfidence_rate", "accuracy"]].round(3).to_string(index=False))
    print(f"\nWrote all summary_* tables to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    main(split=args.split, config=args.config)
