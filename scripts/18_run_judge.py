"""Run ONLY the LLM-judge over existing predictions (protocol §8).

Separate from 04_evaluate.py on purpose: that script also regenerates the accuracy
summaries using the pre-audit default reference (lblimp_, flat scoring), which would
overwrite corrected outputs. The authoritative performance tables come from
scripts/17_study_results.py.

The judge was fixed 2026-07-19 (thinking_budget=0, bigger token budget, retries,
and errors recorded rather than silently becoming NaN). Checkpointed and resumable.
"""
import argparse
import os

import pandas as pd
from _bootstrap import load_cfg

from src.judge import run_judging, DIMENSIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_gemini.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="cap #predictions (debug)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out = cfg["paths"]["out_dir"]
    print(f"judge backend={cfg['judge']['backend']} models={cfg['judge']['models']}")
    if cfg["judge"]["backend"] == "mock":
        raise SystemExit("refusing to run: config uses the MOCK judge — pass "
                         "--config configs/config_gemini.yaml")

    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    preds = pd.read_csv(os.path.join(out, f"predictions_{args.split}.csv"))
    ground = pd.read_csv(os.path.join(out, f"groundedness_{args.split}.csv"))
    if args.limit:
        preds = preds.groupby("prompt_condition").head(args.limit)

    ckpt = os.path.join(out, f"judge_scores_{args.split}.partial.csv")
    judge = run_judging(preds, studies, ground, cfg, checkpoint=ckpt)

    path = os.path.join(out, f"judge_scores_{args.split}.csv")
    judge.to_csv(path, index=False)

    scored = judge[f"score_{DIMENSIONS[0]}"].notna().mean()
    n_err = (judge["judge_error"].fillna("") != "").sum()
    print(f"\nWrote {path} | rows={len(judge)} | scored={100*scored:.1f}% | errors={n_err}")
    if scored < 0.9:
        print("*** WARNING: <90% scored — judge run is NOT usable, inspect judge_error ***")
        print(judge["judge_error"].replace("", pd.NA).dropna().value_counts().head(3).to_string())
    else:
        print("\nmean scores by prompt condition:")
        cols = [f"score_{d}" for d in DIMENSIONS]
        print(judge.groupby("prompt_condition")[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
