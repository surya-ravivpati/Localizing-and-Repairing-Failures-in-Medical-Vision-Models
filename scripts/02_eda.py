"""Exploratory analysis confirming the difficulty/ambiguity signals exist
(protocol §11). Prints distributions; writes eda_summary.csv + label_prevalence.csv.
"""
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.labeling import CHEXPERT_CLASSES


def main():
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    df = pd.read_csv(os.path.join(out, "studies_labeled.csv"))

    print("=" * 60)
    print(f"Usable studies: {len(df)}")
    print(f"Studies with frontal view: {df.has_frontal.sum()}")
    print(f"Mean images/study: {df.n_images.mean():.2f}")

    print("\n--- Difficulty strata ---")
    print(df.difficulty.value_counts().to_string())

    print("\n--- Ambiguity signals ---")
    print(f"Hedged reports:        {df.is_hedged.mean()*100:5.1f}%")
    print(f"Any uncertain label:   {(df.n_uncertain>0).mean()*100:5.1f}%")
    print(f"Mean XXXX scrub density:{df.xxxx_density.mean()*100:5.1f}%")
    print(f">15% XXXX density:     {(df.xxxx_density>0.15).mean()*100:5.1f}%")

    print("\n--- Label prevalence (report-derived) ---")
    prev = []
    for c in CHEXPERT_CLASSES:
        col = f"lbl_{c}"
        pos = (df[col] == 1.0).mean()
        unc = (df[col] == -1.0).mean()
        neg = (df[col] == 0.0).mean()
        na = df[col].isna().mean()
        prev.append({"class": c, "positive": pos, "uncertain": unc,
                     "negative": neg, "not_mentioned": na})
    prev = pd.DataFrame(prev).sort_values("positive", ascending=False)
    print(prev.assign(**{k: (prev[k]*100).round(1) for k in
                         ["positive", "uncertain", "negative", "not_mentioned"]})
          .to_string(index=False))

    print("\n--- n_positive findings per study ---")
    print(df.n_positive.value_counts().sort_index().to_string())

    prev.to_csv(os.path.join(out, "label_prevalence.csv"), index=False)
    summary = {
        "n_studies": len(df),
        "pct_hedged": df.is_hedged.mean(),
        "pct_any_uncertain": (df.n_uncertain > 0).mean(),
        "mean_xxxx_density": df.xxxx_density.mean(),
        "pct_normal": (df.get("lbl_No Finding") == 1.0).mean(),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out, "eda_summary.csv"), index=False)
    print(f"\nWrote eda_summary.csv + label_prevalence.csv to {out}")


if __name__ == "__main__":
    main()
