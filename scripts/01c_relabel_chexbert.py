"""Add CheXbert reference labels to studies_labeled.csv (protocol §3.2 upgrade).

Writes lblcx_<class> columns (findings+impression) so the evaluation can be
re-scored against the field-standard CheXbert labeler instead of the bronze
rule-based one. Idempotent: overwrites any existing lblcx_ columns.
"""
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.labeling import CHEXPERT_CLASSES
from src.labeling_chexbert import label_reports_chexbert


def main():
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    path = os.path.join(out, "studies_labeled.csv")
    df = pd.read_csv(path)
    print(f"labeling {len(df)} reports with CheXbert (CPU, batched) ...")

    labels = label_reports_chexbert(df["report_text"].fillna("").tolist(),
                                    batch_size=32, progress=True)
    lab = pd.DataFrame(labels)
    # align to our canonical class order, prefix lblcx_
    lab = lab[CHEXPERT_CLASSES]
    lab.columns = [f"lblcx_{c}" for c in CHEXPERT_CLASSES]

    # drop any prior lblcx_ cols, then attach
    df = df[[c for c in df.columns if not c.startswith("lblcx_")]]
    df = pd.concat([df.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
    df.to_csv(path, index=False)

    pos = (lab == 1.0).sum().sort_values(ascending=False)
    print("\nCheXbert positive-label prevalence (top):")
    print(pos.head(10).to_string())
    print(f"\nWrote lblcx_ columns to {path}")


if __name__ == "__main__":
    main()
