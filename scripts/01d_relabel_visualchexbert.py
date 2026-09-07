"""Add VisualCheXbert reference labels (lblvcx_*) to studies_labeled.csv (EDIT 3).

Mirrors scripts/01c_relabel_chexbert.py — adds a THIRD labeler alongside the existing
rule-based (lbl_) and CheXbert (lblcx_) columns without touching them. Idempotent.
If the VisualCheXbert checkpoint is not installed, prints how to get it and exits
WITHOUT error; the reference-confidence layer then works with the two existing labelers.
"""
import os

import pandas as pd
from _bootstrap import load_cfg

from src.labeling import CHEXPERT_CLASSES
from src import labeling_visualchexbert as vcx


def main():
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    path = os.path.join(out, "studies_labeled.csv")
    df = pd.read_csv(path)

    if not vcx.available():
        print("VisualCheXbert checkpoint not found — skipping (2-labeler mode).")
        print("To enable the 3rd labeler: place visualCheXbert.pth per "
              "src/labeling_visualchexbert.py, then re-run this script.")
        return

    print(f"labeling {len(df)} reports with VisualCheXbert (CPU, batched) ...")
    labels = vcx.label_reports_visualchexbert(df["report_text"].fillna("").tolist(),
                                              batch_size=32, progress=True)
    lab = pd.DataFrame(labels)[CHEXPERT_CLASSES]
    lab.columns = [f"lblvcx_{c}" for c in CHEXPERT_CLASSES]

    df = df[[c for c in df.columns if not c.startswith("lblvcx_")]]
    df = pd.concat([df.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
    df.to_csv(path, index=False)
    print(f"\nWrote lblvcx_ columns to {path} (rule-based + CheXbert preserved)")


if __name__ == "__main__":
    main()
