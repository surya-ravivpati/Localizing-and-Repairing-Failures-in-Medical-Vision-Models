"""Assign each study to a difficulty stratum from the report/labels only.

Blind to any model output (protocol §11). The ambiguity signal combines:
  - report hedging language (is_hedged),
  - uncertain (-1) structured labels,
  - DISAGREEMENT between two independent label sources:
      (a) the rule-based CheXpert-style labeler (labeling.py), and
      (b) the dataset-provided MeSH/Problems tags.
  - XXXX scrub density (unreliable verification).

Strata: easy_normal, easy_abnormal, intermediate, hard_multi, ambiguous, rare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import xxxx_density

RARE_CLASSES = ["Pneumothorax", "Lung Lesion", "Pleural Other", "Fracture",
                "Enlarged Cardiomediastinum"]


def _mesh_says_normal(problems: str, mesh: str) -> bool:
    p = (str(problems) + " " + str(mesh)).lower()
    return "normal" in p and "no indexing" not in p


def _mesh_says_abnormal(problems: str) -> bool:
    p = str(problems).strip().lower()
    return p not in ("", "normal", "no indexing") and "normal" not in p.split(";")[0]


def assign_difficulty(row: pd.Series) -> str:
    n_pos = int(row["n_positive"])
    n_unc = int(row["n_uncertain"])
    hedged = bool(row["is_hedged"])
    xxxx = xxxx_density(row.get("report_text", ""))

    # Second label source: dataset MeSH/Problems.
    mesh_normal = _mesh_says_normal(row.get("problems", ""), row.get("mesh", ""))
    labeler_normal = (row.get("lbl_No Finding") == 1.0) or (n_pos == 0 and n_unc == 0)

    # Cross-source disagreement on the normal/abnormal call.
    disagree = (mesh_normal != labeler_normal)

    rare_hit = any(row.get(f"lbl_{c}") == 1.0 for c in RARE_CLASSES)

    # Order matters: ambiguity and rarity take precedence over easy/hard buckets.
    if hedged or n_unc > 0 or disagree or xxxx > 0.15:
        return "ambiguous"
    if rare_hit:
        return "rare"
    if mesh_normal and labeler_normal:
        return "easy_normal"
    if n_pos >= 3:
        return "hard_multi"
    if n_pos == 1:
        return "easy_abnormal"
    if n_pos == 2:
        return "intermediate"
    # Fallback: labeler found nothing but MeSH not clearly normal.
    return "intermediate"


def stratify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["xxxx_density"] = df["report_text"].map(xxxx_density)
    df["difficulty"] = df.apply(assign_difficulty, axis=1)
    return df


if __name__ == "__main__":
    import sys, yaml
    from .data import Paths, load_studies
    from .labeling import label_frame
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"))
    p = cfg["paths"]
    df = load_studies(Paths(p["reports_csv"], p["proj_csv"], p["images_dir"]))
    df = label_frame(df)
    df = stratify(df)
    print(df["difficulty"].value_counts())
