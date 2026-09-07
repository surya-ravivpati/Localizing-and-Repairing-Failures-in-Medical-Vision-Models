"""Hallucination & omission metrics (protocol §7.3).

Operational definition:
  - hallucination = a CONTRADICTED claim OR a fabricated specific
    (measurement/prior/device with no report basis).
  - omission = a report-positive finding the model never states (tracked
    separately as a false negative, NOT conflated with hallucination).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .eval_groundedness import extract_claims, adjudicate_claim, _fab_re
from .labeling import CHEXPERT_CLASSES
from .normalize import normalize_dx


def score_hallucination(pred_df: pd.DataFrame, study_df: pd.DataFrame) -> pd.DataFrame:
    truth = study_df.set_index("uid")
    classes = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
    rows = []
    for _, r in pred_df.iterrows():
        resp = json.loads(r["response"])
        ref_row = truth.loc[r["uid"]]
        claims = extract_claims(resp)

        contradicted = sum(1 for c in claims if adjudicate_claim(c, ref_row) == "contradicted")
        text_blob = " ".join([resp.get("explanation", ""),
                              " ".join(resp.get("cited_findings", []) or [])])
        n_fab = len(_fab_re.findall(text_blob))
        hallucinated = (contradicted > 0) or (n_fab > 0)

        # Omission: report-positive classes not present in any model claim/dx.
        ref_pos = {c for c in classes if ref_row.get(f"lbl_{c}") == 1.0}
        stated = set(normalize_dx(resp.get("primary_diagnosis", "")))
        for c in claims:
            if c["presence"] == "present":
                stated.add(c["cls"])
        omitted = ref_pos - stated

        rows.append({
            "uid": r["uid"], "prompt_condition": r["prompt_condition"],
            "difficulty": r["difficulty"],
            "n_contradicted": contradicted, "n_fabricated": n_fab,
            "hallucinated": hallucinated,
            "n_omitted": len(omitted), "n_ref_positive": len(ref_pos),
            "omission_rate": len(omitted) / len(ref_pos) if ref_pos else 0.0,
        })
    return pd.DataFrame(rows)


def hallucination_by_prompt(h_df: pd.DataFrame) -> pd.DataFrame:
    return (h_df.groupby("prompt_condition")
            .agg(hallucination_rate=("hallucinated", "mean"),
                 contradiction_rate=("n_contradicted", "mean"),
                 fabrication_rate=("n_fabricated", "mean"),
                 omission_rate=("omission_rate", "mean"))
            .reset_index())
