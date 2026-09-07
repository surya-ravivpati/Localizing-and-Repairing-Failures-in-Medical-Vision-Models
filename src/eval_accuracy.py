"""Diagnostic accuracy metrics (protocol §7.1).

Per-case correctness (for paired McNemar) + aggregate multi-label P/R/F1 over
the 14 CheXpert classes. Uncertain (-1) labels handled via U-ones/U-zeros/
U-ignore variants.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .labeling import CHEXPERT_CLASSES
from .normalize import normalize_dx, predicted_positive_set


def _pred_set(r) -> set[str]:
    """Predicted positive classes from a prediction row (uses structured
    'findings' when present, else the free-text diagnosis)."""
    try:
        resp = json.loads(r["response"])
    except Exception:
        resp = {"primary_diagnosis": r.get("primary_diagnosis", "")}
    return predicted_positive_set(resp)


# Default reference is the impression-only ("acute") label set (protocol §3.3).
DEFAULT_PREFIX = "lblimp_"

# Classes where bronze-vs-CheXbert inter-labeler agreement is too low to score
# fairly (Cohen's κ < 0.4 in the 754 audit, 2026-07-12): the *reference itself*
# is unreliable here, so scoring against it mostly measures label noise. Reporting
# a "reliable-classes" F1 alongside the full-13 F1 separates model error from
# reference error. Not a default — always report both.
# Recomputed after the bronze negation fix (2026-07-19): only these two remain
# below the κ<0.4 criterion. Pleural Other rose to κ=0.60 and is now scored.
LOW_AGREEMENT_CLASSES = ["Enlarged Cardiomediastinum", "Lung Lesion"]
RELIABLE_CLASSES = [c for c in CHEXPERT_CLASSES
                    if c not in LOW_AGREEMENT_CLASSES and c != "No Finding"]

# CheXpert label hierarchy: a positive child implies the parent is also positive.
# The report labelers (bronze + CheXbert) do NOT propagate this, so scoring a
# child-vs-parent match as a miss double-penalizes a correct read (FP + FN). The
# standard CheXpert evaluation propagates parents on BOTH truth and prediction.
CHEXPERT_HIERARCHY = {
    "Lung Opacity": {"Consolidation", "Edema", "Pneumonia", "Atelectasis",
                     "Lung Lesion"},
}


def propagate_hierarchy(findings: set[str]) -> set[str]:
    """Add parent classes implied by any present child (CheXpert hierarchy)."""
    out = set(findings)
    for parent, children in CHEXPERT_HIERARCHY.items():
        if out & children:
            out.add(parent)
    return out


def _truth_uncertain_set(row: pd.Series,
                         prefix: str = DEFAULT_PREFIX) -> set[str]:
    """Classes the reference marked UNCERTAIN (-1) for this study.

    Needed for a true U-ignore evaluation: those (case, class) cells must be
    MASKED OUT of the confusion matrix, not scored as negatives. A positive-set
    return value cannot express "excluded", which is why `u_policy='ignore'` and
    `'zeros'` were behaving identically before 2026-07-19.
    """
    return {c for c in CHEXPERT_CLASSES if row.get(f"{prefix}{c}") == -1.0}


def _truth_positive_set(row: pd.Series, u_policy: str,
                        prefix: str = DEFAULT_PREFIX) -> set[str]:
    """Set of report-positive classes under an uncertain-label policy.

    NOTE: 'ones' adds uncertain classes as positive; 'zeros' and 'ignore' both
    leave them out here — they differ only in whether the cell is subsequently
    MASKED (see _truth_uncertain_set / scripts/20_uncertain_variants.py).
    """
    pos = set()
    for c in CHEXPERT_CLASSES:
        v = row.get(f"{prefix}{c}")
        if v == 1.0:
            pos.add(c)
        elif v == -1.0 and u_policy == "ones":
            pos.add(c)
    if not any(c != "No Finding" for c in pos):
        pos.add("No Finding")
    return {c for c in pos if c != "No Finding"} or {"No Finding"}


def case_correct(pred, row: pd.Series, u_policy: str = "ignore",
                 prefix: str = DEFAULT_PREFIX) -> bool:
    """Top-1 correct iff predicted class intersects the report-positive set,
    or both sides are normal. `pred` is a set of classes or a free-text string."""
    truth = _truth_positive_set(row, u_policy, prefix)
    pred = pred if isinstance(pred, set) else {c for c in normalize_dx(pred)
                                               if c != "No Finding"}
    if truth == {"No Finding"}:
        return len(pred) == 0
    return len(pred & truth) > 0


def score_accuracy(pred_df: pd.DataFrame, study_df: pd.DataFrame,
                   u_policy: str = "ignore",
                   prefix: str = DEFAULT_PREFIX) -> pd.DataFrame:
    """Return pred_df with a per-row 'correct' column added (vs `prefix` labels)."""
    truth_by_uid = study_df.set_index("uid")
    out = pred_df.copy()
    correct = []
    for _, r in out.iterrows():
        row = truth_by_uid.loc[r["uid"]]
        correct.append(case_correct(_pred_set(r), row, u_policy, prefix))
    out["correct"] = correct
    return out


def multilabel_prf(pred_df: pd.DataFrame, study_df: pd.DataFrame,
                   u_policy: str = "ignore",
                   prefix: str = DEFAULT_PREFIX,
                   classes: list[str] | None = None,
                   hierarchy: bool = False) -> pd.DataFrame:
    """Micro/macro precision/recall/F1 per prompt condition.

    `classes` restricts scoring to a subset (e.g. RELIABLE_CLASSES to exclude the
    low-agreement classes where the reference itself is noisy); defaults to all 13.
    `hierarchy=True` propagates CheXpert parent classes on both sides (the correct
    way to score the parent/child structure — see propagate_hierarchy).
    """
    truth_by_uid = study_df.set_index("uid")
    if classes is None:
        classes = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
    rows = []
    for cond, g in pred_df.groupby("prompt_condition"):
        y_true, y_pred = [], []
        for _, r in g.iterrows():
            row = truth_by_uid.loc[r["uid"]]
            t = _truth_positive_set(row, u_policy, prefix)
            p = _pred_set(r)
            if hierarchy:
                t, p = propagate_hierarchy(t), propagate_hierarchy(p)
            y_true.append([1 if c in t else 0 for c in classes])
            y_pred.append([1 if c in p else 0 for c in classes])
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        for avg in ("micro", "macro"):
            pr, rc, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg, zero_division=0)
            rows.append({"prompt_condition": cond, "average": avg,
                         "precision": pr, "recall": rc, "f1": f1})
    return pd.DataFrame(rows)


def accuracy_f1_summary(pred_df: pd.DataFrame, study_df: pd.DataFrame,
                        u_policy: str = "ignore",
                        prefix: str = DEFAULT_PREFIX) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined per-prompt table: top-1 accuracy + micro/macro precision/recall/F1.

    Returns (summary_table, scored_pred_df with 'correct' column).
    """
    scored = score_accuracy(pred_df, study_df, u_policy, prefix)
    acc = scored.groupby("prompt_condition")["correct"].mean()
    prf = multilabel_prf(scored, study_df, u_policy, prefix)
    micro = prf[prf.average == "micro"].set_index("prompt_condition")
    macro = prf[prf.average == "macro"].set_index("prompt_condition")
    out = pd.DataFrame({
        "accuracy": acc,
        "precision_micro": micro["precision"],
        "recall_micro": micro["recall"],
        "f1_micro": micro["f1"],
        "f1_macro": macro["f1"],
    }).reset_index()
    return out.round(4), scored


def accuracy_by_prompt(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Top-1 accuracy per prompt (requires 'correct' column)."""
    return (pred_df.groupby("prompt_condition")["correct"]
            .agg(["mean", "sum", "count"]).rename(columns={"mean": "accuracy"})
            .reset_index())


def accuracy_by_prompt_difficulty(pred_df: pd.DataFrame) -> pd.DataFrame:
    return (pred_df.groupby(["prompt_condition", "difficulty"])["correct"]
            .mean().reset_index(name="accuracy"))


def multilabel_prf_by_difficulty(pred_df: pd.DataFrame, study_df: pd.DataFrame,
                                 u_policy: str = "ignore",
                                 prefix: str = DEFAULT_PREFIX) -> pd.DataFrame:
    """Micro P/R/F1 per (prompt condition, difficulty stratum)."""
    truth_by_uid = study_df.set_index("uid")
    classes = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
    diff_by_uid = truth_by_uid["difficulty"]
    df = pred_df.copy()
    df["difficulty"] = df["uid"].map(diff_by_uid)
    rows = []
    for (cond, diff), g in df.groupby(["prompt_condition", "difficulty"]):
        y_true, y_pred = [], []
        for _, r in g.iterrows():
            row = truth_by_uid.loc[r["uid"]]
            t = _truth_positive_set(row, u_policy, prefix)
            p = _pred_set(r)
            y_true.append([1 if c in t else 0 for c in classes])
            y_pred.append([1 if c in p else 0 for c in classes])
        pr, rc, f1, _ = precision_recall_fscore_support(
            np.array(y_true), np.array(y_pred), average="micro", zero_division=0)
        rows.append({"prompt_condition": cond, "difficulty": diff, "n": len(g),
                     "precision": pr, "recall": rc, "f1": f1})
    return pd.DataFrame(rows).round(4)


def multilabel_prf_masked(pred_df: pd.DataFrame, study_df: pd.DataFrame,
                          cell_mask, u_policy: str = "ones", prefix: str = "lblcx_",
                          classes: list[str] | None = None, hierarchy: bool = True,
                          pred_fn=None) -> pd.DataFrame:
    """Micro P/R/F1 per prompt, scoring ONLY (study, class) cells for which
    `cell_mask(uid, class)` is True (EDIT 4 — stratify by reference confidence).
    Reuses the U-ignore masking idea: a masked-out cell contributes to no TP/FP/FN.
    `pred_fn` overrides the predictor (e.g. normalize.structured_pred_set)."""
    truth_by_uid = study_df.set_index("uid")
    if classes is None:
        classes = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
    getpred = pred_fn or _pred_set
    rows = []
    for cond, g in pred_df.groupby("prompt_condition"):
        tp = fp = fn = 0
        for _, r in g.iterrows():
            uid = r["uid"]
            t = _truth_positive_set(truth_by_uid.loc[uid], u_policy, prefix)
            p = getpred(r)
            if hierarchy:
                t, p = propagate_hierarchy(t), propagate_hierarchy(p)
            for c in classes:
                if not cell_mask(uid, c):
                    continue
                ti, pi = c in t, c in p
                tp += ti and pi; fp += (not ti) and pi; fn += ti and (not pi)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"prompt_condition": cond, "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4), "n_cells": tp + fp + fn})
    return pd.DataFrame(rows)
