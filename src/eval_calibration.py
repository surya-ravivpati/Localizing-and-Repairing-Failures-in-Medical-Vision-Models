"""Confidence calibration metrics (protocol §7.4).

ECE, MCE, Brier, over/under-confidence, and reliability-diagram bins. Confidence
is the verbalized 0-100 score / 100; correctness comes from eval_accuracy.
Report per prompt AND per difficulty stratum (global ECE hides the H4 effect).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bins(conf: np.ndarray, correct: np.ndarray, n_bins: int):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            out.append((edges[b], edges[b + 1], 0, np.nan, np.nan)); continue
        out.append((edges[b], edges[b + 1], int(m.sum()),
                    conf[m].mean(), correct[m].mean()))
    return out


def calibration_metrics(conf01: np.ndarray, correct: np.ndarray,
                        n_bins: int = 15) -> dict:
    conf01 = np.asarray(conf01, float)
    correct = np.asarray(correct, float)
    n = len(conf01)
    if n == 0:
        return {}
    bins = _bins(conf01, correct, n_bins)
    ece = mce = 0.0
    for lo, hi, cnt, avg_conf, acc in bins:
        if cnt == 0:
            continue
        gap = abs(avg_conf - acc)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
    brier = float(np.mean((conf01 - correct) ** 2))
    over = float(np.mean((conf01 >= 0.5) & (correct == 0)))
    under = float(np.mean((conf01 < 0.5) & (correct == 1)))
    return {"n": n, "ece": ece, "mce": mce, "brier": brier,
            "overconfidence_rate": over, "underconfidence_rate": under,
            "mean_conf": float(conf01.mean()), "accuracy": float(correct.mean())}


def calibration_by_prompt(pred_df: pd.DataFrame, n_bins: int = 15) -> pd.DataFrame:
    rows = []
    for cond, g in pred_df.groupby("prompt_condition"):
        m = calibration_metrics(g["confidence"].values / 100.0,
                                 g["correct"].astype(int).values, n_bins)
        m["prompt_condition"] = cond
        rows.append(m)
    return pd.DataFrame(rows)


def calibration_by_prompt_difficulty(pred_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for (cond, diff), g in pred_df.groupby(["prompt_condition", "difficulty"]):
        m = calibration_metrics(g["confidence"].values / 100.0,
                                 g["correct"].astype(int).values, n_bins)
        m.update({"prompt_condition": cond, "difficulty": diff})
        rows.append(m)
    return pd.DataFrame(rows)


def reliability_bins(pred_df: pd.DataFrame, condition: str, n_bins: int = 10) -> pd.DataFrame:
    g = pred_df[pred_df.prompt_condition == condition]
    bins = _bins(g["confidence"].values / 100.0, g["correct"].astype(int).values, n_bins)
    return pd.DataFrame(bins, columns=["lo", "hi", "count", "mean_conf", "accuracy"])
