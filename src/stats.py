"""Statistical analysis (protocol §10).

Paired tests over the repeated-measures design (same case x 5 prompts):
  - McNemar (paired binary: accuracy, hallucination-free),
  - Wilcoxon signed-rank (paired ordinal/skewed: support-rate, judge scores),
  - bootstrap CIs (ΔECE, Δrate),
  - Benjamini-Hochberg FDR correction,
  - GLMM (mixed-effects logistic) for the prompt x difficulty interaction (H4).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def _paired_pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="uid", columns="prompt_condition", values=value)


def mcnemar_all_pairs(pred_df: pd.DataFrame, value: str = "correct") -> pd.DataFrame:
    piv = _paired_pivot(pred_df, value).dropna()
    conds = list(piv.columns)
    rows = []
    for a, b in itertools.combinations(conds, 2):
        ya, yb = piv[a].astype(int), piv[b].astype(int)
        n01 = int(((ya == 0) & (yb == 1)).sum())
        n10 = int(((ya == 1) & (yb == 0)).sum())
        table = [[0, n01], [n10, 0]]
        res = mcnemar(table, exact=(n01 + n10) < 25)
        rows.append({"a": a, "b": b, "mean_a": ya.mean(), "mean_b": yb.mean(),
                     "n01": n01, "n10": n10, "statistic": res.statistic,
                     "p_value": res.pvalue})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def wilcoxon_all_pairs(df: pd.DataFrame, value: str) -> pd.DataFrame:
    piv = _paired_pivot(df, value)
    conds = list(piv.columns)
    rows = []
    for a, b in itertools.combinations(conds, 2):
        sub = piv[[a, b]].dropna()
        if len(sub) < 5 or (sub[a] - sub[b]).abs().sum() == 0:
            rows.append({"a": a, "b": b, "median_a": sub[a].median(),
                         "median_b": sub[b].median(), "statistic": np.nan,
                         "p_value": 1.0, "cliffs_delta": 0.0}); continue
        stat, p = stats.wilcoxon(sub[a], sub[b])
        rows.append({"a": a, "b": b, "median_a": sub[a].median(),
                     "median_b": sub[b].median(), "statistic": stat, "p_value": p,
                     "cliffs_delta": _cliffs_delta(sub[a].values, sub[b].values)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def _cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def bootstrap_diff(x: np.ndarray, y: np.ndarray, stat=np.mean, n_boot=10000,
                   seed=0) -> dict:
    """Paired bootstrap CI for stat(x) - stat(y) (e.g., ΔECE across cases)."""
    rng = np.random.RandomState(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[i] = stat(x[idx]) - stat(y[idx])
    return {"point": float(stat(x) - stat(y)),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5))}


def glmm_interaction(pred_df: pd.DataFrame, outcome: str = "correct",
                     ref_prompt: str = "direct") -> str:
    """Mixed-effects model: outcome ~ prompt * difficulty + (1|uid).

    Primary fit is a LINEAR mixed model (linear probability model with a random
    case intercept) via statsmodels MixedLM — it converges reliably and its
    interaction coefficients are directly interpretable as probability changes.
    For a strict logistic GLMM, swap to BinomialBayesMixedGLM; for a frequentist
    logistic alternative the except-branch fits cluster-robust Logit. Report the
    logistic version as a robustness check in the manuscript (protocol §10.1).
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    d = pred_df.copy()
    d[outcome] = d[outcome].astype(int)
    d["prompt_condition"] = pd.Categorical(
        d["prompt_condition"],
        categories=[ref_prompt] + [c for c in d.prompt_condition.unique() if c != ref_prompt])
    try:
        md = smf.mixedlm(f"{outcome} ~ C(prompt_condition)*C(difficulty)",
                         d, groups=d["uid"])
        fit = md.fit(method="lbfgs", maxiter=200)
        return str(fit.summary())
    except Exception as e:  # fallback: cluster-robust logistic
        model = smf.logit(f"{outcome} ~ C(prompt_condition)*C(difficulty)", d)
        fit = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["uid"]})
        return f"[GLMM fallback: cluster-robust Logit] {e}\n\n{fit.summary()}"


def dissociation_test(judge_df: pd.DataFrame, ground_df: pd.DataFrame) -> pd.DataFrame:
    """H3: is a prompt more persuasive (coherence) WITHOUT being more faithful?

    Compares each prompt vs 'direct' on judge coherence and on groundedness
    support-rate; flags dissociation = coherence up AND faithfulness not up.
    """
    coh = (judge_df.groupby("prompt_condition")["score_diagnostic_coherence"]
           .mean())
    faith = (judge_df.groupby("prompt_condition")["score_faithfulness"].mean())
    sr = ground_df.groupby("prompt_condition")["support_rate"].mean()
    base = "direct"
    rows = []
    for c in coh.index:
        if c == base:
            continue
        rows.append({"prompt_condition": c,
                     "d_coherence": coh[c] - coh[base],
                     "d_faithfulness": faith[c] - faith[base],
                     "d_support_rate": sr.get(c, np.nan) - sr.get(base, np.nan),
                     "dissociation": (coh[c] - coh[base] > 0) and
                                     (faith[c] - faith[base] <= 0.05)})
    return pd.DataFrame(rows)
