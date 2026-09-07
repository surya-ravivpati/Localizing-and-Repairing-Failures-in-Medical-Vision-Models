"""Claim-level groundedness / reference-consistency (protocol §7.2).

Atomize the model's explanation + extracted_evidence + cited_findings into
finding-claims, then adjudicate each against the report-derived reference into
four states:
    supported | contradicted | unverifiable | unsupported_plausible

Reference-consistency is scored against an INDEPENDENT report (Design A), so
"unverifiable" (report silent) is reported separately and NOT counted as wrong
(reporting-bias mitigation, protocol §3.3).

Claim extraction here is lexicon-based (labeling.LEXICON) — a transparent stand-in
for RadGraph. Swap in RadGraph entities without changing the adjudication logic.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .labeling import LEXICON, CHEXPERT_CLASSES
from .labeling import _neg_re, _unc_re  # reuse negation/uncertainty cues

STATES = ["supported", "contradicted", "unverifiable", "unsupported_plausible"]
FAB_PATTERNS = [r"\d+(\.\d+)?\s?cm", r"\d+(\.\d+)?\s?mm", r"compared to prior",
                r"since (the )?previous", r"new since", r"measured", r"prior study",
                r"\d+(\.\d+)?\s?(cm|mm) (nodule|effusion|diameter)"]
_fab_re = re.compile("|".join(FAB_PATTERNS), re.I)


def extract_claims(resp: dict) -> list[dict]:
    """Return list of {text, cls, presence} claims from a model response."""
    claims = []

    def add(text, presence_hint=None):
        if not text:
            return
        t = str(text)
        neg = bool(_neg_re.search(t))
        unc = bool(_unc_re.search(t))
        presence = presence_hint or ("absent" if neg else ("uncertain" if unc else "present"))
        for cls, patterns in LEXICON.items():
            if any(re.search(p, t, re.I) for p in patterns):
                claims.append({"text": t[:120], "cls": cls, "presence": presence})

    for e in resp.get("extracted_evidence", []) or []:
        add(e.get("finding", ""), e.get("presence"))
    for f in resp.get("cited_findings", []) or []:
        add(f)
    # structured prompt: read the findings object directly
    findings = resp.get("findings")
    if isinstance(findings, dict):
        for name, v in findings.items():
            if isinstance(v, dict):
                pres = str(v.get("presence", "")).lower()
                add(name, pres if pres in ("present", "absent", "uncertain") else None)
    add(resp.get("explanation", ""))
    # de-dup by (cls, presence)
    seen, uniq = set(), []
    for c in claims:
        key = (c["cls"], c["presence"])
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq


def adjudicate_claim(claim: dict, ref_row: pd.Series) -> str:
    cls = claim["cls"]
    ref = ref_row.get(f"lbl_{cls}")
    pres = claim["presence"]
    claim_positive = (pres == "present")
    if np.isnan(ref):
        return "unverifiable"
    ref_positive = (ref == 1.0)
    ref_negative = (ref == 0.0)
    if claim_positive and ref_positive:
        return "supported"
    if (not claim_positive) and ref_negative:
        return "supported"
    if claim_positive and ref_negative:
        return "contradicted"
    if (not claim_positive) and ref_positive:
        return "contradicted"
    return "unsupported_plausible"  # ref uncertain (-1)


def score_groundedness(pred_df: pd.DataFrame, study_df: pd.DataFrame) -> pd.DataFrame:
    """Per-response groundedness stats."""
    truth = study_df.set_index("uid")
    rows = []
    for _, r in pred_df.iterrows():
        resp = json.loads(r["response"])
        ref_row = truth.loc[r["uid"]]
        claims = extract_claims(resp)
        counts = {s: 0 for s in STATES}
        for c in claims:
            counts[adjudicate_claim(c, ref_row)] += 1
        verifiable = counts["supported"] + counts["contradicted"] + counts["unsupported_plausible"]
        support_rate = counts["supported"] / verifiable if verifiable else np.nan
        # fabricated specifics (measurements/priors) anywhere in the text
        text_blob = " ".join([resp.get("explanation", ""),
                              " ".join(resp.get("cited_findings", []) or []),
                              " ".join(e.get("finding", "") for e in
                                       (resp.get("extracted_evidence", []) or []))])
        n_fab = len(_fab_re.findall(text_blob))
        rows.append({
            "uid": r["uid"], "prompt_condition": r["prompt_condition"],
            "difficulty": r["difficulty"], "n_claims": len(claims),
            **counts, "support_rate": support_rate,
            "contradiction_rate": counts["contradicted"] / len(claims) if claims else 0.0,
            "n_fabricated": n_fab,
        })
    return pd.DataFrame(rows)


def groundedness_by_prompt(g_df: pd.DataFrame) -> pd.DataFrame:
    return (g_df.groupby("prompt_condition")
            .agg(support_rate=("support_rate", "mean"),
                 contradiction_rate=("contradiction_rate", "mean"),
                 mean_claims=("n_claims", "mean"),
                 unverifiable=("unverifiable", "mean"))
            .reset_index())
