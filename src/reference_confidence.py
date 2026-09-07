"""Multi-labeler reference-confidence layer (EDIT 3 / EDIT 4).

Combines the three report labelers already produced into the studies table —
rule-based (`lbl_`), CheXbert (`lblcx_`), VisualCheXbert (`lblvcx_`) — WITHOUT
discarding any of them. For each (study, finding) it preserves the individual labels
and derives a consensus, an agreement count, and a reference-confidence level:

    HIGH   — all 3 labelers agree
    MEDIUM — 2 of 3 agree
    LOW    — labelers disagree (only meaningful with 3 labelers)

Labeler-agnostic: it uses whichever `lbl*_<class>` columns exist, so it degrades
cleanly to 2 labelers if VisualCheXbert has not been run (then HIGH = both agree,
MEDIUM/LOW collapse). Reuses the project CHEXPERT_CLASSES taxonomy.
"""
from __future__ import annotations

import pandas as pd

from .labeling import CHEXPERT_CLASSES

# labeler prefix -> friendly name (kept in the preserved per-finding record)
LABELERS = {"lbl_": "rule_based", "lblcx_": "chexbert", "lblvcx_": "visualchexbert"}


def available_labelers(study_df: pd.DataFrame) -> list[str]:
    """Prefixes actually present as columns, in canonical order."""
    return [p for p in LABELERS if any(c.startswith(p) for c in study_df.columns)]


def _state(row, prefix, cls) -> str:
    """PRESENT / ABSENT / UNCERTAIN for one labeler on one (study, finding).
    lbl_/lblcx_ are ternary (CheXpert convention: 1.0/0.0/-1.0). lblvcx_
    (VisualCheXbert) is binary by design — its logreg heads only emit 0/1, it has
    no uncertain state — so it can only ever vote PRESENT or ABSENT."""
    v = row.get(f"{prefix}{cls}")
    if v == 1.0:
        return "PRESENT"
    if v == -1.0:
        return "UNCERTAIN"
    return "ABSENT"


def finding_record(row: pd.Series, cls: str, prefixes: list[str]) -> dict:
    """Preserve individual labels + consensus/agreement/confidence for one
    (study, finding). Individual labels are NEVER discarded (EDIT 3).

    With 3 BINARY voters a genuine tie is mathematically impossible (a majority
    of >=2 always exists), so a naive present/absent vote can never produce LOW.
    Real disagreement instead shows up as one of the ternary-capable labelers
    (rule-based / CheXbert) going UNCERTAIN while the other two split PRESENT vs
    ABSENT — a true 3-way disagreement with no majority. That is what LOW means
    here; it is populated only when this happens.
    """
    states = {LABELERS[p]: _state(row, p, cls) for p in prefixes}
    n = len(states)
    committed = [s for s in states.values() if s != "UNCERTAIN"]
    pos = sum(1 for s in committed if s == "PRESENT")
    neg = len(committed) - pos
    n_uncertain = n - len(committed)

    if n_uncertain == 0:
        agree = max(pos, neg)
        consensus = "PRESENT" if pos >= neg else "ABSENT"
        level = "HIGH" if agree == n else ("MEDIUM" if agree >= (n + 1) // 2 else "LOW")
    elif len(committed) == 0:
        agree, consensus, level = 0, "UNCERTAIN", "LOW"
    else:
        # one labeler uncertain; the rest may still agree (MEDIUM) or split (LOW)
        agree = max(pos, neg)
        consensus = "PRESENT" if pos > neg else ("ABSENT" if neg > pos else "UNCERTAIN")
        level = "MEDIUM" if agree == len(committed) and agree >= 2 else "LOW"

    rec = dict(states)
    rec.update({"consensus": consensus, "agreement": f"{agree}/{n}",
                "confidence": level})
    return rec


def agreement_level(row: pd.Series, cls: str, prefixes: list[str]) -> str:
    return finding_record(row, cls, prefixes)["confidence"]


def confidence_table(study_df: pd.DataFrame) -> pd.DataFrame:
    """Long table (uid, pathology, individual labels, consensus, agreement,
    confidence) across the available labelers."""
    prefixes = available_labelers(study_df)
    classes = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
    rows = []
    for _, row in study_df.iterrows():
        for c in classes:
            rec = finding_record(row, c, prefixes)
            rows.append({"uid": row["uid"], "pathology": c, **rec})
    return pd.DataFrame(rows)


def cell_mask_factory(study_df: pd.DataFrame, level: str, exact: bool = True):
    """Return `mask(uid, class) -> bool`. If `exact`, keep only cells AT `level`
    (for the per-level F1 rows of EDIT 4: HIGH / MEDIUM / LOW separately). If not
    exact, keep cells at `level` or above (cumulative)."""
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    prefixes = available_labelers(study_df)
    by_uid = study_df.set_index("uid")
    target = order[level]

    def mask(uid, cls):
        lvl = order[agreement_level(by_uid.loc[uid], cls, prefixes)]
        return lvl == target if exact else lvl >= target
    return mask
