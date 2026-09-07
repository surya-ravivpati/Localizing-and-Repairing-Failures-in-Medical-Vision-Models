"""Map free-text model diagnoses into the 14-class CheXpert label space.

Fixed, pre-registered normalizer (protocol §7.1): dictionary/substring first,
returns a set of matched classes. "No Finding" is returned for explicit normal
phrasings. Unmappable strings return an empty set and are logged upstream.

For a production run, add an embedding-similarity fallback; the interface
(`normalize_dx`) stays the same.
"""
from __future__ import annotations

import re

from .labeling import LEXICON, CHEXPERT_CLASSES, NEG_CUES

_NORMAL = re.compile(
    r"no acute|normal|no finding|unremarkable|clear lung|indeterminate|"
    r"insufficient evidence|no cardiopulmonary", re.I)

# Audit 2026-07-19: normalize_dx was pure substring matching with NO negation
# scoping, so "No evidence of pneumonia" scored as a POSITIVE Pneumonia
# prediction (68 such cases across the 5 conditions). Findings are now negated
# when a negation cue precedes them within the same clause.
_NEG_RE = re.compile("|".join(NEG_CUES), re.I)
_CLAUSE_SPLIT = re.compile(r",|;|\bwith\b|\band\b|\bbut\b|\.", re.I)


def _clause_hits(clause: str) -> set[str]:
    """Classes asserted (not negated) in one clause."""
    neg = _NEG_RE.search(clause)
    neg_at = neg.start() if neg else None
    hits = set()
    for cls, patterns in LEXICON.items():
        for p in patterns:
            m = re.search(p, clause, re.I)
            if not m:
                continue
            # a finding is negated only if the cue comes BEFORE it in the clause
            if neg_at is not None and m.start() > neg_at:
                continue
            hits.add(cls)
            break
    return hits


def normalize_dx(text: str, negation: bool = True) -> set[str]:
    if not isinstance(text, str) or not text:   # handle NaN / empty error rows
        return set()
    t = text.lower()
    if negation:
        hits = set()
        for clause in _CLAUSE_SPLIT.split(t):
            if clause.strip():
                hits |= _clause_hits(clause)
    else:                                        # legacy behaviour
        hits = {cls for cls, pats in LEXICON.items()
                if any(re.search(p, t, re.I) for p in pats)}
    if not hits and _NORMAL.search(t):
        hits.add("No Finding")
    return hits


_PRESENT = {"present", "yes", "positive", "seen", "visible"}


def _is_present(finding: dict) -> bool:
    """True iff a structured finding entry is marked present (via `presence` or the
    `status` key used by the structured evidence-first prompt), case-insensitively.
    UNCERTAIN and ABSENT are NOT present; a malformed/missing status is NOT present
    (a parse failure must never become a positive finding — EDIT 2)."""
    val = finding.get("presence", finding.get("status", ""))
    return str(val).strip().lower() == "present"


def structured_pred_set(resp: dict, include_uncertain: bool = False) -> set[str]:
    """STRICT structured predictor (EDIT 2): read ONLY the machine-readable `findings`
    table (present, optionally uncertain), mapped to canonical classes — bypassing
    the prose synonym mapper. Robust to minor formatting differences (presence/status,
    upper/lower case). Returns an EMPTY set when there is no valid findings object, so
    a parse failure is never scored as a positive finding.

    Finding keys are resolved in this order: (1) the region-based radiology
    checklist mapping (CHECKLIST_TO_CHEXPERT — e.g. "Effusion" -> {Pleural
    Effusion}, for the `radiology_checklist` condition), (2) an exact/near-exact
    CheXpert class name (the other structured conditions use these as keys
    directly), (3) normalize_dx as a last-resort synonym lookup. Broad region
    headers with no concrete CheXpert mapping (Lungs, Pleura) contribute nothing on
    their own — a PRESENT there is a qualitative signal without a scoreable class.
    """
    findings = resp.get("findings") if isinstance(resp, dict) else None
    if not (isinstance(findings, dict) and findings):
        return set()
    from .prompts import CHECKLIST_TO_CHEXPERT
    pos = set()
    for cls, v in findings.items():
        if not isinstance(v, dict):
            continue
        status = str(v.get("presence", v.get("status", ""))).strip().lower()
        if not (status == "present" or (include_uncertain and status == "uncertain")):
            continue
        key = cls.strip()
        if key in CHECKLIST_TO_CHEXPERT:
            pos |= CHECKLIST_TO_CHEXPERT[key]
        else:
            hit = normalize_dx(key)
            pos |= hit if hit else {key}
    return {c for c in pos if c != "No Finding"}


def predicted_positive_set(resp: dict, use_differential: bool = False,
                           ddx_min_prob: int = 50) -> set[str]:
    """Predicted positive CheXpert classes for a response.

    Reads the richest structured field the prompt produced, so no condition is
    penalised by free-text normalization loss (audit 2026-07-19 — evidence_first
    and ddx emit structured fields that were previously discarded):
      * 'findings'           — structured prompts (name -> {presence, confidence})
      * 'extracted_evidence' — evidence_first ([{finding, location, presence}])
      * 'differential'       — ddx ([{diagnosis, probability}]); OFF by default, a
        differential lists candidate hypotheses rather than asserted findings, so
        counting it changes the semantics. Enable explicitly for a sensitivity
        analysis; `ddx_min_prob` gates which candidates count.
      * else the free-text primary_diagnosis.
    """
    findings = resp.get("findings")
    if isinstance(findings, dict) and findings:
        from .prompts import CHECKLIST_TO_CHEXPERT
        pos = set()
        for cls, v in findings.items():
            # accept either {presence:...} (existing structured) or {status:...}
            # (structured evidence-first, EDIT 1), case-insensitively.
            if isinstance(v, dict) and _is_present(v):
                key = cls.strip()
                if key in CHECKLIST_TO_CHEXPERT:
                    # region-based checklist item (radiology_checklist condition)
                    pos |= CHECKLIST_TO_CHEXPERT[key]
                else:
                    # map the model's finding name into the canonical class space
                    hit = normalize_dx(key)
                    pos |= hit if hit else {key}
        return {c for c in pos if c != "No Finding"}

    ev = resp.get("extracted_evidence")
    if isinstance(ev, list) and ev:
        pos = set()
        for item in ev:
            if not isinstance(item, dict):
                continue
            if str(item.get("presence", "")).lower() in _PRESENT:
                pos |= normalize_dx(str(item.get("finding", "")))
        # union with the free-text conclusion so a stated diagnosis is never lost
        pos |= normalize_dx(resp.get("primary_diagnosis", ""))
        return {c for c in pos if c != "No Finding"}

    if use_differential:
        diff = resp.get("differential")
        if isinstance(diff, list) and diff:
            pos = set()
            for item in diff:
                if not isinstance(item, dict):
                    continue
                try:
                    prob = float(item.get("probability", 0))
                except (TypeError, ValueError):
                    prob = 0
                if prob >= ddx_min_prob:
                    pos |= normalize_dx(str(item.get("diagnosis", "")))
            pos |= normalize_dx(resp.get("primary_diagnosis", ""))
            return {c for c in pos if c != "No Finding"}

    return {c for c in normalize_dx(resp.get("primary_diagnosis", "")) if c != "No Finding"}


def primary_class(text: str) -> str | None:
    """Single best class for top-1 accuracy (first CheXpert-order match)."""
    hits = normalize_dx(text)
    if not hits:
        return None
    for c in CHEXPERT_CLASSES:
        if c in hits:
            return c
    return None
