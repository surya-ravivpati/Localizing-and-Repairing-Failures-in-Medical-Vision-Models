"""Rule-based report labeler over the 14 CheXpert classes.

This is a transparent, dependency-free APPROXIMATION of the CheXpert labeler
(Irvin et al. 2019) / CheXbert (Smit et al. 2020). It performs:
  1. sentence splitting,
  2. lexicon matching per class,
  3. negation + uncertainty scoping (a simplified NegBio-style cue search),
producing per-class states:  1 = positive, 0 = negative, -1 = uncertain,
NaN = not mentioned.

IMPORTANT (protocol §3.2, §23): for a publishable run this should be validated
on the dev set against, and ideally swapped for, the real CheXbert model. It is
provided so the whole pipeline runs deterministically and offline today. Treat
its outputs as a "bronze" reference; use dual-labeler agreement for the primary
analysis. The interface (`label_report`) is stable so CheXbert can drop in.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

CHEXPERT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Lexicons: class -> list of regex phrase patterns (word-boundary matched).
LEXICON: dict[str, list[str]] = {
    "Enlarged Cardiomediastinum": [
        r"enlarged cardiomediastin", r"widened mediastin", r"mediastinal widening",
        r"cardiomediastinal silhouette (is )?enlarged", r"prominent mediastin",
    ],
    "Cardiomegaly": [
        r"cardiomegaly", r"enlarged (cardiac|heart)", r"cardiac (silhouette )?enlarge",
        r"heart size is enlarged", r"enlargement of the (cardiac|heart)",
        # standard synonyms (audit 2026-07-19: CHF mapped to nothing, losing TPs —
        # on CXR congestive heart failure implies an enlarged heart)
        r"congestive heart failure", r"\bchf\b", r"cardiac hypertrophy",
        r"heart (is |appears )?(mildly |moderately |markedly )?enlarged",
    ],
    "Lung Opacity": [
        r"opacit", r"airspace disease", r"air space disease", r"infiltrat",
        r"reticular", r"interstitial marking", r"density", r"densit",
        r"ground.?glass", r"hazy", r"haziness", r"patchy",
    ],
    "Lung Lesion": [
        r"nodul", r"mass", r"lesion", r"lung cancer", r"neoplasm", r"granuloma",
    ],
    "Edema": [
        r"edema", r"vascular congestion", r"pulmonary congestion", r"fluid overload",
        # CHF is the canonical clinical term for cardiogenic pulmonary edema
        r"congestive heart failure", r"\bchf\b", r"kerley",
        r"cephalization", r"pulmonary venous (hypertension|congestion)",
    ],
    "Consolidation": [
        r"consolidat",
    ],
    "Pneumonia": [
        r"pneumonia", r"infectious process", r"bronchopneumonia",
    ],
    "Atelectasis": [
        r"atelecta", r"collapse", r"volume loss",
        r"plate.?like", r"discoid", r"subsegmental",
    ],
    "Pneumothorax": [
        r"pneumothorax", r"pneumothoraces",
    ],
    "Pleural Effusion": [
        r"pleural effusion", r"effusion", r"pleural fluid",
        # NOTE: costophrenic blunting was tried and REVERTED (2026-07-19) — in this
        # corpus it is usually described as *chronic* blunting (pleural scarring),
        # not effusion, and it cost 0.055 kappa against CheXbert.
    ],
    "Pleural Other": [
        r"pleural thickening", r"pleural scarring", r"fibrothorax", r"pleural plaque",
    ],
    "Fracture": [
        r"fracture", r"fractured",
    ],
    "Support Devices": [
        r"catheter", r"pacemaker", r"\bpicc\b", r"endotracheal tube", r"\bett\b",
        r"central line", r"chest tube", r"\bstent\b", r"sternotomy (wire|XXXX)",
        r"\bicd\b", r"port-?a-?cath", r"\btubes?\b", r"\blines?\b", r"\bclips?\b",
        r"\bdevices?\b",
        # "postsurgical changes" alone implies altered anatomy, NOT a visible
        # device — only clips/hardware count (tested 2026-07-19).
        r"sternotomy", r"surgical (clip|hardware)", r"\bwires?\b",
        r"tracheostomy", r"nasogastric", r"pigtail", r"\bcabg\b",
        r"valve replacement", r"prosthe",
    ],
}

NEG_CUES = [
    r"\bno\b", r"\bnot\b", r"without", r"\bnegative for\b", r"no evidence of",
    r"free of", r"absence of", r"\babsent\b", r"resolved", r"\bclear\b",
    r"unremarkable", r"within normal limits", r"\bnormal\b", r"ruled out",
    r"no acute", r"no focal", r"no significant",
]
UNC_CUES = [
    r"\bmay\b", r"possibl", r"cannot (be )?exclude", r"can(no)?t rule out",
    r"questionable", r"\bsuggest", r"concern(ing)? for", r"\bcould\b",
    r"borderline", r"suspect", r"\blikely\b", r"probabl", r"differential",
    r"\bversus\b", r"\bvs\.?\b", r"\beither\b", r"\bmight\b", r"equivocal",
    r"\bappears?\b", r"\bconsider\b", r"if clinically",
]
NORMAL_PHRASES = [
    r"no acute cardiopulmonary", r"normal chest", r"no acute (disease|abnormalit|findings?)",
    r"clear lungs", r"lungs (are )?clear", r"no active disease",
]

_neg_re = re.compile("|".join(NEG_CUES), re.I)
# finding-then-denial, anchored just after the finding mention (see _state_for_mention)
_postneg_re = re.compile(
    r"[\w\s,():;-]{0,40}?\b(?:is|are|was|were)?\s*"
    r"(?:not (?:present|seen|identified|visuali[sz]ed|appreciated)|"
    r"absent|none|negative|unremarkable|resolved|no longer)\b", re.I)
_unc_re = re.compile("|".join(UNC_CUES), re.I)
_normal_re = re.compile("|".join(NORMAL_PHRASES), re.I)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    # split on ., ; and newlines; keep it simple and deterministic.
    parts = re.split(r"(?<=[.;])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _state_for_sentence(sentence: str) -> tuple[bool, bool]:
    """Return (is_negated, is_uncertain) for a sentence (whole-sentence scope)."""
    return bool(_neg_re.search(sentence)), bool(_unc_re.search(sentence))


def _state_for_mention(sentence: str, at: int) -> tuple[bool, bool]:
    """(is_negated, is_uncertain) for a finding mentioned at offset `at`.

    Audit 2026-07-19: whole-sentence scoping wrongly negated findings asserted
    BEFORE a later negation cue — e.g. "Borderline enlargement of the cardiac
    silhouette without acute pulmonary disease" marked Cardiomegaly negative
    (148 such mentions in the corpus). A cue only applies to findings that
    follow it.
    """
    neg = any(m.start() < at for m in _neg_re.finditer(sentence))
    if not neg:
        # POST-POSED negation: the finding is the subject of its own denial —
        # "pleural effusion is not present", "pneumothorax: none", "effusion is
        # absent". Distinct from "cardiomegaly without edema", where the trailing
        # cue negates a DIFFERENT finding, so bare "without" does not count.
        neg = bool(_postneg_re.match(sentence[at:]))
    unc = any(m.start() < at for m in _unc_re.finditer(sentence))
    if not unc:
        # hedges that trail the finding still qualify it ("cardiomegaly, likely")
        unc = bool(_unc_re.search(sentence[at:]))
    return neg, unc


def label_report(text: str) -> dict[str, float]:
    """Return {class: state} where state in {1, 0, -1, NaN}.

    Aggregation rule per class across mentions (CheXpert-like precedence):
      positive > uncertain > negative.  A class never mentioned stays NaN.
    'No Finding' = 1 iff report is clearly normal and no positive pathology found.
    """
    labels: dict[str, float] = {c: np.nan for c in CHEXPERT_CLASSES}
    sentences = split_sentences(text)

    for sent in sentences:
        for cls, patterns in LEXICON.items():
            # earliest mention of this class in the sentence, if any
            starts = [m.start() for p in patterns
                      for m in [re.search(p, sent, re.I)] if m]
            if starts:
                neg, unc = _state_for_mention(sent, min(starts))
                if unc:
                    state = -1.0
                elif neg:
                    state = 0.0
                else:
                    state = 1.0
                prev = labels[cls]
                # precedence: 1 > -1 > 0 > NaN
                order = {1.0: 3, -1.0: 2, 0.0: 1}
                if np.isnan(prev) or order[state] > order.get(prev, 0):
                    labels[cls] = state

    # No Finding logic.
    any_positive = any(labels[c] == 1.0 for c in CHEXPERT_CLASSES if c != "No Finding")
    any_uncertain = any(labels[c] == -1.0 for c in CHEXPERT_CLASSES if c != "No Finding")
    normal_signal = bool(_normal_re.search(text or ""))
    if normal_signal and not any_positive and not any_uncertain:
        labels["No Finding"] = 1.0
    elif any_positive:
        labels["No Finding"] = 0.0
    # else leave NaN (indeterminate)
    return labels


def is_hedged(text: str) -> bool:
    """Report expresses diagnostic uncertainty (protocol §11 ambiguity signal)."""
    return bool(_unc_re.search(text or ""))


def label_frame(df: pd.DataFrame, text_col: str = "report_text",
                prefix: str = "lbl_", summary: bool = True) -> pd.DataFrame:
    """Add one column per CheXpert class (given `prefix`) from `text_col`.

    Two label sets are used in the study (protocol §3.3 report-bias mitigation):
      - prefix 'lbl_'  from findings+impression  -> comprehensive (incl. chronic/
        incidental findings); drives difficulty stratification.
      - prefix 'lblimp_' from the impression only -> the radiologist's ACUTE
        bottom line; the primary accuracy/F1 reference so that a chronic finding
        (old granuloma, pleural thickening, surgical clips) is not scored as a
        model "miss" when the impression itself says 'no acute abnormality'.
    """
    rows = df[text_col].fillna("").map(label_report)
    lab = pd.DataFrame(list(rows), index=df.index)
    lab.columns = [f"{prefix}{c}" for c in CHEXPERT_CLASSES]
    out = pd.concat([df, lab], axis=1)
    if summary:
        path_cols = [f"{prefix}{c}" for c in CHEXPERT_CLASSES if c != "No Finding"]
        out["n_positive"] = (out[path_cols] == 1.0).sum(axis=1)
        out["n_uncertain"] = (out[path_cols] == -1.0).sum(axis=1)
        out["is_hedged"] = df[text_col].fillna("").map(is_hedged)
    return out


if __name__ == "__main__":
    demo = [
        "The cardiac silhouette and mediastinum size are within normal limits. There is no pneumothorax. Normal chest.",
        "Borderline cardiomegaly. Enlarged pulmonary arteries. Clear lungs.",
        "There is a large right pleural effusion. Findings may represent pneumonia.",
    ]
    for t in demo:
        lab = label_report(t)
        pos = {k: v for k, v in lab.items() if not np.isnan(v)}
        print(f"hedged={is_hedged(t)} | {pos}\n  {t}\n")
