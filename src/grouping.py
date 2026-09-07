"""Clinically coherent finding groups for multi-granularity evaluation.

Motivation (audit 2026-07-19): several fine-grained CheXpert distinctions have
inter-labeler κ as low as 0.12 — the *reference itself* cannot reliably separate
them, so a 13-class micro-F1 is partly measuring label noise, not model skill.
Evaluating at a coarser but clinically actionable granularity is a legitimate,
transparent way to report what the model can actually do.

Groups follow the standard CXR review framework (parenchyma / pleura /
cardiomediastinum / devices / bones) — chosen for clinical coherence, NOT
selected to maximize F1. Always report the full ladder, never a single
cherry-picked level.
"""
from __future__ import annotations

# 5 clinical triage groups
CLINICAL_GROUPS: dict[str, set[str]] = {
    "Airspace/Opacity": {"Lung Opacity", "Consolidation", "Edema", "Pneumonia",
                         "Atelectasis", "Lung Lesion"},
    "Cardiomediastinal": {"Cardiomegaly", "Enlarged Cardiomediastinum"},
    "Pleural": {"Pleural Effusion", "Pneumothorax", "Pleural Other"},
    "Support Devices": {"Support Devices"},
    "Fracture": {"Fracture"},
}

# 3 core pathology groups (excludes devices + bones: not acute cardiopulmonary)
CORE_PATHOLOGY_GROUPS: dict[str, set[str]] = {
    k: v for k, v in CLINICAL_GROUPS.items()
    if k in ("Airspace/Opacity", "Cardiomediastinal", "Pleural")
}

# binary triage: any acute abnormality (support devices are not pathology)
ACUTE_EXCLUDE = {"Support Devices"}


def to_groups(findings: set[str], groups: dict[str, set[str]]) -> set[str]:
    """Map a fine-grained finding set to the group level."""
    return {g for g, members in groups.items() if findings & members}


def any_acute(findings: set[str], exclude: set[str] = ACUTE_EXCLUDE) -> bool:
    return bool(set(findings) - exclude - {"No Finding"})
