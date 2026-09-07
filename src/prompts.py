"""The five prompt conditions (protocol §5).

Each builder returns (system, user) strings. The system preamble is IDENTICAL
across conditions; only the reasoning scaffold in the user turn changes. All
conditions demand the same JSON schema so outputs are directly comparable.
"""
from __future__ import annotations

PROMPT_VERSION = "v1.0"

SYSTEM_PREAMBLE = (
    "You are assisting with a research study on chest radiograph interpretation. "
    "You are shown one or more chest X-ray images and (optionally) a short clinical "
    "indication. Report only what is supported by the image. If a finding is not "
    "visible or is ambiguous, say so. Do not invent measurements, priors, or clinical "
    "history. Confidence is an integer 0-100 where: 90-100 = near-certain, "
    "70-89 = probable, 50-69 = favored but uncertain, 30-49 = possible, "
    "0-29 = unlikely/absent. Return ONLY valid JSON matching the requested fields."
)

_SCHEMA_HINT = (
    'Return JSON with keys: "primary_diagnosis" (string), "confidence" (int 0-100), '
    '"explanation" (string)'
)


def _indication(ind) -> str:
    # a missing indication arrives as float NaN, not "" — guard before .strip()
    ind = ind.strip() if isinstance(ind, str) else ""
    return ind if ind else "None provided"


def direct(indication: str) -> tuple[str, str]:
    u = (
        f"Indication: {_indication(indication)}\n"
        "Give your single most likely primary diagnosis (or "
        '"No acute cardiopulmonary finding"), a one-sentence explanation, and your '
        f"confidence (0-100).\n{_SCHEMA_HINT}."
    )
    return SYSTEM_PREAMBLE, u


def cot(indication: str) -> tuple[str, str]:
    u = (
        f"Indication: {_indication(indication)}\n"
        "Think step by step about the visible anatomy (lungs, pleura, heart/mediastinum, "
        "bones, devices) BEFORE concluding. Then give the primary diagnosis, explanation, "
        "and confidence (0-100). Put the step-by-step reasoning in \"reasoning\".\n"
        f"{_SCHEMA_HINT}, and \"reasoning\" (string)."
    )
    return SYSTEM_PREAMBLE, u


def ddx(indication: str) -> tuple[str, str]:
    u = (
        f"Indication: {_indication(indication)}\n"
        "Produce a ranked differential of up to 5 diagnoses. For each: name, brief "
        "supporting rationale from the image, and a probability (0-100). Probabilities "
        "need not sum to 100. Then state the single most likely primary diagnosis and "
        "overall confidence.\n"
        f"{_SCHEMA_HINT}, and \"differential\" (array of "
        '{"diagnosis","probability","rationale"}).'
    )
    return SYSTEM_PREAMBLE, u


def evidence_first(indication: str) -> tuple[str, str]:
    u = (
        f"Indication: {_indication(indication)}\n"
        "STEP 1 - EVIDENCE: List every discrete visual finding you can actually see, each "
        'as {finding, location, presence: present/absent/uncertain}. Include pertinent '
        "negatives.\n"
        "STEP 2 - DIAGNOSIS: Using ONLY the evidence in Step 1, state the primary "
        "diagnosis, an explanation that references specific Step-1 items, and confidence.\n"
        f"{_SCHEMA_HINT}, and \"extracted_evidence\" (array of "
        '{"finding","location","presence"}).'
    )
    return SYSTEM_PREAMBLE, u


def uncertainty_first(indication: str) -> tuple[str, str]:
    u = (
        f"Indication: {_indication(indication)}\n"
        "STEP 1 - UNCERTAINTY: State image-quality limits and which regions are ambiguous "
        "or non-diagnostic. Decide whether a confident read is possible.\n"
        "STEP 2 - DIAGNOSIS: Give primary diagnosis (or explicit \"indeterminate / "
        "insufficient evidence\"), explanation, and a calibrated confidence (0-100) that "
        "reflects Step 1. Abstention is allowed and encouraged when warranted.\n"
        f"{_SCHEMA_HINT}, \"uncertainty_statement\" (string), and \"abstained\" (bool)."
    )
    return SYSTEM_PREAMBLE, u


# The 13 CheXpert pathologies the structured prompt asks about (No Finding is
# implied when all are absent). Matches labeling.CHEXPERT_CLASSES ordering.
STRUCTURED_FINDINGS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]


def structured(indication: str) -> tuple[str, str]:
    """#1: structured multi-label read. Forces the model to CONSIDER each of the
    13 findings (directly attacks under-calling) and removes free-text->class
    normalization loss — the prediction is already in the label space."""
    checklist = ", ".join(STRUCTURED_FINDINGS)
    u = (
        f"Indication: {_indication(indication)}\n"
        "Systematically assess the chest radiograph for EACH of the following "
        f"findings, one at a time: {checklist}.\n"
        "For every finding, decide present / absent / uncertain and give a "
        "confidence 0-100. Do not default to 'absent' — look specifically for each.\n"
        'Return JSON with: "findings" (object mapping each finding name to '
        '{"presence":"present"|"absent"|"uncertain","confidence":int}), '
        '"primary_diagnosis" (string: the single most important positive finding, '
        'or "No acute cardiopulmonary finding" if all absent), '
        '"confidence" (int 0-100 for the primary), and "explanation" (string).'
    )
    return SYSTEM_PREAMBLE, u


def structured_balanced(indication: str) -> tuple[str, str]:
    """Structured scan + base-rate priming to curb over-calling. Keeps the
    systematic 13-finding checklist (recall) but requires a clear abnormality to
    mark 'present' and routes equivocal calls to 'uncertain' (not scored positive),
    improving precision/specificity on the normal-heavy base rate."""
    checklist = ", ".join(STRUCTURED_FINDINGS)
    u = (
        f"Indication: {_indication(indication)}\n"
        "This is a chest radiograph from a general outpatient collection in which "
        "MOST studies are normal. Systematically assess for EACH of the following, "
        f"one at a time: {checklist}.\n"
        "Mark a finding 'present' ONLY if you can clearly and confidently identify a "
        "definite abnormality. If it is subtle, equivocal, or you are unsure, mark "
        "'uncertain'. Otherwise mark 'absent'. Do NOT over-call: a clear/normal chest "
        "is the single most common result, so avoid reporting borderline findings as "
        "present.\n"
        'Return JSON with: "findings" (object mapping each finding name to '
        '{"presence":"present"|"absent"|"uncertain","confidence":int}), '
        '"primary_diagnosis" (string: the single most important finding marked '
        'present, or "No acute cardiopulmonary finding" if none), '
        '"confidence" (int 0-100), and "explanation" (string).'
    )
    return SYSTEM_PREAMBLE, u


def structured_soft(indication: str) -> tuple[str, str]:
    """Balanced scan with a softened presence bar. Audit of structured_balanced
    showed it routes visible-but-subtle findings (cardiomegaly, opacity,
    atelectasis — the 3 most-missed classes vs CheXbert) to 'uncertain', which
    is scored as a miss. This variant keeps the base-rate priming but tells the
    model to commit to 'present' whenever there is visible evidence, reserving
    'uncertain' for genuinely indeterminate reads."""
    checklist = ", ".join(STRUCTURED_FINDINGS)
    u = (
        f"Indication: {_indication(indication)}\n"
        "This is a chest radiograph from a general outpatient collection in which "
        "most studies are normal. Systematically assess for EACH of the following, "
        f"one at a time: {checklist}.\n"
        "Mark a finding 'present' if you can see visible evidence for it, even if "
        "mild — e.g. mark Cardiomegaly present if the cardiothoracic ratio exceeds "
        "0.5 on a frontal view, and mark subtle opacities or streaky atelectasis "
        "present if visible. Reserve 'uncertain' for findings you genuinely cannot "
        "decide on from the image, and 'absent' when the region looks normal. "
        "Do not invent findings that are not visible.\n"
        'Return JSON with: "findings" (object mapping each finding name to '
        '{"presence":"present"|"absent"|"uncertain","confidence":int}), '
        '"primary_diagnosis" (string: the single most important finding marked '
        'present, or "No acute cardiopulmonary finding" if none), '
        '"confidence" (int 0-100), and "explanation" (string).'
    )
    return SYSTEM_PREAMBLE, u


# Per-class discriminators for the 6 classes the model systematically OVER-calls
# vs CheXbert (n_pred >> n_true in the 754 audit): specific radiographic criteria
# that must be met before marking 'present'. Attacks BIAS (the error type that
# survives ensembling) rather than variance.
_OVERCALL_GUIDANCE = (
    "Apply these stricter criteria for findings that are commonly over-read — mark "
    "them 'present' ONLY if the specific criterion is met, otherwise 'absent':\n"
    "- Enlarged Cardiomediastinum: only if the mediastinal silhouette is genuinely "
    "widened; do NOT mark it merely because the heart is enlarged (that is "
    "Cardiomegaly). It is rarely a distinct finding.\n"
    "- Consolidation: only for dense, homogeneous airspace opacity with air "
    "bronchograms — not for any faint or patchy opacity (that is Lung Opacity).\n"
    "- Edema: only for interstitial/alveolar signs (Kerley lines, perihilar "
    "vascular congestion, cephalization) — not for isolated opacities.\n"
    "- Atelectasis: only for discrete linear/streaky/band-like opacity or volume "
    "loss — not for every basilar density.\n"
    "- Support Devices: only if you can actually see a tube, line, catheter, "
    "pacemaker, or wire — not inferred from clinical context.\n"
    "- Lung Opacity: reserve for a real focal or diffuse opacity; a clear lung is "
    "the most common result.\n"
)


def structured_targeted(indication: str) -> tuple[str, str]:
    """structured_soft's commit-if-visible recall, PLUS per-class discriminators for
    the 6 systematically over-called classes (the audit's bias source). Goal: keep
    soft's recall on the well-behaved classes while cutting the false positives that
    dominate the precision loss — the one lever left that targets bias, not variance."""
    checklist = ", ".join(STRUCTURED_FINDINGS)
    u = (
        f"Indication: {_indication(indication)}\n"
        "This is a chest radiograph from a general outpatient collection in which "
        "most studies are normal. Systematically assess for EACH of the following, "
        f"one at a time: {checklist}.\n"
        "Mark a finding 'present' if you can see visible evidence for it, even if "
        "mild — e.g. mark Cardiomegaly present if the cardiothoracic ratio exceeds "
        "0.5 on a frontal view. Reserve 'uncertain' for findings you genuinely "
        "cannot decide on, and 'absent' when the region looks normal. Do not invent "
        "findings that are not visible.\n"
        f"{_OVERCALL_GUIDANCE}"
        'Return JSON with: "findings" (object mapping each finding name to '
        '{"presence":"present"|"absent"|"uncertain","confidence":int}), '
        '"primary_diagnosis" (string: the single most important finding marked '
        'present, or "No acute cardiopulmonary finding" if none), '
        '"confidence" (int 0-100), and "explanation" (string).'
    )
    return SYSTEM_PREAMBLE, u


def evidence_structured(indication: str) -> tuple[str, str]:
    """Follow-up (EDIT 1): a MORE STRUCTURED version of evidence_first — the best
    existing condition. Same principle (identify visual evidence BEFORE concluding),
    but the visual-findings stage becomes a machine-evaluable table keyed by the
    canonical classes, so it can be scored directly without prose->synonym mapping
    (see normalize.structured_pred_set). Added as a NEW condition; `evidence_first`
    is left unchanged so Experiment 1 stays reproducible."""
    checklist = ", ".join(STRUCTURED_FINDINGS)
    u = (
        f"Indication: {_indication(indication)}\n"
        "Work in order and DO NOT let the indication decide the diagnosis — the "
        "diagnosis must follow from what you actually see.\n"
        "1. IMAGE QUALITY — projection, rotation, inspiration, exposure.\n"
        "2. VISUAL FINDINGS — review cardiomediastinal silhouette, lungs, pleura, "
        "bones, and devices, and for EACH of these findings record a status of "
        f"PRESENT / ABSENT / UNCERTAIN with (where applicable) laterality, location, "
        f"and the visual evidence: {checklist}.\n"
        "3. INTEGRATE the findings marked PRESENT.\n"
        "4. IMPRESSION — at most 3 abnormalities, taken only from step 2.\n"
        "5. CONFIDENCE.\n"
        'Return JSON with: "image_quality" (object: projection, rotation, '
        'inspiration, exposure), "findings" (object mapping each finding name to '
        '{"status":"PRESENT"|"ABSENT"|"UNCERTAIN","laterality":string,'
        '"location":string,"visual_evidence":string}), "impression" (array of at '
        'most 3 strings), "primary_diagnosis" (string; the most important PRESENT '
        'finding or "No acute cardiopulmonary finding"), "confidence" (int 0-100).'
    )
    return SYSTEM_PREAMBLE, u


# Exact region-based checklist requested by the user (distinct from
# STRUCTURED_FINDINGS, which uses CheXpert class names directly). Each item maps
# to one or more CheXpert classes for scoring — see CHECKLIST_TO_CHEXPERT below.
RADIOLOGY_CHECKLIST = [
    "Cardiomediastinal silhouette", "Lungs", "Pleura", "Airspace opacity",
    "Interstitial opacity", "Focal lesion", "Pneumothorax", "Effusion",
    "Atelectasis", "Edema", "Bones", "Devices",
]

# How each checklist item maps into the CheXpert label space for scoring. Items
# that are broad region headers (Lungs, Pleura) map to no single CheXpert class on
# their own — a PRESENT status there is a qualitative "something is wrong here"
# signal without a specific class, so they are excluded from _pred_set() scoring
# and only the SPECIFIC-finding items below are scored directly (Cardiomediastinal
# silhouette, Airspace/Interstitial opacity, Focal lesion, Pneumothorax, Effusion,
# Atelectasis, Edema, Bones, Devices all map to >=1 concrete class).
CHECKLIST_TO_CHEXPERT = {
    "Cardiomediastinal silhouette": {"Cardiomegaly", "Enlarged Cardiomediastinum"},
    "Airspace opacity": {"Lung Opacity", "Consolidation", "Pneumonia"},
    "Interstitial opacity": {"Lung Opacity", "Edema"},
    "Focal lesion": {"Lung Lesion"},
    "Pneumothorax": {"Pneumothorax"},
    "Effusion": {"Pleural Effusion"},
    "Atelectasis": {"Atelectasis"},
    "Edema": {"Edema"},
    "Bones": {"Fracture"},
    "Devices": {"Support Devices"},
}


def radiology_checklist(indication: str) -> tuple[str, str]:
    """Follow-up: the EXACT region-based checklist as specified (distinct
    checklist wording from evidence_structured, which uses CheXpert class names
    directly). Same principle — force PRESENT/ABSENT/UNCERTAIN + laterality/
    location/evidence per item BEFORE integrating into a diagnosis — but with the
    conventional radiology-read framing (region headers + specific finding types)
    rather than the project's internal taxonomy. Score Step 2 directly via
    normalize.structured_pred_set / CHECKLIST_TO_CHEXPERT, not via prose mapping."""
    checklist = "\n".join(f"{i}. {c}" for i, c in enumerate(RADIOLOGY_CHECKLIST, 1))
    u = (
        f"Indication: {_indication(indication)}\n"
        "Work through this standardized checklist in order. The diagnosis must "
        "follow from what you find in Step 2 — do NOT let the indication decide "
        "the diagnosis.\n\n"
        "Step 1 — Image quality: projection, rotation, inspiration, exposure.\n\n"
        f"Step 2 — Visual findings. For EACH of the following, one at a time:\n{checklist}\n"
        "For each: mark PRESENT / ABSENT / UNCERTAIN, and where applicable give "
        "laterality, location, and the specific visual evidence.\n\n"
        "Step 3 — Integrate the findings marked PRESENT.\n\n"
        "Step 4 — Impression: at most 3 abnormalities, drawn only from Step 2.\n\n"
        "Step 5 — Confidence.\n\n"
        'Return JSON with: "image_quality" (object: projection, rotation, '
        'inspiration, exposure), "findings" (object mapping each of the 12 '
        'checklist item names above to {"status":"PRESENT"|"ABSENT"|"UNCERTAIN",'
        '"laterality":string,"location":string,"visual_evidence":string}), '
        '"impression" (array of at most 3 strings), "primary_diagnosis" (string; '
        'the most important PRESENT finding or "No acute cardiopulmonary finding"), '
        '"confidence" (int 0-100), and "explanation" (string, one sentence).'
    )
    return SYSTEM_PREAMBLE, u


BUILDERS = {
    "direct": direct,
    "cot": cot,
    "ddx": ddx,
    "evidence_first": evidence_first,
    "uncertainty_first": uncertainty_first,
    "structured": structured,
    "structured_balanced": structured_balanced,
    "structured_soft": structured_soft,
    "structured_targeted": structured_targeted,
    "evidence_structured": evidence_structured,      # follow-up: structured evidence-first
    "radiology_checklist": radiology_checklist,      # follow-up: exact region-based checklist
}


def build(condition: str, indication: str) -> tuple[str, str]:
    return BUILDERS[condition](indication)


# --------------------------------------------------------------------------- #
# FREE-TEXT prompts for small local VLMs (e.g. MedGemma 4B).
# A 4B model's perception collapses under strict-JSON instructions (it defaults
# to "normal" or hallucinates), but reads competently in free-text. These keep
# EACH condition's reasoning scaffold but ask for a plain radiology narrative;
# findings are recovered downstream with negation-aware normalize_dx.
# --------------------------------------------------------------------------- #
_FT_SYS = (
    "You are a radiologist reading a chest X-ray. Report findings by their standard "
    "radiology names (e.g. cardiomegaly, pleural effusion, atelectasis, "
    "consolidation, pulmonary edema, pneumothorax, lung opacity, fracture, support "
    "device). Report only what the image supports."
)

# Shared systematic-review requirement. The 4B model defaults to "normal" whenever
# a prompt offers an easy out; forcing an explicit region-by-region pass makes
# EVERY condition actually engage the image (the fix for direct/uncertainty_first
# scoring 0). Each condition then layers its own distinct reasoning scaffold on
# top, so the comparison still measures reasoning FORMAT, not "did it enumerate".
_FT_SCAN = ("Examine each region in turn — heart and mediastinum, both lungs, "
            "pleura and costophrenic angles, bones, and any devices — and name "
            "each abnormality you can see. ")


def _ft(condition: str, indication: str) -> tuple[str, str]:
    ind = _indication(indication)
    scaffold = {
        "direct":
            _FT_SCAN + "Then give your single most likely primary diagnosis.",
        "cot":
            "Reason step by step: for each region, state what you observe and "
            "whether it is normal or abnormal, working through heart/mediastinum, "
            "lungs, pleura, bones, devices. Then give your overall impression and "
            "the primary diagnosis.",
        "ddx":
            _FT_SCAN + "Then list a ranked differential of the most likely "
            "diagnoses, and state the single most likely one with its supporting "
            "findings.",
        "evidence_first":
            "STEP 1 — list every visual finding you can see, region by region "
            "(heart/mediastinum, lungs, pleura, bones, devices), each as "
            "present or absent. STEP 2 — using only that evidence, name the "
            "abnormal findings and the primary diagnosis.",
        "uncertainty_first":
            _FT_SCAN + "For each abnormality, say whether you are confident or "
            "it is equivocal. Then give a calibrated primary diagnosis, naming "
            "every finding you do see even if mild.",
        "evidence_structured":
            "First note image quality (projection, rotation, inspiration, exposure). "
            "Then " + _FT_SCAN.lower() +
            "For each abnormality give its status (present/absent/uncertain), "
            "laterality, location, and the visual evidence. Then an impression of at "
            "most 3 abnormalities, and the primary diagnosis based only on what you "
            "saw. Do not infer the diagnosis from the indication.",
        "radiology_checklist":
            "First note image quality (projection, rotation, inspiration, exposure). "
            "Then go through this checklist one item at a time: cardiomediastinal "
            "silhouette, lungs, pleura, airspace opacity, interstitial opacity, "
            "focal lesion, pneumothorax, effusion, atelectasis, edema, bones, "
            "devices — for each say present/absent/uncertain, laterality, location, "
            "and the visual evidence. Then an impression of at most 3 abnormalities "
            "drawn only from that checklist, and the primary diagnosis based only on "
            "what you saw. Do not infer the diagnosis from the indication.",
        "perception_only":
            "Examine this chest radiograph systematically. " + _FT_SCAN +
            "For each finding give its status and the visual evidence. Do NOT give "
            "an overall diagnosis or impression — report only the individual "
            "findings; a separate step will reason from them later.",
    }[condition]
    return _FT_SYS, f"Indication: {ind}\n{scaffold}"


def build_freetext(condition: str, indication: str) -> tuple[str, str]:
    """Free-text variant of the five pre-registered conditions (no JSON demand)."""
    return _ft(condition, indication)


# --------------------------------------------------------------------------- #
# TWO-STAGE perception/reasoning separation (reviewer follow-up).
#
# evidence_structured still generates the diagnosis in the SAME pass as the
# findings, so a wrong answer could come from bad perception (didn't see it) or
# bad reasoning (saw it, concluded wrong) and the single response can't
# distinguish them. This splits it into two INDEPENDENT calls:
#   Stage A (perception_only):     image -> findings ONLY, diagnosis forbidden.
#   Stage B (diagnose_from_findings): frozen Stage-A findings, as TEXT, with NO
#     image access -> diagnosis only.
# Scoring Stage A's findings directly measures perception; scoring Stage B's
# diagnosis (produced from frozen, already-graded findings) measures reasoning
# conditioned on whatever was actually perceived.
# --------------------------------------------------------------------------- #

def perception_only(indication: str) -> tuple[str, str]:
    """Stage A: image -> structured findings. No diagnosis, impression, or
    interpretation permitted — only per-finding observations."""
    checklist = ", ".join(STRUCTURED_FINDINGS)
    u = (
        f"Indication: {_indication(indication)}\n"
        "Examine this chest radiograph systematically. For EACH of the following "
        f"findings, decide present / absent / uncertain, with the specific visual "
        f"evidence: {checklist}.\n"
        "Do NOT state an overall diagnosis, impression, or interpretation. Report "
        "ONLY the individual findings — a later, separate step will reason from "
        "them.\n"
        'Return JSON with: "findings" (object mapping each finding name to '
        '{"presence":"present"|"absent"|"uncertain","visual_evidence":string,'
        '"confidence":int}), "primary_diagnosis" (always the literal string '
        '"DEFERRED" — do not diagnose here), "confidence" (int 0-100, your overall '
        'certainty in the findings above), "explanation" (string, one line on image '
        'quality only, no diagnostic content).'
    )
    return SYSTEM_PREAMBLE, u


BUILDERS["perception_only"] = perception_only   # two-stage Stage A: findings, no dx


_DIAGNOSE_SYSTEM = (
    "You are a radiologist reasoning from a colleague's documented findings. You do "
    "NOT have access to the original image — reason ONLY from the findings listed "
    "below. Return ONLY valid JSON matching the requested fields."
)


def _serialize_findings(findings: dict) -> str:
    """Render a Stage-A findings dict as plain text for the Stage-B text-only call."""
    lines = []
    for name in STRUCTURED_FINDINGS:
        f = findings.get(name)
        if not isinstance(f, dict):
            lines.append(f"- {name}: not assessed")
            continue
        pres = str(f.get("presence", "absent")).upper()
        ev = f.get("visual_evidence", "")
        lines.append(f"- {name}: {pres}" + (f" (evidence: {ev})" if ev and pres == "PRESENT" else ""))
    return "\n".join(lines)


def diagnose_from_findings_cot(findings: dict, indication: str) -> tuple[str, str]:
    """Stage B, ELABORATED reasoning (Experiment 5). Identical inputs to
    `diagnose_from_findings` — the same frozen findings, still no image — but the
    model must reason explicitly before committing: weigh the findings, consider
    what they are collectively consistent with, name and dismiss alternatives, and
    only then conclude. This is the "strong reasoning" level of the 2x2; the visual
    factor is set upstream by which Stage-A run supplies the findings."""
    u = (
        f"Indication: {_indication(indication)}\n"
        "The following findings were documented from systematic review of a chest "
        f"radiograph:\n{_serialize_findings(findings)}\n\n"
        "Reason step by step before concluding, using ONLY these findings:\n"
        "1. Which findings are present, and how do they relate to one another?\n"
        "2. What single clinical picture are they collectively most consistent with?\n"
        "3. What alternative would explain them, and why is it less likely?\n"
        "4. Only then, commit to the primary diagnosis (or \"No acute "
        "cardiopulmonary finding\" if all findings are absent).\n"
        "Do not assume anything not listed.\n"
        'Return JSON with: "reasoning" (string, steps 1-3), "primary_diagnosis" '
        '(string), "confidence" (int 0-100), "explanation" (string).'
    )
    return _DIAGNOSE_SYSTEM, u


def diagnose_from_findings(findings: dict, indication: str) -> tuple[str, str]:
    """Stage B: frozen findings (text) -> diagnosis. No image is sent with this
    call — the caller must use a text-only backend path."""
    u = (
        f"Indication: {_indication(indication)}\n"
        "The following findings were documented from systematic review of a chest "
        f"radiograph:\n{_serialize_findings(findings)}\n\n"
        "Based ONLY on these findings — do not assume anything not listed — state "
        "the primary diagnosis (or \"No acute cardiopulmonary finding\" if all "
        "findings are absent), a one-sentence explanation citing the specific "
        "findings that support it, and your confidence.\n"
        'Return JSON with: "primary_diagnosis" (string), "confidence" (int 0-100), '
        '"explanation" (string).'
    )
    return _DIAGNOSE_SYSTEM, u
