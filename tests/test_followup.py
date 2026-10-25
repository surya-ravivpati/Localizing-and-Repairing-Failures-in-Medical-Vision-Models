"""Tests for the follow-up edits: structured evidence-first parsing, 3-labeler
consensus, and reference-confidence levels. API-free. Run: pytest tests/test_followup.py -q
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import prompts
from src.normalize import structured_pred_set, predicted_positive_set
from src.reference_confidence import finding_record, agreement_level, cell_mask_factory


# ---- EDIT 1: new condition added, existing ones preserved ----
def test_evidence_first_unchanged_and_structured_added():
    assert "evidence_first" in prompts.BUILDERS            # untouched (Exp 1)
    assert "evidence_structured" in prompts.BUILDERS       # new
    _, u = prompts.build("evidence_structured", "chest pain")
    assert "IMAGE QUALITY" in u and "PRESENT" in u and "JSON" in u
    _, uf = prompts.build_freetext("evidence_structured", "")  # MedGemma path
    assert uf and "JSON" not in uf


# ---- EDIT 2: structured status parsing ----
def test_structured_status_present_absent_uncertain():
    resp = {"findings": {
        "Cardiomegaly": {"status": "PRESENT"},
        "Edema": {"status": "ABSENT"},
        "Atelectasis": {"status": "UNCERTAIN"},
        "Pleural Effusion": {"presence": "present"}}}       # both keys supported
    got = structured_pred_set(resp)
    assert got == {"Cardiomegaly", "Pleural Effusion"}      # not absent/uncertain
    assert "Atelectasis" in structured_pred_set(resp, include_uncertain=True)

def test_structured_case_insensitive_and_existing_eval_reads_status():
    resp = {"findings": {"Consolidation": {"status": "present"}}}
    assert structured_pred_set(resp) == {"Consolidation"}
    # the EXISTING framework predictor also reads `status` now (feeds in directly)
    assert "Consolidation" in predicted_positive_set(resp)

def test_parse_failure_is_never_positive():
    assert structured_pred_set({}) == set()                 # no findings object
    assert structured_pred_set({"findings": "garbage"}) == set()
    assert structured_pred_set({"findings": {"X": "notadict"}}) == set()
    assert structured_pred_set({"findings": {"Edema": {"status": ""}}}) == set()
    assert structured_pred_set(None) == set()


# ---- EDIT 3: 3-labeler consensus + confidence ----
def _row(**kw):
    return pd.Series(kw)

def test_consensus_high_medium_low_three_labelers():
    pre = ["lbl_", "lblcx_", "lblvcx_"]
    # 3/3 present -> HIGH
    r = _row(**{"lbl_Edema": 1.0, "lblcx_Edema": 1.0, "lblvcx_Edema": 1.0})
    rec = finding_record(r, "Edema", pre)
    assert rec["confidence"] == "HIGH" and rec["consensus"] == "PRESENT" and rec["agreement"] == "3/3"
    assert rec["rule_based"] == "PRESENT" and rec["visualchexbert"] == "PRESENT"   # preserved
    # 2/3 present -> MEDIUM
    r = _row(**{"lbl_Cardiomegaly": 1.0, "lblcx_Cardiomegaly": 1.0, "lblvcx_Cardiomegaly": 0.0})
    rec = finding_record(r, "Cardiomegaly", pre)
    assert rec["confidence"] == "MEDIUM" and rec["consensus"] == "PRESENT" and rec["agreement"] == "2/3"
    # individual labels are never discarded
    assert rec["visualchexbert"] == "ABSENT"

def test_two_labeler_degradation():
    pre = ["lbl_", "lblcx_"]
    r = _row(**{"lbl_Edema": 1.0, "lblcx_Edema": 1.0})
    assert agreement_level(r, "Edema", pre) == "HIGH"       # both agree
    r = _row(**{"lbl_Edema": 1.0, "lblcx_Edema": 0.0})
    assert agreement_level(r, "Edema", pre) == "MEDIUM"     # split 1/1 -> majority tie


# ---- EDIT 4: exact-level masking ----
def test_cell_mask_exact_level():
    s = pd.DataFrame([{"uid": 1, "lbl_Edema": 1.0, "lblcx_Edema": 1.0, "lblvcx_Edema": 1.0,
                       "lbl_Cardiomegaly": 1.0, "lblcx_Cardiomegaly": 1.0, "lblvcx_Cardiomegaly": 0.0}])
    high = cell_mask_factory(s, "HIGH", exact=True)
    med = cell_mask_factory(s, "MEDIUM", exact=True)
    assert high(1, "Edema") and not high(1, "Cardiomegaly")
    assert med(1, "Cardiomegaly") and not med(1, "Edema")


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"])
